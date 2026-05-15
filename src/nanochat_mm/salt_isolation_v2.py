"""Per-user salt isolation — v2, with sharded perceptual embedding.

The v1 probe (experiment #9) found that salting the Engram hash alone is
not sufficient: the shared `vis_perc_emb`/`aud_perc_emb` table is keyed
by raw code, so user A's insertion at code C and user B's insertion at
the same code C write to the same row. That row's gradient leaks the
marker across users regardless of the Engram salt.

v2 fix: shard the perceptual embedding table per user bucket. The
effective row index becomes `bucket * V + code`, where the bucket is
derived from the user salt. Both the Engram hash *and* the perc-emb
lookup are now salt-aware, so two users at the same code map to fully
disjoint storage in both the gate (Engram table) and the input
embedding (perc_emb table).

This is a minimally invasive change implemented as a thin subclass of
QwenEngramBolt; the rest of the pipeline (generic-NTP pretrain,
surgical insertion, evaluation) is reused.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity
from qwen_engram_bolt import (
    QwenEngramBolt, build_fixed_context, get_touched_rows, MODEL_ID, DEVICE,
)
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)

SALT_USER_A = 0xCAFEBABE
SALT_USER_B = 0xDEADC0DE
N_USER_BUCKETS = 8


class SaltAwareBolt(QwenEngramBolt):
    """QwenEngramBolt with per-user-bucket sharded perceptual embeddings.

    The vis/aud perceptual embedding tables are inflated to V * n_buckets
    rows. The current user bucket is set via `set_user_bucket(salt)`,
    which routes lookups for code `c` to row `bucket * V + c`.
    """
    def __init__(self, *args, n_user_buckets=N_USER_BUCKETS, **kw):
        super().__init__(*args, **kw)
        self.n_user_buckets = n_user_buckets
        V_vis = self.vis_perc_emb.num_embeddings
        V_aud = self.aud_perc_emb.num_embeddings
        self.V_vis_per_user = V_vis
        self.V_aud_per_user = V_aud

        old_vis_w = self.vis_perc_emb.weight.detach().clone()
        old_aud_w = self.aud_perc_emb.weight.detach().clone()
        new_vis = nn.Embedding(V_vis * n_user_buckets, self.hidden_size)
        new_aud = nn.Embedding(V_aud * n_user_buckets, self.hidden_size)
        with torch.no_grad():
            for b in range(n_user_buckets):
                new_vis.weight[b * V_vis:(b + 1) * V_vis] = old_vis_w
                new_aud.weight[b * V_aud:(b + 1) * V_aud] = old_aud_w
        new_vis.to(dtype=torch.bfloat16, device=DEVICE)
        new_aud.to(dtype=torch.bfloat16, device=DEVICE)
        self.vis_perc_emb = new_vis
        self.aud_perc_emb = new_aud
        self._user_bucket = 0

    def set_user_bucket(self, salt: int):
        # Reduce salt to a bucket in [0, n_user_buckets). Salt 0 → bucket 0
        # so global (unsalted) operation is preserved.
        if int(salt) == 0:
            self._user_bucket = 0
        else:
            # Mix the salt with a prime so neighbouring salts don't collide.
            mixed = (int(salt) ^ (int(salt) >> 16)) * 2654435761
            self._user_bucket = int(mixed) % self.n_user_buckets

    def build_inputs_embeds(self, input_ids, modality_ids):
        B, T = input_ids.shape
        device = input_ids.device
        emb = torch.zeros(B, T, self.hidden_size, device=device, dtype=torch.bfloat16)
        m_text = (modality_ids == MODALITY_TEXT)
        m_vis = (modality_ids == MODALITY_VISION)
        m_aud = (modality_ids == MODALITY_AUDIO)
        if m_text.any():
            text_ids = torch.where(m_text, input_ids, torch.zeros_like(input_ids))
            text_emb = self.qwen.get_input_embeddings()(text_ids)
            emb = emb + m_text.unsqueeze(-1).to(emb.dtype) * text_emb
        if m_vis.any():
            vis_ids = torch.where(m_vis, input_ids, torch.zeros_like(input_ids))
            vis_ids = vis_ids + self._user_bucket * self.V_vis_per_user
            vis_emb = self.vis_perc_emb(vis_ids)
            emb = emb + m_vis.unsqueeze(-1).to(emb.dtype) * vis_emb
        if m_aud.any():
            aud_ids = torch.where(m_aud, input_ids, torch.zeros_like(input_ids))
            aud_ids = aud_ids + self._user_bucket * self.V_aud_per_user
            aud_emb = self.aud_perc_emb(aud_ids)
            emb = emb + m_aud.unsqueeze(-1).to(emb.dtype) * aud_emb
        return emb


def insert_with_salt(bolt, code, modality_id, marker, tok, salt,
                      max_steps=60, lr=1.0, T=24):
    """Surgical insertion with both Engram-hash salt and bucketed perc_emb."""
    eng = bolt.engram.engrams[str(modality_id)]
    eng.user_salt = salt
    bolt.set_user_bucket(salt)
    eng.reset_cache()

    input_ids, modality_ids_t = build_fixed_context(code, modality_id, tok, marker, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids_t = modality_ids_t.to(DEVICE)
    touched = get_touched_rows(eng, code, input_ids)
    if modality_id == MODALITY_VISION:
        perc_emb_param = bolt.vis_perc_emb.weight
        V_per_user = bolt.V_vis_per_user
    else:
        perc_emb_param = bolt.aud_perc_emb.weight
        V_per_user = bolt.V_aud_per_user
    bucket = bolt._user_bucket
    bucketed_row = bucket * V_per_user + int(code)

    params_to_opt = [eng.tables[ks].embedding.weight for ks in touched] + [perc_emb_param]
    opt = torch.optim.SGD(params_to_opt, lr=lr, momentum=0.0)

    target = torch.tensor([marker], dtype=torch.long, device=DEVICE)
    last_loss = float("inf"); steps_taken = 0
    for step in range(max_steps):
        logits = bolt(input_ids, modality_ids_t)
        last = logits[:, -1, :]
        loss = F.cross_entropy(last, target)
        last_loss = float(loss.item())
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            # Engram: only touched rows
            for ks, rows in touched.items():
                W = eng.tables[ks].embedding.weight
                if W.grad is None: continue
                mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.grad.dtype)
                row_idx = torch.tensor(sorted(rows), device=W.device, dtype=torch.long)
                mask[row_idx] = 1.0
                W.grad.mul_(mask)
            # Perc-emb: ONLY the bucketed row for this code, not the unbucketed one.
            if perc_emb_param.grad is not None:
                pmask = torch.zeros(perc_emb_param.shape[0], 1,
                                     device=perc_emb_param.device,
                                     dtype=perc_emb_param.grad.dtype)
                pmask[bucketed_row] = 1.0
                perc_emb_param.grad.mul_(pmask)
        opt.step()
        steps_taken = step + 1
        if last_loss < 0.5:
            break

    eng.user_salt = 0
    bolt.set_user_bucket(0)
    eng.reset_cache()
    return steps_taken, last_loss


def query_with_salt(bolt, code, modality_id, tok, salt, markers, T=24):
    eng = bolt.engram.engrams[str(modality_id)]
    eng.user_salt = salt
    bolt.set_user_bucket(salt)
    eng.reset_cache()
    inp, mids = build_fixed_context(code, modality_id, tok, marker_text_id=0, T=T)
    inp = inp.to(DEVICE); mids = mids.to(DEVICE)
    with torch.no_grad():
        logits = bolt(inp, mids)
        marker_logits = torch.stack([logits[0, -1, m] for m in markers])
        pred_idx = int(marker_logits.argmax().item())
    eng.user_salt = 0
    bolt.set_user_bucket(0)
    eng.reset_cache()
    return pred_idx


def main():
    print("=" * 70)
    print("Per-user salt isolation v2 (sharded perc_emb)")
    print("=" * 70)
    print(f"\nN user buckets: {N_USER_BUCKETS}")

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri_large.npz")
    _, _, ev_emb, ev_pid = split_by_identity(aud['emb'], aud['pid'])
    K = 64
    apply_fn = fit_naive_rq(aud['emb'], n_levels=1, k_per=K)

    bolt = SaltAwareBolt(qwen, tok, V_vis=K, V_aud=K, engram_attach_layer=24,
                          n_user_buckets=N_USER_BUCKETS).to(DEVICE)
    bolt.install_hook()
    print("[pretrain] generic-NTP 200 steps at bucket 0 ...")
    pretrain_generic(bolt, aud['emb'], aud['pid'], apply_fn, MODALITY_AUDIO, tok,
                     n_steps=200, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    by_id = defaultdict(list)
    for i, p in enumerate(ev_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    user_a_ids = ids_sorted[:5]
    user_b_ids = ids_sorted[5:10]
    marker_offset_a = 30001
    marker_offset_b = 30001  # SAME marker offsets; isolation MUST come from salt
    print(f"\nUser A: {len(user_a_ids)} identities, salt={hex(SALT_USER_A)}")
    print(f"User B: {len(user_b_ids)} identities, salt={hex(SALT_USER_B)}")
    print(f"Both use marker offset {marker_offset_a}; rely on salt isolation.")
    print(f"  → bucket A = {(SALT_USER_A ^ (SALT_USER_A >> 16)) * 2654435761 % N_USER_BUCKETS}")
    print(f"  → bucket B = {(SALT_USER_B ^ (SALT_USER_B >> 16)) * 2654435761 % N_USER_BUCKETS}\n")

    rng = np.random.default_rng(42)
    for k, pid in enumerate(user_a_ids):
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_code_arr = apply_fn(ev_emb[idxs[0]][None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        insert_with_salt(bolt, reg_code, MODALITY_AUDIO, marker_offset_a + k, tok, SALT_USER_A)
    for k, pid in enumerate(user_b_ids):
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_code_arr = apply_fn(ev_emb[idxs[0]][None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        insert_with_salt(bolt, reg_code, MODALITY_AUDIO, marker_offset_b + k, tok, SALT_USER_B)

    results = {}
    for user_label, user_ids, my_salt, my_offset, other_salt in [
        ("A", user_a_ids, SALT_USER_A, marker_offset_a, SALT_USER_B),
        ("B", user_b_ids, SALT_USER_B, marker_offset_b, SALT_USER_A),
    ]:
        my_markers = list(range(my_offset, my_offset + len(user_ids)))
        correct_in = 0; correct_cross = 0; total = 0
        for k, pid in enumerate(user_ids):
            idxs = list(by_id[pid]); rng.shuffle(idxs)
            for qi in idxs[1:4]:
                q_code_arr = apply_fn(ev_emb[qi][None])[0]
                q_code = int(q_code_arr.item() if hasattr(q_code_arr, 'item') else q_code_arr)
                pred_in = query_with_salt(bolt, q_code, MODALITY_AUDIO, tok,
                                          my_salt, my_markers)
                pred_cross = query_with_salt(bolt, q_code, MODALITY_AUDIO, tok,
                                             other_salt, my_markers)
                if pred_in == k: correct_in += 1
                if pred_cross == k: correct_cross += 1
                total += 1
        in_acc = correct_in / total if total else 0
        cross_acc = correct_cross / total if total else 0
        print(f"User {user_label}: in-salt retr = {in_acc:.3f}, cross-salt leak = {cross_acc:.3f}  "
              f"(chance = {1 / len(user_ids):.3f})")
        results[user_label] = {
            "in_salt_retrieval": in_acc,
            "cross_salt_retrieval": cross_acc,
            "chance": 1 / len(user_ids),
        }

    results["config"] = {
        "n_user_buckets": N_USER_BUCKETS,
        "fix": "sharded perc_emb + engram hash salt",
        "salt_A_hex": hex(SALT_USER_A),
        "salt_B_hex": hex(SALT_USER_B),
    }
    out = Path("/home/ubuntu/multimodal-user-memory/results/salt_isolation_v2.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {out}")
    print("\nINTERPRETATION:")
    print("  in-salt >> cross-salt → salt provides isolation")
    print("  cross-salt ≈ chance   → users are fully isolated")
    print("  cross-salt ≈ in-salt  → leak persists (v1 result)")


if __name__ == "__main__":
    sys.exit(main())
