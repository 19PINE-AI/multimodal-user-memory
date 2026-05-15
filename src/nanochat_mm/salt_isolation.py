"""Per-user salt isolation test.

Setup: two users (A and B). Each has their own set of identities to
register. They share the same Engram physically but use distinct
per-user salts in the hash so their effective address spaces are
disjoint.

Test: register user A's identities with salt_A; register user B's
identities with salt_B. Query at salt_A should retrieve user A's
identities ONLY (and give chance for user B's, since B's markers were
inserted at different rows). Vice versa.

This validates the multi-tenant property: per-user salt gives clean
isolation without per-user model copies.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_AUDIO
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity
from qwen_engram_bolt import (
    QwenEngramBolt, surgical_insert, build_fixed_context, MODEL_ID, DEVICE,
)
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)


SALT_USER_A = 0xCAFEBABE
SALT_USER_B = 0xDEADC0DE


def query_with_salt(bolt, code, modality_id, tok, salt, marker_offset, N_registered):
    eng = bolt.engram.engrams[str(modality_id)]
    eng.user_salt = salt
    eng.reset_cache()
    inp, mids = build_fixed_context(code, modality_id, tok, marker_text_id=0, T=24)
    inp = inp.to(DEVICE); mids = mids.to(DEVICE)
    markers = list(range(marker_offset, marker_offset + N_registered))
    with torch.no_grad():
        logits = bolt(inp, mids)
        marker_logits = torch.stack([logits[0, -1, m] for m in markers])
        pred_idx = int(marker_logits.argmax().item())
    eng.user_salt = 0  # reset
    eng.reset_cache()
    return pred_idx


def insert_with_salt(bolt, code, modality_id, marker, tok, salt, max_steps=60):
    eng = bolt.engram.engrams[str(modality_id)]
    eng.user_salt = salt
    eng.reset_cache()
    surgical_insert(bolt, code, modality_id, marker, tok, max_steps=max_steps, lr=1.0, T=24)
    eng.user_salt = 0
    eng.reset_cache()


def main():
    print("=" * 70)
    print("Per-user salt isolation test")
    print("=" * 70)

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

    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K, engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    print("[pretrain] generic-NTP 200 steps ...")
    pretrain_generic(bolt, aud['emb'], aud['pid'], apply_fn, MODALITY_AUDIO, tok,
                      n_steps=200, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    by_id = defaultdict(list)
    for i, p in enumerate(ev_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())

    # Assign first 5 identities to user A, next 5 to user B
    user_a_ids = ids_sorted[:5]
    user_b_ids = ids_sorted[5:10]
    marker_offset_a = 30001
    marker_offset_b = 30001  # SAME marker offsets — collision possible without salt isolation
    # The salt should give disjoint address spaces so this is fine.

    print(f"\nUser A: {len(user_a_ids)} identities, salt={hex(SALT_USER_A)}")
    print(f"User B: {len(user_b_ids)} identities, salt={hex(SALT_USER_B)}")
    print(f"Both use marker offset 30001 — relying on salt isolation.\n")

    rng = np.random.default_rng(42)
    # ---- Register all of A's identities ----
    for k, pid in enumerate(user_a_ids):
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_code_arr = apply_fn(ev_emb[idxs[0]][None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        insert_with_salt(bolt, reg_code, MODALITY_AUDIO, marker_offset_a + k, tok, SALT_USER_A)
    # ---- Register all of B's identities ----
    for k, pid in enumerate(user_b_ids):
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_code_arr = apply_fn(ev_emb[idxs[0]][None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        insert_with_salt(bolt, reg_code, MODALITY_AUDIO, marker_offset_b + k, tok, SALT_USER_B)

    # ---- Query: each user's identities should be retrievable with their own salt ----
    # And cross-salt queries should give "random" results
    results = {}
    for user_label, user_ids, my_salt, my_offset, other_salt, other_offset in [
        ("A", user_a_ids, SALT_USER_A, marker_offset_a, SALT_USER_B, marker_offset_b),
        ("B", user_b_ids, SALT_USER_B, marker_offset_b, SALT_USER_A, marker_offset_a),
    ]:
        correct_in_user = 0; total = 0
        correct_cross = 0
        for k, pid in enumerate(user_ids):
            idxs = list(by_id[pid]); rng.shuffle(idxs)
            # 3 queries per identity (not the registration sample)
            for qi in idxs[1:4]:
                q_code_arr = apply_fn(ev_emb[qi][None])[0]
                q_code = int(q_code_arr.item() if hasattr(q_code_arr, 'item') else q_code_arr)
                # Query with own salt (in-user) — should retrieve k correctly
                pred = query_with_salt(bolt, q_code, MODALITY_AUDIO, tok,
                                          my_salt, my_offset, len(user_ids))
                if pred == k: correct_in_user += 1
                # Query with OTHER user's salt (cross-user) — should give ~chance
                pred_cross = query_with_salt(bolt, q_code, MODALITY_AUDIO, tok,
                                                 other_salt, other_offset, len(user_ids))
                if pred_cross == k: correct_cross += 1
                total += 1
        in_acc = correct_in_user / total if total else 0
        cross_acc = correct_cross / total if total else 0
        print(f"User {user_label}: in-salt retrieval = {in_acc:.3f}, cross-salt 'leak' = {cross_acc:.3f}  "
              f"(chance = {1/len(user_ids):.3f})")
        results[user_label] = {
            "in_salt_retrieval": in_acc,
            "cross_salt_retrieval": cross_acc,
            "chance": 1 / len(user_ids),
        }

    out = Path("/home/ubuntu/multimodal-user-memory/results/salt_isolation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {out}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print(" If in-salt retrieval >> cross-salt 'leak', salt isolation works.")
    print(" If cross-salt is at chance, users are properly isolated.")


if __name__ == "__main__":
    sys.exit(main())
