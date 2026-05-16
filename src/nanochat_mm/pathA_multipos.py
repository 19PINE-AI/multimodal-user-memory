"""Multi-position perceptual codes — expand Path A's address space from K to K^T_perc.

First-principles: Path A's hash address space is determined by the n-gram
hash over input positions. With one perceptual code at one position, the
address space reduces to K. With T_perc perceptual codes at consecutive
positions (e.g., a residual quantization of the embedding into T_perc
sub-codes), the address space becomes K^T_perc — matching or exceeding N
without any K-saturation.

For N=1000 IDs:
  K=32, T_perc=2 → 1024 addresses  (✓ matches N exactly)
  K=16, T_perc=3 → 4096 addresses  (4× headroom)
  K=8,  T_perc=4 → 4096 addresses  (4× headroom, smaller per-code)

The intra-id consistency requirement also changes: instead of one code
matching, we need at least 2 of T_perc codes to match for a partial hit.
The Engram's n-gram hash is naturally robust to per-position drops
because of its multi-head structure (different heads use different
n-gram orders).

This script:
  1. Builds T_perc-level residual quantizer on the train embeddings.
  2. Modifies build_fixed_context to use T_perc perceptual positions.
  3. Runs Path A at large N with this expanded address space.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import (
    MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO,
)
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import QwenEngramBolt, MODEL_ID, DEVICE
from real_encoder_train import fit_naive_rq

torch.manual_seed(42); np.random.seed(42)


def fit_residual_rq(emb_np, n_levels, k_per, seed=42):
    """Returns apply_fn: emb → codes of shape [N, n_levels]."""
    return fit_naive_rq(emb_np, n_levels=n_levels, k_per=k_per, seed=seed)


def build_multipos_context(codes, modality_id, tok, marker_text_id, T=24):
    """Like build_fixed_context but places T_perc perceptual codes at the
    last T_perc positions of the context. codes: shape [T_perc] of ints
    each in [0, K).
    """
    codes = list(codes)
    T_perc = len(codes)
    prompt = "You see"
    pref_ids = tok.encode(prompt, add_special_tokens=False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    # Reserve last T_perc positions for perceptual codes
    pref = list(pref_ids) + [pad_id] * (T - T_perc - len(pref_ids))
    pref = pref[: T - T_perc]
    input_ids = pref + [int(c) for c in codes]
    mids = [MODALITY_TEXT] * (T - T_perc) + [int(modality_id)] * T_perc
    return (torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            torch.tensor(mids, dtype=torch.long).unsqueeze(0))


def surgical_insert_multipos(bolt, codes, modality_id, marker_text_id,
                              tok, max_steps=80, lr=1.0, T=24):
    """Surgical insertion with multi-position codes."""
    from qwen_engram_bolt import get_touched_rows
    eng = bolt.engram.engrams[str(modality_id)]
    input_ids, modality_ids = build_multipos_context(
        codes, modality_id, tok, marker_text_id, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
    # get_touched_rows uses the LAST position by default; with T_perc multiple
    # positions we want all of them — sum of touched rows
    touched_all = defaultdict(set)
    inp_np = input_ids.cpu().numpy()
    hashes_per_layer = eng.hash_mapping.hash_all_layers(inp_np, user_salt=int(eng.user_salt))
    T_perc = len(codes)
    for lid, h in hashes_per_layer.items():
        tbl = eng.tables[str(lid)]
        offsets = tbl.offsets.cpu().numpy()
        for pos in range(T - T_perc, T):
            local = h[0, pos]
            global_rows = local + offsets
            touched_all[str(lid)].update(int(r) for r in global_rows.tolist())

    if modality_id == MODALITY_VISION:
        perc_emb_param = bolt.vis_perc_emb.weight
    else:
        perc_emb_param = bolt.aud_perc_emb.weight
    params = [eng.tables[ks].embedding.weight for ks in touched_all] + [perc_emb_param]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.0)

    target = torch.tensor([marker_text_id], dtype=torch.long, device=DEVICE)
    last_loss = float("inf"); steps_taken = 0
    for step in range(max_steps):
        logits = bolt(input_ids, modality_ids)
        loss = F.cross_entropy(logits[:, -1, :], target)
        last_loss = float(loss.item())
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            for ks, rows in touched_all.items():
                W = eng.tables[ks].embedding.weight
                if W.grad is None: continue
                mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.grad.dtype)
                mask[torch.tensor(sorted(rows), device=W.device, dtype=torch.long)] = 1.0
                W.grad.mul_(mask)
            if perc_emb_param.grad is not None:
                pmask = torch.zeros(perc_emb_param.shape[0], 1,
                                     device=perc_emb_param.device,
                                     dtype=perc_emb_param.grad.dtype)
                for c in codes:
                    pmask[int(c)] = 1.0
                perc_emb_param.grad.mul_(pmask)
        opt.step()
        steps_taken = step + 1
        if last_loss < 0.5: break
    return steps_taken, last_loss


def evaluate_multipos(bolt, apply_fn, ev_emb, ev_pid, modality_id, tok,
                      *, T_perc, K, N_subset=None, n_queries_per_id=3,
                      max_steps=60, lr=1.0, T=24, marker_offset=30001):
    by_id = defaultdict(list)
    for i, p in enumerate(ev_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None: ids_sorted = ids_sorted[:N_subset]
    marker_ids = list(range(marker_offset, marker_offset + len(ids_sorted)))
    markers = {pid: marker_ids[i] for i, pid in enumerate(ids_sorted)}

    eng = bolt.engram.engrams[str(modality_id)]
    eng_snap = {ks: tbl.embedding.weight.detach().clone() for ks, tbl in eng.tables.items()}
    perc_snap = (bolt.aud_perc_emb if modality_id == MODALITY_AUDIO else bolt.vis_perc_emb).weight.detach().clone()

    # Pre-quantize all eval embeddings
    all_codes = apply_fn(ev_emb)  # shape [N, T_perc]
    if all_codes.ndim == 1:
        all_codes = all_codes[:, None]

    rng = np.random.default_rng(99)
    register_codes = {}
    insert_stats = []
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_codes = all_codes[idxs[0]]
        register_codes[pid] = reg_codes
        steps, fl = surgical_insert_multipos(
            bolt, reg_codes, modality_id, markers[pid], tok,
            max_steps=max_steps, lr=lr, T=T,
        )
        insert_stats.append((steps, fl))

    correct = 0; total = 0
    code_match_t = 0; code_match_c = 0
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        q_idxs = idxs[1:1 + n_queries_per_id]
        for qi in q_idxs:
            q_codes = all_codes[qi]
            input_ids, modality_ids = build_multipos_context(
                q_codes, modality_id, tok, marker_text_id=0, T=T)
            input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
            with torch.no_grad():
                logits = bolt(input_ids, modality_ids)
                last = logits[0, -1, :]
                marker_logits = torch.stack([last[m] for m in marker_ids])
                pred_local = int(marker_logits.argmax().item())
                pred_pid = ids_sorted[pred_local]
            total += 1
            ok = (pred_pid == pid)
            if ok: correct += 1
            # All T_perc codes must match for "code-match" condition
            if np.array_equal(q_codes, register_codes[pid]):
                code_match_t += 1
                if ok: code_match_c += 1

    with torch.no_grad():
        for ks, w in eng_snap.items():
            eng.tables[ks].embedding.weight.copy_(w)
        if modality_id == MODALITY_AUDIO:
            bolt.aud_perc_emb.weight.copy_(perc_snap)
        else:
            bolt.vis_perc_emb.weight.copy_(perc_snap)
    return {
        "N_registered": len(ids_sorted), "N_queries": total,
        "retrieval_at_1": correct / total if total else 0.0,
        "code_match_retr": code_match_c / code_match_t if code_match_t else float("nan"),
        "fraction_code_match": code_match_t / total if total else 0.0,
        "avg_insert_steps": float(np.mean([s for s, _ in insert_stats])),
        "avg_insert_loss": float(np.mean([l for _, l in insert_stats])),
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "v-xc-id-face"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    T_perc = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    Ns_str = sys.argv[5] if len(sys.argv) > 5 else "20,100,300,500,700"
    Ns = [int(x) for x in Ns_str.split(",")]

    print("=" * 70)
    print(f"Multi-position Path A — mode={mode}  K={K}  T_perc={T_perc}  seed={seed}")
    print(f"  Address space: K^T_perc = {K**T_perc} (>= N to avoid hash collisions)")
    print("=" * 70)

    torch.manual_seed(seed); np.random.seed(seed)
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    if mode == "v-xc-id-face":
        d = np.load(EMB / "arcface_face_combined.npz")
    else:
        from id_codebook_v2 import MODE_PATHS
        d = np.load(EMB / MODE_PATHS[mode][0])
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    n_eval_ids = len(set(ev_pid.tolist()))
    Ns = [N for N in Ns if N <= n_eval_ids]
    modality_id = MODALITY_VISION if mode.startswith("v-") else MODALITY_AUDIO
    print(f"  data: train {len(set(tr_pid.tolist()))} IDs / {len(tr_emb)} samp, "
          f"eval {n_eval_ids} IDs / {len(ev_emb)} samp; modality={modality_id}")

    # Fit residual RQ at K per level, T_perc levels
    print(f"\n[fit] residual RQ with T_perc={T_perc} levels at K={K} each")
    apply_fn = fit_residual_rq(tr_emb, n_levels=T_perc, k_per=K, seed=seed)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()
    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                            engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    print("[pretrain] generic-NTP 400 steps (only first level used for input embedding)")
    # Pretrain with the FIRST level's codes (for simplicity)
    def apply_level0(e):
        c = apply_fn(e)
        return c[:, 0] if c.ndim == 2 else c
    pretrain_generic(bolt, tr_emb, tr_pid, apply_level0, modality_id, tok,
                     n_steps=400, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    print(f"\n[eval — multi-position]")
    print(f"{'N':>5} | {'retr@1':>8} | {'code-match':>11} | {'frac-code':>10} | "
          f"{'elapsed':>8}")
    print("-" * 55)
    results = {}
    for N in Ns:
        t0 = time.time()
        r = evaluate_multipos(bolt, apply_fn, ev_emb, ev_pid, modality_id, tok,
                                T_perc=T_perc, K=K, N_subset=N,
                                n_queries_per_id=3, max_steps=60, lr=1.0, T=24)
        elapsed = time.time() - t0
        print(f"{N:>5} | {r['retrieval_at_1']:>8.3f} | {r['code_match_retr']:>11.3f} | "
              f"{r['fraction_code_match']:>10.3f} | {elapsed:>7.0f}s")
        results[N] = {**r, "elapsed_s": elapsed}

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/"
                f"pathA_multipos_{mode}_K{K}_T{T_perc}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump({"mode": mode, "K": K, "T_perc": T_perc, "address_space": K**T_perc,
                    "seed": seed, "Ns": Ns, "results": {str(k): v for k, v in results.items()}},
                   f, indent=2, default=str)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    sys.exit(main())
