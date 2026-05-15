"""Path A with top-K codebook insertion.

The codebook miss rate is the binding constraint: P(query and registration
quantise to the same code) is 0.4–0.7. When a cross-condition probe lands
on a different code than its registration, the Engram address misses
entirely and no mechanism strength helps.

Top-K insertion attacks this directly: at registration time, write the
marker at the top-K nearest codes (not just argmin). The Engram tables
and perc_emb rows are updated for K codes per identity rather than 1.
At query time, argmin is still used — but the marker is now reachable
from K different addresses, so the probability of a hit goes up roughly
K-fold (modulo overlap with other registrations).

Trade-off: writing to K codes per identity raises the per-identity row
footprint by K× and increases inter-id collision risk. We pick K=3 as
a balance.

Drop-in replacement for `pathA_idcb_run.py`: same arguments, same
output filename convention.
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
from engram_module_mm import MODALITY_VISION, MODALITY_AUDIO
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import (
    QwenEngramBolt, build_fixed_context, get_touched_rows, MODEL_ID, DEVICE,
)
from id_codebook_v2 import load_pipeline_apply, MODE_PATHS

torch.manual_seed(42); np.random.seed(42)


MODE_TO_MODALITY = {
    "a-xr-id": MODALITY_AUDIO,
    "a-scn":   MODALITY_AUDIO,
    "a-para":  MODALITY_AUDIO,
    "v-xc-id": MODALITY_VISION,
    "v-sty":   MODALITY_VISION,
    "v-sty-clip": MODALITY_VISION,
}


def topk_codes(emb_np, centroids_t, top_k=3):
    """Returns the top-K nearest codes for each row, sorted nearest-first."""
    x = torch.from_numpy(emb_np.astype(np.float32)).to(DEVICE)
    x = F.normalize(x, dim=-1)
    d2 = (x.pow(2).sum(-1, keepdim=True)
          - 2 * x @ centroids_t.t()
          + centroids_t.pow(2).sum(-1))
    return d2.topk(top_k, dim=-1, largest=False).indices.cpu().numpy()


def surgical_insert_one(bolt, code_token, modality_id, marker_text_id,
                         tok, max_steps=80, lr=1.0, T=24):
    """One-code surgical insertion — single code → single row write.
    Re-implements `qwen_engram_bolt.surgical_insert` (avoiding circular import)."""
    import torch.nn.functional as F
    eng = bolt.engram.engrams[str(modality_id)]
    input_ids, modality_ids = build_fixed_context(code_token, modality_id, tok, marker_text_id, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
    touched = get_touched_rows(eng, code_token, input_ids)
    if modality_id == MODALITY_VISION:
        perc_emb_param = bolt.vis_perc_emb.weight
    else:
        perc_emb_param = bolt.aud_perc_emb.weight
    params_to_opt = [eng.tables[ks].embedding.weight for ks in touched] + [perc_emb_param]
    opt = torch.optim.SGD(params_to_opt, lr=lr, momentum=0.0)
    target = torch.tensor([marker_text_id], dtype=torch.long, device=DEVICE)
    last_loss = float("inf"); steps_taken = 0
    for step in range(max_steps):
        logits = bolt(input_ids, modality_ids)
        loss = F.cross_entropy(logits[:, -1, :], target)
        last_loss = float(loss.item())
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            for ks, rows in touched.items():
                W = eng.tables[ks].embedding.weight
                if W.grad is None: continue
                mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.grad.dtype)
                mask[torch.tensor(sorted(rows), device=W.device, dtype=torch.long)] = 1.0
                W.grad.mul_(mask)
            if perc_emb_param.grad is not None:
                pmask = torch.zeros(perc_emb_param.shape[0], 1,
                                     device=perc_emb_param.device,
                                     dtype=perc_emb_param.grad.dtype)
                pmask[int(code_token)] = 1.0
                perc_emb_param.grad.mul_(pmask)
        opt.step()
        steps_taken = step + 1
        if last_loss < 0.5: break
    return steps_taken, last_loss


def evaluate_topk(bolt, codebook_apply, centroids_t, eval_emb, eval_pid,
                   modality_id, tok, *, top_k=3, N_subset=None,
                   n_queries_per_id=None, max_steps=80, lr=1.0, T=24,
                   marker_offset=30001):
    """Evaluate with top-K insertion: each identity is registered at its
    top-K codes (in order). At query, argmin code is used.
    """
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None: ids_sorted = ids_sorted[:N_subset]
    marker_ids = list(range(marker_offset, marker_offset + len(ids_sorted)))
    markers = {pid: marker_ids[i] for i, pid in enumerate(ids_sorted)}

    eng = bolt.engram.engrams[str(modality_id)]
    eng_snap = {ks: tbl.embedding.weight.detach().clone() for ks, tbl in eng.tables.items()}
    perc_snap = (bolt.aud_perc_emb if modality_id == MODALITY_AUDIO else bolt.vis_perc_emb).weight.detach().clone()

    rng = np.random.default_rng(99)
    register_topk_codes = {}; code_to_pid = defaultdict(set); insert_stats = []
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_emb = eval_emb[idxs[0]]
        # Get top-K codes for this registration
        tk = topk_codes(reg_emb[None], centroids_t, top_k=top_k)[0]
        register_topk_codes[pid] = tk
        # Insert at each top-K code (different codes get the SAME marker)
        for code in tk:
            steps, fl = surgical_insert_one(
                bolt, int(code), modality_id, markers[pid],
                tok, max_steps=max_steps, lr=lr, T=T,
            )
            insert_stats.append((steps, fl))
            code_to_pid[int(code)].add(pid)

    correct = 0; total = 0
    code_match_c = 0; code_match_t = 0
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        q_idxs = idxs[1:]
        if n_queries_per_id is not None: q_idxs = q_idxs[:n_queries_per_id]
        for qi in q_idxs:
            q_emb = eval_emb[qi]
            q_code_arr = codebook_apply(q_emb[None])
            q_code = int(q_code_arr.item() if hasattr(q_code_arr, 'item') else q_code_arr[0])
            input_ids, modality_ids = build_fixed_context(q_code, modality_id, tok,
                                                            marker_text_id=0, T=T)
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
            # Code-match here means: q_code is in pid's top-K registered codes
            if q_code in set(register_topk_codes[pid].tolist()):
                code_match_t += 1
                if ok: code_match_c += 1

    # Restore tables
    with torch.no_grad():
        for ks, w in eng_snap.items():
            eng.tables[ks].embedding.weight.copy_(w)
        if modality_id == MODALITY_AUDIO:
            bolt.aud_perc_emb.weight.copy_(perc_snap)
        else:
            bolt.vis_perc_emb.weight.copy_(perc_snap)

    return {
        "N_registered": len(ids_sorted), "N_queries": total,
        "retrieval_at_1": correct / total if total > 0 else 0.0,
        "code_match_retr": code_match_c / code_match_t if code_match_t > 0 else float("nan"),
        "fraction_code_match": code_match_t / total if total > 0 else 0.0,
        "N_collision_codes": len([c for c, ps in code_to_pid.items() if len(ps) > 1]),
        "avg_insert_steps": float(np.mean([s for s, _ in insert_stats])),
        "avg_insert_loss": float(np.mean([l for _, l in insert_stats])),
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "a-xr-id"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    use_stable = (sys.argv[4] == "stable") if len(sys.argv) > 4 else False
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

    print("=" * 70)
    print(f"Path A + id-codebook v2 + top-K — mode={mode}  K={K}  top_k={top_k}  "
          f"stable={use_stable}  seed={seed}")
    print("=" * 70)

    torch.manual_seed(seed); np.random.seed(seed)
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary, _ = MODE_PATHS[mode]
    d = np.load(EMB / primary)
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)

    # Load the appropriate codebook
    cb_dir = Path("/home/ubuntu/multimodal-user-memory/runs/codebooks")
    if use_stable:
        cb_path = cb_dir / f"stable_codebook_{mode}_K{K}.pt"
    else:
        cb_path = cb_dir / f"id_v2_codebook_{mode}_K{K}.pt"
    print(f"  loading codebook from {cb_path}")
    apply_fn = load_pipeline_apply(cb_path)

    # Extract centroids tensor for top-K lookup
    state = torch.load(cb_path, map_location=DEVICE, weights_only=False)
    centroids_t = torch.from_numpy(state["centroids"].astype(np.float32)).to(DEVICE)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    modality_id = MODE_TO_MODALITY[mode]
    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                            engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    print(f"  bolt built (K={K}, top_k={top_k})")

    print(f"\n[pretrain] generic-NTP 400 steps  modality={modality_id}")
    losses = pretrain_generic(bolt, tr_emb, tr_pid, apply_fn, modality_id, tok,
                              n_steps=400, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)
    print(f"  pretrain final loss: {float(np.mean(losses[-30:])):.4f}")

    print("\n[eval top-K]")
    Ns = [5, 10, 20]; nq = 5
    results = {}
    for N in Ns:
        if N > len(set(ev_pid)): continue
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate_topk(bolt, apply_fn, centroids_t, ev_emb, ev_pid, modality_id, tok,
                            top_k=top_k, N_subset=N, n_queries_per_id=nq,
                            max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f}  on {100*r['fraction_code_match']:.0f}%  "
              f"collisions={r['N_collision_codes']}")
        results[N] = {"rag": rag, **r}

    suffix = "stable_" if use_stable else ""
    out = Path(f"/home/ubuntu/multimodal-user-memory/results/pathA_idcb_topk{top_k}_{suffix}{mode}_K{K}.json")
    with open(out, "w") as f:
        json.dump({"mode": mode, "K": K, "top_k": top_k,
                    "codebook": str(cb_path), "seed": seed,
                    "results": results}, f, indent=2, default=str)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    sys.exit(main())
