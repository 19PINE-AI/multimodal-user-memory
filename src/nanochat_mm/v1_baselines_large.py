"""v1 baselines on the LARGE perceptual-embedding sets.

Reproduces the v1 hash-keyed Engram retrieval protocol (first-write-wins
and chained-with-RAG-cheat) on the larger 158-vision / 58-audio embedding
sets. Gives a fair head-to-head against Path A's scaling numbers.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/ for engram_retrieval.py
from engram_retrieval import (
    fit_naive_rq, per_user_engram_retrieval, embedding_rag_baseline,
)
from v2_retrieval import split_by_identity

SEED = 42
np.random.seed(SEED)

LARGE = "/home/ubuntu/multimodal-user-memory/runs/embeddings"


def main():
    print("=" * 70)
    print("v1 baselines on LARGE data (for fair comparison vs Path A)")
    print("=" * 70)

    aud = np.load(f"{LARGE}/ecapa_libri_large.npz")
    vis = np.load(f"{LARGE}/arcface_lfw_large.npz")
    aud_tr, _, aud_ev, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr, _, vis_ev, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])

    Ns = [5, 10, 20, 40, 60]; nq = 5
    results = {}

    # Codebook configs from v1 sweep (we know which ones are competitive)
    configs = [(1, 32), (1, 128), (2, 16), (2, 64), (3, 16)]

    for name, train_emb, eval_emb, eval_pid in [
        ("audio", aud_tr, aud_ev, aud_ev_pid),
        ("vision", vis_tr, vis_ev, vis_ev_pid),
    ]:
        print(f"\n[{name}] eval ids: {len(set(eval_pid))}; train ids: {len(set(_ for _ in []))} (placeholder)")
        # Embedding RAG ceiling
        rag = {N: embedding_rag_baseline(eval_emb, eval_pid, N_subset=N, n_queries_per_id=nq) for N in Ns}
        for N in Ns:
            if rag[N] is not None:
                print(f"  RAG ceiling N={N}: {rag[N]:.4f}")

        # Engram across configs
        mod_results = {"rag": rag, "engram": {}}
        for n_levels, k_per in configs:
            cfg = f"L{n_levels}_K{k_per}"
            eff_K = k_per ** n_levels
            print(f"\n  [{cfg} eff_K={eff_K}]")
            apply_fn = fit_naive_rq(train_emb, n_levels, k_per)
            cfg_results = {}
            for N in Ns:
                if N > len(set(eval_pid)): continue
                m = per_user_engram_retrieval(eval_emb, eval_pid, apply_fn, N_subset=N, n_queries_per_id=nq)
                if m is None: continue
                cfg_results[N] = m
                print(f"    N={N:>2}  first-write={m['retrieval_first_write_wins']:.3f}  "
                      f"chained={m['retrieval_chained_disambig']:.3f}  miss={m['missed_no_slot_hit']:.3f}  "
                      f"slots_used={m['num_slots_used']}  max-occ={m['max_slot_occupancy']}")
            mod_results["engram"][cfg] = cfg_results
        results[name] = mod_results

    out = Path("/home/ubuntu/multimodal-user-memory/results/v1_baselines_large.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Headline: best v1 first-write vs best v1 chained vs Path A best
    pa_scaling = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_scaling.json"))
    pa_k64 = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_scaling_k64.json"))
    pa_ste = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_ste_k64.json"))
    print("\n" + "=" * 100)
    print("HEADLINE — large-data fair comparison: parametric retrievers")
    print("=" * 100)
    print(f"{'modality':>8} | {'N':>3} | {'RAG':>5} | {'v1 first-write':>14} | {'v1 chained (RAG cheat)':>22} | "
          f"{'Path A best':>11}")
    print("-" * 100)
    for name in ["audio", "vision"]:
        for N in Ns:
            if N > len(set(aud_ev_pid)) and name == "audio": continue
            if N > len(set(vis_ev_pid)) and name == "vision": continue
            rag_v = results[name]["rag"].get(N, float("nan"))
            best_fw = max((results[name]["engram"][c].get(N, {}).get("retrieval_first_write_wins", 0.0)
                            for c in results[name]["engram"]), default=0.0)
            best_ch = max((results[name]["engram"][c].get(N, {}).get("retrieval_chained_disambig", 0.0)
                            for c in results[name]["engram"]), default=0.0)
            # Best Path A: pick from scaling K32, K64, K64+STE
            pa_best = max(
                pa_scaling.get(name, {}).get(str(N), {}).get("retrieval_at_1", 0.0),
                pa_k64.get(name, {}).get(str(N), {}).get("retrieval_at_1", 0.0),
                pa_ste.get(name, {}).get(str(N), {}).get("retrieval_at_1", 0.0),
            )
            print(f"{name:>8} | {N:>3} | {rag_v:>5.3f} | {best_fw:>14.3f} | {best_ch:>22.3f} | {pa_best:>11.3f}")


if __name__ == "__main__":
    sys.exit(main())
