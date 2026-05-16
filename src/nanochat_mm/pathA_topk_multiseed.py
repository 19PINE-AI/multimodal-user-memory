"""Multi-seed top-K Path A verification.

The single-seed top-K=3 run on A-SCN at N=5 lifted retr@1 from 0.43 → 0.56
(+0.128). This script verifies that across 5 seeds before adding it to
the paper. Same harness as pathA_multiseed.py but with top-K insertion.
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
from engram_module_mm import MODALITY_AUDIO, MODALITY_VISION
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import QwenEngramBolt, MODEL_ID, DEVICE
from id_codebook_v2 import MODE_PATHS, load_pipeline_apply
from pathA_topk_run import evaluate_topk, MODE_TO_MODALITY


def run_one_seed(mode, K, top_k, seed, qwen, tok):
    torch.manual_seed(seed); np.random.seed(seed)
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary, _ = MODE_PATHS[mode]
    d = np.load(EMB / primary)
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    # Use the single-seed codebook (already saved); varies the bolt/pretrain seed
    cb_path = Path(f"/home/ubuntu/multimodal-user-memory/runs/codebooks/"
                    f"id_v2_codebook_{mode}_K{K}.pt")
    apply_fn = load_pipeline_apply(cb_path)
    state = torch.load(cb_path, map_location=DEVICE, weights_only=False)
    centroids_t = torch.from_numpy(state["centroids"].astype(np.float32)).to(DEVICE)

    modality_id = MODE_TO_MODALITY[mode]
    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                            engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    pretrain_generic(bolt, tr_emb, tr_pid, apply_fn, modality_id, tok,
                     n_steps=400, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    Ns = [5, 10, 20]; nq = 5
    out = {}
    for N in Ns:
        if N > len(set(ev_pid)): continue
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate_topk(bolt, apply_fn, centroids_t, ev_emb, ev_pid, modality_id, tok,
                            top_k=top_k, N_subset=N, n_queries_per_id=nq,
                            max_steps=80, lr=1.0, T=24)
        out[N] = {"rag": rag, "retr@1": r["retrieval_at_1"],
                   "code_match": r["code_match_retr"],
                   "frac_code_match": r["fraction_code_match"]}
    bolt.remove_hook()
    del bolt
    torch.cuda.empty_cache()
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "a-scn"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    seeds = list(range(42, 42 + n_seeds))

    print("=" * 70)
    print(f"Top-K multi-seed — mode={mode}  K={K}  top_k={top_k}  n_seeds={n_seeds}")
    print(f"  seeds: {seeds}")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    all_results = {}
    for s in seeds:
        print(f"\n--- seed = {s} ---")
        try:
            t0 = time.time()
            all_results[s] = run_one_seed(mode, K, top_k, s, qwen, tok)
            for N, r in all_results[s].items():
                print(f"  N={N:>3}  retr@1={r['retr@1']:.3f}  RAG={r['rag']:.3f}  "
                      f"code-match={r['code_match']:.3f}  frac={r['frac_code_match']:.3f}")
            print(f"  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  seed {s} failed: {e}")
            all_results[s] = {"error": str(e)}

    print("\n" + "=" * 70)
    print(f"AGGREGATE — mean ± std across {n_seeds} seeds")
    print("=" * 70)
    print(f"{'N':>4} | {'mean retr@1':>13} | {'std':>5} | {'min':>5} | {'max':>5} | {'RAG':>5}")
    print("-" * 60)
    aggregate = {}
    for N in [5, 10, 20]:
        vals = []; rags = []
        for s in seeds:
            if N in all_results.get(s, {}) and "retr@1" in all_results[s][N]:
                vals.append(all_results[s][N]["retr@1"])
                rags.append(all_results[s][N]["rag"])
        if vals:
            mean = float(np.mean(vals)); std = float(np.std(vals))
            rag_mean = float(np.mean(rags))
            print(f"{N:>4} | {mean:>13.3f} | {std:>5.3f} | {min(vals):>5.3f} | "
                  f"{max(vals):>5.3f} | {rag_mean:>5.3f}")
            aggregate[N] = {"mean": mean, "std": std, "min": min(vals),
                             "max": max(vals), "vals": vals, "rag_vals": rags,
                             "rag_mean": rag_mean}

    out_path = Path(f"/home/ubuntu/multimodal-user-memory/results/"
                    f"pathA_topk{top_k}_multiseed_{mode}_K{K}.json")
    with open(out_path, "w") as f:
        json.dump({"mode": mode, "K": K, "top_k": top_k, "seeds": seeds,
                    "per_seed": all_results,
                    "aggregate": {str(k): v for k, v in aggregate.items()}},
                   f, indent=2, default=str)
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    sys.exit(main())
