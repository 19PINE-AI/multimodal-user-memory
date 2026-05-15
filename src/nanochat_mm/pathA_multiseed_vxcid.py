"""V-XC-ID multi-seed verification with adapter+largepool codebook.

Unlike `pathA_multiseed.py` (which fits naive k-means per seed), V-XC-ID
uses the v2 design: an adapter trained on LFW-XXL minus eval IDs, plus
k-means in the adapter output space. Each seed varies:
  - Adapter init + SupCon batch sampling
  - K-means init (seed=seed)
  - Path A pretrain (torch seed)
  - The eval rng inside evaluate() is held at 99 (data-side variance
    is the same across runs).
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
from engram_module_mm import MODALITY_VISION
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import QwenEngramBolt, evaluate, MODEL_ID, DEVICE
from id_codebook_v2 import (
    InvarianceAdapter, train_adapter, kmeans_centroids, MODE_PATHS,
)


def fit_adapter_largepool_codebook(seed, K=64):
    """Train adapter on LFW-XXL minus eval IDs, k-means in adapter space.
    Returns apply_fn for the resulting pipeline."""
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary_p, larger_p = MODE_PATHS["v-xc-id"]
    primary = np.load(EMB / primary_p)
    larger = np.load(EMB / larger_p)
    _, _, ev_emb, ev_pid = split_by_identity(primary["emb"].astype(np.float32),
                                                primary["pid"])
    ev_id_set = set(ev_pid.tolist())
    l_emb = larger["emb"].astype(np.float32); l_pid = larger["pid"]
    keep = ~np.isin(l_pid, list(ev_id_set))
    l_emb_clean = l_emb[keep]; l_pid_clean = l_pid[keep]

    torch.manual_seed(seed); np.random.seed(seed)
    adapter = train_adapter(l_emb_clean, l_pid_clean, hidden=None, alpha=0.5,
                              dropout=0.1, n_steps=2000, batch=128, lr=1e-3,
                              temperature=0.1, seed=seed)
    src_t = torch.from_numpy(l_emb_clean.astype(np.float32)).to(DEVICE)
    src_t = F.normalize(src_t, dim=-1)
    with torch.no_grad():
        z = adapter(src_t)
    centroids_np = kmeans_centroids(z.cpu().numpy(), K, seed=seed)
    centroids_t = torch.from_numpy(centroids_np.astype(np.float32)).to(DEVICE)

    @torch.no_grad()
    def apply(emb_np):
        x = torch.from_numpy(emb_np.astype(np.float32)).to(DEVICE)
        x = F.normalize(x, dim=-1)
        z = adapter(x)
        d2 = (z.pow(2).sum(-1, keepdim=True)
              - 2 * z @ centroids_t.t()
              + centroids_t.pow(2).sum(-1))
        return d2.argmin(-1).cpu().numpy()
    return apply, ev_emb, ev_pid


def run_one_seed(seed, K, qwen, tok):
    apply_fn, ev_emb, ev_pid = fit_adapter_largepool_codebook(seed=seed, K=K)

    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary = np.load(EMB / MODE_PATHS["v-xc-id"][0])
    tr_emb, tr_pid, _, _ = split_by_identity(primary["emb"].astype(np.float32),
                                              primary["pid"])

    torch.manual_seed(seed); np.random.seed(seed)
    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                            engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    pretrain_generic(bolt, tr_emb, tr_pid, apply_fn, MODALITY_VISION, tok,
                     n_steps=400, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)
    Ns = [5, 10, 20]; nq = 5
    out = {}
    for N in Ns:
        if N > len(set(ev_pid)): continue
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate(bolt, apply_fn, ev_emb, ev_pid, MODALITY_VISION, tok,
                       N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        out[N] = {"rag": rag, "retr@1": r["retrieval_at_1"],
                   "code_match": r["code_match_retr"],
                   "frac_code_match": r["fraction_code_match"]}
    bolt.remove_hook()
    del bolt
    torch.cuda.empty_cache()
    return out


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seeds = list(range(42, 42 + n_seeds))
    print("=" * 70)
    print(f"V-XC-ID multi-seed (adapter+largepool) — K={K}  n_seeds={n_seeds}")
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
            all_results[s] = run_one_seed(s, K, qwen, tok)
            for N, r in all_results[s].items():
                print(f"  N={N:>3}  retr@1={r['retr@1']:.3f}  RAG={r['rag']:.3f}  "
                      f"code-match={r['code_match']:.3f}")
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

    out_path = Path("/home/ubuntu/multimodal-user-memory/results/") / f"pathA_multiseed_v-xc-id_K{K}.json"
    with open(out_path, "w") as f:
        json.dump({"mode": "v-xc-id", "K": K, "seeds": seeds,
                    "per_seed": all_results,
                    "aggregate": {str(k): v for k, v in aggregate.items()}},
                   f, indent=2, default=str)
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    sys.exit(main())
