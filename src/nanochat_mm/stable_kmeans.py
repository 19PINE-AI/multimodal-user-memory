"""Stable k-means via restart-and-select.

The single-seed k-means in `id_codebook_v2.kmeans_centroids` is sensitive
to the random init: different seeds can produce centroid sets whose
eval same-code rate varies by 10+ points. Multi-seed Path A inherits
that variance (std 0.10–0.13 on retr@1).

This module runs k-means with N inits and selects the centroid set that
maximises the within-identity same-code rate on the training pairs.
The output is a "stable" codebook — same data, lower variance.

Usage:
  centroids = stable_kmeans(emb, pid, K=32, n_inits=20)
  # → centroids: np.ndarray [K, D]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _kmeans_one(x_np, K, seed):
    import faiss
    km = faiss.Kmeans(x_np.shape[1], K, niter=30, verbose=False, seed=seed)
    km.train(x_np.astype(np.float32))
    return km.centroids.astype(np.float32)


def _assign(x_np, centroids):
    """Hard argmin assignment. Returns codes shape [N]."""
    x = x_np.astype(np.float32)
    d2 = (x ** 2).sum(-1, keepdims=True) - 2 * x @ centroids.T + (centroids ** 2).sum(-1)
    return d2.argmin(-1)


def _same_code_rate(codes, pid):
    """Among all same-id pairs, fraction with matching codes."""
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)
    n_same = 0; n_pairs = 0
    for pid_v, samps in by_id.items():
        if len(samps) < 2: continue
        for i in range(len(samps)):
            for j in range(i + 1, len(samps)):
                if codes[samps[i]] == codes[samps[j]]: n_same += 1
                n_pairs += 1
    return n_same / n_pairs if n_pairs else 0.0


def _inter_id_collision_rate(codes, pid):
    """Among different-id pairs, fraction with matching codes (lower is better)."""
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)
    ids = list(by_id.keys())
    code_arr = np.asarray(codes)
    n_diff = 0; n_diff_same = 0
    # Pair first-sample of each pair of ids (cheap approx)
    for i in range(len(ids)):
        a = by_id[ids[i]][0]
        for j in range(i + 1, len(ids)):
            b = by_id[ids[j]][0]
            if code_arr[a] == code_arr[b]: n_diff_same += 1
            n_diff += 1
    return n_diff_same / n_diff if n_diff else 0.0


def stable_kmeans(emb, pid, K, n_inits=20, score_fn=None, verbose=True):
    """Run k-means N times with different seeds; pick the centroid set with
    the best score on the training data.

    Default score = train_same_code - 0.3 * inter_id_collision (matches the
    selection criterion used elsewhere in id_codebook_v2).
    """
    if score_fn is None:
        score_fn = lambda c, p: _same_code_rate(c, p) - 0.3 * _inter_id_collision_rate(c, p)
    best_centroids = None; best_score = -np.inf; best_seed = None
    history = []
    for s in range(n_inits):
        # Use distinct primes to avoid clustering accidents from incidentally
        # close seeds
        seed = 42 + s * 17
        try:
            centroids = _kmeans_one(emb, K, seed=seed)
        except RuntimeError as e:
            if verbose: print(f"  init seed={seed}: faiss error: {e}")
            continue
        codes = _assign(emb, centroids)
        sc = score_fn(codes, pid)
        history.append({"seed": seed, "score": float(sc)})
        if sc > best_score:
            best_score = sc; best_centroids = centroids; best_seed = seed
        if verbose and (s + 1) % 5 == 0:
            print(f"    [stable_kmeans] init {s+1}/{n_inits}  score={sc:.3f}  best={best_score:.3f}")
    if verbose:
        scores = [h["score"] for h in history]
        print(f"  stable_kmeans({n_inits} inits): "
              f"best score={best_score:.3f} at seed={best_seed}; "
              f"mean={float(np.mean(scores)):.3f}, std={float(np.std(scores)):.3f}, "
              f"max-min={max(scores) - min(scores):.3f}")
    return best_centroids, history


def fit_stable_kmeans_apply(emb, pid, *, K, n_inits=20):
    """Returns (apply_fn, centroids_np) compatible with id_codebook_v2's
    `load_pipeline_apply` interface (no adapter)."""
    centroids_np, hist = stable_kmeans(emb, pid, K=K, n_inits=n_inits, verbose=True)
    centroids_t = torch.from_numpy(centroids_np.astype(np.float32)).to(DEVICE)
    @torch.no_grad()
    def apply(emb_np):
        x = torch.from_numpy(emb_np.astype(np.float32)).to(DEVICE)
        x = F.normalize(x, dim=-1)
        d2 = (x.pow(2).sum(-1, keepdim=True)
              - 2 * x @ centroids_t.t()
              + centroids_t.pow(2).sum(-1))
        return d2.argmin(-1).cpu().numpy()
    return apply, centroids_np, hist


if __name__ == "__main__":
    """Standalone diagnostic — compare stable vs single-seed k-means.

    Usage:
      python3 stable_kmeans.py <mode> <K>
    """
    from v2_retrieval import split_by_identity
    from id_codebook_v2 import (
        MODE_PATHS, evaluate_same_code_rate, evaluate_cross_id_collision,
        save_pipeline,
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "a-xr-id"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    n_inits = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary, _ = MODE_PATHS[mode]
    d = np.load(EMB / primary)
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)

    print("=" * 70)
    print(f"Stable k-means diagnostic — mode={mode} K={K} n_inits={n_inits}")
    print(f"  train {len(set(tr_pid))} IDs / {len(tr_emb)} samples")
    print("=" * 70)

    # Single-seed reference (seed=42)
    centroids_single = _kmeans_one(tr_emb, K, seed=42)
    tr_codes_s = _assign(tr_emb, centroids_single)
    ev_codes_s = _assign(ev_emb, centroids_single)
    tr_same_s = _same_code_rate(tr_codes_s, tr_pid)
    ev_same_s = _same_code_rate(ev_codes_s, ev_pid)
    ev_inter_s = _inter_id_collision_rate(ev_codes_s, ev_pid)
    print(f"\n[single-seed=42] train_same_code={tr_same_s:.3f}  "
          f"eval_same_code={ev_same_s:.3f}  eval_inter_coll={ev_inter_s:.3f}")

    # Stable k-means
    print(f"\n[stable_kmeans] running {n_inits} inits...")
    centroids_stable, hist = stable_kmeans(tr_emb, tr_pid, K=K, n_inits=n_inits, verbose=False)
    tr_codes_st = _assign(tr_emb, centroids_stable)
    ev_codes_st = _assign(ev_emb, centroids_stable)
    tr_same_st = _same_code_rate(tr_codes_st, tr_pid)
    ev_same_st = _same_code_rate(ev_codes_st, ev_pid)
    ev_inter_st = _inter_id_collision_rate(ev_codes_st, ev_pid)
    print(f"[stable] train_same_code={tr_same_st:.3f}  "
          f"eval_same_code={ev_same_st:.3f}  eval_inter_coll={ev_inter_st:.3f}")

    scores = [h["score"] for h in hist]
    print(f"\n  score histogram (train metric): "
          f"min={min(scores):.3f}  med={np.median(scores):.3f}  max={max(scores):.3f}  "
          f"std={np.std(scores):.3f}")
    print(f"  eval same-code LIFT (stable - single): {ev_same_st - ev_same_s:+.3f}")

    # Save the stable codebook in the v2 pipeline format
    out_dir = Path("/home/ubuntu/multimodal-user-memory/runs/codebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pipeline(out_dir / f"stable_codebook_{mode}_K{K}.pt",
                   None, centroids_stable, tr_emb.shape[1], 0.5, None)
    print(f"\n[saved] {out_dir}/stable_codebook_{mode}_K{K}.pt")
