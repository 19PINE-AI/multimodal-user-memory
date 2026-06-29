"""How many slots to remember M faces/voices? (encoder-space, the design that works)

The first version of this study asked a FROZEN LM to match a query face against
faces packed into its token space. That fails at chance (a frozen LM cannot do
perceptual matching in token space -- see CAPACITY_FINDINGS.md). The fix is the
mechanism that actually works (AttMem): match in the encoder's COSINE space.

So the real "tokens per identity" question is: compress M registered face/voice
keys into k prototype slots, then recognise a cross-condition query by cosine.
  k >= M : one slot per identity (= AttMem) -> recognition works.
  k <  M : slots merge identities -> only ~k of the M remain recoverable.

This is pure encoder-space (no LM), so it runs on CPU and isolates the capacity
question cleanly. Sweep M x k for faces (ArcFace) and voices (ECAPA).

Usage:
  python3 set_memory.py --emb_file runs/embeddings/arcface_lfw_xxxl.npz \
      --m_eval 2 4 8 16 32 64 --k_eval 2 4 8 16 32 64 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/multimodal-user-memory")
log = logging.getLogger("set_memory")


def load_emb(path):
    d = np.load(path)
    emb = d["emb"].astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    by = {}
    for i, p in enumerate(pid):
        by.setdefault(str(p), []).append(i)
    ids = [p for p, ix in by.items() if len(ix) >= 2]
    return emb, by, ids


def spherical_kmeans(X, k, seed, iters=50):
    """Cluster L2-normalised rows of X into k prototypes by cosine. k>=N -> identity."""
    n = len(X)
    if k >= n:
        return np.arange(n), X.copy()
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(n, k, replace=False)].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        new = (X @ cent.T).argmax(1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            m = assign == j
            if m.any():
                c = X[m].mean(0)
                cent[j] = c / (np.linalg.norm(c) + 1e-8)
    return assign, cent


def eval_cell(emb, by, ids, seed, M, k):
    """Register M identities (1 key each), compress keys to k prototypes, then
    recognise a cross-condition query of each. recall@1 over identities."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(ids, size=M, replace=False)
    reg, qry = [], []
    for p in sel:
        ix = list(by[str(p)]); rng.shuffle(ix)
        reg.append(ix[0]); qry.append(ix[1])
    K = emb[reg]; Q = emb[qry]                              # [M,D], identities 0..M-1
    _, cent = spherical_kmeans(K, k, seed)                  # k prototype slots
    proto_id = (cent @ K.T).argmax(1)                       # each slot -> its representative id
    pred = proto_id[(Q @ cent.T).argmax(1)]                # query -> nearest slot -> id
    return float((pred == np.arange(M)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb_file", default=str(ROOT / "runs/embeddings/arcface_lfw_xxxl.npz"))
    ap.add_argument("--m_eval", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--k_eval", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_sets", type=int, default=20, help="eval sets per (M,k,seed)")
    ap.add_argument("--out", default=str(ROOT / "results" / "set_memory.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    emb, by, ids = load_emb(args.emb_file)
    name = Path(args.emb_file).stem
    log.info("%s: %d identities (>=2), dim=%d", name, len(ids), emb.shape[1])
    rows = []
    for M in args.m_eval:
        if M > len(ids):
            continue
        for k in args.k_eval:
            vals = []
            for s in args.seeds:
                for t in range(args.n_sets):
                    vals.append(eval_cell(emb, by, ids, s * 100 + t, M, k))
            rows.append({"M": M, "k": k, "recall_mean": float(np.mean(vals)),
                         "recall_std": float(np.std(vals)), "tokens_per_id": k / M})
        log.info("M=%2d done", M)
    Path(args.out).write_text(json.dumps({"emb": name, "rows": rows}, indent=2))

    Ms = sorted({r["M"] for r in rows}); ks = sorted({r["k"] for r in rows})
    grid = {(r["M"], r["k"]): r["recall_mean"] for r in rows}
    print(f"\n=== SET-MEMORY recall@1: compress M {name} into k prototype slots ===")
    print("  M\\k " + "".join(f"{k:>7}" for k in ks))
    for M in Ms:
        print(f"  {M:>3} " + "".join(
            f"{grid[(M,k)]:>7.2f}" if (M, k) in grid else "   -   " for k in ks))
    print(f"\nwrote {args.out}")
    print("Read: recall holds for k>=M (~1 slot/identity); drops ~k/M when k<M.")


if __name__ == "__main__":
    main()
