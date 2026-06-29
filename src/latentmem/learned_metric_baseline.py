"""Learned-metric baseline: is AttMem's win just metric learning on the encoder?

AttMem at eval ranks by cosine over the RAW encoder embeddings (its only learned
parts are the soft value-blend + marker geometry); it never learns a similarity.
The fair question a reviewer asks: train a similarity on the same encoder features
(same train identities) and see if it matches AttMem.

We compare, on the SAME 50/50 identity split and the SAME recall@1 protocol the
paper uses (1 registration photo per id, cross-condition queries, cosine-NN over
the N registered keys):
  raw      cosine over encoder embeddings           (= RAG / embedding retrieval)
  lda      regularised LDA fit on train identities   (closed-form learned metric)
  whiten   within-class whitening fit on train       (simpler learned transform)
  contrast contrastive linear head fit on train      (SGD learned metric)
  attmem   (read from the paper's result JSON)

If a learned metric matches/beats AttMem, the win is metric learning (cheaper to
do directly). If AttMem still wins, the parametric readout adds something real.

Usage: python3 learned_metric_baseline.py --emb arcface_face_xxxl --attmem_json <path>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/multimodal-user-memory")
EMB = ROOT / "runs" / "embeddings"


def split_by_identity(emb, pid, train_frac=0.5):
    rng = np.random.RandomState(42)
    uniq = sorted(set(pid.tolist())); rng.shuffle(uniq)
    n_tr = int(len(uniq) * train_frac); tr = set(uniq[:n_tr])
    m = np.array([str(p) in tr for p in pid])
    return emb[m], pid[m], emb[~m], pid[~m]


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def recall_at_N(eval_emb, eval_pid, W, N, n_queries=3, seed=99):
    """Project by W, then the paper's cosine-NN recall@1 over N registered ids.
    `seed` controls the registration/query draw (paper protocol uses 99)."""
    by = defaultdict(list)
    for i, p in enumerate(eval_pid):
        by[str(p)].append(i)
    ids = sorted(by.keys())[:N]
    rng = np.random.default_rng(seed)
    reg, reg_lab, qs = [], [], []
    for p in ids:
        ix = list(by[p]); rng.shuffle(ix)
        reg.append(eval_emb[ix[0]]); reg_lab.append(p)
        for qi in ix[1:1 + n_queries]:
            qs.append((eval_emb[qi], p))
    if not qs:
        return 0.0
    R = l2(np.stack(reg) @ W); Q = l2(np.stack([q[0] for q in qs]) @ W)
    pred = (Q @ R.T).argmax(1)
    return float(np.mean([reg_lab[pred[k]] == qs[k][1] for k in range(len(qs))]))


def recall_multiseed(eval_emb, eval_pid, W, N, seeds):
    """Mean +/- std of recall@1 across eval draws (registration/query sampling)."""
    vals = [recall_at_N(eval_emb, eval_pid, W, N, seed=s) for s in seeds]
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)


def fit_lda(X, y, max_dim=256, reg=1e-2):
    D = X.shape[1]; mu = X.mean(0); classes = sorted(set(y.tolist()))
    Sw = np.zeros((D, D)); Sb = np.zeros((D, D))
    for c in classes:
        Xc = X[y == c]
        if len(Xc) < 2:
            continue
        muc = Xc.mean(0); d = Xc - muc
        Sw += d.T @ d
        e = (muc - mu)[:, None]; Sb += len(Xc) * (e @ e.T)
    Sw += reg * np.trace(Sw) / D * np.eye(D)
    from scipy.linalg import eigh
    vals, vecs = eigh(Sb, Sw)
    dim = min(max_dim, len(classes) - 1, D)
    return vecs[:, -dim:]


def fit_whiten(X, y, reg=1e-2):
    D = X.shape[1]; classes = sorted(set(y.tolist())); Sw = np.zeros((D, D))
    for c in classes:
        Xc = X[y == c]
        if len(Xc) < 2:
            continue
        d = Xc - Xc.mean(0); Sw += d.T @ d
    Sw = Sw / max(1, len(X)) + reg * np.eye(D)
    from scipy.linalg import fractional_matrix_power
    return np.real(fractional_matrix_power(Sw, -0.5))


def fit_contrastive(X, y, dim=256, steps=2000, lr=1e-2, tau=0.1, seed=0):
    import torch
    dev = "cpu"  # tiny linear; avoid contended GPU
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    labs = np.array([str(p) for p in y]); by = defaultdict(list)
    for i, p in enumerate(labs):
        by[p].append(i)
    multi = [p for p, ix in by.items() if len(ix) >= 2]
    D = X.shape[1]
    W = torch.nn.Parameter(torch.eye(D, dim, device=dev) + 0.01 * torch.randn(D, dim, device=dev))
    opt = torch.optim.Adam([W], lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        bp = rng.choice(multi, size=min(64, len(multi)), replace=False)
        ai, pi = [], []
        for p in bp:
            a, b = rng.choice(by[p], 2, replace=False); ai.append(a); pi.append(b)
        A = torch.nn.functional.normalize(Xt[ai] @ W, dim=1)
        P = torch.nn.functional.normalize(Xt[pi] @ W, dim=1)
        logits = (A @ P.T) / tau                      # InfoNCE: positive on diagonal
        loss = torch.nn.functional.cross_entropy(
            logits, torch.arange(len(ai), device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="arcface_face_xxxl")
    ap.add_argument("--attmem_json", default="")
    ap.add_argument("--ns", type=int, nargs="+", default=[5, 10, 20, 50, 100, 300, 700, 1000])
    ap.add_argument("--eval_seeds", type=int, default=20, help="number of eval draws")
    ap.add_argument("--eval_seeds_base", type=int, default=90)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = np.load(EMB / f"{args.emb}.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    tr_e, tr_p, ev_e, ev_p = split_by_identity(emb, pid)
    n_ev = len(set(ev_p.tolist()))
    print(f"{args.emb}: {len(set(tr_p.tolist()))} train ids, {n_ev} eval ids, dim={emb.shape[1]}")

    metrics = {"raw": np.eye(emb.shape[1])}
    print("fitting LDA ..."); metrics["lda"] = fit_lda(tr_e, tr_p)
    print("fitting whiten ..."); metrics["whiten"] = fit_whiten(tr_e, tr_p)
    print("fitting contrastive ..."); metrics["contrast"] = fit_contrastive(tr_e, tr_p)

    attmem = {}
    if args.attmem_json and Path(args.attmem_json).exists():
        a = json.load(open(args.attmem_json))["results"]
        attmem = {int(N): v["attmem"] for N, v in a.items()}

    seeds = list(range(args.eval_seeds_base, args.eval_seeds_base + args.eval_seeds))
    Ns = [N for N in args.ns if N <= n_ev]
    rows = {m: {N: recall_multiseed(ev_e, ev_p, W, N, seeds) for N in Ns}
            for m, W in metrics.items()}
    print(f"\n=== LEARNED-METRIC BASELINE ({args.emb}) recall@1, "
          f"mean+/-std over {len(seeds)} eval draws ===")
    cols = ["raw", "lda", "whiten", "contrast"] + (["attmem"] if attmem else [])
    print(f"{'N':>5} | " + " ".join(f"{c:>13}" for c in cols))
    for N in Ns:
        vals = [f"{rows[m][N][0]:.3f}+-{rows[m][N][1]:.3f}"
                for m in ["raw", "lda", "whiten", "contrast"]]
        if attmem:
            vals.append(f"{attmem.get(N, float('nan')):.3f}")
        print(f"{N:>5} | " + " ".join(f"{v:>13}" for v in vals))
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"emb": args.emb, "eval_seeds": seeds,
             "rows": {m: {N: {"mean": rows[m][N][0], "std": rows[m][N][1]} for N in Ns}
                      for m in metrics}, "attmem": attmem}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
