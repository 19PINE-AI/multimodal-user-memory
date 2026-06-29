"""Exp 5 -- is k-means optimal for the perceptual capacity law?

set_memory.py compresses M registered identity-keys into k hard prototype slots
(spherical k-means) and recognises a cross-condition query by nearest slot. The
empirical law is recall ~ min(1, k/M). Question for a reviewer: is that a k-means
artifact, or fundamental? We test a *learned* compressor against k-means.

Two budgets:
  learned_hard : k slot vectors, each identity hard-assigned to one slot (the slot
                 reports one representative id). STRICT k-slot budget, O(k) storage,
                 O(1)/identity. Optimised by gradient descent on noise-augmented
                 queries (self-supervised; never sees the real eval query).
  learned_soft : k slot vectors + an M x k soft code per identity (reconstruction
                 = softmax(code) @ slots). RELAXED budget: O(M*k) storage, no longer
                 O(1)/identity. This is the only way to beat the hard pigeonhole.

If learned_hard ~ k-means ~ min(1,k/M), the law is fundamental (pigeonhole: k
slots can keep at most ~k of M identities separable). learned_soft shows the price
of beating it is O(M) storage -- at which point you may as well keep the M keys.

CPU only (small). Usage: python3 learned_compressor.py --emb arcface_lfw_xxxl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/multimodal-user-memory")
EMB = ROOT / "runs" / "embeddings"


def l2(X, ax=-1):
    return X / (np.linalg.norm(X, axis=ax, keepdims=True) + 1e-9)


def spherical_kmeans(X, k, seed, iters=50):
    n = len(X)
    if k >= n:
        return X.copy()
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(n, k, replace=False)].copy()
    assign = np.full(n, -1)
    for _ in range(iters):
        new = (X @ cent.T).argmax(1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            m = assign == j
            if m.any():
                cent[j] = l2(X[m].mean(0), ax=0)
    return cent


def recall_from_slots(slots, K, Q):
    """Hard slot budget: each slot -> its representative reg id; query -> slot -> id."""
    proto_id = (slots @ K.T).argmax(1)
    pred = proto_id[(Q @ slots.T).argmax(1)]
    return float((pred == np.arange(len(K))).mean())


def learned_hard(K, k, seed, sigma, steps=400, lr=0.05):
    """Learn k slot vectors by gradient descent; hard slot->id at eval.
    Self-supervised: augment reg keys with sigma-noise to mimic cross-condition
    queries, train slots so an augmented key's nearest slot is dominated by its id."""
    import torch
    M, D = K.shape
    rng = np.random.default_rng(seed)
    S = torch.tensor(spherical_kmeans(K, k, seed), dtype=torch.float32, requires_grad=True)
    Kt = torch.tensor(K, dtype=torch.float32)
    opt = torch.optim.Adam([S], lr=lr)
    ids = torch.arange(M)
    for _ in range(steps):
        noise = torch.tensor(rng.normal(0, sigma, size=(M, D)), dtype=torch.float32)
        Qa = torch.nn.functional.normalize(Kt + noise, dim=1)        # augmented queries
        Sn = torch.nn.functional.normalize(S, dim=1)
        # soft slot membership of each reg key -> soft slot->id responsibility
        rk = torch.softmax((Kt @ Sn.T) * 10.0, dim=0)               # [M,k] id-mass per slot
        qslot = torch.softmax((Qa @ Sn.T) * 10.0, dim=1)            # [M,k] query->slot
        pid = qslot @ rk.T                                          # [M,M] query->id score
        loss = torch.nn.functional.cross_entropy(pid, ids)
        opt.zero_grad(); loss.backward(); opt.step()
    return torch.nn.functional.normalize(S, dim=1).detach().numpy()


def learned_soft(K, k, seed, sigma, steps=400, lr=0.05):
    """Relaxed budget: k slots + M x k soft codes. Reconstruction = softmax(code)@slots.
    Predict query id by nearest reconstruction. O(M*k) storage."""
    import torch
    M, D = K.shape
    rng = np.random.default_rng(seed)
    S = torch.tensor(spherical_kmeans(K, min(k, M), seed) if k < M else K.copy(),
                     dtype=torch.float32)
    if S.shape[0] < k:  # pad
        extra = torch.tensor(rng.normal(0, 1, (k - S.shape[0], D)), dtype=torch.float32)
        S = torch.cat([S, extra])
    S = S.clone().requires_grad_(True)
    code = torch.zeros(M, k, requires_grad=True)
    Kt = torch.tensor(K, dtype=torch.float32)
    opt = torch.optim.Adam([S, code], lr=lr)
    ids = torch.arange(M)
    for _ in range(steps):
        noise = torch.tensor(rng.normal(0, sigma, size=(M, D)), dtype=torch.float32)
        Qa = torch.nn.functional.normalize(Kt + noise, dim=1)
        recon = torch.nn.functional.normalize(torch.softmax(code, 1) @ S, dim=1)  # [M,D]
        sims = Qa @ recon.T                                        # [M,M]
        loss = torch.nn.functional.cross_entropy(sims * 10.0, ids)
        opt.zero_grad(); loss.backward(); opt.step()
    recon = torch.nn.functional.normalize(torch.softmax(code, 1) @ S, dim=1).detach().numpy()
    return recon  # M reconstructions (one per id)


def recall_soft(recon, Q):
    pred = (Q @ recon.T).argmax(1)
    return float((pred == np.arange(len(recon))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="arcface_lfw_xxxl")
    ap.add_argument("--cells", default="8:4,16:8,32:8,32:16,64:16",
                    help="M:k cells where k<M (room to beat the law)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = np.load(EMB / f"{args.emb}.npz")
    emb = l2(d["emb"].astype(np.float32))
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    by = defaultdict(list)
    for i, p in enumerate(pid):
        by[str(p)].append(i)
    ids_all = [p for p in by if len(by[p]) >= 2]

    # estimate cross-condition noise sigma: mean ||reg-query|| over same-id pairs
    rng0 = np.random.default_rng(7)
    diffs = []
    for p in rng0.choice(ids_all, size=min(200, len(ids_all)), replace=False):
        ix = list(by[p]); rng0.shuffle(ix)
        diffs.append(np.linalg.norm(emb[ix[0]] - emb[ix[1]]))
    sigma = float(np.mean(diffs)) / np.sqrt(emb.shape[1])
    print(f"{args.emb}: {len(ids_all)} ids, dim={emb.shape[1]}, est sigma={sigma:.4f}")

    cells = [tuple(int(x) for x in c.split(":")) for c in args.cells.split(",")]
    print(f"\n=== LEARNED COMPRESSOR vs k-means (recall@1, mean over {len(args.seeds)} seeds) ===")
    print(f"{'M':>4} {'k':>4} | {'min(1,k/M)':>10} {'kmeans':>8} {'learn_hard':>11} {'learn_soft':>11}")
    rows = []
    for M, k in cells:
        km, lh, ls = [], [], []
        for s in args.seeds:
            rng = np.random.default_rng(1000 + s)
            sel = rng.choice(ids_all, size=M, replace=False)
            reg, qry = [], []
            for p in sel:
                ix = list(by[str(p)]); rng.shuffle(ix)
                reg.append(ix[0]); qry.append(ix[1])
            K = emb[reg]; Q = emb[qry]
            km.append(recall_from_slots(spherical_kmeans(K, k, s), K, Q))
            lh.append(recall_from_slots(learned_hard(K, k, s, sigma), K, Q))
            ls.append(recall_soft(learned_soft(K, k, s, sigma), Q))
        law = min(1.0, k / M)
        r = {"M": M, "k": k, "law": law, "kmeans": float(np.mean(km)),
             "learned_hard": float(np.mean(lh)), "learned_soft": float(np.mean(ls))}
        rows.append(r)
        print(f"{M:>4} {k:>4} | {law:>10.3f} {r['kmeans']:>8.3f} "
              f"{r['learned_hard']:>11.3f} {r['learned_soft']:>11.3f}")
    print("\nRead: learned_hard ~ kmeans ~ min(1,k/M) => the law is fundamental "
          "(pigeonhole). learned_soft beats it only by paying O(M*k) storage.")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"emb": args.emb, "sigma": sigma, "rows": rows}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
