"""Quantiser bakeoff on held-out identities.

Compares four quantisers on the same held-out evaluation protocol:
  A. naive   — residual k-means (no learning)
  B. recon   — RQ-VAE trained with reconstruction + commitment only
  C. cls     — RQ-VAE trained with reconstruction + commitment + identity classifier  (the failed §3 baseline)
  D. contr   — RQ-VAE trained with reconstruction + commitment + NT-Xent contrastive on same-identity pairs

Setup:
  - Identities split 50/50 into TRAIN (quantiser fit) and HELDOUT (eval).
  - For (C) and (D), positives come from same-identity pairs within TRAIN.
  - All four quantised codes evaluated on HELDOUT — intra/inter collision rates and ratio.

We're looking for: does ANY learned variant beat naive on held-out at scale (eff_K >= 1024)?
"""

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from learned_rqvae import (
    ResidualVQ, extract_or_load_audio_embeddings,
    extract_or_load_vision_embeddings,
)
from rqvae_heldout import split_by_identity, naive_kmeans_heldout, collision_stats

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

EPOCHS = 600
LR = 3e-3
TEMP = 0.1  # NT-Xent temperature
BATCH_FRAC = 1.0  # full-batch (small dataset)


def make_same_id_pairs(pid_int_arr):
    """Return list of (i, j) index pairs with same identity, i < j."""
    by_id = defaultdict(list)
    for i, p in enumerate(pid_int_arr):
        by_id[int(p)].append(i)
    pairs = []
    for p, idxs in by_id.items():
        if len(idxs) >= 2:
            pairs.extend(list(itertools.combinations(idxs, 2)))
    return pairs


def train_recon(train_emb, n_levels, k_per):
    D = train_emb.shape[1]
    rvq = ResidualVQ(D, n_levels, k_per).to(DEVICE)
    opt = torch.optim.AdamW(rvq.parameters(), lr=LR, weight_decay=1e-4)
    x = torch.from_numpy(train_emb).to(DEVICE)
    for _ in range(EPOCHS):
        q, _, cl = rvq(x)
        loss = F.mse_loss(q, x) + 0.25 * cl
        opt.zero_grad(); loss.backward(); opt.step()
    return rvq


def train_cls(train_emb, train_pid_int, n_levels, k_per, lambda_cls=1.0):
    D = train_emb.shape[1]
    n_classes = int(train_pid_int.max()) + 1
    rvq = ResidualVQ(D, n_levels, k_per).to(DEVICE)
    head = nn.Linear(D, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(list(rvq.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-4)
    x = torch.from_numpy(train_emb).to(DEVICE)
    y = torch.from_numpy(train_pid_int).long().to(DEVICE)
    for _ in range(EPOCHS):
        q, _, cl = rvq(x)
        loss = F.mse_loss(q, x) + 0.25 * cl + lambda_cls * F.cross_entropy(head(q), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return rvq


def train_contrastive(train_emb, train_pid_int, n_levels, k_per, lambda_con=1.0):
    """NT-Xent on quantised reps. Positives = same-identity pair; negatives = rest of batch."""
    D = train_emb.shape[1]
    rvq = ResidualVQ(D, n_levels, k_per).to(DEVICE)
    opt = torch.optim.AdamW(rvq.parameters(), lr=LR, weight_decay=1e-4)
    x_all = torch.from_numpy(train_emb).to(DEVICE)

    pairs = make_same_id_pairs(train_pid_int)
    if not pairs:
        return rvq  # not enough data
    pair_arr = np.array(pairs, dtype=np.int64)
    rng = np.random.RandomState(SEED)

    batch_size = min(64, len(pair_arr))
    for epoch in range(EPOCHS):
        # Sample a batch of pairs
        idx = rng.choice(len(pair_arr), size=batch_size, replace=len(pair_arr) < batch_size)
        b = pair_arr[idx]
        anchors = x_all[b[:, 0]]
        positives = x_all[b[:, 1]]

        q_a, _, cl_a = rvq(anchors)
        q_p, _, cl_p = rvq(positives)

        # Reconstruction part on both
        recon = F.mse_loss(q_a, anchors) + F.mse_loss(q_p, positives)
        commit = cl_a + cl_p

        # NT-Xent: build 2N x 2N similarity matrix
        z = torch.cat([F.normalize(q_a, dim=-1), F.normalize(q_p, dim=-1)], dim=0)  # (2B, D)
        sim = z @ z.t() / TEMP  # (2B, 2B)
        N = q_a.size(0)
        labels = torch.arange(2 * N, device=DEVICE)
        labels = (labels + N) % (2 * N)  # positive of i is i+N (and vice versa)
        # Mask self
        mask = torch.eye(2 * N, device=DEVICE, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        con_loss = F.cross_entropy(sim, labels)

        loss = recon + 0.25 * commit + lambda_con * con_loss
        opt.zero_grad(); loss.backward(); opt.step()
    return rvq


def quantise(rvq, emb_np):
    rvq.eval()
    with torch.no_grad():
        x = torch.from_numpy(emb_np).to(DEVICE)
        _, codes, _ = rvq(x)
    return codes.cpu().numpy()


def main():
    cache_dir = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    results = {}

    configs = [(2, 16), (2, 64), (3, 16), (3, 32), (4, 16)]

    for modality, extractor in [
        ("audio", lambda: extract_or_load_audio_embeddings(cache_dir / "ecapa_libri.npz")),
        ("vision", lambda: extract_or_load_vision_embeddings(cache_dir / "arcface_lfw.npz")),
    ]:
        print(f"\n{'=' * 70}\n{modality.upper()}\n{'=' * 70}")
        emb, pid = extractor()
        train_emb, train_pid, test_emb, test_pid = split_by_identity(emb, pid, train_frac=0.5)
        # int-encoded training pids (held-out pids stay as strings; we only need int for cls/contr training)
        _, train_pid_int = np.unique(train_pid, return_inverse=True)
        print(f"  Train: {len(train_emb)} embs / {len(set(train_pid))} ids   |   Test: {len(test_emb)} embs / {len(set(test_pid))} ids")

        modality_results = {}
        for n_levels, k_per in configs:
            cfg = f"L{n_levels}_K{k_per}"
            eff_K = k_per ** n_levels
            print(f"\n  [{cfg} eff_K={eff_K}]")

            # A. naive
            naive_apply = naive_kmeans_heldout(train_emb, n_levels, k_per)
            codes_naive = naive_apply(test_emb)
            stats_naive = collision_stats(codes_naive, test_pid)
            print(f"    naive  : intra={stats_naive[0]:.4f}  inter={stats_naive[1]:.4f}  ratio={stats_naive[2]:.2f}")

            # B. reconstruction only
            rvq_r = train_recon(train_emb, n_levels, k_per)
            codes_r = quantise(rvq_r, test_emb)
            stats_r = collision_stats(codes_r, test_pid)
            print(f"    recon  : intra={stats_r[0]:.4f}  inter={stats_r[1]:.4f}  ratio={stats_r[2]:.2f}")

            # C. classifier
            rvq_c = train_cls(train_emb, train_pid_int, n_levels, k_per)
            codes_c = quantise(rvq_c, test_emb)
            stats_c = collision_stats(codes_c, test_pid)
            print(f"    cls    : intra={stats_c[0]:.4f}  inter={stats_c[1]:.4f}  ratio={stats_c[2]:.2f}")

            # D. contrastive
            rvq_d = train_contrastive(train_emb, train_pid_int, n_levels, k_per)
            codes_d = quantise(rvq_d, test_emb)
            stats_d = collision_stats(codes_d, test_pid)
            print(f"    contr  : intra={stats_d[0]:.4f}  inter={stats_d[1]:.4f}  ratio={stats_d[2]:.2f}")

            modality_results[cfg] = {
                "eff_K": eff_K,
                "naive":  dict(zip(("intra", "inter", "ratio"), [float(v) if v != float("inf") else None for v in stats_naive])),
                "recon":  dict(zip(("intra", "inter", "ratio"), [float(v) if v != float("inf") else None for v in stats_r])),
                "cls":    dict(zip(("intra", "inter", "ratio"), [float(v) if v != float("inf") else None for v in stats_c])),
                "contr":  dict(zip(("intra", "inter", "ratio"), [float(v) if v != float("inf") else None for v in stats_d])),
            }
        results[modality] = modality_results

    out = Path("/home/ubuntu/multimodal-user-memory/results/quantiser_bakeoff.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Headline
    print("\n" + "=" * 100)
    print("BAKEOFF HEADLINE — held-out ratio (intra / inter; higher = more discriminative)")
    print("=" * 100)
    print(f"{'modality':>8} | {'config':>10} | {'eff_K':>6} | {'naive':>7} | {'recon':>7} | {'cls':>7} | {'contr':>7} | best")
    print("-" * 100)
    for mod, mr in results.items():
        for cfg, d in mr.items():
            r_naive = d['naive']['ratio'] if d['naive']['ratio'] is not None else float('inf')
            r_recon = d['recon']['ratio'] if d['recon']['ratio'] is not None else float('inf')
            r_cls = d['cls']['ratio'] if d['cls']['ratio'] is not None else float('inf')
            r_contr = d['contr']['ratio'] if d['contr']['ratio'] is not None else float('inf')
            ratios = {'naive': r_naive, 'recon': r_recon, 'cls': r_cls, 'contr': r_contr}
            best = max(ratios.items(), key=lambda kv: kv[1])
            print(f"{mod:>8} | {cfg:>10} | {d['eff_K']:>6} | {r_naive:>7.2f} | {r_recon:>7.2f} | {r_cls:>7.2f} | {r_contr:>7.2f} | {best[0]}({best[1]:.2f})")

    print(f"\n[done] Wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
