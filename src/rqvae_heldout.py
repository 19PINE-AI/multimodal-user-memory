"""Held-out generalisation check for learned RQ-VAE.

Concern: the §3 finding (learned RQ-VAE beats naive at depth) was
measured with the identity classifier trained on the same identities
we evaluate on. For real user memory the quantiser must generalise to
*unseen* identities — new users the agent has never met during
quantiser training.

Protocol:
  1. Split identities 50/50 into TRAIN (quantiser fit) and HELDOUT.
  2. Train learned RQ-VAE only on TRAIN identities (classifier sees only TRAIN ids).
  3. Quantise HELDOUT embeddings → compute intra/inter collision stats.
  4. Also train naive residual k-means on TRAIN embeddings (no labels) →
     quantise HELDOUT → same stats.
  5. Compare. Both should generalise (no label leakage); the learned one
     should keep its advantage *if* the identity-discrimination axes
     it allocated transfer.

Win condition: learned beats naive on held-out identities at deep configs
(L>=3 or L2_K>=64), at least within 2-3pp of the in-domain numbers.
"""

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
    ResidualVQ, naive_rq_kmeans, collision_stats,
    extract_or_load_audio_embeddings, extract_or_load_vision_embeddings,
)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


def split_by_identity(emb, pid, train_frac=0.5):
    rng = np.random.RandomState(SEED)
    unique = sorted(set(pid.tolist()))
    rng.shuffle(unique)
    n_train = int(len(unique) * train_frac)
    train_ids = set(unique[:n_train])
    train_mask = np.array([str(p) in train_ids for p in pid])
    return emb[train_mask], pid[train_mask], emb[~train_mask], pid[~train_mask]


def train_rqvae_heldout(train_emb, train_pid, n_levels, k_per_level, epochs=400, lr=3e-3, lambda_cls=1.0):
    D = train_emb.shape[1]
    unique_pids, pid_ints = np.unique(train_pid, return_inverse=True)
    n_classes = len(unique_pids)
    x = torch.from_numpy(train_emb).to(DEVICE)
    y = torch.from_numpy(pid_ints).long().to(DEVICE)
    rvq = ResidualVQ(D, n_levels, k_per_level).to(DEVICE)
    cls_head = nn.Linear(D, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(list(rvq.parameters()) + list(cls_head.parameters()), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        rvq.train()
        q, codes, cl = rvq(x)
        recon_loss = F.mse_loss(q, x)
        logits = cls_head(q)
        cls_loss = F.cross_entropy(logits, y)
        loss = recon_loss + 0.25 * cl + lambda_cls * cls_loss
        opt.zero_grad(); loss.backward(); opt.step()
    return rvq


def quantise(rvq, emb_np):
    rvq.eval()
    with torch.no_grad():
        x = torch.from_numpy(emb_np).to(DEVICE)
        _, codes, _ = rvq(x)
    return codes.cpu().numpy()


def naive_kmeans_heldout(train_emb, n_levels, k_per_level):
    """Train k-means on train_emb, return a function that quantises new embeddings."""
    import faiss
    D = train_emb.shape[1]
    centroids = []
    residual = train_emb.copy()
    for L in range(n_levels):
        km = faiss.Kmeans(D, k_per_level, niter=20, verbose=False, seed=SEED + L)
        km.train(residual)
        _, c = km.index.search(residual, 1)
        c = c.squeeze(1)
        centroids.append(km.centroids.copy())
        residual = residual - km.centroids[c]

    def apply(emb_np):
        residual = emb_np.copy()
        codes = np.zeros((len(emb_np), n_levels), dtype=np.int64)
        for L, c_arr in enumerate(centroids):
            # nearest centroid
            d2 = (residual ** 2).sum(1, keepdims=True) - 2 * residual @ c_arr.T + (c_arr ** 2).sum(1)
            idx = d2.argmin(1)
            codes[:, L] = idx
            residual = residual - c_arr[idx]
        return codes
    return apply


def main():
    cache_dir = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    results = {}
    for modality, extractor in [
        ("audio", lambda: extract_or_load_audio_embeddings(cache_dir / "ecapa_libri.npz")),
        ("vision", lambda: extract_or_load_vision_embeddings(cache_dir / "arcface_lfw.npz")),
    ]:
        print(f"\n{'='*60}\n{modality.upper()} (held-out generalisation)\n{'='*60}")
        emb, pid = extractor()
        train_emb, train_pid, test_emb, test_pid = split_by_identity(emb, pid, train_frac=0.5)
        print(f"  Train: {len(train_emb)} embs, {len(set(train_pid))} identities")
        print(f"  Test:  {len(test_emb)} embs, {len(set(test_pid))} identities  (DISJOINT from train)")
        modality_results = {}
        for n_levels, k_per in [(2, 16), (2, 64), (3, 16), (4, 16)]:
            cfg = f"L{n_levels}_K{k_per}"
            eff_K = k_per ** n_levels
            print(f"\n  config: {n_levels} levels x {k_per} codes  (eff_K={eff_K})")

            # naive — fit on train, quantise test
            apply_naive = naive_kmeans_heldout(train_emb, n_levels, k_per)
            n_codes = apply_naive(test_emb)
            n_intra, n_inter, n_ratio = collision_stats(n_codes, test_pid)
            print(f"  [naive]   intra={n_intra:.4f}  inter={n_inter:.4f}  ratio={n_ratio:.2f}")

            # learned — fit on train (with classifier over train identities), quantise test
            print(f"  [learned] training on train identities ...")
            rvq = train_rqvae_heldout(train_emb, train_pid, n_levels, k_per)
            l_codes = quantise(rvq, test_emb)
            l_intra, l_inter, l_ratio = collision_stats(l_codes, test_pid)
            print(f"  [learned] intra={l_intra:.4f}  inter={l_inter:.4f}  ratio={l_ratio:.2f}")

            modality_results[cfg] = {
                "eff_K": eff_K,
                "naive": {"intra": float(n_intra), "inter": float(n_inter), "ratio": float(n_ratio) if n_ratio != float("inf") else None},
                "learned": {"intra": float(l_intra), "inter": float(l_inter), "ratio": float(l_ratio) if l_ratio != float("inf") else None},
                "delta_intra": float(l_intra - n_intra),
            }
        results[modality] = modality_results

    out = Path("/home/ubuntu/multimodal-user-memory/results/rqvae_heldout.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("HELD-OUT GENERALISATION: learned RQ-VAE vs naive RQ k-means")
    print("=" * 70)
    print(f"{'modality':>10} | {'config':>10} | {'eff_K':>8} | {'naive':>8} | {'learned':>8} | {'Δ':>7}")
    print("-" * 70)
    for mod, mr in results.items():
        for cfg, d in mr.items():
            print(f"{mod:>10} | {cfg:>10} | {d['eff_K']:>8} | {d['naive']['intra']:>8.4f} | {d['learned']['intra']:>8.4f} | {d['delta_intra']:>+7.4f}")


if __name__ == "__main__":
    sys.exit(main())
