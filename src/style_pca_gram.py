"""Style encoder fix: PCA-project Gram features to a tractable dim.

v2 found Gram-matrix style descriptors (174K dim) give top-1 painter
recall 0.44 — a real lift over DINOv2's 0.24. But k-means quantisation
at 174K dim is degenerate. PCA to ~512 should preserve style signal
while making the codebook learnable.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
np.random.seed(SEED)


def main():
    print("=" * 70)
    print("Style: PCA-project Gram features to 512-dim")
    print("=" * 70)

    d = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/gram_wikiart_distinctive.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"]
    print(f"  loaded: emb {emb.shape}, {len(set(pid))} painters")

    # PCA — cap to min(n_samples - 1, 100) since we only have 120 samples
    from sklearn.decomposition import PCA
    target_dim = min(100, emb.shape[0] - 1)
    pca = PCA(n_components=target_dim, random_state=SEED)
    pca.fit(emb)
    emb_reduced = pca.transform(emb).astype(np.float32)
    emb_reduced /= np.linalg.norm(emb_reduced, axis=1, keepdims=True) + 1e-9
    print(f"  PCA -> {emb_reduced.shape}; explained variance ratio sum = {pca.explained_variance_ratio_.sum():.3f}")

    # Same eval as sanity_style
    sims = emb_reduced @ emb_reduced.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    nn_same = (pid[nn] == pid).mean()
    print(f"\n[sanity] PCA-Gram top-1 painter recall: {nn_same:.4f}")

    by_p = defaultdict(list)
    for i, p in enumerate(pid): by_p[p].append(i)
    intra, inter = [], []
    for p, idxs in by_p.items():
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                intra.append(sims[idxs[i], idxs[j]])
    rng = np.random.RandomState(SEED)
    for _ in range(5000):
        i, j = rng.randint(0, len(emb_reduced), size=2)
        if pid[i] != pid[j]: inter.append(sims[i, j])
    print(f"[sanity] intra-painter cosine: mean={np.mean(intra):.4f}")
    print(f"[sanity] inter-painter cosine: mean={np.mean(inter):.4f}")

    import faiss
    print(f"\n[sanity] Quantisation on PCA-Gram-{target_dim}d:")
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    results = {}
    for K in [4, 8, 16, 32, 64]:
        if K > len(emb_reduced): continue
        km = faiss.Kmeans(target_dim, K, niter=20, verbose=False, seed=SEED)
        km.train(emb_reduced)
        _, codes = km.index.search(emb_reduced, 1); codes = codes.squeeze(1)
        intra_agree = 0; intra_pairs = 0
        for p, idxs in by_p.items():
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    intra_pairs += 1
                    if codes[idxs[i]] == codes[idxs[j]]: intra_agree += 1
        rng2 = np.random.RandomState(SEED + K)
        inter_coll = 0; inter_pairs = 0
        for _ in range(10000):
            i, j = rng2.randint(0, len(emb_reduced), size=2)
            if pid[i] != pid[j]:
                inter_pairs += 1
                if codes[i] == codes[j]: inter_coll += 1
        intra_rate = intra_agree / max(intra_pairs, 1)
        inter_rate = inter_coll / max(inter_pairs, 1)
        ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
        print(f"  {K:>4d} | {intra_rate:>12.4f} | {inter_rate:>12.4f} | {ratio:>8.2f}")
        results[K] = (intra_rate, inter_rate, ratio)

    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/style_pca_gram.npz")
    np.savez(out_emb, emb=emb_reduced, pid=pid)
    print(f"\n[done] saved {out_emb}")

    import json
    with open("/home/ubuntu/multimodal-user-memory/results/sanity_style_pca_gram.json", "w") as f:
        json.dump({
            "encoder": "VGG16-Gram + PCA-512",
            "top1_recall": float(nn_same),
            "intra_mean": float(np.mean(intra)),
            "inter_mean": float(np.mean(inter)),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "quantisation": {str(K): {"intra_agree": float(a), "inter_coll": float(b),
                                       "ratio": float(r) if r != float("inf") else None}
                             for K, (a, b, r) in results.items()},
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
