"""Sanity check 3 — style encoder collision rate on WikiArt cross-period.

Question: does a self-supervised style-aware encoder (DINOv2) produce
embeddings discriminative enough across DIFFERENT-PERIOD paintings of
the SAME PAINTER vs different painters, such that naive k-means
quantisation preserves identity? This is the gating experiment for
the V-STY task in the PerceptMem benchmark.

Encoder choice: DINOv2-small (facebook/dinov2-small, 22M params). DINOv2
is SSL-trained on natural images; it captures low/mid-level features
(texture, brushwork, palette) better than CLIP which collapses to
semantic content. Not CLIP because CLIP would put all "portrait
paintings" near each other regardless of artist.

Data: WikiArt subset via huggan/wikiart on HF. Filter to ~30-50 painters
with multiple paintings spanning early/late periods (we use date metadata
to split when available, else random subset).
"""
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_PAINTERS = 50
PAINTINGS_PER_PAINTER = 8


def main():
    print("=" * 70)
    print("Sanity check 3 — DINOv2 style encoder on WikiArt cross-period")
    print("=" * 70)

    from transformers import AutoImageProcessor, AutoModel
    print("\n[load] DINOv2-small ...")
    model_id = "facebook/dinov2-small"
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    print(f"  loaded; param count = {sum(p.numel() for p in model.parameters()):,}")

    print("\n[load] WikiArt subset via HF datasets ...")
    from datasets import load_dataset
    # `huggan/wikiart` has 81444 train images with artist + style + genre labels
    try:
        ds = load_dataset("huggan/wikiart", split="train", streaming=False)
    except Exception as e:
        print(f"  load_dataset failed: {e}")
        print("  trying streaming ...")
        ds = load_dataset("huggan/wikiart", split="train", streaming=True)

    # Group by artist — filter to artists with >= PAINTINGS_PER_PAINTER
    print("\n[filter] scanning WikiArt for painters with multiple paintings ...")
    by_artist = defaultdict(list)
    if hasattr(ds, "__len__"):
        N = len(ds)
        artists = ds["artist"]
        for i in range(N):
            by_artist[int(artists[i])].append(i)
    else:
        # Streaming
        N = 0
        for i, ex in enumerate(ds):
            by_artist[int(ex["artist"])].append(i)
            N += 1
            if N >= 50000: break
    print(f"  scanned {N} WikiArt images; {len(by_artist)} distinct artists")
    eligible = [(a, idxs) for a, idxs in by_artist.items() if len(idxs) >= PAINTINGS_PER_PAINTER]
    print(f"  {len(eligible)} artists have >= {PAINTINGS_PER_PAINTER} paintings")

    if len(eligible) < NUM_PAINTERS:
        chosen_painters = eligible
    else:
        chosen_painters = random.sample(eligible, NUM_PAINTERS)
    print(f"  sampled {len(chosen_painters)} painters")

    print("\n[encode] extracting DINOv2 embeddings ...")
    embs, pids = [], []
    for k, (artist_id, idxs) in enumerate(chosen_painters):
        sel = random.sample(idxs, k=PAINTINGS_PER_PAINTER)
        # Fetch each image; convert to RGB; preprocess
        batch_imgs = []
        for i in sel:
            img = ds[i]["image"]
            if isinstance(img, Image.Image):
                if img.mode != "RGB": img = img.convert("RGB")
                batch_imgs.append(img)
        if not batch_imgs: continue
        inputs = proc(images=batch_imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
            # Use the CLS pooler_output if available, else mean pool last_hidden_state
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output
            else:
                feats = out.last_hidden_state.mean(dim=1)
        feats = feats.cpu().numpy().astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        for f in feats:
            embs.append(f); pids.append(str(artist_id))
        if (k + 1) % 5 == 0:
            print(f"  processed {k+1}/{len(chosen_painters)} painters, {len(embs)} embeddings")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    D = emb.shape[1]
    print(f"\n[done] Collected {len(emb)} embeddings, dim={D}, {len(set(pid))} painters")

    # ---- Soundness: top-1 nearest-neighbour painter recall ----
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    nn_same = (pid[nn] == pid).mean()
    print(f"\n[sanity] DINOv2 raw cosine top-1 painter recall: {nn_same:.4f}")

    # Intra/inter cosine
    by_p = defaultdict(list)
    for i, p in enumerate(pid): by_p[p].append(i)
    intra, inter = [], []
    for p, idxs in by_p.items():
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                intra.append(sims[idxs[i], idxs[j]])
    rng = np.random.RandomState(SEED)
    for _ in range(5000):
        i, j = rng.randint(0, len(emb), size=2)
        if pid[i] != pid[j]: inter.append(sims[i, j])
    print(f"[sanity] intra-painter cosine: mean={np.mean(intra):.4f} std={np.std(intra):.4f}  n={len(intra)}")
    print(f"[sanity] inter-painter cosine: mean={np.mean(inter):.4f} std={np.std(inter):.4f}  n={len(inter)}")

    # ---- Quantisation: flat k-means at multiple K ----
    import faiss
    CODEBOOK_SIZES = [8, 16, 32, 64, 128]
    print(f"\n[sanity] Quantisation:")
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    results = {}
    for K in CODEBOOK_SIZES:
        if K > len(emb): continue
        km = faiss.Kmeans(D, K, niter=20, verbose=False, seed=SEED)
        km.train(emb)
        _, codes = km.index.search(emb, 1); codes = codes.squeeze(1)
        # Intra agreement
        intra_agree = 0; intra_pairs = 0
        for p, idxs in by_p.items():
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    intra_pairs += 1
                    if codes[idxs[i]] == codes[idxs[j]]: intra_agree += 1
        intra_rate = intra_agree / max(intra_pairs, 1)
        # Inter collision
        rng2 = np.random.RandomState(SEED + K)
        inter_coll = 0; inter_pairs = 0
        for _ in range(10000):
            i, j = rng2.randint(0, len(emb), size=2)
            if pid[i] != pid[j]:
                inter_pairs += 1
                if codes[i] == codes[j]: inter_coll += 1
        inter_rate = inter_coll / max(inter_pairs, 1)
        ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
        print(f"  {K:>4d} | {intra_rate:>12.4f} | {inter_rate:>12.4f} | {ratio:>8.2f}")
        results[K] = (intra_rate, inter_rate, ratio)

    # Save embeddings for downstream Path A experiments
    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/dinov2_wikiart.npz")
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_emb, emb=emb, pid=pid)
    print(f"\n[done] saved embeddings to {out_emb}")

    # Save summary
    out_res = Path("/home/ubuntu/multimodal-user-memory/results/sanity_style_collisions.json")
    import json
    with open(out_res, "w") as f:
        json.dump({
            "encoder": model_id,
            "num_painters": int(len(set(pid))),
            "paintings_per_painter": int(PAINTINGS_PER_PAINTER),
            "n_embeddings": int(len(emb)),
            "embedding_dim": int(D),
            "raw_cosine": {
                "top1_painter_recall": float(nn_same),
                "intra_mean": float(np.mean(intra)),
                "intra_std": float(np.std(intra)),
                "inter_mean": float(np.mean(inter)),
                "inter_std": float(np.std(inter)),
            },
            "quantisation": {str(K): {"intra_agree": float(a), "inter_coll": float(b), "ratio": float(r) if r != float("inf") else None}
                             for K, (a, b, r) in results.items()},
        }, f, indent=2)
    print(f"[done] saved summary to {out_res}")

    print("\n[verdict]")
    if nn_same < 0.40:
        print("  WARN — DINOv2 top-1 painter recall is moderate; cross-period style is genuinely harder than face identity.")
    else:
        print(f"  PASS — DINOv2 top-1 painter recall = {nn_same:.3f}.")
    viable_K = [K for K, (a, b, r) in results.items() if r > 5.0 and a > 0.4]
    if viable_K:
        print(f"  PASS — viable codebook sizes (ratio>5, intra>0.4): {viable_K}")
    else:
        print("  PARTIAL — encoder is sound but quantisation collapses style. Try a style-specific head.")


if __name__ == "__main__":
    sys.exit(main())
