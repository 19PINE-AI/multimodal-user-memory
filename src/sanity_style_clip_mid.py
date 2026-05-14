"""Style encoder via CLIP INTERMEDIATE layers (not final).

Hypothesis: CLIP final-layer features (the standard pooler output) collapse
style into semantics — that's why DINOv2/CLIP top-1 painter recall was
only 0.24. Earlier CLIP layers preserve more low-level features
(texture, brushwork, palette) before semantic abstraction takes over.

Try: extract CLIP-ViT-B/32 hidden states at multiple intermediate layers,
mean-pool, concatenate, optionally project to lower dim. Compare to:
  - DINOv2 final: top-1 0.24 (baseline)
  - VGG-Gram + PCA-100: top-1 0.42 (best so far)
"""
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
    print("Style — CLIP intermediate-layer features")
    print("=" * 70)

    from transformers import CLIPModel, CLIPProcessor
    model_id = "openai/clip-vit-base-patch32"
    print(f"\n[load] {model_id}")
    proc = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
    n_layers = model.config.vision_config.num_hidden_layers
    print(f"  vision layers: {n_layers}, hidden_size: {model.config.vision_config.hidden_size}")

    from datasets import load_dataset
    ds = load_dataset("huggan/wikiart", split="train")
    artist_col = ds["artist"]
    by_artist = defaultdict(list)
    for i, a in enumerate(artist_col):
        by_artist[int(a)].append(i)
    eligible = [(a, idxs) for a, idxs in by_artist.items() if len(idxs) >= PAINTINGS_PER_PAINTER]
    chosen = random.sample(eligible, k=min(NUM_PAINTERS, len(eligible)))
    print(f"  sampled {len(chosen)} painters")

    # We want intermediate layers — middle of the stack typically preserves more low-level
    target_layers = [3, 6, 9]  # ViT-B/32 has 12 layers; layers 3, 6, 9 are mid-net
    print(f"  using CLIP vision layers: {target_layers}")

    print("\n[encode] CLIP intermediate features ...")
    embs, pids = [], []
    for k, (artist_id, idxs) in enumerate(chosen):
        sel = random.sample(idxs, k=PAINTINGS_PER_PAINTER)
        batch_imgs = []
        for i in sel:
            img = ds[i]["image"]
            if not isinstance(img, Image.Image): continue
            if img.mode != "RGB": img = img.convert("RGB")
            batch_imgs.append(img)
        if not batch_imgs: continue
        inputs = proc(images=batch_imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"],
                                              output_hidden_states=True)
        hidden_states = vision_out.hidden_states  # tuple of length n_layers+1
        # Mean-pool over patches at each target layer
        feats_per_layer = []
        for L in target_layers:
            h = hidden_states[L]  # [B, P, D]
            f = h.mean(dim=1)  # [B, D] mean over patches
            feats_per_layer.append(f.cpu().numpy().astype(np.float32))
        # Concatenate
        feats = np.concatenate(feats_per_layer, axis=1)  # [B, D*n_layers]
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        for f in feats:
            embs.append(f); pids.append(str(artist_id))
        if (k + 1) % 10 == 0:
            print(f"  processed {k+1}/{len(chosen)} painters, {len(embs)} embeddings, dim={feats.shape[1]}")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    D = emb.shape[1]
    print(f"\n[done] {emb.shape}, {len(set(pid))} painters, dim={D}")

    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    nn_same = (pid[nn] == pid).mean()
    print(f"\n[sanity] CLIP-mid top-1 painter recall: {nn_same:.4f}  (baseline DINOv2=0.24, Gram+PCA=0.42)")

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
    print(f"[sanity] intra-painter cosine: {np.mean(intra):.4f}  inter-painter cosine: {np.mean(inter):.4f}")

    import faiss
    print(f"\n[sanity] quantisation:")
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    results = {}
    for K in [8, 16, 32, 64, 128]:
        if K > len(emb): continue
        km = faiss.Kmeans(D, K, niter=20, verbose=False, seed=SEED)
        km.train(emb)
        _, codes = km.index.search(emb, 1); codes = codes.squeeze(1)
        intra_agree = 0; intra_pairs = 0
        for p, idxs in by_p.items():
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    intra_pairs += 1
                    if codes[idxs[i]] == codes[idxs[j]]: intra_agree += 1
        rng2 = np.random.RandomState(SEED + K)
        inter_coll = 0; inter_pairs = 0
        for _ in range(10000):
            i, j = rng2.randint(0, len(emb), size=2)
            if pid[i] != pid[j]:
                inter_pairs += 1
                if codes[i] == codes[j]: inter_coll += 1
        intra_rate = intra_agree / max(intra_pairs, 1)
        inter_rate = inter_coll / max(inter_pairs, 1)
        ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
        print(f"  {K:>4d} | {intra_rate:>12.4f} | {inter_rate:>12.4f} | {ratio:>8.2f}")
        results[K] = (intra_rate, inter_rate, ratio)

    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/clip_mid_wikiart.npz")
    np.savez(out_emb, emb=emb, pid=pid)
    print(f"\n[done] saved {out_emb}")

    import json
    with open("/home/ubuntu/multimodal-user-memory/results/sanity_style_clip_mid.json", "w") as f:
        json.dump({
            "encoder": f"{model_id} mid layers {target_layers}",
            "top1_recall": float(nn_same),
            "intra_mean": float(np.mean(intra)),
            "inter_mean": float(np.mean(inter)),
            "quantisation": {str(K): {"intra_agree": float(a), "inter_coll": float(b),
                                       "ratio": float(r) if r != float("inf") else None}
                             for K, (a, b, r) in results.items()},
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
