"""Sanity check 3 v2 — style with distinctive painters and Gram-matrix descriptors.

v1 finding: DINOv2 features on 50 random WikiArt painters give intra-cosine
0.44 vs inter-cosine 0.32 (only 12pp gap). Two hypotheses:
  (a) DINOv2 captures genre as much as style — random painters span
      same genres (portraits, landscapes), so intra and inter overlap.
  (b) DINOv2 features are too semantic.

This script tries both fixes:
  1. Restrict to artists with HIGHLY distinctive visual styles
     (Pollock, Mondrian, Monet, Van Gogh, Picasso, ...).
  2. Replace DINOv2 mean-pooled features with Gram-matrix style descriptors
     (Gatys et al. 2016 NST) extracted from a VGG-like backbone — the
     canonical style-vs-content disentanglement.

Use DINOv2 with FIRST-LAYER (lower-level) features as a middle ground.
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

# Highly distinctive-style painters from WikiArt; targets we KNOW have unique style
DISTINCTIVE_PAINTERS = [
    "vincent-van-gogh", "claude-monet", "pablo-picasso", "jackson-pollock",
    "piet-mondrian", "salvador-dali", "henri-matisse", "wassily-kandinsky",
    "edvard-munch", "paul-cezanne", "paul-gauguin", "andy-warhol",
    "georgia-okeeffe", "marc-chagall", "pierre-auguste-renoir", "edgar-degas",
    "frida-kahlo", "joan-miro", "rene-magritte", "mark-rothko",
    "gustav-klimt", "egon-schiele", "francisco-goya", "diego-rivera",
    "winslow-homer",
]


def gram_matrix(features):
    """Gram matrix of [B, C, H, W] features. Returns flattened upper triangle."""
    B, C, H, W = features.shape
    f = features.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1, 2)) / (C * H * W)  # [B, C, C]
    # Take upper triangle (excluding diagonal) — symmetric, so half is enough
    iu = torch.triu_indices(C, C, offset=0)
    return G[:, iu[0], iu[1]]  # [B, C*(C+1)/2]


def main():
    print("=" * 70)
    print("Sanity check 3 v2 — distinctive painters + Gram-matrix style")
    print("=" * 70)

    print("\n[load] VGG-16 (ImageNet pretrained) for Gram features ...")
    from torchvision.models import vgg16, VGG16_Weights
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
    # Use multiple layer activations
    layer_idx = [3, 8, 15, 22]  # conv layers in VGG16.features
    print(f"  using activations at layer indices {layer_idx}")

    print("\n[load] WikiArt; filter to distinctive painters ...")
    from datasets import load_dataset
    ds = load_dataset("huggan/wikiart", split="train")
    print(f"  total images: {len(ds)}")
    artist_names = ds.features["artist"].names if hasattr(ds.features["artist"], "names") else None
    if artist_names is None:
        print("  no artist names index; falling back to numeric IDs (using ANY artist)")
        artist_names = []
    print(f"  num artist names: {len(artist_names)}")
    # Find distinctive painter indices
    name_to_id = {n.lower().replace(" ", "-").replace("_", "-"): i for i, n in enumerate(artist_names)}
    distinctive_ids = []
    matched_names = []
    for p in DISTINCTIVE_PAINTERS:
        for k, v in name_to_id.items():
            if p in k or k in p:
                distinctive_ids.append(v)
                matched_names.append(artist_names[v])
                break
    print(f"  matched {len(distinctive_ids)} distinctive painters: {matched_names}")
    if not distinctive_ids:
        print("  no matches; aborting")
        return

    artist_col = ds["artist"]
    by_artist = defaultdict(list)
    for i, a in enumerate(artist_col):
        if int(a) in distinctive_ids:
            by_artist[int(a)].append(i)
    print(f"  painting counts per painter: {[(artist_names[k], len(v)) for k, v in by_artist.items()]}")

    PAINTINGS_PER_PAINTER = 8
    eligible = [(a, idxs) for a, idxs in by_artist.items() if len(idxs) >= PAINTINGS_PER_PAINTER]
    print(f"  {len(eligible)} painters have >= {PAINTINGS_PER_PAINTER} paintings")

    # Preprocess: resize + normalise like ImageNet
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\n[encode] Gram-matrix features ...")
    embs, pids = [], []
    with torch.no_grad():
        for k, (artist_id, idxs) in enumerate(eligible):
            sel = random.sample(idxs, k=PAINTINGS_PER_PAINTER)
            for i in sel:
                img = ds[i]["image"]
                if not isinstance(img, Image.Image): continue
                if img.mode != "RGB": img = img.convert("RGB")
                x = tf(img).unsqueeze(0).to(DEVICE)
                # Extract activations at chosen layers
                grams = []
                cur = x
                for li, layer in enumerate(vgg):
                    cur = layer(cur)
                    if li in layer_idx:
                        g = gram_matrix(cur)[0]  # [C*(C+1)/2]
                        # Normalise per-layer to balance scale across layers
                        g = g / (g.norm() + 1e-9)
                        grams.append(g.cpu().numpy())
                feat = np.concatenate(grams).astype(np.float32)
                feat = feat / (np.linalg.norm(feat) + 1e-9)
                embs.append(feat); pids.append(artist_names[artist_id])
            if (k + 1) % 5 == 0:
                print(f"  processed {k+1}/{len(eligible)} painters, {len(embs)} embs, dim={len(embs[0])}")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    D = emb.shape[1]
    print(f"\n[done] {emb.shape}, {len(set(pid))} painters, dim={D}")

    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    print(f"\n[sanity] Gram-style top-1 painter recall: {(pid[nn] == pid).mean():.4f}")

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
    print(f"[sanity] intra-painter cosine: mean={np.mean(intra):.4f}")
    print(f"[sanity] inter-painter cosine: mean={np.mean(inter):.4f}")

    import faiss
    print(f"\n[sanity] Quantisation:")
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    results = {}
    for K in [4, 8, 16, 32, 64]:
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

    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings") / "gram_wikiart_distinctive.npz"
    np.savez(out_emb, emb=emb, pid=pid)
    print(f"\n[done] saved {out_emb}")

    out_res = Path("/home/ubuntu/multimodal-user-memory/results") / "sanity_style_v2_distinctive.json"
    import json
    with open(out_res, "w") as f:
        json.dump({
            "encoder": "VGG16-gram-matrix",
            "painters_targeted": DISTINCTIVE_PAINTERS,
            "painters_matched": [artist_names[i] for i in distinctive_ids if any(idxs for k, idxs in by_artist.items() if k == i)],
            "n_embeddings": int(len(emb)),
            "n_painters": int(len(set(pid))),
            "embedding_dim": int(D),
            "raw_cosine": {
                "top1_painter_recall": float((pid[nn] == pid).mean()),
                "intra_mean": float(np.mean(intra)),
                "inter_mean": float(np.mean(inter)),
            },
            "quantisation": {str(K): {"intra_agree": float(a), "inter_coll": float(b),
                                       "ratio": float(r) if r != float("inf") else None}
                             for K, (a, b, r) in results.items()},
        }, f, indent=2)
    print(f"[done] saved summary")


if __name__ == "__main__":
    sys.exit(main())
