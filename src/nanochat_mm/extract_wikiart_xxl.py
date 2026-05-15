"""Extract CLIP-mid features from huggan/wikiart for a larger painter pool.

The current style_pca_gram.npz uses 15 painters, clip_mid_wikiart.npz uses 50.
The full huggan/wikiart has 129 painters (one labelled "Unknown Artist"
which we drop) — enough data to break the cross-condition style data
ceiling.

Sampling protocol: take up to N works per painter (default 30). Keep
painters that have at least min_per_artist works. Train/eval split is
identity-disjoint via v2_retrieval.split_by_identity.

Encoder: same CLIP-ViT mid-layers (3, 6, 9) concat as the existing
clip_mid_wikiart.npz, so the new corpus drops cleanly into the existing
v-sty-clip pipeline.
"""
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_OUT = "/home/ubuntu/multimodal-user-memory/runs/embeddings/clip_mid_wikiart_xxl.npz"


def extract_clip_mid_features(images, processor, model, batch=8):
    """Reproduce the clip-mid encoder used in sanity_style_clip_mid.py:
    concatenate hidden states at layers 3, 6, 9 (post-LN), mean-pool over
    patches, then L2-normalize."""
    all_emb = []
    n = len(images); t0 = time.time()
    for i in range(0, n, batch):
        batch_imgs = images[i:i + batch]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.vision_model(
                **inputs, output_hidden_states=True,
            )
        hs = out.hidden_states  # tuple of (n_layers+1) tensors [B, T, D]
        # Take layers 3, 6, 9 (skip patch token 0 = [CLS])
        h3 = hs[3][:, 1:, :].mean(1)
        h6 = hs[6][:, 1:, :].mean(1)
        h9 = hs[9][:, 1:, :].mean(1)
        z = torch.cat([h3, h6, h9], dim=-1)
        z = F.normalize(z, dim=-1)
        all_emb.append(z.cpu().numpy())
        if (i // batch + 1) % 20 == 0:
            print(f"    encoded {i+batch}/{n}  ({time.time()-t0:.0f}s)")
    return np.concatenate(all_emb, axis=0).astype(np.float32)


def main():
    n_per_artist = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    min_per_artist = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_artists = int(sys.argv[3]) if len(sys.argv) > 3 else 129

    print(f"WikiArt XXL extraction: {n_per_artist} works/artist, "
          f"min {min_per_artist}, max {max_artists} artists")

    print("\nLoading CLIP-ViT-B/16...")
    from transformers import CLIPProcessor, CLIPModel
    model_name = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
    print("  CLIP loaded.")

    print("\nLoading huggan/wikiart (non-streaming, cached locally)...")
    from datasets import load_dataset
    ds = load_dataset("huggan/wikiart", split="train")  # cached at ~/.cache/huggingface
    artist_names = ds.features["artist"].names
    print(f"  loaded {len(ds)} examples")

    # Group indices by artist using the column-wise read (fast — no image decode yet)
    print("  building artist→indices map (no image decode yet)...")
    artist_col = ds["artist"]  # fast column read
    by_artist_idx = defaultdict(list)
    for i, a in enumerate(artist_col):
        if artist_names[a] == "Unknown Artist":
            continue
        by_artist_idx[a].append(i)
    print(f"  {len(by_artist_idx)} known-artist labels found")

    # Take up to n_per_artist indices per artist, then decode just those images
    chosen_indices = []
    chosen_artists = []
    n_full = 0
    for a, idxs in by_artist_idx.items():
        if len(idxs) < min_per_artist:
            continue
        take = idxs[:n_per_artist]
        chosen_indices.extend(take)
        chosen_artists.extend([a] * len(take))
        n_full += 1
        if n_full >= max_artists: break
    print(f"  selected {len(chosen_indices)} examples across {n_full} artists "
          f"(≥ {min_per_artist} works each)")

    # Decode only the chosen images
    print("\n  decoding selected images...")
    t0 = time.time()
    images = []
    for k, idx in enumerate(chosen_indices):
        images.append(ds[idx]["image"])
        if (k + 1) % 500 == 0:
            print(f"    decoded {k+1}/{len(chosen_indices)} ({time.time()-t0:.0f}s)")
    print(f"  decoded {len(images)} images ({time.time()-t0:.0f}s)")
    keep = {a: [images[i] for i, ca in enumerate(chosen_artists) if ca == a]
             for a in set(chosen_artists)}
    print(f"\nKept {len(keep)} artists")

    # Encode
    print("\nEncoding with CLIP-mid (concat layers 3, 6, 9)...")
    all_images = []; all_artists = []
    for a_idx, imgs in keep.items():
        for img in imgs:
            all_images.append(img); all_artists.append(a_idx)
    print(f"  total {len(all_images)} images to encode")
    emb = extract_clip_mid_features(all_images, processor, model, batch=8)
    pid = np.array(all_artists, dtype=np.int64)
    print(f"  emb shape: {emb.shape}  pid shape: {pid.shape}")

    os.makedirs(os.path.dirname(EMB_OUT), exist_ok=True)
    np.savez(EMB_OUT, emb=emb, pid=pid)
    print(f"\n[saved] {EMB_OUT}")
    # Print summary
    from collections import Counter
    cnt = Counter(pid.tolist())
    print(f"  {len(cnt)} artists, mean {np.mean(list(cnt.values())):.1f} works/artist "
          f"(min {min(cnt.values())}, max {max(cnt.values())})")


if __name__ == "__main__":
    sys.exit(main())
