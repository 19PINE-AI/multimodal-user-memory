"""Sanity check 1 — ArcFace + quantisation collision rate on LFW cross-condition faces.

Visual analogue of sanity_ecapa_collisions.py. Question: when ArcFace
embeddings (designed for cross-condition identity discrimination) are
quantised into a discrete codebook, does the same person under
different photo conditions (lighting, expression, angle) map to the
same code more often than two different people map to the same code?

Data: LFW (Labelled Faces in the Wild), fetched via sklearn. People
with >=10 photos give us natural cross-condition variation.

Model: ArcFace R50 (w600k_r50.onnx) from the buffalo_l pack.
"""

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

NUM_PEOPLE = 30
PHOTOS_PER_PERSON = 8
MIN_PHOTOS_PER_PERSON = 10  # sklearn LFW filter
CODEBOOK_SIZES = [8, 16, 32, 64, 128]
RQ_CONFIGS = [(2, 16), (2, 64), (3, 16), (4, 16), (4, 64)]
SEED = 42

ARCFACE_PATH = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"

random.seed(SEED)
np.random.seed(SEED)


def preprocess(img: np.ndarray) -> np.ndarray:
    """sklearn LFW gives HxWx3 float in [0,1] (if color=True), already centred-aligned.
    ArcFace expects 112x112 BGR, scaled to (x - 127.5) / 128."""
    import cv2

    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    # sklearn LFW is RGB; ArcFace wants BGR (cv2 convention)
    img = img[..., ::-1]
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    arr = img.astype(np.float32)
    arr = (arr - 127.5) / 128.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr[None]  # 1xCxHxW


def main():
    print("[sanity] Fetching LFW via sklearn (this caches ~250MB at ~/scikit_learn_data) ...")
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=MIN_PHOTOS_PER_PERSON, color=True, resize=1.0)
    print(f"[sanity] LFW loaded: {lfw.images.shape[0]} photos, {len(lfw.target_names)} people")
    print(f"[sanity] image shape: {lfw.images.shape[1:]}")  # (H, W, 3) float

    # Group by person id
    by_person = defaultdict(list)
    for i, t in enumerate(lfw.target):
        by_person[int(t)].append(i)
    eligible = [(p, idxs) for p, idxs in by_person.items() if len(idxs) >= PHOTOS_PER_PERSON]
    print(f"[sanity] {len(eligible)} people have >= {PHOTOS_PER_PERSON} photos")
    chosen = random.sample(eligible, k=min(NUM_PEOPLE, len(eligible)))
    print(f"[sanity] Sampled {len(chosen)} people")

    print(f"[sanity] Loading ArcFace from {ARCFACE_PATH} ...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(str(ARCFACE_PATH), providers=providers)
    input_name = sess.get_inputs()[0].name
    actual_providers = sess.get_providers()
    print(f"[sanity] ONNX providers in use: {actual_providers}")

    # Extract embeddings
    embeddings = []
    for person_id, photo_idxs in chosen:
        sample_idxs = random.sample(photo_idxs, k=PHOTOS_PER_PERSON)
        for i in sample_idxs:
            img = lfw.images[i]  # HxWx3 float in [0,1]
            try:
                arr = preprocess(img)
                emb = sess.run(None, {input_name: arr})[0][0]
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                embeddings.append((person_id, i, emb))
            except Exception as e:
                print(f"[sanity] skip person {person_id} photo {i}: {e}")
        if len(embeddings) % 40 == 0:
            print(f"[sanity] processed {len(embeddings)} embeddings so far")

    print(f"[sanity] Collected {len(embeddings)} embeddings across {len(chosen)} people")

    pid_arr = np.array([e[0] for e in embeddings])
    emb_matrix = np.stack([e[2] for e in embeddings]).astype(np.float32)
    D = emb_matrix.shape[1]
    print(f"[sanity] Embedding dim = {D}")

    # ---- Soundness probe: top-1 nearest-neighbour identity recall ----
    sims = emb_matrix @ emb_matrix.T
    np.fill_diagonal(sims, -np.inf)
    nn_idx = sims.argmax(axis=1)
    nn_same = (pid_arr[nn_idx] == pid_arr).mean()
    print(f"\n[sanity] ArcFace raw cosine top-1 identity recall: {nn_same:.4f} (expect > 0.95)")

    # Intra/inter cosine
    pid_to_indices = defaultdict(list)
    for i, p in enumerate(pid_arr):
        pid_to_indices[int(p)].append(i)
    intra_sims, inter_sims = [], []
    for p, idxs in pid_to_indices.items():
        if len(idxs) >= 2:
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    intra_sims.append(sims[idxs[i], idxs[j]])
    rng = np.random.RandomState(SEED)
    for _ in range(5000):
        i, j = rng.randint(0, len(embeddings), size=2)
        if pid_arr[i] != pid_arr[j]:
            inter_sims.append(sims[i, j])
    intra_sims = np.array(intra_sims)
    inter_sims = np.array(inter_sims)
    print(f"[sanity] Cosine intra-identity:  mean={intra_sims.mean():.4f}  std={intra_sims.std():.4f}  n={len(intra_sims)}")
    print(f"[sanity] Cosine inter-identity:  mean={inter_sims.mean():.4f}  std={inter_sims.std():.4f}  n={len(inter_sims)}")

    # ---- Quantisation ----
    import faiss

    def collision_stats(codes):
        intra_agree = 0; intra_pairs = 0
        for p, idxs in pid_to_indices.items():
            if len(idxs) >= 2:
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        intra_pairs += 1
                        if tuple(np.atleast_1d(codes[idxs[i]])) == tuple(np.atleast_1d(codes[idxs[j]])):
                            intra_agree += 1
        intra_rate = intra_agree / intra_pairs if intra_pairs > 0 else 0.0
        inter_coll = 0; inter_pairs = 0
        rng2 = np.random.RandomState(SEED + 17)
        for _ in range(20000):
            i, j = rng2.randint(0, len(embeddings), size=2)
            if pid_arr[i] != pid_arr[j]:
                inter_pairs += 1
                if tuple(np.atleast_1d(codes[i])) == tuple(np.atleast_1d(codes[j])):
                    inter_coll += 1
        inter_rate = inter_coll / inter_pairs if inter_pairs > 0 else 0.0
        ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
        return intra_rate, inter_rate, ratio

    print("\n[sanity] Quantisation results:")
    print("  [flat k-means]")
    results = {}
    for K in CODEBOOK_SIZES:
        if K > len(embeddings):
            continue
        kmeans = faiss.Kmeans(D, K, niter=20, verbose=False, seed=SEED)
        kmeans.train(emb_matrix)
        _, codes = kmeans.index.search(emb_matrix, 1)
        codes = codes.squeeze(1)
        intra_rate, inter_rate, ratio = collision_stats(codes)
        print(f"  K={K:>5d} | intra={intra_rate:.4f} | inter={inter_rate:.4f} | ratio={ratio:8.2f}")
        results[f"flat_{K}"] = (intra_rate, inter_rate, ratio)

    print("  [residual product quantisation]")
    for n_levels, k_per in RQ_CONFIGS:
        if k_per > len(embeddings):
            continue
        residual = emb_matrix.copy()
        level_codes = np.zeros((len(embeddings), n_levels), dtype=np.int64)
        for L in range(n_levels):
            km = faiss.Kmeans(D, k_per, niter=20, verbose=False, seed=SEED + L)
            km.train(residual)
            _, codes_L = km.index.search(residual, 1)
            codes_L = codes_L.squeeze(1)
            level_codes[:, L] = codes_L
            residual = residual - km.centroids[codes_L]
        intra_rate, inter_rate, ratio = collision_stats(level_codes)
        eff_K = k_per ** n_levels
        print(f"  RQ levels={n_levels} k={k_per} (eff_K={eff_K:>10d}) | intra={intra_rate:.4f} | inter={inter_rate:.4f} | ratio={ratio:8.2f}")
        results[f"rq_{n_levels}_{k_per}"] = (intra_rate, inter_rate, ratio)

    # ---- Save ----
    out_dir = Path("/home/ubuntu/multimodal-user-memory/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_dir / "sanity_arcface_collisions.json", "w") as f:
        json.dump({
            "config": {
                "num_people": NUM_PEOPLE,
                "photos_per_person": PHOTOS_PER_PERSON,
                "min_photos_per_person": MIN_PHOTOS_PER_PERSON,
                "codebook_sizes": CODEBOOK_SIZES,
                "rq_configs": RQ_CONFIGS,
                "seed": SEED,
                "n_embeddings_collected": len(embeddings),
                "embedding_dim": int(D),
            },
            "raw_cosine": {
                "top1_identity_recall": float(nn_same),
                "intra_mean": float(intra_sims.mean()),
                "intra_std": float(intra_sims.std()),
                "inter_mean": float(inter_sims.mean()),
                "inter_std": float(inter_sims.std()),
            },
            "quantisation": {k: {"intra_agree": float(a), "inter_coll": float(b), "ratio": float(r) if r != float("inf") else None}
                             for k, (a, b, r) in results.items()},
        }, f, indent=2)
    print(f"\n[sanity] Wrote results to {out_dir / 'sanity_arcface_collisions.json'}")

    print("\n[sanity] Verdict:")
    if nn_same < 0.85:
        print("  FAIL — ArcFace nearest-neighbour recall too low on this domain.")
        return 1
    viable = [k for k, (a, b, r) in results.items() if r > 5.0 and a > 0.5]
    if viable:
        print(f"  PASS — viable codebooks (ratio > 5, intra-agree > 0.5): {viable}")
        return 0
    else:
        print("  PARTIAL — encoder sound but quantiser collapses identity. Tune RQ depth.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
