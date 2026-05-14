"""Sanity check 2 — ECAPA-TDNN + quantisation collision rate on LibriSpeech.

Question: when we quantise ECAPA-TDNN speaker embeddings into a discrete
codebook (the proposed perceptual hash address), does the same speaker
across different recording sessions map to the same code more often
than two different speakers map to the same code?

This is the gating experiment from research_plan.md §11 for the audio
modality. If intra-speaker code agreement << inter-speaker collision,
the perceptual-Engram mechanism is unsound for audio and the quantiser
needs rethinking before any further investment.

Data: LibriSpeech test-clean (already at ~/data/LibriSpeech/test-clean).
Each speaker has multiple chapters → cross-chapter pairs simulate
"same speaker, different recording session."

Metrics reported per codebook size K:
  - Intra-speaker code-agreement rate: P(code(x) == code(y) | speaker(x) == speaker(y), x != y)
  - Inter-speaker collision rate:      P(code(x) == code(y) | speaker(x) != speaker(y))
  - Discriminability ratio: intra / inter (higher is better; >>1 is the win condition)
  - Top-1 nearest-neighbour speaker recall (cosine, pre-quantisation, as a soundness check)
"""

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

LIBRI_ROOT = Path.home() / "data" / "LibriSpeech" / "test-clean"
NUM_SPEAKERS = 40
UTTERANCES_PER_SPEAKER = 8  # need >=2 to test intra-speaker agreement
MIN_CHAPTERS_PER_SPEAKER = 2
# Flat k-means codebook sizes (must be <= total embeddings for k-means to be sensible)
CODEBOOK_SIZES = [8, 16, 32, 64, 128]
# Residual product-quantiser configs: list of (n_levels, codebook_per_level)
# Effective codebook = codebook_per_level ** n_levels
RQ_CONFIGS = [(2, 16), (2, 64), (3, 16), (4, 16), (4, 64)]
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def find_speakers_with_multi_chapter(root: Path, min_chapters: int):
    speakers = []
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir():
            continue
        chapters = [c for c in spk_dir.iterdir() if c.is_dir()]
        if len(chapters) >= min_chapters:
            speakers.append(spk_dir)
    return speakers


def collect_utterances(speaker_dir: Path, n: int):
    """Pick n utterances, biased toward chapter diversity."""
    chapters = sorted([c for c in speaker_dir.iterdir() if c.is_dir()])
    by_chapter = []
    for ch in chapters:
        flacs = sorted(ch.glob("*.flac"))
        if flacs:
            by_chapter.append(flacs)
    if not by_chapter:
        return []
    out = []
    i = 0
    # Round-robin across chapters until we have n
    while len(out) < n:
        flacs = by_chapter[i % len(by_chapter)]
        if flacs:
            choice = random.choice(flacs)
            if choice not in out:
                out.append(choice)
        i += 1
        if i > 200:  # safety
            break
    return out[:n]


def main():
    print(f"[sanity] LibriSpeech root: {LIBRI_ROOT}")
    print(f"[sanity] Picking {NUM_SPEAKERS} speakers with >= {MIN_CHAPTERS_PER_SPEAKER} chapters each")
    speakers = find_speakers_with_multi_chapter(LIBRI_ROOT, MIN_CHAPTERS_PER_SPEAKER)
    print(f"[sanity] Found {len(speakers)} eligible speakers in test-clean")
    speakers = random.sample(speakers, k=min(NUM_SPEAKERS, len(speakers)))
    print(f"[sanity] Sampled {len(speakers)} speakers")

    print("[sanity] Loading SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb)...")
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/home/ubuntu/multimodal-user-memory/runs/pretrained-ecapa",
        run_opts={"device": device},
    )
    print(f"[sanity] Loaded on {device}.")

    # Collect embeddings
    embeddings = []  # list of (speaker_id, chapter_id, utterance_path, emb_np)
    for spk_idx, spk_dir in enumerate(speakers):
        utts = collect_utterances(spk_dir, UTTERANCES_PER_SPEAKER)
        if not utts:
            continue
        for u in utts:
            try:
                # soundfile reads FLAC natively without TorchCodec/ffmpeg
                data, sr = sf.read(str(u), always_2d=False, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                wav = torch.from_numpy(data).unsqueeze(0)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                # Truncate to first 6s to keep batches small
                wav = wav[:, : 16000 * 6]
                emb = encoder.encode_batch(wav.to(device)).squeeze().cpu().numpy()
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                embeddings.append((spk_dir.name, u.parent.name, u.name, emb))
            except Exception as e:
                print(f"[sanity] skip {u}: {e}")
        if (spk_idx + 1) % 5 == 0:
            print(f"[sanity] processed {spk_idx + 1} speakers, {len(embeddings)} embeddings so far")

    print(f"[sanity] Collected {len(embeddings)} embeddings across {len(speakers)} speakers")

    spk_ids = np.array([e[0] for e in embeddings])
    emb_matrix = np.stack([e[3] for e in embeddings]).astype(np.float32)
    D = emb_matrix.shape[1]
    print(f"[sanity] Embedding dim = {D}")

    # ---- Soundness probe: top-1 nearest-neighbour speaker recall ----
    # For each embedding, find its nearest other embedding and check if it's the same speaker.
    sims = emb_matrix @ emb_matrix.T
    np.fill_diagonal(sims, -np.inf)
    nn_idx = sims.argmax(axis=1)
    nn_same_spk = (spk_ids[nn_idx] == spk_ids).mean()
    print(f"\n[sanity] ECAPA-TDNN raw cosine top-1 speaker recall: {nn_same_spk:.4f} (expect > 0.95)")

    # Intra/inter cosine stats
    spk_to_indices = defaultdict(list)
    for i, s in enumerate(spk_ids):
        spk_to_indices[s].append(i)
    intra_sims = []
    inter_sims = []
    for s, idxs in spk_to_indices.items():
        if len(idxs) >= 2:
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    intra_sims.append(sims[idxs[i], idxs[j]])
    # Sample inter
    rng = np.random.RandomState(SEED)
    sample_size = min(5000, len(embeddings) * 50)
    for _ in range(sample_size):
        i, j = rng.randint(0, len(embeddings), size=2)
        if spk_ids[i] != spk_ids[j]:
            inter_sims.append(sims[i, j])
    intra_sims = np.array(intra_sims)
    inter_sims = np.array(inter_sims)
    print(f"[sanity] Cosine intra-speaker:  mean={intra_sims.mean():.4f}  std={intra_sims.std():.4f}  n={len(intra_sims)}")
    print(f"[sanity] Cosine inter-speaker:  mean={inter_sims.mean():.4f}  std={inter_sims.std():.4f}  n={len(inter_sims)}")

    # ---- Quantisation: k-means → discrete codes ----
    # For each codebook size K, fit k-means on the embeddings, then measure
    # intra-speaker code agreement rate vs inter-speaker collision rate.
    import faiss

    print(f"\n[sanity] Quantisation results across codebook sizes:")
    print(f"{'K':>7} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    print("-" * 50)

    def collision_stats(codes):
        intra_agree = 0; intra_pairs = 0
        for s, idxs in spk_to_indices.items():
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
            if spk_ids[i] != spk_ids[j]:
                inter_pairs += 1
                if tuple(np.atleast_1d(codes[i])) == tuple(np.atleast_1d(codes[j])):
                    inter_coll += 1
        inter_rate = inter_coll / inter_pairs if inter_pairs > 0 else 0.0
        ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
        return intra_rate, inter_rate, ratio

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

    # ---- Residual product quantisation (RQ-VAE-like, but k-means based) ----
    # At each level we cluster the residual, then store an integer per level.
    # Effective codebook = codebook_per_level ** n_levels (only "tuple agreement" counts as collision).
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
            centroids = km.centroids  # (k_per, D)
            residual = residual - centroids[codes_L]

        # Two embeddings have the same code iff they agree on ALL levels (full tuple)
        intra_rate, inter_rate, ratio = collision_stats(level_codes)
        eff_K = k_per ** n_levels
        print(f"  RQ levels={n_levels} k={k_per} (eff_K={eff_K:>10d}) | intra={intra_rate:.4f} | inter={inter_rate:.4f} | ratio={ratio:8.2f}")
        results[f"rq_{n_levels}_{k_per}"] = (intra_rate, inter_rate, ratio)

    # ---- Save results ----
    out_dir = Path("/home/ubuntu/multimodal-user-memory/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_dir / "sanity_ecapa_collisions.json", "w") as f:
        json.dump({
            "config": {
                "num_speakers": NUM_SPEAKERS,
                "utterances_per_speaker": UTTERANCES_PER_SPEAKER,
                "codebook_sizes": CODEBOOK_SIZES,
                "seed": SEED,
                "n_embeddings_collected": len(embeddings),
            },
            "raw_cosine": {
                "top1_speaker_recall": float(nn_same_spk),
                "intra_mean": float(intra_sims.mean()),
                "intra_std": float(intra_sims.std()),
                "inter_mean": float(inter_sims.mean()),
                "inter_std": float(inter_sims.std()),
            },
            "quantisation": {k: {"intra_agree": float(a), "inter_coll": float(b), "ratio": float(r) if r != float("inf") else None}
                             for k, (a, b, r) in results.items()},
        }, f, indent=2)
    print(f"\n[sanity] Wrote results to {out_dir / 'sanity_ecapa_collisions.json'}")

    # ---- Interpretation ----
    print("\n[sanity] Verdict:")
    if nn_same_spk < 0.85:
        print("  FAIL — ECAPA-TDNN nearest-neighbour recall is too low; not a viable encoder for this domain.")
        return 1
    viable_codebooks = [K for K, (a, b, r) in results.items() if r > 5.0 and a > 0.5]
    if viable_codebooks:
        print(f"  PASS — viable codebook sizes (ratio > 5, intra-agree > 0.5): {viable_codebooks}")
        return 0
    else:
        print("  PARTIAL — encoder is sound but flat quantisation collapses identity. Try RQ-VAE or learned product quantiser.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
