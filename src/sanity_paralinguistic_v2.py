"""Sanity check 5 (revised) — paralinguistic STATE encoder on RAVDESS.

Right test: do same-EMOTION utterances cluster together (regardless of speaker)
more than different-emotion utterances?

This tests speaker-invariance (positives are different speakers same emotion)
+ state-discrimination (negatives are different emotions). Exactly the
property the paralinguistic-memory sub-modality needs.

Encoder: emotion-finetuned wav2vec2 if available, else features from a
general speech model with emotion-discriminative classifier.
"""
import io
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EMOTIONS = 8     # RAVDESS has 8: neutral, calm, happy, sad, angry, fearful, disgust, surprised
CLIPS_PER_EMOTION = 60  # cap (RAVDESS has ~60 each from 24 speakers × ~2-3 takes)


def main():
    print("=" * 70)
    print("Sanity check 5 (revised) — paralinguistic STATE encoder on RAVDESS")
    print("=" * 70)

    from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification
    # Prefer an emotion-tuned model so penultimate features are state-discriminative
    cand_models = [
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "superb/wav2vec2-base-superb-er",
        "facebook/wav2vec2-base-960h",  # last-resort general features
    ]
    model = None; model_id = None; proc = None
    for mid in cand_models:
        try:
            proc = AutoFeatureExtractor.from_pretrained(mid)
            try:
                model = AutoModelForAudioClassification.from_pretrained(mid).to(DEVICE).eval()
            except Exception:
                model = AutoModel.from_pretrained(mid).to(DEVICE).eval()
            model_id = mid
            print(f"  loaded {mid}")
            break
        except Exception as e:
            print(f"  {mid} failed: {type(e).__name__}: {str(e)[:80]}")
    if model is None:
        print("  no audio model available")
        return

    from datasets import load_dataset, Audio
    ds = load_dataset("xbgoose/ravdess", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  loaded RAVDESS: {len(ds)} clips, columns={ds.column_names}")

    emotions_col = ds["emotion"]
    by_emo = defaultdict(list)
    for i, e in enumerate(emotions_col):
        by_emo[str(e)].append(i)
    print(f"  emotions distribution: {[(e, len(idxs)) for e, idxs in by_emo.items()]}")

    print("\n[encode] extracting features ...")
    import soundfile as sf
    embs, emo_labels, spk_labels = [], [], []
    actors_col = ds["actor"]
    for emo, idxs in by_emo.items():
        if len(idxs) < 4: continue
        # Cap per emotion
        sel = random.sample(idxs, k=min(CLIPS_PER_EMOTION, len(idxs)))
        for i in sel:
            ad = ds[i]["audio"]
            if not (isinstance(ad, dict) and ad.get("bytes") is not None): continue
            try:
                wav, sr = sf.read(io.BytesIO(ad["bytes"]), dtype="float32")
            except Exception: continue
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
            wav = wav[:16000 * 5]
            inputs = proc(wav, sampling_rate=16000, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                if hasattr(model, "wav2vec2"):
                    out = model.wav2vec2(**inputs)
                    hidden = out.last_hidden_state
                else:
                    out = model(**inputs, output_hidden_states=False)
                    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else None
                if hidden is None: continue
            feat = hidden.mean(dim=1).cpu().numpy()[0]
            feat = feat / (np.linalg.norm(feat) + 1e-9)
            embs.append(feat); emo_labels.append(str(emo)); spk_labels.append(str(actors_col[i]))

    emb = np.stack(embs).astype(np.float32)
    emo = np.array(emo_labels)
    spk = np.array(spk_labels)
    D = emb.shape[1]
    print(f"\n[done] {emb.shape}, {len(set(emo))} emotions, {len(set(spk))} speakers, dim={D}")

    # Evaluate: emotion clustering (state) AND speaker clustering (should be lower)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)

    print(f"\n[sanity] top-1 same-EMOTION recall: {(emo[nn] == emo).mean():.4f}  (higher = better; we want state-discriminative)")
    print(f"[sanity] top-1 same-SPEAKER recall: {(spk[nn] == spk).mean():.4f}  (lower = better; speaker invariance)")

    # Intra/inter for emotion
    by_e = defaultdict(list)
    for i, e in enumerate(emo): by_e[e].append(i)
    intra_emo, inter_emo = [], []
    for e, idxs in by_e.items():
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                intra_emo.append(sims[idxs[i], idxs[j]])
    rng = np.random.RandomState(SEED)
    for _ in range(5000):
        i, j = rng.randint(0, len(emb), size=2)
        if emo[i] != emo[j]: inter_emo.append(sims[i, j])
    print(f"[sanity] intra-emotion cosine: mean={np.mean(intra_emo):.4f}")
    print(f"[sanity] inter-emotion cosine: mean={np.mean(inter_emo):.4f}")

    # Quantisation by emotion
    import faiss
    print(f"\n[sanity] Quantisation (grouped by EMOTION):")
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    results = {}
    for K in [8, 16, 32, 64]:
        if K > len(emb): continue
        km = faiss.Kmeans(D, K, niter=20, verbose=False, seed=SEED)
        km.train(emb)
        _, codes = km.index.search(emb, 1); codes = codes.squeeze(1)
        intra_agree = 0; intra_pairs = 0
        for e, idxs in by_e.items():
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    intra_pairs += 1
                    if codes[idxs[i]] == codes[idxs[j]]: intra_agree += 1
        rng2 = np.random.RandomState(SEED + K)
        inter_coll = 0; inter_pairs = 0
        for _ in range(10000):
            i, j = rng2.randint(0, len(emb), size=2)
            if emo[i] != emo[j]:
                inter_pairs += 1
                if codes[i] == codes[j]: inter_coll += 1
        intra_rate = intra_agree / max(intra_pairs, 1)
        inter_rate = inter_coll / max(inter_pairs, 1)
        ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
        print(f"  {K:>4d} | {intra_rate:>12.4f} | {inter_rate:>12.4f} | {ratio:>8.2f}")
        results[K] = (intra_rate, inter_rate, ratio)

    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings") / "wav2vec_paralinguistic_v2.npz"
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_emb, emb=emb, pid=emo, speaker=spk)
    print(f"\n[done] saved to {out_emb}")

    out_res = Path("/home/ubuntu/multimodal-user-memory/results") / "sanity_paralinguistic_v2.json"
    import json
    with open(out_res, "w") as f:
        json.dump({
            "encoder": model_id,
            "n_embeddings": int(len(emb)),
            "n_emotions": int(len(set(emo))),
            "n_speakers": int(len(set(spk))),
            "raw_cosine": {
                "top1_same_emotion_recall": float((emo[nn] == emo).mean()),
                "top1_same_speaker_recall": float((spk[nn] == spk).mean()),
                "intra_emotion_mean": float(np.mean(intra_emo)),
                "inter_emotion_mean": float(np.mean(inter_emo)),
            },
            "quantisation_by_emotion": {str(K): {"intra_agree": float(a), "inter_coll": float(b),
                                                  "ratio": float(r) if r != float("inf") else None}
                                         for K, (a, b, r) in results.items()},
        }, f, indent=2)
    print(f"[done] saved summary")


if __name__ == "__main__":
    sys.exit(main())
