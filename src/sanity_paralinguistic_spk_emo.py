"""Sanity / data prep — paralinguistic with speaker x emotion identities.

Reframes RAVDESS for the per-user state memory task: each (speaker_id,
emotion) pair is a distinct 'identity', giving 24*8 = 192 identity
classes with ~7-8 clips each. Now N>=5 evaluation is unblocked.

This corresponds to the natural paralinguistic memory framing: the
agent has previously seen 'speaker X in state Y' and at query time has
to retrieve which (speaker, state) the new clip matches.

Encoder: same wav2vec2-emotion as sanity_paralinguistic_v2 — features
are state-discriminative within speaker and speaker-discriminative
across states, so the (speaker, emotion) pair is fully distinguishable.
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


def main():
    print("=" * 70)
    print("Paralinguistic — speaker x emotion identities on RAVDESS")
    print("=" * 70)

    from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification
    model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    print(f"\n[load] {model_id}")
    proc = AutoFeatureExtractor.from_pretrained(model_id)
    try:
        model = AutoModelForAudioClassification.from_pretrained(model_id).to(DEVICE).eval()
    except Exception:
        model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()

    from datasets import load_dataset, Audio
    ds = load_dataset("xbgoose/ravdess", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  loaded {len(ds)} clips")

    emotions = ds["emotion"]
    actors = ds["actor"]

    # Group by (actor, emotion); we want enough samples per pair
    by_se = defaultdict(list)
    for i, (a, e) in enumerate(zip(actors, emotions)):
        by_se[(int(a), str(e))].append(i)
    print(f"  (speaker, emotion) pairs: {len(by_se)}")
    cnt = [len(v) for v in by_se.values()]
    print(f"  clips per pair: min={min(cnt)} max={max(cnt)} mean={sum(cnt)/len(cnt):.1f}")

    # Cap clips per pair to balance
    CLIPS_PER_PAIR = 5
    eligible = [(k, v) for k, v in by_se.items() if len(v) >= CLIPS_PER_PAIR]
    print(f"  pairs with >= {CLIPS_PER_PAIR} clips: {len(eligible)}")

    print("\n[encode] wav2vec2-emotion features ...")
    import soundfile as sf
    embs, pids = [], []
    for k, ((actor, emo), idxs) in enumerate(eligible):
        sel = random.sample(idxs, k=CLIPS_PER_PAIR)
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
                else:
                    out = model(**inputs)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else None
                if hidden is None: continue
            f = hidden.mean(dim=1).cpu().numpy()[0]
            f = f / (np.linalg.norm(f) + 1e-9)
            embs.append(f); pids.append(f"a{actor}_{emo}")
        if (k + 1) % 30 == 0:
            print(f"  processed {k+1}/{len(eligible)} pairs, {len(embs)} embeddings")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    print(f"\n[done] {emb.shape}, {len(set(pid))} (speaker,emotion) identities")

    # Sanity: top-1 NN within (s,e) pairs
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    nn_same = (pid[nn] == pid).mean()
    print(f"[sanity] top-1 same-(s,e) recall: {nn_same:.4f}")

    by_p = defaultdict(list)
    for i, p in enumerate(pid): by_p[p].append(i)
    intra, inter = [], []
    for p, idxs in by_p.items():
        if len(idxs) < 2: continue
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                intra.append(sims[idxs[i], idxs[j]])
    rng = np.random.RandomState(SEED)
    for _ in range(5000):
        i, j = rng.randint(0, len(emb), size=2)
        if pid[i] != pid[j]: inter.append(sims[i, j])
    print(f"[sanity] intra-(s,e) cosine: {np.mean(intra):.4f}  inter-(s,e) cosine: {np.mean(inter):.4f}")

    import faiss
    print(f"\n[sanity] quantisation:")
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    results = {}
    for K in [16, 32, 64, 128]:
        if K > len(emb): continue
        km = faiss.Kmeans(emb.shape[1], K, niter=20, verbose=False, seed=SEED)
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

    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/wav2vec_para_spk_emo.npz")
    np.savez(out_emb, emb=emb, pid=pid)
    print(f"\n[done] saved {out_emb}")

    import json
    with open("/home/ubuntu/multimodal-user-memory/results/sanity_paralinguistic_spk_emo.json", "w") as f:
        json.dump({
            "encoder": model_id, "n_embeddings": int(len(emb)),
            "n_identities": int(len(set(pid))),
            "top1_recall": float(nn_same),
            "intra_mean": float(np.mean(intra)), "inter_mean": float(np.mean(inter)),
            "quantisation": {str(K): {"intra_agree": float(a), "inter_coll": float(b),
                                       "ratio": float(r) if r != float("inf") else None}
                             for K, (a, b, r) in results.items()},
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
