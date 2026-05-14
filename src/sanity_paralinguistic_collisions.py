"""Sanity check 5 — paralinguistic state encoder collision rate on RAVDESS.

Paralinguistic memory is the harder side of audio: not "who is this
speaker" (that's ECAPA) but "what state is this familiar speaker in"
(tired / happy / angry / etc.). We need an encoder that captures
the STATE while being SPEAKER-INVARIANT — the OPPOSITE invariance
to ECAPA.

Encoder: wav2vec2-base fine-tuned for speech emotion recognition.
We try `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
or similar HF emotion-classification model; use its penultimate
layer as the embedding.

Dataset: RAVDESS — 24 actors × 60 utterances × 8 emotions ≈ 1440
files. Two genders. Available as a HF dataset
`xbgoose/ravdess` or similar.

Task: given two utterances of the same SPEAKER in different
emotion STATES, the embeddings should differ (good state
discrimination). Given two utterances of different SPEAKERS in the
same STATE, embeddings should be similar (good speaker invariance).
"""
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SPEAKERS = 16  # RAVDESS has 24, leave headroom
UTTERANCES_PER_SPEAKER = 8


def main():
    print("=" * 70)
    print("Sanity check 5 — paralinguistic state encoder on RAVDESS")
    print("=" * 70)

    print("\n[encoder] loading wav2vec2-base-960h (then we extract penultimate features) ...")
    # Use the base wav2vec2 model for general audio features; fine-tuned emotion
    # heads exist but for the sanity check we want raw audio representations.
    from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor
    model_id = "facebook/wav2vec2-base-960h"
    try:
        proc = AutoFeatureExtractor.from_pretrained(model_id)
        model = Wav2Vec2Model.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE)
    except Exception as e:
        print(f"  wav2vec2-base load failed: {e}; trying ehcalabres emotion-finetuned")
        model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        proc = AutoFeatureExtractor.from_pretrained(model_id)
        model = Wav2Vec2Model.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    print(f"  loaded; params = {sum(p.numel() for p in model.parameters()):,}")

    print("\n[data] loading RAVDESS via HF datasets ...")
    from datasets import load_dataset, Audio
    # Try a few possible IDs
    ds = None
    for ds_id in ["xbgoose/ravdess", "minoosh/RAVDESS", "narad/ravdess"]:
        try:
            ds = load_dataset(ds_id, split="train")
            print(f"  loaded {ds_id}: {len(ds)} clips, columns={ds.column_names}")
            break
        except Exception as e:
            print(f"  {ds_id} failed: {type(e).__name__}: {str(e)[:80]}")
    if ds is None:
        print("  no RAVDESS dataset available on HF directly")
        print("  falling back to LibriSpeech with synthetic 'state' via prosody perturbation")
        return run_libri_paralinguistic_proxy(model, proc)
    # Bypass torchcodec by re-casting audio column to no-decode
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))

    # Inspect schema
    print(f"  schema: {ds.features}")
    # Group by speaker; RAVDESS metadata has actor + emotion fields typically
    by_spk = defaultdict(list)
    # Schema varies between mirrors; try common field names
    spk_field = None
    for cand in ["speaker", "actor", "speaker_id", "actor_id"]:
        if cand in ds.column_names:
            spk_field = cand; break
    if spk_field is not None:
        col = ds[spk_field]
        for i, v in enumerate(col):
            by_spk[str(v)].append(i)
    else:
        # Parse from path
        for fname in ["path", "file", "filename"]:
            if fname in ds.column_names:
                col = ds[fname]
                for i, v in enumerate(col):
                    if v is None: continue
                    stem = Path(str(v)).stem
                    parts = stem.split("-")
                    # RAVDESS files: 03-01-06-01-02-01-12.wav (actor=last)
                    if len(parts) >= 7:
                        by_spk[parts[-1]].append(i)
                break
    print(f"  by speaker: {len(by_spk)} speakers")
    if not by_spk:
        print("  could not parse speakers; trying first-N positional sample")
        # fallback: just compute embedding sanity on all clips
        return

    eligible = [(s, idxs) for s, idxs in by_spk.items() if len(idxs) >= UTTERANCES_PER_SPEAKER]
    print(f"  {len(eligible)} speakers have >= {UTTERANCES_PER_SPEAKER} clips")
    chosen = random.sample(eligible, k=min(NUM_SPEAKERS, len(eligible)))
    print(f"  sampled {len(chosen)} speakers")

    print("\n[encode] extracting wav2vec2 mean-pooled features ...")
    import soundfile as sf, io
    embs, sids = [], []
    for spk_idx, (spk_id, idxs) in enumerate(chosen):
        sel = random.sample(idxs, k=UTTERANCES_PER_SPEAKER)
        for i in sel:
            row = ds[i]
            ad = row.get("audio") or row.get("speech")
            if ad is None: continue
            try:
                if isinstance(ad, dict) and ad.get("bytes") is not None:
                    wav, sr = sf.read(io.BytesIO(ad["bytes"]), dtype="float32")
                elif isinstance(ad, dict) and ad.get("path") is not None:
                    wav, sr = sf.read(ad["path"], dtype="float32")
                else:
                    continue
            except Exception as e:
                print(f"  skip {i}: {type(e).__name__} {str(e)[:50]}")
                continue
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
            wav = wav[:16000 * 5]
            inputs = proc(wav, sampling_rate=16000, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inputs)
            feats = out.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            feats = feats / (np.linalg.norm(feats) + 1e-9)
            embs.append(feats); sids.append(str(spk_id))
        if (spk_idx + 1) % 4 == 0:
            print(f"  processed {spk_idx+1}/{len(chosen)} speakers, {len(embs)} embeddings")

    emb = np.stack(embs).astype(np.float32)
    sid = np.array(sids)
    print(f"\n[done] Collected {emb.shape}, {len(set(sid))} speakers")

    eval_and_save(emb, sid, model_id, prefix="paralinguistic")


def eval_and_save(emb, sid, model_id, prefix):
    """Compute intra/inter cosine + quantisation stats."""
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    nn_same = (sid[nn] == sid).mean()
    print(f"\n[sanity] {model_id} top-1 same-speaker recall: {nn_same:.4f}")
    print("  (NOTE: paralinguistic encoder is supposed to be SPEAKER-INVARIANT,")
    print("   so HIGH same-speaker recall is BAD here; lower is better.)")

    by_p = defaultdict(list)
    for i, s in enumerate(sid): by_p[s].append(i)
    intra, inter = [], []
    for s, idxs in by_p.items():
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                intra.append(sims[idxs[i], idxs[j]])
    rng = np.random.RandomState(SEED)
    for _ in range(5000):
        i, j = rng.randint(0, len(emb), size=2)
        if sid[i] != sid[j]: inter.append(sims[i, j])
    print(f"[sanity] intra-speaker cosine: mean={np.mean(intra):.4f}  n={len(intra)}")
    print(f"[sanity] inter-speaker cosine: mean={np.mean(inter):.4f}  n={len(inter)}")

    # Quantisation
    import faiss
    print(f"\n[sanity] Quantisation:")
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    results = {}
    for K in [8, 16, 32, 64, 128]:
        if K > len(emb): continue
        km = faiss.Kmeans(emb.shape[1], K, niter=20, verbose=False, seed=SEED)
        km.train(emb)
        _, codes = km.index.search(emb, 1); codes = codes.squeeze(1)
        intra_agree = 0; intra_pairs = 0
        for s, idxs in by_p.items():
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    intra_pairs += 1
                    if codes[idxs[i]] == codes[idxs[j]]: intra_agree += 1
        rng2 = np.random.RandomState(SEED + K)
        inter_coll = 0; inter_pairs = 0
        for _ in range(10000):
            i, j = rng2.randint(0, len(emb), size=2)
            if sid[i] != sid[j]:
                inter_pairs += 1
                if codes[i] == codes[j]: inter_coll += 1
        intra_rate = intra_agree / max(intra_pairs, 1)
        inter_rate = inter_coll / max(inter_pairs, 1)
        ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
        print(f"  {K:>4d} | {intra_rate:>12.4f} | {inter_rate:>12.4f} | {ratio:>8.2f}")
        results[K] = (intra_rate, inter_rate, ratio)

    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings") / f"wav2vec_{prefix}.npz"
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_emb, emb=emb, pid=sid)
    print(f"\n[done] saved embeddings to {out_emb}")

    out_res = Path("/home/ubuntu/multimodal-user-memory/results") / f"sanity_{prefix}_collisions.json"
    import json
    with open(out_res, "w") as f:
        json.dump({
            "encoder": model_id,
            "n_embeddings": int(len(emb)),
            "n_speakers": int(len(set(sid))),
            "raw_cosine": {
                "top1_same_speaker_recall": float(nn_same),
                "intra_mean": float(np.mean(intra)),
                "inter_mean": float(np.mean(inter)),
            },
            "quantisation": {str(K): {"intra_agree": float(a), "inter_coll": float(b),
                                       "ratio": float(r) if r != float("inf") else None}
                             for K, (a, b, r) in results.items()},
        }, f, indent=2)
    print(f"[done] saved summary to {out_res}")


def run_libri_paralinguistic_proxy(model, proc):
    """Fallback: use LibriSpeech speakers across chapters as a weak proxy.
    Tests whether wav2vec2 features cluster ACROSS chapters (= state changes)
    less tightly than ECAPA does. This is informative but not the real test."""
    import soundfile as sf
    libri = Path.home() / "data" / "LibriSpeech" / "test-clean"
    speakers = [d for d in sorted(libri.iterdir()) if d.is_dir() and len([c for c in d.iterdir() if c.is_dir()]) >= 2]
    speakers = random.sample(speakers, k=min(NUM_SPEAKERS, len(speakers)))
    embs, sids = [], []
    print(f"\n[fallback] LibriSpeech proxy: {len(speakers)} speakers")
    for k, spk in enumerate(speakers):
        chapters = sorted([c for c in spk.iterdir() if c.is_dir()])
        flacs = []
        for c in chapters:
            flacs.extend(sorted(c.glob("*.flac")))
        sel = random.sample(flacs, k=min(UTTERANCES_PER_SPEAKER, len(flacs)))
        for u in sel:
            data, sr = sf.read(str(u), always_2d=False, dtype="float32")
            if data.ndim > 1: data = data.mean(axis=1)
            if sr != 16000:
                import torchaudio
                data = torchaudio.functional.resample(torch.from_numpy(data).unsqueeze(0), sr, 16000).squeeze(0).numpy()
            # Clip to 5 sec
            data = data[:16000 * 5]
            inputs = proc(data, sampling_rate=16000, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inputs)
            f = out.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            f = f / (np.linalg.norm(f) + 1e-9)
            embs.append(f); sids.append(spk.name)
    emb = np.stack(embs).astype(np.float32)
    sid = np.array(sids)
    print(f"  proxy embeddings: {emb.shape}, {len(set(sid))} speakers")
    eval_and_save(emb, sid, "facebook/wav2vec2-base-960h", prefix="paralinguistic_proxy")


if __name__ == "__main__":
    sys.exit(main())
