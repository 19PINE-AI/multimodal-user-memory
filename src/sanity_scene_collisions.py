"""Sanity check 4 — acoustic scene encoder on ESC-50 (room/scene class proxy).

The full PerceptMem A-SCN task is "is this the same room as last week"
— cross-recording same-scene. DCASE TAU Urban Acoustic Scenes is the
ideal benchmark; for the sanity check we use ESC-50 (50 environment
classes, 40 clips each, ~2GB total). The clips per class are recorded
in different acoustic conditions, so "same class across clips" proxies
"same scene across recordings."

Encoder: AST (Audio Spectrogram Transformer) or PANNs CNN14.
We try AST first via HF: `MIT/ast-finetuned-audioset-10-10-0.4593`
which is a strong general audio classifier. Use its penultimate
mean-pooled features.

Win condition: same-scene-class intra-cosine >> different-scene-class
inter-cosine, and naive k-means K~32 separates well.
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SCENES = 40        # ESC-50 has 50
CLIPS_PER_SCENE = 8    # ESC-50 has 40


def main():
    print("=" * 70)
    print("Sanity check 4 — AST acoustic-scene encoder on ESC-50")
    print("=" * 70)

    from transformers import AutoFeatureExtractor, AutoModel
    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    print(f"\n[load] {model_id} ...")
    proc = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    print(f"  params = {sum(p.numel() for p in model.parameters()):,}")

    print("\n[data] loading ESC-50 ...")
    from datasets import load_dataset, Audio
    ds = None
    for ds_id in ["ashraq/esc50", "dynamic-superb/EnvironmentalSoundClassification_AnimalsMix_ESC50-Animals"]:
        try:
            ds = load_dataset(ds_id, split="train")
            print(f"  loaded {ds_id}: {len(ds)} clips")
            print(f"  columns: {ds.column_names}")
            break
        except Exception as e:
            print(f"  {ds_id} failed: {type(e).__name__}: {str(e)[:80]}")
    if ds is None:
        print("[fail] no acoustic-scene dataset available; aborting sanity 4")
        return
    # Bypass HF's torchcodec-based audio decoder (broken in this env);
    # we will manually decode via soundfile from the audio bytes.
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))

    # Identify label field
    label_field = None
    for f in ["category", "label", "target", "class_name", "esc10_class"]:
        if f in ds.column_names:
            label_field = f
            break
    if label_field is None:
        # Use 'filename' parsing as last resort
        print("  no label field; aborting")
        return
    print(f"  using label field: {label_field}")

    # Iterate via column access to avoid per-row audio decode
    labels_col = ds[label_field]
    by_lbl = defaultdict(list)
    for i, lbl in enumerate(labels_col):
        by_lbl[str(lbl)].append(i)
    print(f"  {len(by_lbl)} distinct labels")
    eligible = [(l, idxs) for l, idxs in by_lbl.items() if len(idxs) >= CLIPS_PER_SCENE]
    print(f"  {len(eligible)} labels have >= {CLIPS_PER_SCENE} clips")
    chosen = random.sample(eligible, k=min(NUM_SCENES, len(eligible)))
    print(f"  sampled {len(chosen)} scenes")

    print("\n[encode] extracting AST features ...")
    import soundfile as sf
    import io
    embs, lbls = [], []
    for k, (lbl, idxs) in enumerate(chosen):
        sel = random.sample(idxs, k=CLIPS_PER_SCENE)
        # Batch encode for efficiency
        batch_wavs = []
        for i in sel:
            row = ds[i]
            ad = row.get("audio")
            if ad is None: continue
            # ad has 'bytes' and 'path' when decode=False
            if isinstance(ad, dict) and ad.get("bytes") is not None:
                wav, sr = sf.read(io.BytesIO(ad["bytes"]), dtype="float32")
            elif isinstance(ad, dict) and ad.get("path") is not None:
                wav, sr = sf.read(ad["path"], dtype="float32")
            else:
                continue
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
            # Pad/truncate to 5s
            target = 16000 * 5
            if len(wav) >= target: wav = wav[:target]
            else: wav = np.pad(wav, (0, target - len(wav)))
            batch_wavs.append(wav)
        if not batch_wavs: continue
        inputs = proc(batch_wavs, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        # AST returns last_hidden_state [B, T, D]; mean-pool
        feats = out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        for f in feats:
            embs.append(f); lbls.append(str(lbl))
        if (k + 1) % 10 == 0:
            print(f"  processed {k+1}/{len(chosen)} scenes, {len(embs)} embeddings")

    emb = np.stack(embs).astype(np.float32)
    lbl = np.array(lbls)
    D = emb.shape[1]
    print(f"\n[done] Collected {emb.shape}, {len(set(lbl))} scenes, dim={D}")

    # Top-1 NN
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    nn_same = (lbl[nn] == lbl).mean()
    print(f"\n[sanity] AST top-1 same-scene recall: {nn_same:.4f}")

    by_l = defaultdict(list)
    for i, l in enumerate(lbl): by_l[l].append(i)
    intra, inter = [], []
    for l, idxs in by_l.items():
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                intra.append(sims[idxs[i], idxs[j]])
    rng = np.random.RandomState(SEED)
    for _ in range(5000):
        i, j = rng.randint(0, len(emb), size=2)
        if lbl[i] != lbl[j]: inter.append(sims[i, j])
    print(f"[sanity] intra-scene cosine: mean={np.mean(intra):.4f}  n={len(intra)}")
    print(f"[sanity] inter-scene cosine: mean={np.mean(inter):.4f}  n={len(inter)}")

    import faiss
    print(f"\n[sanity] Quantisation:")
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    results = {}
    for K in [8, 16, 32, 64, 128]:
        if K > len(emb): continue
        km = faiss.Kmeans(D, K, niter=20, verbose=False, seed=SEED)
        km.train(emb)
        _, codes = km.index.search(emb, 1); codes = codes.squeeze(1)
        intra_agree = 0; intra_pairs = 0
        for l, idxs in by_l.items():
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    intra_pairs += 1
                    if codes[idxs[i]] == codes[idxs[j]]: intra_agree += 1
        rng2 = np.random.RandomState(SEED + K)
        inter_coll = 0; inter_pairs = 0
        for _ in range(10000):
            i, j = rng2.randint(0, len(emb), size=2)
            if lbl[i] != lbl[j]:
                inter_pairs += 1
                if codes[i] == codes[j]: inter_coll += 1
        intra_rate = intra_agree / max(intra_pairs, 1)
        inter_rate = inter_coll / max(inter_pairs, 1)
        ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
        print(f"  {K:>4d} | {intra_rate:>12.4f} | {inter_rate:>12.4f} | {ratio:>8.2f}")
        results[K] = (intra_rate, inter_rate, ratio)

    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings") / "ast_esc50.npz"
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_emb, emb=emb, pid=lbl)
    print(f"\n[done] saved embeddings to {out_emb}")

    out_res = Path("/home/ubuntu/multimodal-user-memory/results") / "sanity_scene_collisions.json"
    import json
    with open(out_res, "w") as f:
        json.dump({
            "encoder": model_id,
            "n_embeddings": int(len(emb)),
            "n_scenes": int(len(set(lbl))),
            "embedding_dim": int(D),
            "raw_cosine": {
                "top1_same_scene_recall": float(nn_same),
                "intra_mean": float(np.mean(intra)),
                "inter_mean": float(np.mean(inter)),
            },
            "quantisation": {str(K): {"intra_agree": float(a), "inter_coll": float(b),
                                       "ratio": float(r) if r != float("inf") else None}
                             for K, (a, b, r) in results.items()},
        }, f, indent=2)
    print(f"[done] saved summary to {out_res}")


if __name__ == "__main__":
    sys.exit(main())
