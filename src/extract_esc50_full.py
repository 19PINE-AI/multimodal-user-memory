"""Extract AST features on ALL 50 ESC-50 classes with 20 clips each."""
import io
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import torchaudio

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIPS_PER_SCENE = 20  # was 8


def main():
    print("Loading ESC-50 (full 50 classes, 20 clips each) ...")
    from transformers import AutoFeatureExtractor, AutoModel
    from datasets import load_dataset, Audio
    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    proc = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE).eval()

    ds = load_dataset("ashraq/esc50", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    labels = ds["category"]
    by_lbl = defaultdict(list)
    for i, l in enumerate(labels):
        by_lbl[str(l)].append(i)
    print(f"  {len(by_lbl)} classes")

    embs, lbls = [], []
    for k, (lbl, idxs) in enumerate(sorted(by_lbl.items())):
        sel = random.sample(idxs, k=min(CLIPS_PER_SCENE, len(idxs)))
        batch_wavs = []
        for i in sel:
            ad = ds[i]["audio"]
            if not (isinstance(ad, dict) and ad.get("bytes") is not None): continue
            try:
                wav, sr = sf.read(io.BytesIO(ad["bytes"]), dtype="float32")
            except: continue
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != 16000:
                wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
            wav = wav[:16000 * 5] if len(wav) >= 16000*5 else np.pad(wav, (0, 16000*5 - len(wav)))
            batch_wavs.append(wav)
        if not batch_wavs: continue
        inputs = proc(batch_wavs, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        feats = out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        for f in feats:
            embs.append(f); lbls.append(str(lbl))
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(by_lbl)} classes, {len(embs)} embs")

    emb = np.stack(embs).astype(np.float32); lbl = np.array(lbls)
    out = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/ast_esc50_full.npz")
    np.savez(out, emb=emb, pid=lbl)
    print(f"\n[done] {emb.shape}, {len(set(lbl))} scenes -> {out}")


if __name__ == "__main__":
    sys.exit(main())
