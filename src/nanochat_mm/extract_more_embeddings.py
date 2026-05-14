"""Re-extract perceptual embeddings at larger scale.

Goals:
  - Vision: fetch all LFW identities with >= 10 photos (158 of them via
    sklearn).
  - Audio: extract from BOTH test-clean and test-other speakers with
    >= 2 chapters, to ~50+ speakers total.

Saves to runs/embeddings/{arcface_lfw_large.npz, ecapa_libri_large.npz}
with the SAME npz layout (emb, pid) as the original.
"""
import sys
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
import cv2

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIR = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")


def extract_vision_large(min_faces=10, photos_per_id=10):
    print(f"[vision] fetching LFW (color=True, min_faces_per_person={min_faces}) ...")
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, color=True, resize=1.0)
    print(f"[vision] LFW: {lfw.images.shape[0]} photos, {len(lfw.target_names)} people")

    by_person = defaultdict(list)
    for i, t in enumerate(lfw.target):
        by_person[int(t)].append(i)
    eligible = sorted([(p, idxs) for p, idxs in by_person.items() if len(idxs) >= photos_per_id])
    print(f"[vision] {len(eligible)} people have >= {photos_per_id} photos")

    sess = ort.InferenceSession(
        "/home/ubuntu/.insightface/models/buffalo_l/w600k_r50.onnx",
        providers=['CPUExecutionProvider'],
    )
    inp_name = sess.get_inputs()[0].name

    embs, pids = [], []
    for k, (pid, idxs) in enumerate(eligible):
        sel = random.sample(idxs, k=photos_per_id)
        for i in sel:
            img = lfw.images[i]
            img = (img * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            arr = ((img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
            e = sess.run(None, {inp_name: arr})[0][0]
            e = e / (np.linalg.norm(e) + 1e-9)
            embs.append(e); pids.append(str(pid))
        if (k + 1) % 20 == 0:
            print(f"  processed {k+1}/{len(eligible)} identities, {len(embs)} embeddings")
    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    out = EMB_DIR / "arcface_lfw_large.npz"
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=emb, pid=pid)
    print(f"[vision] wrote {emb.shape} embeddings, {len(set(pids))} identities -> {out}")
    return emb, pid


def extract_audio_large(min_chapters=2, utts_per_spk=10):
    print(f"[audio] scanning LibriSpeech test-clean + test-other ...")
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/home/ubuntu/multimodal-user-memory/runs/pretrained-ecapa",
        run_opts={"device": DEVICE},
    )
    libri_root = Path.home() / "data" / "LibriSpeech"
    speakers = []
    for subset in ["test-clean", "test-other"]:
        subset_dir = libri_root / subset
        if not subset_dir.exists(): continue
        for spk_dir in sorted(subset_dir.iterdir()):
            if not spk_dir.is_dir(): continue
            chapters = [c for c in spk_dir.iterdir() if c.is_dir()]
            if len(chapters) >= min_chapters:
                speakers.append(spk_dir)
    print(f"[audio] {len(speakers)} eligible speakers across test-clean + test-other")

    embs, pids = [], []
    for k, spk_dir in enumerate(speakers):
        chapters = sorted([c for c in spk_dir.iterdir() if c.is_dir()])
        flacs = []
        for c in chapters:
            flacs.extend(sorted(c.glob("*.flac")))
        sel = random.sample(flacs, k=min(utts_per_spk, len(flacs)))
        for u in sel:
            try:
                data, sr = sf.read(str(u), always_2d=False, dtype="float32")
                if data.ndim > 1: data = data.mean(axis=1)
                wav = torch.from_numpy(data).unsqueeze(0)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                wav = wav[:, :16000 * 6]
                e = enc.encode_batch(wav.to(DEVICE)).squeeze().cpu().numpy()
                e = e / (np.linalg.norm(e) + 1e-9)
                embs.append(e); pids.append(spk_dir.name)
            except Exception as ex:
                pass
        if (k + 1) % 10 == 0:
            print(f"  processed {k+1}/{len(speakers)} speakers, {len(embs)} embeddings")
    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    out = EMB_DIR / "ecapa_libri_large.npz"
    np.savez(out, emb=emb, pid=pid)
    print(f"[audio] wrote {emb.shape} embeddings, {len(set(pids))} identities -> {out}")
    return emb, pid


def main():
    print("Re-extracting embeddings at larger scale")
    print("=" * 60)
    extract_vision_large(min_faces=10, photos_per_id=10)
    print()
    extract_audio_large(min_chapters=2, utts_per_spk=10)


if __name__ == "__main__":
    sys.exit(main())
