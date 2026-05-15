"""Robustness probes — stress tests for cross-condition variation.

1. Audio: register clean clip; query a NOISY/FILTERED version of clip
   (simulates cross-channel like phone vs studio). Test ECAPA's
   robustness AND Path A's mechanism.

2. Vision: register clean face; query an OCCLUDED version (random
   black box). Test ArcFace robustness AND Path A's mechanism.

We just measure intra/inter cosine and code-match consistency, not full
Path A retrieval — these probes characterise the ENCODER's robustness
which directly determines Path A's match-fraction.
"""
import io
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)


def main():
    print("=" * 70)
    print("Robustness probes")
    print("=" * 70)

    # ----- AUDIO: cross-microphone via additive noise / bandpass filter -----
    print("\n[AUDIO] cross-microphone simulation on LibriSpeech ...")
    import soundfile as sf
    import torch, torchaudio
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/home/ubuntu/multimodal-user-memory/runs/pretrained-ecapa",
        run_opts={"device": DEVICE},
    )

    libri = Path.home() / "data" / "LibriSpeech" / "test-clean"
    speakers = [d for d in sorted(libri.iterdir()) if d.is_dir()][:15]

    def encode_with_perturbation(wav, sr, mode):
        wav = wav[:16000*6]
        if mode == "clean":
            pass
        elif mode == "phone":
            # Simulate phone band: 300-3400 Hz bandpass + add modest noise
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav_t = torchaudio.functional.bandpass_biquad(wav_t, 16000, central_freq=1850, Q=0.8)
            wav_t = wav_t + 0.02 * torch.randn_like(wav_t)
            wav = wav_t.squeeze(0).numpy()
        elif mode == "noisy":
            wav = wav + 0.05 * np.random.randn(len(wav)).astype(np.float32)
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        return enc.encode_batch(wav_t.to(DEVICE)).squeeze().cpu().numpy()

    speaker_pairs = []
    for spk in speakers[:10]:
        chapters = sorted([c for c in spk.iterdir() if c.is_dir()])
        if not chapters: continue
        flacs = list(chapters[0].glob("*.flac"))[:1]
        for u in flacs:
            data, sr = sf.read(str(u), always_2d=False, dtype="float32")
            if data.ndim > 1: data = data.mean(axis=1)
            if sr != 16000:
                data = torchaudio.functional.resample(torch.from_numpy(data).unsqueeze(0), sr, 16000).squeeze(0).numpy()
            clean_emb = encode_with_perturbation(data, 16000, "clean")
            phone_emb = encode_with_perturbation(data, 16000, "phone")
            noisy_emb = encode_with_perturbation(data, 16000, "noisy")
            clean_emb = clean_emb / (np.linalg.norm(clean_emb) + 1e-9)
            phone_emb = phone_emb / (np.linalg.norm(phone_emb) + 1e-9)
            noisy_emb = noisy_emb / (np.linalg.norm(noisy_emb) + 1e-9)
            speaker_pairs.append((spk.name, clean_emb, phone_emb, noisy_emb))

    print(f"  collected {len(speaker_pairs)} speaker clean+phone+noisy triples")

    intra_phone = [float(clean @ phone) for _, clean, phone, _ in speaker_pairs]
    intra_noisy = [float(clean @ noisy) for _, clean, _, noisy in speaker_pairs]
    print(f"  ECAPA same-speaker (clean vs phone-filtered): mean cosine = {np.mean(intra_phone):.4f}")
    print(f"  ECAPA same-speaker (clean vs +noise):         mean cosine = {np.mean(intra_noisy):.4f}")

    # K-means on clean embeddings; check code consistency under perturbation
    clean_embs = np.stack([p[1] for p in speaker_pairs])
    phone_embs = np.stack([p[2] for p in speaker_pairs])
    noisy_embs = np.stack([p[3] for p in speaker_pairs])
    import faiss
    K = 4  # small N=10; K=4 is the most we can fit
    km = faiss.Kmeans(clean_embs.shape[1], K, niter=20, verbose=False, seed=SEED)
    km.train(clean_embs)
    _, codes_clean = km.index.search(clean_embs, 1); codes_clean = codes_clean.squeeze(1)
    _, codes_phone = km.index.search(phone_embs, 1); codes_phone = codes_phone.squeeze(1)
    _, codes_noisy = km.index.search(noisy_embs, 1); codes_noisy = codes_noisy.squeeze(1)
    print(f"  code stability clean→phone: {float((codes_clean == codes_phone).mean()):.3f}")
    print(f"  code stability clean→noisy: {float((codes_clean == codes_noisy).mean()):.3f}")

    # ----- VISION: cross-occlusion -----
    print("\n[VISION] cross-occlusion on LFW ...")
    import cv2, onnxruntime as ort
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=10, color=True, resize=1.0)
    by_id = defaultdict(list)
    for i, t in enumerate(lfw.target):
        by_id[int(t)].append(i)
    eligible = sorted([(p, idxs) for p, idxs in by_id.items() if len(idxs) >= 2])[:20]

    sess = ort.InferenceSession(
        "/home/ubuntu/.insightface/models/buffalo_l/w600k_r50.onnx",
        providers=['CPUExecutionProvider'],
    )
    inp_name = sess.get_inputs()[0].name

    def encode_face(img, occlusion=None):
        img = (img * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        if occlusion == "eye_mask":
            img[20:50, :, :] = 0  # black box over eyes
        elif occlusion == "mouth_mask":
            img[70:100, :, :] = 0  # black box over mouth
        elif occlusion == "noise":
            img = (img.astype(np.int32) + np.random.randint(-40, 40, img.shape)).clip(0, 255).astype(np.uint8)
        arr = ((img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        e = sess.run(None, {inp_name: arr})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)

    intra_eye = []; intra_mouth = []; intra_noise = []
    code_eye = []; code_mouth = []; code_noise = []
    clean_embs_v = []
    eye_embs_v = []; mouth_embs_v = []; noise_embs_v = []
    for pid, idxs in eligible:
        i = idxs[0]
        clean = encode_face(lfw.images[i])
        eye = encode_face(lfw.images[i], "eye_mask")
        mouth = encode_face(lfw.images[i], "mouth_mask")
        noise = encode_face(lfw.images[i], "noise")
        intra_eye.append(float(clean @ eye))
        intra_mouth.append(float(clean @ mouth))
        intra_noise.append(float(clean @ noise))
        clean_embs_v.append(clean)
        eye_embs_v.append(eye); mouth_embs_v.append(mouth); noise_embs_v.append(noise)
    print(f"  ArcFace same-face (clean vs eye-mask):    mean cosine = {np.mean(intra_eye):.4f}")
    print(f"  ArcFace same-face (clean vs mouth-mask):  mean cosine = {np.mean(intra_mouth):.4f}")
    print(f"  ArcFace same-face (clean vs random noise): mean cosine = {np.mean(intra_noise):.4f}")

    clean_v = np.stack(clean_embs_v); eye_v = np.stack(eye_embs_v); mouth_v = np.stack(mouth_embs_v); noise_v = np.stack(noise_embs_v)
    K_v = 8  # 20 vision samples
    km2 = faiss.Kmeans(clean_v.shape[1], K_v, niter=20, verbose=False, seed=SEED)
    km2.train(clean_v)
    _, c_clean = km2.index.search(clean_v, 1); c_clean = c_clean.squeeze(1)
    _, c_eye = km2.index.search(eye_v, 1); c_eye = c_eye.squeeze(1)
    _, c_mouth = km2.index.search(mouth_v, 1); c_mouth = c_mouth.squeeze(1)
    _, c_noise = km2.index.search(noise_v, 1); c_noise = c_noise.squeeze(1)
    print(f"  code stability clean→eye-mask:    {float((c_clean == c_eye).mean()):.3f}")
    print(f"  code stability clean→mouth-mask:  {float((c_clean == c_mouth).mean()):.3f}")
    print(f"  code stability clean→noise:       {float((c_clean == c_noise).mean()):.3f}")

    out = Path("/home/ubuntu/multimodal-user-memory/results/robustness_probes.json")
    import json
    with open(out, "w") as f:
        json.dump({
            "audio": {
                "clean_vs_phone_cosine": float(np.mean(intra_phone)),
                "clean_vs_noisy_cosine": float(np.mean(intra_noisy)),
                "code_stability_phone": float((codes_clean == codes_phone).mean()),
                "code_stability_noisy": float((codes_clean == codes_noisy).mean()),
            },
            "vision": {
                "clean_vs_eye_mask_cosine": float(np.mean(intra_eye)),
                "clean_vs_mouth_mask_cosine": float(np.mean(intra_mouth)),
                "clean_vs_noise_cosine": float(np.mean(intra_noise)),
                "code_stability_eye": float((c_clean == c_eye).mean()),
                "code_stability_mouth": float((c_clean == c_mouth).mean()),
                "code_stability_noise": float((c_clean == c_noise).mean()),
            },
        }, f, indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    sys.exit(main())
