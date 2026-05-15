"""Extract ECAPA-TDNN embeddings from VoxCeleb1 test set for cross-channel.

VoxCeleb1 file naming: id10270+5r0dWxy17C8+00001.wav
  - speaker:  id10270  (40 unique speakers in test set)
  - session:  5r0dWxy17C8  (YouTube video; different sessions = different channels)
  - utterance: 00001
"""
import io
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset, Audio

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UTTS_PER_SPK_SESSION = 2  # 2 per (speaker, session) pair to keep total manageable
MIN_SESSIONS_PER_SPK = 2  # for cross-session test


def main():
    print("Loading VoxCeleb1 test ...")
    ds = load_dataset("Codec-SUPERB/Voxceleb1_test_original", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  {len(ds)} clips")

    # Group by (speaker, session)
    by_spk_sess = defaultdict(list)
    for i, fn in enumerate(ds["id"]):
        parts = str(fn).split("+")
        if len(parts) < 2: continue
        spk, sess = parts[0], parts[1]
        by_spk_sess[(spk, sess)].append(i)

    # Group sessions by speaker
    spk_sessions = defaultdict(list)
    for (spk, sess), idxs in by_spk_sess.items():
        spk_sessions[spk].append((sess, idxs))
    eligible_spk = [spk for spk, sessions in spk_sessions.items() if len(sessions) >= MIN_SESSIONS_PER_SPK]
    print(f"  {len(eligible_spk)} speakers with >= {MIN_SESSIONS_PER_SPK} sessions")

    # Load ECAPA-TDNN
    print("Loading ECAPA-TDNN ...")
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/home/ubuntu/multimodal-user-memory/runs/pretrained-ecapa",
        run_opts={"device": DEVICE},
    )

    embs, pids, sessions = [], [], []
    for k, spk in enumerate(eligible_spk):
        # Pick top sessions (largest), take a couple of utts each
        spk_sess_sorted = sorted(spk_sessions[spk], key=lambda x: -len(x[1]))
        n_sess = min(4, len(spk_sess_sorted))  # cap at 4 sessions per speaker
        for sess, idxs in spk_sess_sorted[:n_sess]:
            sel = random.sample(idxs, k=min(UTTS_PER_SPK_SESSION, len(idxs)))
            for i in sel:
                ad = ds[i]["audio"]
                if not (isinstance(ad, dict) and ad.get("bytes") is not None): continue
                try:
                    wav, sr = sf.read(io.BytesIO(ad["bytes"]), dtype="float32")
                except Exception: continue
                if wav.ndim > 1: wav = wav.mean(axis=1)
                wav_t = torch.from_numpy(wav).unsqueeze(0)
                if sr != 16000:
                    wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
                wav_t = wav_t[:, :16000 * 6]
                e = enc.encode_batch(wav_t.to(DEVICE)).squeeze().cpu().numpy()
                e = e / (np.linalg.norm(e) + 1e-9)
                embs.append(e); pids.append(spk); sessions.append(sess)
        if (k + 1) % 10 == 0:
            print(f"  processed {k+1}/{len(eligible_spk)} speakers, {len(embs)} embeddings")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    sess_arr = np.array(sessions)
    out = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_voxceleb1.npz")
    np.savez(out, emb=emb, pid=pid, session=sess_arr)
    print(f"\n[done] {emb.shape}, {len(set(pid))} speakers, "
          f"{len(set(sess_arr))} sessions total -> {out}")


if __name__ == "__main__":
    sys.exit(main())
