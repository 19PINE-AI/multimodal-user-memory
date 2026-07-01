"""Robust audio loading that bypasses torchcodec.

torchcodec 0.7 (the only build available for the installed torch 2.10 nightly) is ABI-
incompatible with that torch, and downgrading torch would break the VLM/encoder stack.
The reliable fix is to load HuggingFace audio datasets with Audio(decode=False) and decode
the raw bytes ourselves (soundfile for wav/flac; ffmpeg subprocess otherwise). This makes
otherwise-blocked datasets (VoxConverse, AMI, ...) usable via partial streaming, with no
bulk download and no torchcodec.
"""
import io, subprocess, tempfile, os
import numpy as np
import soundfile as sf


def decode_bytes(b, target_sr=16000):
    """Decode audio bytes to mono float32 at target_sr, via soundfile or ffmpeg."""
    try:
        wav, sr = sf.read(io.BytesIO(b), dtype="float32")
    except Exception:
        with tempfile.TemporaryDirectory() as td:
            ip, op = f"{td}/in", f"{td}/out.wav"
            open(ip, "wb").write(b)
            subprocess.run(["ffmpeg", "-y", "-i", ip, "-ar", str(target_sr), "-ac", "1", op],
                           capture_output=True)
            wav, sr = sf.read(op, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != target_sr:
        import torch, torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav)[None], sr, target_sr).squeeze(0).numpy()
    return wav.astype(np.float32)


def stream_dataset(name, split="test", config=None, decode_audio=False):
    """Stream a dataset partially (no full download) with audio decoding disabled."""
    from datasets import load_dataset, Audio
    d = (load_dataset(name, config, split=split, streaming=True) if config
         else load_dataset(name, split=split, streaming=True))
    if not decode_audio:
        d = d.cast_column("audio", Audio(decode=False))
    return d
