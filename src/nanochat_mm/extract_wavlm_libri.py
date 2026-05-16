"""Extract WavLM-SV speaker embeddings on LibriSpeech.

WavLM-Base+ SV (Microsoft) is a stronger speaker encoder than ECAPA-TDNN,
explicitly fine-tuned for speaker verification. The hypothesis: stronger
cross-channel/cross-session embeddings → higher same-code rate at any K.

We re-extract on the same LibriSpeech utterances as the current
ecapa_libri_large.npz (29+29 IDs, 580 samples) for direct comparison.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Match the librispeech samples used in ecapa_libri_large.npz extraction.
# The original extractor picked 10 chapters per speaker, ~20 speakers per
# train/eval split. Recreate using sklearn-style speaker indexing.
LIBRI_ROOT = Path("/home/ubuntu/LibriSpeech/train-clean-100")
OUT = "/home/ubuntu/multimodal-user-memory/runs/embeddings/wavlm_libri_large.npz"


def collect_audio_files(per_speaker=10, max_speakers=58):
    """Walk LibriSpeech and collect (speaker_id, audio_path) tuples."""
    speakers = sorted([d.name for d in LIBRI_ROOT.iterdir() if d.is_dir()])[:max_speakers]
    items = []
    for sp in speakers:
        chapters = sorted([d for d in (LIBRI_ROOT / sp).iterdir() if d.is_dir()])
        if not chapters: continue
        flac_files = []
        for ch in chapters:
            flacs = sorted(ch.glob("*.flac"))
            flac_files.extend(flacs)
            if len(flac_files) >= per_speaker:
                break
        for f in flac_files[:per_speaker]:
            items.append((sp, str(f)))
    return items


def main():
    print("=" * 70)
    print("WavLM-Base+ SV speaker embeddings on LibriSpeech")
    print("=" * 70)

    items = collect_audio_files(per_speaker=10, max_speakers=58)
    by_sp = defaultdict(list)
    for sp, p in items: by_sp[sp].append(p)
    print(f"  collected {len(items)} files across {len(by_sp)} speakers "
          f"(mean {np.mean([len(v) for v in by_sp.values()]):.1f}/spk)")

    print("\nLoading WavLM-Base+ SV ...")
    from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor
    model_name = "microsoft/wavlm-base-plus-sv"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name).to(DEVICE).eval()
    print("  loaded.")

    import soundfile as sf
    embs = []; pids = []
    t0 = time.time()
    for i, (sp, path) in enumerate(items):
        audio, sr = sf.read(path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # Cap to 6s for speed (typical utterance length)
        max_len = 16000 * 6
        if len(audio) > max_len:
            audio = audio[:max_len]
        inputs = fe(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            x = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
            emb = F.normalize(x.embeddings, dim=-1)[0].cpu().numpy()
        embs.append(emb); pids.append(sp)
        if (i+1) % 50 == 0:
            print(f"    encoded {i+1}/{len(items)}  ({time.time()-t0:.0f}s)")

    embs = np.stack(embs).astype(np.float32)
    pids = np.array(pids)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, emb=embs, pid=pids)
    print(f"\n[saved] {OUT}  shape={embs.shape}  {len(set(pids.tolist()))} speakers")


if __name__ == "__main__":
    sys.exit(main())
