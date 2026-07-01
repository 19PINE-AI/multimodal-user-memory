"""Vision+audio (video) identity: does storing BOTH a face and a voice as a fused key
beat either modality alone for cross-condition recall? Uses real RAVDESS video (24 actors
recorded on video across 8 emotions), so each identity has a genuine (face, voice) pair.

Per clip we extract a face frame (ArcFace) and the audio (ECAPA). We register each actor
from one clip and recognise a held-out clip of a DIFFERENT emotion, comparing:
  face-only  : ArcFace cosine
  voice-only : ECAPA cosine
  fused (AV) : sum of the two cosines (equal weight)
This is the vision+audio regime: the memory carries both channels for one identity.

Usage: python3 eval_av_fusion.py [M] [n_clips_per_actor] [n_draws]
"""
import sys, io, json, subprocess, tempfile, glob, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from face_encoder import FaceDetector, ArcFaceEncoderBGR

DEV = os.environ.get("ATTMEM_DEVICE", "cuda"); SR = 16000
RAV = Path("/home/ubuntu/multimodal-user-memory/data/ravdess_av")
CACHE = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/ravdess_av.npz")
from PIL import Image


from PIL import ImageFilter

def extract(mp4):
    """Return (face_bgr112 or None, wav16k) via ffmpeg into temp files."""
    with tempfile.TemporaryDirectory() as td:
        fp, ap = f"{td}/f.png", f"{td}/a.wav"
        subprocess.run(["ffmpeg", "-y", "-i", mp4, "-vf", "select=eq(n\\,30)",
                        "-vframes", "1", fp], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", mp4, "-ar", str(SR), "-ac", "1", ap],
                       capture_output=True)
        face = Image.open(fp).convert("RGB") if os.path.exists(fp) else None
        wav, _ = sf.read(ap, dtype="float32") if os.path.exists(ap) else (np.zeros(SR), SR)
        return face, wav[:SR * 4] if len(wav) >= SR * 4 else np.pad(wav, (0, SR * 4 - len(wav)))


def degrade_face(pil):
    """Simulate a distant/tiny face: severe downscale (to ~14px) then back up, plus blur.
    RAVDESS faces are studio-frontal and trivial for ArcFace at full res, so only strong
    resolution loss (a realistic far-camera condition) degrades identity."""
    w, h = pil.size
    small = pil.resize((14, 14), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return small.filter(ImageFilter.GaussianBlur(radius=3))


def degrade_audio(wav, rng, snr_db=0):
    """Simulate a noisy channel: additive white noise at the given SNR."""
    p = np.mean(wav ** 2) + 1e-9
    n = rng.standard_normal(len(wav)).astype(np.float32)
    n *= np.sqrt(p / (10 ** (snr_db / 10)) / (np.mean(n ** 2) + 1e-9))
    return (wav + n).astype(np.float32)


def build_cache(M, n_clips):
    from speechbrain.inference.speaker import EncoderClassifier
    det = FaceDetector(gpu=(DEV == "cuda"))
    arc = ArcFaceEncoderBGR(providers=(("CUDAExecutionProvider", "CPUExecutionProvider")
                                        if DEV == "cuda" else ("CPUExecutionProvider",)))
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           run_opts={"device": DEV})
    def voice(wav):
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(wav).float()[None].to(DEV)).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    actors = sorted(glob.glob(str(RAV / "Actor_*")))
    F, Fb, V, Vn, PID = [], [], [], [], []
    rng = np.random.default_rng(0)
    for ad in actors:
        aid = os.path.basename(ad).split("_")[1]
        clips = sorted(glob.glob(f"{ad}/*.mp4")); rng.shuffle(clips); clips = clips[:n_clips]
        for c in clips:
            face, wav = extract(c)
            if face is None: continue
            a = det.detect_align(face)
            if a is None: continue
            ab = det.detect_align(degrade_face(face))          # blurred-face variant
            F.append(arc.encode_bgr112(a))
            Fb.append(arc.encode_bgr112(ab) if ab is not None else arc.encode_pil(degrade_face(face)))
            V.append(voice(wav)); Vn.append(voice(degrade_audio(wav, rng)))
            PID.append(aid)
        print(f"  actor {aid}: {PID.count(aid)} clips", flush=True)
    F = np.stack(F).astype(np.float32); Fb = np.stack(Fb).astype(np.float32)
    V = np.stack(V).astype(np.float32); Vn = np.stack(Vn).astype(np.float32)
    PID = np.array(PID)
    np.savez(CACHE, face=F, face_blur=Fb, voice=V, voice_noise=Vn, pid=PID)
    print(f"cached {F.shape[0]} clips, {len(set(PID.tolist()))} actors -> {CACHE}")
    return F, Fb, V, Vn, PID


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n_clips = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_draws = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    need = ["face", "face_blur", "voice", "voice_noise", "pid"]
    if CACHE.exists() and all(k in np.load(CACHE, allow_pickle=True) for k in need):
        d = np.load(CACHE, allow_pickle=True)
        F, Fb, V, Vn, PID = d["face"], d["face_blur"], d["voice"], d["voice_noise"], d["pid"]
    else:
        F, Fb, V, Vn, PID = build_cache(M, n_clips)
    def l2(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    F, Fb, V, Vn = l2(F), l2(Fb), l2(V), l2(Vn)
    by = defaultdict(list)
    for i, p in enumerate(PID): by[str(p)].append(i)
    actors = [p for p in by if len(by[p]) >= 2]

    # three query conditions: both clean, face degraded (blur), audio degraded (noise).
    # registration is always clean face+voice; only the query channel degrades.
    conds = {"clean": (F, V), "face degraded": (Fb, V), "audio degraded": (F, Vn)}

    def draw(seed, Qfsrc, Qvsrc):
        rng = np.random.default_rng(seed); pick = list(actors); rng.shuffle(pick); pick = pick[:M]
        reg, qry = [], []
        for p in pick:
            ix = list(by[p]); rng.shuffle(ix); reg.append(ix[0]); qry.append(ix[1])
        Rf, Rv = F[reg], V[reg]; Qf, Qv = Qfsrc[qry], Qvsrc[qry]
        gt = np.arange(len(pick)); sf_ = Qf @ Rf.T; sv = Qv @ Rv.T
        return (float((sf_.argmax(1) == gt).mean()),
                float((sv.argmax(1) == gt).mean()),
                float(((sf_ + sv).argmax(1) == gt).mean()))

    names = ["face-only", "voice-only", "fused (AV)"]
    out = {"M": M, "n_draws": n_draws, "n_actors": len(actors), "chance": 1.0 / M, "conditions": {}}
    print(f"\n=== Audiovisual identity + robustness (RAVDESS, M={M}, {len(actors)} actors, {n_draws} draws) ===")
    for cond, (Qf, Qv) in conds.items():
        res = np.array([draw(s, Qf, Qv) for s in range(1000, 1000 + n_draws)])
        row = {}
        for j, nm in enumerate(names):
            m = res[:, j].mean(); ci = 1.96 * res[:, j].std(ddof=1) / np.sqrt(len(res))
            row[nm] = {"recall": float(m), "ci95": float(ci)}
        out["conditions"][cond] = row
        print(f"  [{cond:14}] " + "  ".join(f"{nm} {row[nm]['recall']:.3f}" for nm in names))
    # keep top-level clean numbers for backward-compatible figure code
    out.update(out["conditions"]["clean"])
    print(f"  chance ~{1/M:.3f}")
    Path("/home/ubuntu/multimodal-user-memory/results/av_fusion.json").write_text(json.dumps(out, indent=2))
    print("wrote results/av_fusion.json")


if __name__ == "__main__":
    main()
