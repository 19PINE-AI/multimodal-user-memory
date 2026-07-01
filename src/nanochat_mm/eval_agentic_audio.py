"""Pure-audio grounding: the audio analog of the vision ablation. A multi-speaker clip
is built by concatenating K single-speaker utterances (real VoxCeleb speakers) in random
order. The audio-LLM (Qwen2.5-Omni) grounds the referenced speaker to a time span --- the
direct analog of the VLM grounding a bounding box --- and ECAPA identifies that span,
matched against M registered speakers (each registered from a different, clean clip).

Methods (recall@1 over the M registered speakers):
  oracle : ECAPA on the target's true segment                     (ceiling)
  agentic: Omni grounds the K-th speaker's time span -> ECAPA     (ours)
  whole  : ECAPA on the whole K-speaker clip                      (store-only floor)
Also reports grounding accuracy (predicted span midpoint lands in the true segment).

Usage: python3 eval_agentic_audio.py [M] [K] [seed]
"""
import sys, re, io, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, soundfile as sf, torchaudio

DEV = "cuda"; OMNI = "Qwen/Qwen2.5-Omni-7B"; SR = 16000
SEG = 4  # seconds per speaker segment
ORD = ["first", "second", "third", "fourth", "fifth"]


def load_wav(ad, secs):
    wav, sr = sf.read(io.BytesIO(ad["bytes"]), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav)[None], sr, SR).squeeze(0).numpy()
    n = SR * secs
    return wav[:n] if len(wav) >= n else np.pad(wav, (0, n - len(wav)))


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    from datasets import load_dataset, Audio
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    ds = load_dataset("Codec-SUPERB/Voxceleb1_test_original", split="test").cast_column("audio", Audio(decode=False))
    by = defaultdict(list)
    for i, fn in enumerate(ds["id"]):
        by[fn.split("+")[0]].append(i)
    ids = [s for s, v in by.items() if len(v) >= 3]
    rng = np.random.default_rng(SEED); rng.shuffle(ids); ids = ids[:M]
    audio = ds["audio"]

    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           run_opts={"device": DEV})
    def emb(wav):
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(wav).float()[None].to(DEV)).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    oproc = Qwen2_5OmniProcessor.from_pretrained(OMNI)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        OMNI, dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True).eval()
    try: omni.disable_talker()
    except Exception: pass

    def ground_span(clip, pos, nspk):
        secs = len(clip) / SR
        q = (f"This {secs:.0f}-second audio contains {nspk} different people speaking one "
             f"after another. When does the {ORD[pos]} speaker speak? "
             f"Answer with only their start and end time in seconds as [start, end].")
        conv = [{"role": "user", "content": [{"type": "audio", "audio": clip},
                 {"type": "text", "text": q}]}]
        text = oproc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = oproc(text=text, audio=[clip], return_tensors="pt", sampling_rate=SR).to(DEV)
        with torch.no_grad():
            out = omni.generate(**inp, max_new_tokens=32, do_sample=False, return_audio=False)
        dec = oproc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        nums = re.findall(r'(\d+\.?\d*)', dec)
        if len(nums) >= 2:
            a, b = float(nums[0]), float(nums[1])
            if b > a: return a, b
        return None

    # register each speaker from a clean single clip (cross-session query later)
    reg = {}
    for k, s in enumerate(ids):
        reg[s] = emb(load_wav(audio[by[s][0]], SEG))
    R = np.stack([reg[s] for s in ids])

    methods = ["oracle", "agentic", "whole"]
    hit = {m: 0 for m in methods}; tot = 0; ground_ok = 0
    for k, s in enumerate(ids):
        pos = int(rng.integers(0, K))
        others = [rng.choice([q for q in ids if q != s]) for _ in range(K - 1)]
        # build the K-speaker clip: target's SECOND clip + distractors' clips
        segs, boundaries, oi = [], [], 0
        for j in range(K):
            spk = s if j == pos else others[oi]; oi += 0 if j == pos else 1
            src = by[spk][1] if j == pos else by[spk][int(rng.integers(0, len(by[spk])))]
            w = load_wav(audio[src], SEG)
            boundaries.append((len(np.concatenate(segs)) if segs else 0,
                               (len(np.concatenate(segs)) if segs else 0) + len(w)))
            segs.append(w)
        clip = np.concatenate(segs).astype(np.float32)
        t0, t1 = boundaries[pos]
        qw = {"oracle": clip[t0:t1], "whole": clip}
        # agentic grounding
        span = ground_span(clip, pos, K)
        if span is not None:
            gs, ge = int(span[0] * SR), int(span[1] * SR)
            gs = max(0, min(gs, len(clip) - SR)); ge = max(gs + SR, min(ge, len(clip)))
            mid = (gs + ge) / 2
            ground_ok += int(t0 <= mid <= t1)
            qw["agentic"] = clip[gs:ge]
        else:
            qw["agentic"] = clip
        for m in methods:
            q = emb(qw[m]); pred = int((q @ R.T).argmax())
            hit[m] += int(ids[pred] == s)
        tot += 1
        if k % 10 == 0: print(f"  {k}/{M}", flush=True)

    print(f"\n=== Pure-audio grounding (Omni+ECAPA, VoxCeleb, M={M}, K={K}, seed={SEED}) ===")
    res = {m: hit[m] / tot for m in methods}
    for m in methods: print(f"  {m:8} recall@1 : {res[m]:.3f}")
    res["grounding_acc"] = ground_ok / tot
    print(f"  grounding accuracy : {res['grounding_acc']:.3f}   chance ~{1/M:.3f}")
    res.update({"M": M, "K": K, "seed": SEED, "chance": 1.0 / M})
    Path(f"/home/ubuntu/multimodal-user-memory/results/agentic_audio_K{K}_s{SEED}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote results/agentic_audio_K{K}_s{SEED}.json")


if __name__ == "__main__":
    main()
