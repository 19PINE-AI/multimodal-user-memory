"""Text-only memory baseline vs parametric encoder, for the three AUDIO modalities
(speaker, vocal tone, acoustic scene), paired on identical draws.

Parallel to the face caption baseline: Qwen2.5-Omni describes each clip for the modality
(its best shot at a re-identification note), a sentence encoder embeds the note, and
recognition is cosine-NN over the notes. Scored paired against the modality's perceptual
encoder (ECAPA / wav2vec2-emotion / AST) on the same registrations and queries.

Expected: text fails on perceptual identity (speaker timbre, personal vocal-tone baseline)
but works on the NAMEABLE acoustic-scene category -- the router boundary, demonstrated.

Usage: python3 text_baseline_audio.py [POOL] [N] [draws]
"""
import sys, io, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, soundfile as sf, torchaudio

DEV = "cuda"
OMNI = "Qwen/Qwen2.5-Omni-7B"
SR = 16000


def load_wav(ad, secs):
    wav, sr = sf.read(io.BytesIO(ad["bytes"]), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav)[None], sr, SR).squeeze(0).numpy()
    n = SR * secs
    return wav[:n] if len(wav) >= n else np.pad(wav, (0, n - len(wav)))


# ---- perceptual encoders (one loaded at a time) -------------------------------------
def ecapa_encoder():
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         run_opts={"device": DEV})
    def f(wav):
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(wav).float()[None].to(DEV)).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)
    return f

def emotion_encoder():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    mid = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    proc = AutoFeatureExtractor.from_pretrained(mid)
    model = AutoModelForAudioClassification.from_pretrained(mid).to(DEV).eval()
    def f(wav):
        inp = proc(wav, sampling_rate=SR, return_tensors="pt").to(DEV)
        with torch.no_grad():
            e = model.wav2vec2(inp.input_values).last_hidden_state.mean(1)[0].cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)
    return f

def ast_encoder():
    from transformers import AutoFeatureExtractor, AutoModel
    mid = "MIT/ast-finetuned-audioset-10-10-0.4593"
    proc = AutoFeatureExtractor.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid).to(DEV).eval()
    def f(wav):
        inp = proc(wav, sampling_rate=SR, return_tensors="pt").to(DEV)
        with torch.no_grad():
            e = model(**inp).pooler_output[0].cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)
    return f


# ---- modality configs ---------------------------------------------------------------
def cfg_speaker():
    from datasets import load_dataset, Audio
    ds = load_dataset("Codec-SUPERB/Voxceleb1_test_original", split="test").cast_column("audio", Audio(decode=False))
    by = defaultdict(lambda: defaultdict(list))  # spk -> sess -> idxs
    for i, fn in enumerate(ds["id"]):
        p = fn.split("+"); by[p[0]][p[1]].append(i)
    ids = [s for s, sess in by.items() if len(sess) >= 2]   # cross-session
    def two_views(spk):
        sess = list(by[spk].keys())[:2]
        return [by[spk][sess[0]][0], by[spk][sess[1]][0]]
    prompt = ("Describe this person's voice for speaker identification: pitch, timbre, "
              "accent, and speaking style. Ignore the words spoken. 2-3 sentences.")
    return ds, ids, two_views, prompt, ecapa_encoder, 6, "speaker"

def cfg_tone():
    from datasets import load_dataset, Audio
    ds = load_dataset("xbgoose/ravdess", split="train").cast_column("audio", Audio(decode=False))
    by = defaultdict(list)
    for i, (a, e) in enumerate(zip(ds["actor"], ds["emotion"])):
        by[f"{a}_{e}"].append(i)
    ids = [k for k, v in by.items() if len(v) >= 2]
    def two_views(k): return [by[k][0], by[k][1]]
    prompt = ("Describe the emotional tone and vocal delivery of this speaker for later "
              "matching: mood, energy, and affect. 2-3 sentences.")
    return ds, ids, two_views, prompt, emotion_encoder, 3, "tone"

def cfg_scene():
    from datasets import load_dataset, Audio
    ds = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
    by = defaultdict(list)
    for i, c in enumerate(ds["category"]): by[c].append(i)
    ids = [k for k, v in by.items() if len(v) >= 2]
    def two_views(k): return [by[k][0], by[k][1]]
    prompt = "What is this sound? Describe the acoustic scene in one or two sentences."
    return ds, ids, two_views, prompt, ast_encoder, 5, "scene"


def main():
    POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    draws = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from sentence_transformers import SentenceTransformer
    oproc = Qwen2_5OmniProcessor.from_pretrained(OMNI)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        OMNI, dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True).eval()
    try: omni.disable_talker()
    except Exception: pass
    sent = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)

    def describe(wav, prompt):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": wav},
                 {"type": "text", "text": prompt}]}]
        text = oproc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = oproc(text=text, audio=[wav], return_tensors="pt", sampling_rate=SR).to(DEV)
        with torch.no_grad():
            out = omni.generate(**inp, max_new_tokens=80, do_sample=False, return_audio=False)
        return oproc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    results = {}
    for cfg in [cfg_speaker, cfg_tone, cfg_scene]:
        ds, ids, two_views, prompt, enc_fac, secs, name = cfg()
        rng = np.random.default_rng(0); rng.shuffle(ids); ids = ids[:POOL]
        print(f"\n[{name}] pool={len(ids)} (>=2 views), encoding+describing ...", flush=True)
        enc = enc_fac()
        caps, encs = {}, {}
        for k, p in enumerate(ids):
            v0, v1 = two_views(p)
            w0, w1 = load_wav(ds[v0]["audio"], secs), load_wav(ds[v1]["audio"], secs)
            caps[p] = [describe(w0, prompt), describe(w1, prompt)]
            encs[p] = [enc(w0), enc(w1)]
            if k % 10 == 0: print(f"  {name} {k}/{len(ids)}", flush=True)
        del enc; torch.cuda.empty_cache()
        txt = {p: sent.encode(caps[p], normalize_embeddings=True) for p in ids}

        def draw(seed):
            r = np.random.default_rng(seed); pick = list(ids); r.shuffle(pick); pick = pick[:N]
            Rt = np.stack([txt[p][0] for p in pick]); Qt = np.stack([txt[p][1] for p in pick])
            Ra = np.stack([encs[p][0] for p in pick]); Qa = np.stack([encs[p][1] for p in pick])
            gt = np.arange(len(pick))
            return float(((Qt @ Rt.T).argmax(1) == gt).mean()), float(((Qa @ Ra.T).argmax(1) == gt).mean())

        Nn = min(N, len(ids))
        res = [draw(s) for s in range(3000, 3000 + draws)]
        tt = np.array([x[0] for x in res]); aa = np.array([x[1] for x in res])
        ci = lambda x: 1.96 * float(np.std(x, ddof=1)) / np.sqrt(len(x))
        results[name] = {"N": Nn, "pool": len(ids), "draws": draws,
                         "text": {"recall": float(tt.mean()), "ci95": ci(tt)},
                         "encoder": {"recall": float(aa.mean()), "ci95": ci(aa)},
                         "chance": 1.0 / Nn, "example_caption": caps[ids[0]][0]}
        print(f"  [{name}] text {tt.mean():.3f}+/-{ci(tt):.3f}  encoder {aa.mean():.3f}+/-{ci(aa):.3f}  chance {1.0/Nn:.3f}")
        print(f"  [{name}] example: {caps[ids[0]][0][:150]}")

    Path("/home/ubuntu/multimodal-user-memory/results/text_baseline_audio.json").write_text(json.dumps(results, indent=2))
    print("\nwrote results/text_baseline_audio.json")


if __name__ == "__main__":
    main()
