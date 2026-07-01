"""Real-conversation speaker grounding on VoxConverse (in-the-wild YouTube conversations
with ground-truth diarization). Within a conversation, we enroll each speaker from their
earlier turns (ECAPA) and, from a held-out window containing several speakers, recognise a
target speaker three ways:
  oracle : ECAPA on the target's ground-truth turns in the window   (ceiling)
  agentic: an audio-LLM grounds the referenced speaker's span -> ECAPA
  whole  : ECAPA on the whole multi-speaker window                  (store-only floor)
recall@1 is over the enrolled speakers of that conversation. This is the pure-audio
grounding result on real conversational data (natural turn-taking, overlap, noise),
loaded via partial streaming with the torchcodec bypass.

Usage: python3 eval_conversation_grounding.py [n_convos] [seed] [dataset]
  dataset = voxconverse (default) | ami
"""
import sys, re, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from audio_loader import stream_dataset, decode_bytes

DEV = "cuda"; OMNI = "Qwen/Qwen2.5-Omni-7B"; SR = 16000
ORD = ["first", "second", "third", "fourth", "fifth"]
MINSEG = 1.2  # seconds; ignore very short turns


def speaker_turns(row):
    """Group ground-truth turns by speaker: {spk: [(start,end), ...]}."""
    ts, te, sp = row["timestamps_start"], row["timestamps_end"], row["speakers"]
    by = defaultdict(list)
    for a, b, s in zip(ts, te, sp):
        if b - a >= MINSEG: by[s].append((float(a), float(b)))
    return by


DATASETS = {"voxconverse": ("diarizers-community/voxconverse", None),
            "ami": ("diarizers-community/ami", "ihm")}


def main():
    n_convos = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dskey = sys.argv[3] if len(sys.argv) > 3 else "voxconverse"
    ds_name, ds_cfg = DATASETS[dskey]
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           run_opts={"device": DEV})
    def emb(wav):
        if len(wav) < SR // 2: wav = np.pad(wav, (0, SR // 2 - len(wav)))
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
        q = (f"This {secs:.0f}-second clip has {nspk} people speaking in conversation. "
             f"When does the {ORD[pos]} distinct speaker (by order of first speaking) talk? "
             f"Answer only their start and end time in seconds as [start, end].")
        conv = [{"role": "user", "content": [{"type": "audio", "audio": clip},
                 {"type": "text", "text": q}]}]
        text = oproc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = oproc(text=text, audio=[clip], return_tensors="pt", sampling_rate=SR).to(DEV)
        with torch.no_grad():
            out = omni.generate(**inp, max_new_tokens=32, do_sample=False, return_audio=False)
        dec = oproc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        n = re.findall(r'(\d+\.?\d*)', dec)
        return (float(n[0]), float(n[1])) if len(n) >= 2 and float(n[1]) > float(n[0]) else None

    ds = stream_dataset(ds_name, split="test", config=ds_cfg)
    rng = np.random.default_rng(SEED)
    hit = {"oracle": 0, "agentic": 0, "whole": 0}; tot = 0; ground_ok = 0
    seg = lambda w, a, b: w[int(a * SR):int(b * SR)]
    done = 0
    for row in ds:
        if done >= n_convos: break
        by = speaker_turns(row)
        spk = [s for s in by if len(by[s]) >= 3]
        if len(spk) < 2: continue
        wav = decode_bytes(row["audio"]["bytes"])
        rng.shuffle(spk); spk = spk[:4]
        # enroll each speaker from their first two turns
        enroll = {s: emb(np.concatenate([seg(wav, a, b) for a, b in by[s][:2]])) for s in spk}
        R = np.stack([enroll[s] for s in spk])
        target = spk[0]
        # build a window: the target's 3rd turn + one turn from each other speaker, in time order
        picks = [(by[target][2], target)] + [(by[s][rng.integers(len(by[s]))], s) for s in spk[1:]]
        picks.sort(key=lambda x: x[0][0])
        window = np.concatenate([seg(wav, a, b) for (a, b), _ in picks])
        # local boundaries within the window
        bnds, acc = [], 0
        for (a, b), s in picks:
            L = len(seg(wav, a, b)); bnds.append((acc, acc + L, s)); acc += L
        pos = [i for i, (_, _, s) in enumerate(bnds) if s == target][0]
        t0, t1, _ = bnds[pos]
        qw = {"oracle": window[t0:t1], "whole": window}
        span = ground_span(window, pos, len(picks))
        if span is not None:
            gs, ge = max(0, int(span[0] * SR)), min(len(window), int(span[1] * SR))
            if ge <= gs: ge = min(len(window), gs + SR)
            ground_ok += int(t0 <= (gs + ge) / 2 <= t1); qw["agentic"] = window[gs:ge]
        else:
            qw["agentic"] = window
        for m in qw:
            q = emb(qw[m]); hit[m] += int(spk[int((q @ R.T).argmax())] == target)
        tot += 1; done += 1
        if done % 5 == 0: print(f"  {done}/{n_convos}", flush=True)

    print(f"\n=== Real-conversation grounding ({dskey}, {tot} meetings, seed={SEED}) ===")
    res = {m: hit[m] / max(1, tot) for m in hit}
    for m in ["oracle", "agentic", "whole"]:
        print(f"  {m:8} recall@1 : {res[m]:.3f}")
    res["grounding_acc"] = ground_ok / max(1, tot); res["n_convos"] = tot; res["dataset"] = dskey
    print(f"  grounding accuracy : {res['grounding_acc']:.3f}")
    tag = "" if dskey == "voxconverse" else f"{dskey}_"
    Path(f"/home/ubuntu/multimodal-user-memory/results/conversation_grounding_{tag}s{SEED}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote results/conversation_grounding_{tag}s{SEED}.json")


if __name__ == "__main__":
    main()
