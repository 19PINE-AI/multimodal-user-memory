"""Comprehensive evaluation of the agentic perceptual-memory pipeline:
   VLM localizes the referent  ->  ArcFace encodes the crop  ->  AttMem/KV match.

Referential recognition on multi-face scenes. For each identity we build a register
scene and a query scene, each a K-face composite with the target at a referred
position ("the i-th person from the left") and random distractors. We register one
view per identity and recognise the other, recall@1 over the M registered keys
(cosine == the AttMem attention read; the KV-cache realisation changes cost, not this
number).

Methods:
  oracle    : crop the TRUE target box -> ArcFace            (ceiling)
  agentic   : VLM grounds the referent -> crop its box -> ArcFace
  whole     : ArcFace on the WHOLE composite (context-blind) (floor)
Reported: end-to-end recall@1, grounding accuracy (crop lands on target column),
and a prefill-cost comparison (KV token vs vector-DB round-trip).

Usage: python3 eval_agentic_perceptual.py [M] [K]
"""
import sys, re, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))
from face_encoder import ArcFaceEncoder

DEV = "cuda"; import os as _os; VLM = _os.environ.get("ATTMEM_VLM","Qwen/Qwen2.5-VL-3B-Instruct"); S = 224
ORD = ["first", "second", "third", "fourth"]


def composite(imgs):
    K = len(imgs); canvas = Image.new("RGB", (S * K, S))
    for i, im in enumerate(imgs):
        canvas.paste(im.convert("RGB").resize((S, S)), (i * S, 0))
    return canvas


def parse_box(txt):
    m = re.findall(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', txt)
    if not m: return None
    return [int(v) for v in m[0]]


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from datasets import load_dataset
    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM, torch_dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True).eval()
    enc = ArcFaceEncoder()

    ds = load_dataset("ljnlonoljpiljm/agedb", split="train")
    by = defaultdict(list)
    for i, ident in enumerate(ds["identity"]):
        by[str(ident)].append(i)
    ids = [p for p in by if len(by[p]) >= 2]
    rng = np.random.default_rng(0); rng.shuffle(ids); ids = ids[:M]
    img = ds["image"]

    def vlm_box(comp, pos):
        # processed image is canvas size (multiples of 28) -> coords in canvas px
        q = (f"Locate the face of the {ORD[pos]} person from the left of the image. "
             f"Output only its bounding box as [x1,y1,x2,y2].")
        msgs = [{"role": "user", "content": [{"type": "image", "image": comp},
                 {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[comp], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
        gen = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return parse_box(gen)

    def crop_box(comp, box):
        if box is None: return comp
        x1, y1, x2, y2 = box; W, H = comp.size
        x1 = max(0, min(W - 1, x1)); x2 = max(x1 + 1, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(y1 + 1, min(H, y2))
        return comp.crop((x1, y1, x2, y2))

    # build register + query scenes per identity, run all methods
    rec = {m: {"reg": [], "qry": [], "lab": []} for m in ["oracle", "agentic", "agentic_snap", "whole"]}
    ground_ok = 0; ground_tot = 0
    for k, p in enumerate(ids):
        ix = list(by[p]); rng.shuffle(ix)
        for role, ti in [("reg", ix[0]), ("qry", ix[1])]:
            pos = int(rng.integers(0, K))                       # target position in the scene
            others = [rng.choice(by[rng.choice([q for q in ids if q != p])]) for _ in range(K - 1)]
            faces = []; oi = 0
            for j in range(K):
                if j == pos: faces.append(img[ti])
                else: faces.append(img[others[oi]]); oi += 1
            comp = composite(faces)
            # oracle: the true target column
            oracle_crop = comp.crop((pos * S, 0, (pos + 1) * S, S))
            rec["oracle"][role].append(enc.encode_pil(oracle_crop))
            # whole composite
            rec["whole"][role].append(enc.encode_pil(comp))
            # agentic: VLM grounds the position
            box = vlm_box(comp, pos)
            rec["agentic"][role].append(enc.encode_pil(crop_box(comp, box)))
            # agentic_snap: snap the VLM box-center to a column, crop the full (aligned)
            # column -> isolates SELECTION ("which person") from box-precision
            if box is not None:
                cx = (box[0] + box[2]) / 2; col = int(min(K - 1, max(0, cx // S)))
                ground_tot += 1; ground_ok += int(col == pos)
            else:
                col = 0
            rec["agentic_snap"][role].append(enc.encode_pil(comp.crop((col * S, 0, (col + 1) * S, S))))
            if role == "reg": rec["oracle"]["lab"].append(k)
        if k % 10 == 0: print(f"  {k}/{M}")

    def recall(reg, qry, lab):
        R = np.stack(reg); R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
        Q = np.stack(qry); Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        pred = (Q @ R.T).argmax(1)
        return float(np.mean([lab[pred[i]] == lab[i] for i in range(len(qry))]))

    lab = rec["oracle"]["lab"]
    print(f"\n=== Agentic perceptual memory: referential recognition (M={M}, K={K}) ===")
    res = {}
    for m in ["oracle", "agentic_snap", "agentic", "whole"]:
        r = recall(rec[m]["reg"], rec[m]["qry"], lab); res[m] = r
        print(f"  {m:8} recall@1 : {r:.3f}")
    g = ground_ok / max(1, ground_tot)
    print(f"  grounding accuracy (VLM crop in target column): {g:.3f}")
    print(f"  [reference] in-model context-query (falsifier) : 0.251   chance ~{1/M:.3f}")
    print(f"\n  cost: AttMem KV = 1 token/identity, residual recall (no round-trip).")
    print(f"        vector-DB    = round-trip + re-prefill (name ~1 tok; re-examined image ~256-1280 tok).")
    res.update({"grounding_acc": g, "M": M, "K": K, "inmodel_query": 0.251})
    Path("/home/ubuntu/multimodal-user-memory/results/agentic_perceptual.json").write_text(json.dumps(res, indent=2))
    print("wrote results/agentic_perceptual.json")


if __name__ == "__main__":
    main()
