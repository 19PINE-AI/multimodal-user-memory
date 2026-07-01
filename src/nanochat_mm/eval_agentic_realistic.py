"""Realistic-scene grounding: harder than the spaced gray composites. Faces are pasted
onto a real photographic background at JITTERED positions and VARYING scales (optionally
in a 2D layout), so the VLM must resolve a positional reference among faces that are not
neatly cell-aligned. This stresses the grounding step the ablation relies on.

Scene: canvas = a dimmed real image (a random unrelated dataset photo) at W x H; K faces
pasted at jittered centers with scale in [0.7, 1.05] x FACE. Reference is by left-to-right
rank of face centers (2D: row-major reading order). Methods:
  oracle_align : correct face region -> RetinaFace align -> ArcFace       (ceiling)
  agentic_align: VLM grounds the referent -> its region -> align -> ArcFace (ours)
  whole        : ArcFace on the whole cluttered canvas                     (store-only floor)
Reports recall@1, grounding accuracy (VLM box -> correct face by rank), detector hit-rate.

Usage: ATTMEM_VLM=Qwen/Qwen2.5-VL-7B-Instruct python3 eval_agentic_realistic.py [M] [K] [seed] [layout]
  layout = row | grid
"""
import sys, re, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch

sys.path.insert(0, str(Path(__file__).parent))
from face_encoder import ArcFaceEncoderBGR, FaceDetector

DEV = "cuda"; VLM = os.environ.get("ATTMEM_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
FACE = 150
ORD = ["first", "second", "third", "fourth", "fifth", "sixth"]


def parse_box(t):
    m = re.findall(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', t)
    return [int(v) for v in m[0]] if m else None


def build_scene(faces, bg, rng, layout):
    """Paste K faces on a dimmed real background at jittered pos/scale. Returns the
    composite and the list of (cx, cy, x1, y1, x2, y2) placements in paste order."""
    K = len(faces)
    if layout == "grid":
        cols = int(np.ceil(np.sqrt(K))); rows = int(np.ceil(K / cols))
    else:
        cols, rows = K, 1
    cw, ch = 240, 240
    W, H = cols * cw, rows * ch
    canvas = ImageEnhance.Brightness(bg.convert("RGB").resize((W, H))).enhance(0.55)
    places = []
    for i, im in enumerate(faces):
        r, c = divmod(i, cols)
        s = float(rng.uniform(0.70, 1.05)); fs = int(FACE * s)
        f = im.convert("RGB").resize((fs, fs))
        # jitter within the cell, keeping the face fully inside
        jx = int(rng.uniform(0, cw - fs)); jy = int(rng.uniform(0, ch - fs))
        x0, y0 = c * cw + jx, r * ch + jy
        canvas.paste(f, (x0, y0))
        places.append((x0 + fs / 2, y0 + fs / 2, x0, y0, x0 + fs, y0 + fs))
    return canvas, places


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    layout = sys.argv[4] if len(sys.argv) > 4 else "row"
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from datasets import load_dataset
    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM, torch_dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True).eval()
    enc = ArcFaceEncoderBGR(); det = FaceDetector()

    ds = load_dataset("ljnlonoljpiljm/agedb", split="train")
    by = defaultdict(list)
    for i, ident in enumerate(ds["identity"]):
        by[str(ident)].append(i)
    ids = [p for p in by if len(by[p]) >= 2]
    rng = np.random.default_rng(SEED); rng.shuffle(ids); ids = ids[:M]
    img = ds["image"]; nimg = len(img)

    def rank_of(places, pos):
        """reading-order rank (left->right, top->bottom) of the face pasted at index pos."""
        order = sorted(range(len(places)), key=lambda i: (round(places[i][1] / 120), places[i][0]))
        return order.index(pos)

    def region_crop(comp, place):
        _, _, x1, y1, x2, y2 = place
        return comp.crop((int(x1), int(y1), int(x2), int(y2)))

    def encode_aligned(pil):
        a = det.detect_align(pil)
        return enc.encode_bgr112(a) if a is not None else enc.encode_pil(pil)

    def vlm_box(comp, target_rank):
        q = (f"Locate the face of the {ORD[target_rank]} person in reading order "
             f"(left to right, top to bottom). Output only its bounding box as [x1,y1,x2,y2].")
        msgs = [{"role": "user", "content": [{"type": "image", "image": comp},
                 {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[comp], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
        return parse_box(proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0])

    methods = ["oracle_align", "agentic_align", "whole"]
    rec = {m: {"reg": [], "qry": []} for m in methods}; lab = []
    ground_ok = ground_tot = det_hit = det_tot = 0
    for k, p in enumerate(ids):
        ix = list(by[p]); rng.shuffle(ix)
        for role, ti in [("reg", ix[0]), ("qry", ix[1])]:
            pos = int(rng.integers(0, K))
            others = [rng.choice(by[rng.choice([q for q in ids if q != p])]) for _ in range(K - 1)]
            faces = []; oi = 0
            for j in range(K):
                if j == pos: faces.append(img[ti])
                else: faces.append(img[others[oi]]); oi += 1
            bg = img[int(rng.integers(0, nimg))]
            comp, places = build_scene(faces, bg, rng, layout)
            target_rank = rank_of(places, pos)
            rec["oracle_align"][role].append(encode_aligned(region_crop(comp, places[pos])))
            rec["whole"][role].append(enc.encode_pil(comp))
            box = vlm_box(comp, target_rank)
            if box is not None:
                bx, by_ = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                # nearest face center to the VLM box -> which pasted face
                dists = [ (bx - pl[0])**2 + (by_ - pl[1])**2 for pl in places ]
                picked = int(np.argmin(dists))
                ground_tot += 1; ground_ok += int(picked == pos)
                W, H = comp.size
                x1, y1, x2, y2 = box
                crop = comp.crop((max(0, x1), max(0, y1), min(W, max(x1 + 1, x2)), min(H, max(y1 + 1, y2))))
                a = det.detect_align(crop); det_tot += 1; det_hit += int(a is not None)
                rec["agentic_align"][role].append(enc.encode_bgr112(a) if a is not None else enc.encode_pil(crop))
            else:
                rec["agentic_align"][role].append(enc.encode_pil(comp))
            if role == "reg": lab.append(k)
        if k % 10 == 0: print(f"  {k}/{M}", flush=True)

    def recall(reg, qry):
        R = np.stack(reg); R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
        Q = np.stack(qry); Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        pred = (Q @ R.T).argmax(1)
        return float(np.mean([lab[pred[i]] == lab[i] for i in range(len(qry))]))

    print(f"\n=== Realistic-scene grounding ({VLM.split('/')[-1]}, M={M}, K={K}, {layout}, seed={SEED}) ===")
    res = {}
    for m in methods:
        res[m] = recall(rec[m]["reg"], rec[m]["qry"]); print(f"  {m:14} recall@1 : {res[m]:.3f}")
    res["grounding_acc"] = ground_ok / max(1, ground_tot)
    res["detector_hit_rate"] = det_hit / max(1, det_tot)
    print(f"  grounding accuracy : {res['grounding_acc']:.3f}   detector hit-rate : {res['detector_hit_rate']:.3f}")
    res.update({"VLM": VLM, "M": M, "K": K, "layout": layout, "seed": SEED})
    tag = f"{layout}_{VLM.split('/')[-1]}_K{K}_s{SEED}"
    Path(f"/home/ubuntu/multimodal-user-memory/results/agentic_realistic_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote results/agentic_realistic_{tag}.json")


if __name__ == "__main__":
    main()
