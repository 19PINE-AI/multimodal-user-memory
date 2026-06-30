"""Production agentic perceptual-memory eval with the realistic pipeline:
   VLM localize -> RetinaFace detect+align inside the region -> ArcFace -> match.

Scenes place each face inside a padded cell (background margin) so a real detector
can find it (un-aligned, in-context). Methods:
  oracle_align : correct cell -> RetinaFace align -> ArcFace        (ceiling)
  agentic_align: VLM grounds referent -> its cell -> RetinaFace align -> ArcFace
  agentic_crop : VLM box -> resize -> ArcFace (no re-align)         (box-precision-limited)
  whole        : ArcFace on the whole scene                         (context-blind floor)
Reports recall@1, grounding accuracy, and detector hit-rate.

Usage: ATTMEM_VLM=Qwen/Qwen2.5-VL-32B-Instruct python3 eval_agentic_production.py [M] [K]
"""
import sys, re, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch

sys.path.insert(0, str(Path(__file__).parent))
from face_encoder import ArcFaceEncoderBGR, FaceDetector

DEV = "cuda"; VLM = os.environ.get("ATTMEM_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
FACE = 160; CELL = 240; ORD = ["first", "second", "third", "fourth"]


def scene(faces):
    """Each face centered in a CELL with gray margin; K cells in a row."""
    K = len(faces); canvas = Image.new("RGB", (CELL * K, CELL), (128, 128, 128))
    for i, im in enumerate(faces):
        f = im.convert("RGB").resize((FACE, FACE))
        canvas.paste(f, (i * CELL + (CELL - FACE) // 2, (CELL - FACE) // 2))
    return canvas


def parse_box(t):
    m = re.findall(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', t)
    return [int(v) for v in m[0]] if m else None


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
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
    img = ds["image"]

    def encode_aligned(pil_cell):
        a = det.detect_align(pil_cell)
        if a is None:                                    # fallback: tight resize
            return enc.encode_pil(pil_cell)
        return enc.encode_bgr112(a)

    def vlm_box(comp, pos):
        q = (f"Locate the face of the {ORD[pos]} person from the left. "
             f"Output only its bounding box as [x1,y1,x2,y2].")
        msgs = [{"role": "user", "content": [{"type": "image", "image": comp},
                 {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[comp], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
        return parse_box(proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0])

    methods = ["oracle_align", "agentic_align", "agentic_crop", "whole"]
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
            comp = scene(faces)
            cell = lambda c: comp.crop((c * CELL, 0, (c + 1) * CELL, CELL))
            rec["oracle_align"][role].append(encode_aligned(cell(pos)))
            rec["whole"][role].append(enc.encode_pil(comp))
            box = vlm_box(comp, pos)
            if box is not None:
                cx = (box[0] + box[2]) / 2; col = int(min(K - 1, max(0, cx // CELL)))
                ground_tot += 1; ground_ok += int(col == pos)
                x1, y1, x2, y2 = box; W, H = comp.size
                crop = comp.crop((max(0, x1), max(0, y1), min(W, max(x1 + 1, x2)), min(H, max(y1 + 1, y2))))
            else:
                col = 0; crop = comp
            rec["agentic_crop"][role].append(enc.encode_pil(crop))
            a = det.detect_align(cell(col)); det_tot += 1; det_hit += int(a is not None)
            rec["agentic_align"][role].append(enc.encode_bgr112(a) if a is not None else enc.encode_pil(cell(col)))
            if role == "reg": lab.append(k)
        if k % 10 == 0: print(f"  {k}/{M}", flush=True)

    def recall(reg, qry):
        R = np.stack(reg); R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
        Q = np.stack(qry); Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        pred = (Q @ R.T).argmax(1)
        return float(np.mean([lab[pred[i]] == lab[i] for i in range(len(qry))]))

    print(f"\n=== Production agentic perceptual memory ({VLM.split('/')[-1]}, M={M}, K={K}) ===")
    res = {}
    for m in methods:
        res[m] = recall(rec[m]["reg"], rec[m]["qry"]); print(f"  {m:14} recall@1 : {res[m]:.3f}")
    res["grounding_acc"] = ground_ok / max(1, ground_tot)
    res["detector_hit_rate"] = det_hit / max(1, det_tot)
    print(f"  grounding accuracy : {res['grounding_acc']:.3f}   detector hit-rate : {res['detector_hit_rate']:.3f}")
    print(f"  [ref] in-model context-query 0.251   chance ~{1/M:.3f}")
    res.update({"VLM": VLM, "M": M, "K": K, "seed": SEED})
    tag = f"{VLM.split('/')[-1]}_K{K}_s{SEED}"
    Path(f"/home/ubuntu/multimodal-user-memory/results/agentic_prod_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote results/agentic_prod_{tag}.json")


if __name__ == "__main__":
    main()
