"""Agentic perceptual memory on a 2nd vision domain: paintings (style/artist).

Mirrors the face pipeline but with a non-face encoder and no landmark alignment, to show
the VLM localization step generalizes beyond faces. A collage places K paintings (by
different artists) each in a padded cell; the task is to recognize the painting in a
referenced position as belonging to a registered artist (a DIFFERENT painting by that
artist was registered). Methods:
  oracle_crop : correct cell -> CLIP                          (localization ceiling)
  agentic_crop: VLM grounds the referent -> its box -> CLIP   (agentic)
  whole       : CLIP on the whole collage                     (context-blind floor)
Recognition ceiling is CLIP's style/artist ability (modest); the point is that
agentic_crop ~ oracle_crop >> whole and grounding accuracy is high, i.e. localization
transfers to a new visual domain.

Usage: ATTMEM_VLM=Qwen/Qwen2.5-VL-7B-Instruct python3 eval_agentic_paintings.py [M] [K] [seed]
"""
import sys, re, os, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch

DEV = "cuda"; VLM = os.environ.get("ATTMEM_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPID = "openai/clip-vit-large-patch14"
CELL = 240; ART = 200; ORD = ["first", "second", "third", "fourth"]


def scene(arts):
    K = len(arts); canvas = Image.new("RGB", (CELL * K, CELL), (128, 128, 128))
    for i, im in enumerate(arts):
        f = im.convert("RGB").resize((ART, ART))
        canvas.paste(f, (i * CELL + (CELL - ART) // 2, (CELL - ART) // 2))
    return canvas


def parse_box(t):
    m = re.findall(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', t)
    return [int(v) for v in m[0]] if m else None


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              CLIPModel, CLIPProcessor)
    from datasets import load_dataset
    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM, torch_dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True).eval()
    clip = CLIPModel.from_pretrained(CLIPID).to(DEV).eval()
    cproc = CLIPProcessor.from_pretrained(CLIPID)

    def clip_emb(pil):
        inp = cproc(images=pil.convert("RGB"), return_tensors="pt").to(DEV)
        with torch.no_grad():
            e = clip.get_image_features(**inp)[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    ds = load_dataset("huggan/wikiart", split="train")
    by = defaultdict(list)
    for i, a in enumerate(ds["artist"]):
        by[str(a)].append(i)
    ids = [p for p in by if len(by[p]) >= 2]
    rng = np.random.default_rng(SEED); rng.shuffle(ids); ids = ids[:M]
    img = ds["image"]

    def vlm_box(comp, pos):
        q = (f"Locate the painting in the {ORD[pos]} position from the left. "
             f"Output only its bounding box as [x1,y1,x2,y2].")
        msgs = [{"role": "user", "content": [{"type": "image", "image": comp},
                 {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[comp], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
        return parse_box(proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0])

    methods = ["oracle_crop", "agentic_crop", "whole"]
    rec = {m: {"reg": [], "qry": []} for m in methods}; lab = []
    ground_ok = ground_tot = 0
    for k, p in enumerate(ids):
        ix = list(by[p]); rng.shuffle(ix)
        for role, ti in [("reg", ix[0]), ("qry", ix[1])]:
            pos = int(rng.integers(0, K))
            others = [rng.choice(by[rng.choice([q for q in ids if q != p])]) for _ in range(K - 1)]
            arts = []; oi = 0
            for j in range(K):
                if j == pos: arts.append(img[ti])
                else: arts.append(img[others[oi]]); oi += 1
            comp = scene(arts)
            cell = lambda c: comp.crop((c * CELL, 0, (c + 1) * CELL, CELL))
            rec["oracle_crop"][role].append(clip_emb(cell(pos)))
            rec["whole"][role].append(clip_emb(comp))
            box = vlm_box(comp, pos)
            if box is not None:
                cx = (box[0] + box[2]) / 2; col = int(min(K - 1, max(0, cx // CELL)))
                ground_tot += 1; ground_ok += int(col == pos)
                x1, y1, x2, y2 = box; W, H = comp.size
                crop = comp.crop((max(0, x1), max(0, y1), min(W, max(x1 + 1, x2)), min(H, max(y1 + 1, y2))))
            else:
                crop = comp
            rec["agentic_crop"][role].append(clip_emb(crop))
            if role == "reg": lab.append(k)
        if k % 10 == 0: print(f"  {k}/{M}", flush=True)

    def recall(reg, qry):
        R = np.stack(reg); Q = np.stack(qry)
        pred = (Q @ R.T).argmax(1)
        return float(np.mean([lab[pred[i]] == lab[i] for i in range(len(qry))]))

    print(f"\n=== Agentic paintings ({VLM.split('/')[-1]}, M={M}, K={K}, seed={SEED}) ===")
    res = {}
    for m in methods:
        res[m] = recall(rec[m]["reg"], rec[m]["qry"]); print(f"  {m:14} recall@1 : {res[m]:.3f}")
    res["grounding_acc"] = ground_ok / max(1, ground_tot)
    print(f"  grounding accuracy : {res['grounding_acc']:.3f}   chance ~{1/M:.3f}")
    res.update({"VLM": VLM, "M": M, "K": K, "seed": SEED, "encoder": "CLIP-ViT-L/14"})
    tag = f"{VLM.split('/')[-1]}_K{K}_s{SEED}"
    (REPO_ROOT / "results" / f"agentic_paint_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote results/agentic_paint_{tag}.json")


if __name__ == "__main__":
    main()
