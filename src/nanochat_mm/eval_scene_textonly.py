"""Scene-level text-only baseline, matched to the exact cluttered scenes used by the
grounded-memory ablation. Constructs the SAME composites (same seed/M/K) and, instead of
grounding+encoding, has the VLM describe the referenced referent *from the whole scene*
for re-identification; a sentence encoder matches the notes (reg vs cross-condition qry).

This is the honest text-only-on-scenes bar for Figure 5: the captioner must both find the
referent in context AND describe it discriminatively, with text as the only channel.

Usage: ATTMEM_VLM=Qwen/Qwen2.5-VL-7B-Instruct python3 eval_scene_textonly.py [domain] [M] [K] [seed]
  domain = faces | paintings
"""
import sys, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch

DEV = "cuda"; VLM = os.environ.get("ATTMEM_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
ORD = ["first", "second", "third", "fourth", "fifth"]

# face scene geometry (matches eval_agentic_production.py)
F_FACE, F_CELL = 160, 240
# painting scene geometry (matches eval_agentic_paintings.py)
P_ART, P_CELL = 200, 240

FACE_PROMPT = ("Look at the {ord} person from the left. Describe that person's face for "
               "later identification: age, gender, skin tone, hair, face shape, and any "
               "distinctive features. Ignore the other people. 2-3 sentences.")
PAINT_PROMPT = ("Look at the painting in the {ord} position from the left. Describe that "
                "painting for later identification of its artist: style, period, palette, "
                "brushwork, and subject. Ignore the other paintings. 2-3 sentences.")


def scene(imgs, cell, inner):
    K = len(imgs); canvas = Image.new("RGB", (cell * K, cell), (128, 128, 128))
    for i, im in enumerate(imgs):
        f = im.convert("RGB").resize((inner, inner))
        canvas.paste(f, (i * cell + (cell - inner) // 2, (cell - inner) // 2))
    return canvas


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "faces"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM, torch_dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True).eval()
    sent = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)

    if domain == "faces":
        ds = load_dataset("ljnlonoljpiljm/agedb", split="train"); idkey = "identity"
        cell, inner, prompt = F_CELL, F_FACE, FACE_PROMPT
    else:
        ds = load_dataset("huggan/wikiart", split="train"); idkey = "artist"
        cell, inner, prompt = P_CELL, P_ART, PAINT_PROMPT
    by = defaultdict(list)
    for i, ident in enumerate(ds[idkey]):
        by[str(ident)].append(i)
    ids = [p for p in by if len(by[p]) >= 2]
    rng = np.random.default_rng(SEED); rng.shuffle(ids); ids = ids[:M]
    img = ds["image"]

    def describe(comp, pos):
        q = prompt.format(ord=ORD[pos])
        msgs = [{"role": "user", "content": [{"type": "image", "image": comp},
                 {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[comp], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=96, do_sample=False)
        return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    notes = {"reg": [], "qry": []}; lab = []
    for k, p in enumerate(ids):
        ix = list(by[p]); rng.shuffle(ix)
        for role, ti in [("reg", ix[0]), ("qry", ix[1])]:
            pos = int(rng.integers(0, K))
            others = [rng.choice(by[rng.choice([q for q in ids if q != p])]) for _ in range(K - 1)]
            items = []; oi = 0
            for j in range(K):
                if j == pos: items.append(img[ti])
                else: items.append(img[others[oi]]); oi += 1
            comp = scene(items, cell, inner)
            notes[role].append(describe(comp, pos))
            if role == "reg": lab.append(k)
        if k % 10 == 0: print(f"  {domain} {k}/{M}", flush=True)

    R = sent.encode(notes["reg"], normalize_embeddings=True)
    Q = sent.encode(notes["qry"], normalize_embeddings=True)
    pred = (Q @ R.T).argmax(1)
    recall = float(np.mean([lab[pred[i]] == lab[i] for i in range(len(Q))]))

    print(f"\n=== Scene-level text-only ({domain}, {VLM.split('/')[-1]}, M={M}, K={K}, seed={SEED}) ===")
    print(f"  text-only (scene caption) recall@1 : {recall:.3f}   chance ~{1/M:.3f}")
    print(f"  example: {notes['reg'][0][:140]}")
    res = {"domain": domain, "VLM": VLM, "M": M, "K": K, "seed": SEED,
           "scene_text_recall": recall, "chance": 1.0 / M, "example": notes["reg"][0]}
    tag = f"{domain}_{VLM.split('/')[-1]}_K{K}_s{SEED}"
    Path(f"/home/ubuntu/multimodal-user-memory/results/scene_textonly_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote results/scene_textonly_{tag}.json")


if __name__ == "__main__":
    main()
