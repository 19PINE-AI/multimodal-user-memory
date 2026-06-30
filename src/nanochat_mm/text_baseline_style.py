"""Text-only baseline vs CLIP for painting style (artist identity), paired on identical
WikiArt draws. A VLM describes each painting's artist/style as a re-identification note;
a sentence encoder embeds it; recognition is cosine-NN over notes. Scored against CLIP.

Usage: ATTMEM_VLM=Qwen/Qwen2.5-VL-7B-Instruct python3 text_baseline_style.py [POOL] [N] [draws]
"""
import sys, os, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

DEV = "cuda"; VLM = os.environ.get("ATTMEM_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
CLIPID = "openai/clip-vit-large-patch14"
PROMPT = ("Describe this painting for later identification of its artist: style, period, "
          "palette, brushwork, and subject. 2-3 sentences.")


def main():
    POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    draws = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              CLIPModel, CLIPProcessor)
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM, torch_dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True).eval()
    clip = CLIPModel.from_pretrained(CLIPID).to(DEV).eval()
    cproc = CLIPProcessor.from_pretrained(CLIPID)
    sent = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)

    def clip_emb(pil):
        inp = cproc(images=pil.convert("RGB"), return_tensors="pt").to(DEV)
        with torch.no_grad():
            e = clip.get_image_features(**inp)[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def caption(pil):
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil.convert("RGB")},
                 {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[pil.convert("RGB")], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=96, do_sample=False)
        return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    ds = load_dataset("huggan/wikiart", split="train")
    by = defaultdict(list)
    for i, a in enumerate(ds["artist"]): by[str(a)].append(i)
    ids = [p for p in by if len(by[p]) >= 2]
    rng = np.random.default_rng(0); rng.shuffle(ids); ids = ids[:POOL]
    img = ds["image"]

    caps, encs = {}, {}
    for k, p in enumerate(ids):
        ix = list(by[p]); r = np.random.default_rng(100 + k); r.shuffle(ix); v0, v1 = ix[0], ix[1]
        caps[p] = [caption(img[v0]), caption(img[v1])]
        encs[p] = [clip_emb(img[v0]), clip_emb(img[v1])]
        if k % 10 == 0: print(f"  style {k}/{len(ids)}", flush=True)
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
    out = {"modality": "style", "N": Nn, "pool": len(ids), "draws": draws,
           "text": {"recall": float(tt.mean()), "ci95": ci(tt)},
           "encoder": {"recall": float(aa.mean()), "ci95": ci(aa)},
           "chance": 1.0 / Nn, "example_caption": caps[ids[0]][0]}
    print(f"\n=== Text-only vs CLIP, WikiArt style/artist (N={Nn}, {draws} draws) ===")
    print(f"  text {tt.mean():.3f}+/-{ci(tt):.3f}  CLIP {aa.mean():.3f}+/-{ci(aa):.3f}  chance {1.0/Nn:.3f}")
    print(f"  example: {caps[ids[0]][0][:150]}")
    Path("/home/ubuntu/multimodal-user-memory/results/text_baseline_style.json").write_text(json.dumps(out, indent=2))
    print("wrote results/text_baseline_style.json")


if __name__ == "__main__":
    main()
