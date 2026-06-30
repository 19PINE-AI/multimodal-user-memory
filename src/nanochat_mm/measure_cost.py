"""Exp A: per-perception token/compute cost. A VLM that keeps a perception in
context pays its full vision-token bill (hundreds to >1k tokens per image); AttMem
stores the perception as one bank row and reads it as ONE marker token. We measure
the real vision-token count for face crops in Qwen2.5-VL and compute the context and
prefill-FLOP scaling for a K-perception conversation.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from transformers import AutoProcessor

VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
CTX = 128_000  # context window


def vis_tokens(proc, size):
    img = Image.fromarray((np.random.default_rng(0).integers(0, 255, (size, size, 3))).astype("uint8"))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "x"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    out = proc(text=[text], images=[img], return_tensors="pt")
    # count image-pad tokens
    img_tok = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n = int((out["input_ids"][0] == img_tok).sum())
    if "image_grid_thw" in out:
        thw = out["image_grid_thw"][0].tolist()
        return n, thw
    return n, None


def main():
    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    print(f"=== vision tokens per image, {VLM} ===")
    print(f"{'crop px':>8}  {'vision tokens':>14}  {'grid t,h,w':>14}")
    counts = {}
    for s in [112, 224, 336, 448, 672]:
        n, thw = vis_tokens(proc, s)
        counts[s] = n
        print(f"{s:>5}px  {n:>14}  {str(thw):>14}")
    # use 224px (a typical face crop) as the headline
    nvis = counts[224]
    print(f"\n=== per-perception prefill cost (face crop ~224px = {nvis} vision tokens) ===")
    print(f"  image-in-context (VLM) : {nvis} tokens / perception")
    print(f"  AttMem (marker token)  : 1 token / perception   -> {nvis}x fewer prefill tokens")
    print(f"  attention is O(tokens^2), so prefill compute ratio ~ {nvis}x (per-token) and grows with context")
    print(f"\n=== K-perception conversation: tokens in context ===")
    print(f"  {'K':>5}  {'image-in-context':>18}  {'AttMem':>8}")
    for K in [1, 5, 10, 50, 100, 1000]:
        print(f"  {K:>5}  {K*nvis:>18,}  {K:>8}")
    print(f"\n=== max perceptions that fit in a {CTX//1000}k context window ===")
    print(f"  image-in-context : {CTX // nvis} images   (then OOM / truncation)")
    print(f"  AttMem           : unbounded (bank is external to the context; only K query markers appear)")
    import json
    Path("/home/ubuntu/multimodal-user-memory/results/cost.json").write_text(json.dumps(
        {"model": VLM, "vis_tokens_by_size": counts, "headline_nvis_224": nvis,
         "ctx": CTX, "max_images_in_ctx": CTX // nvis}, indent=2))
    print("\nwrote results/cost.json")


if __name__ == "__main__":
    main()
