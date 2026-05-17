"""Propositional control: verify AttMem hook does not regress text-only recall.

Critical "no regression" win condition from research_plan.md section 5.3.
We test that:
  1. With AttMem hook installed but no perceptual positions in the batch,
     the LM's text-only outputs are byte-identical to vanilla Qwen.
  2. With the bank populated (vision + audio IDs registered) but the input
     is text-only, outputs are still byte-identical.

If both pass, the bolt-on is truly transparent to text.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE


PROMPTS = [
    "The capital of France is",
    "In a hole in the ground there lived",
    "Mathematics is the science of",
    "When my friend told me the news, I",
    "The square root of 144 is",
    "A list of programming languages: Python, Java,",
    "If a triangle has three equal sides, it is called",
    "I went to the bakery and bought some fresh",
]


def vanilla_logits(qwen, tok, prompts, max_new=8):
    """Run vanilla Qwen on each prompt, get top-K predicted tokens and logits."""
    out = []
    for p in prompts:
        ids = tok.encode(p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            r = qwen(input_ids=ids, use_cache=False)
            last = r.logits[0, -1, :]
            top = last.topk(20)
        out.append({
            "prompt": p,
            "top20_ids": top.indices.cpu().tolist(),
            "top20_logits": top.values.cpu().float().tolist(),
        })
    return out


def bolt_text_logits(bolt, tok, prompts):
    """Run bolt forward with all-text modality_ids (no perceptual positions)."""
    out = []
    for p in prompts:
        ids = tok.encode(p, return_tensors="pt").to(DEVICE)
        B, T = ids.shape
        modality_ids = torch.zeros(B, T, dtype=torch.long, device=DEVICE)  # all TEXT
        with torch.no_grad():
            logits = bolt(modality_ids, ids, {})  # no perceptual keys
            last = logits[0, -1, :]
            top = last.topk(20)
        out.append({
            "prompt": p,
            "top20_ids": top.indices.cpu().tolist(),
            "top20_logits": top.values.cpu().float().tolist(),
        })
    return out


def compare(a, b, label):
    """Compare top-K and logit values between two runs."""
    print(f"\n=== {label} ===")
    n_exact_top1 = 0
    n_exact_top5 = 0
    n_exact_top20 = 0
    max_abs_logit_diff = 0.0
    for x, y in zip(a, b):
        t1a, t1b = x["top20_ids"][0], y["top20_ids"][0]
        if t1a == t1b: n_exact_top1 += 1
        top5_match = x["top20_ids"][:5] == y["top20_ids"][:5]
        if top5_match: n_exact_top5 += 1
        top20_match = x["top20_ids"] == y["top20_ids"]
        if top20_match: n_exact_top20 += 1
        l_a = np.array(x["top20_logits"])
        l_b = np.array(y["top20_logits"])
        diff = float(np.abs(l_a - l_b).max())
        max_abs_logit_diff = max(max_abs_logit_diff, diff)
    n = len(a)
    print(f"  top-1 exact match: {n_exact_top1}/{n}")
    print(f"  top-5 exact match: {n_exact_top5}/{n}")
    print(f"  top-20 exact match: {n_exact_top20}/{n}")
    print(f"  max |logit diff| (top-20): {max_abs_logit_diff:.6f}")
    return {"top1": n_exact_top1, "top5": n_exact_top5, "top20": n_exact_top20,
            "max_logit_diff": max_abs_logit_diff, "n": n}


def main():
    print("Loading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    print("\n[1/3] Vanilla Qwen baseline ...")
    vanilla = vanilla_logits(qwen, tok, PROMPTS)

    print("\n[2/3] Bolt installed, empty bank, text-only inputs ...")
    bolt = QwenAttMemBolt(qwen, tok, vision_key_dim=512, audio_key_dim=192,
                          attach_layer=33, attach_lm_head=True).to(DEVICE)
    bolt.install_hook()
    bolt_empty = bolt_text_logits(bolt, tok, PROMPTS)
    r1 = compare(vanilla, bolt_empty, "Vanilla vs bolt-with-empty-bank (text-only)")

    print("\n[3/3] Bolt installed, populated bank, text-only inputs ...")
    # Populate both banks with 100 random IDs
    vis_keys = torch.randn(100, 512, device=DEVICE)
    aud_keys = torch.randn(100, 192, device=DEVICE)
    vis_markers = list(range(30001, 30101))
    aud_markers = list(range(30101, 30201))
    bolt.insert_batch(MODALITY_VISION, vis_keys, vis_markers)
    bolt.insert_batch(MODALITY_AUDIO, aud_keys, aud_markers)
    bolt_populated = bolt_text_logits(bolt, tok, PROMPTS)
    r2 = compare(vanilla, bolt_populated, "Vanilla vs bolt-with-populated-bank (text-only)")

    print("\n=== Summary ===")
    pass_1 = r1["top20"] == r1["n"] and r1["max_logit_diff"] < 1e-4
    pass_2 = r2["top20"] == r2["n"] and r2["max_logit_diff"] < 1e-4
    print(f"  Empty bank, text-only: {'PASS — byte identical to vanilla' if pass_1 else 'FAIL — diverges'}")
    print(f"  Populated bank, text-only: {'PASS — byte identical to vanilla' if pass_2 else 'FAIL — diverges'}")
    print(f"\nConclusion: AttMem bolt {'IS' if pass_1 and pass_2 else 'IS NOT'} transparent to text-only inputs.")


if __name__ == "__main__":
    main()
