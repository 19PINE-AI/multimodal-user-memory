"""Propositional control suite — research_plan §4.2 (4).

Verifies that adding the bolt-on Engram does NOT regress text-only recall.

Construction:
  - Build a short set of (prefix, target) text-recall items where the
    target follows naturally from the prefix in the LM's distribution.
  - Score: top-1 next-token match against the gold target.
  - Compare: frozen Qwen alone vs Qwen + MultimodalEngramSet bolt-on
    (NO perceptual content; the bolt-on operates on text only).
  - The bolt-on should be effectively a no-op on pure text (no perc_emb
    activations; the Engram's text-side hash is initialised to zero
    residual after generic-NTP).

A regression here means the bolt-on architecture hurts the base model's
text capability — a load-bearing claim of the paper (Path A is
non-invasive to the frozen LM).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_TEXT
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity
from qwen_engram_bolt import QwenEngramBolt, MODEL_ID, DEVICE
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)


# Text-recall items: each is (prefix, expected_top1_next_token).
# Chosen to cover common factual / commonsense / linguistic completions
# that a 3 B base model should reliably get right.
ITEMS = [
    ("The capital of France is", " Paris"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("Two plus two equals", " four"),
    ("Shakespeare wrote a famous play about a Danish prince called", " Hamlet"),
    ("Water boils at one hundred degrees", " Celsius"),
    ("The chemical symbol for water is", " H"),
    ("The author of '1984' is George", " Orwell"),
    ("The currency of Japan is the", " yen"),
    ("The opposite of black is", " white"),
    ("The first president of the United States was George", " Washington"),
    ("The fastest land animal is the", " cheetah"),
    ("DNA is short for", " deoxy"),
    ("The longest river in the world is the", " Nile"),
    ("The Great Wall is located in", " China"),
    ("A triangle has", " three"),
    ("Photosynthesis uses light from the", " sun"),
    ("The Eiffel Tower is in", " Paris"),
    ("Mount Everest is the world's tallest", " mountain"),
    ("Albert Einstein developed the theory of", " relativity"),
    ("Pi is approximately three point", " one"),
    ("The Mona Lisa was painted by Leonardo da", " Vinci"),
    ("Plants take in carbon", " dioxide"),
    ("A square has four equal", " sides"),
    ("The Pacific is an", " ocean"),
    ("Cats are members of the family", " F"),  # Felidae
    ("Honey is made by", " bees"),
    ("The boiling point of water at sea level is one hundred", " degrees"),
    ("The Earth orbits around the", " sun"),
    ("Hydrogen is the lightest", " element"),
    ("The largest country by area is", " Russia"),
]


def score_top1(model, tok, items, hook_callable=None):
    """For each (prefix, expected_token), check if expected is the top-1
    next token. Returns the fraction correct.

    hook_callable: an optional context manager for installing/uninstalling
    a forward hook (Engram bolt-on) before/after the forward pass.
    """
    correct = 0
    detail = []
    for prefix, expected in items:
        ids = tok.encode(prefix, return_tensors="pt").to(DEVICE)
        # Capture the first token id of `expected` (with leading space).
        exp_ids = tok.encode(expected, add_special_tokens=False)
        if not exp_ids:
            continue
        target_id = exp_ids[0]
        with torch.no_grad():
            if hook_callable is not None:
                with hook_callable():
                    out = model(input_ids=ids)
            else:
                out = model(input_ids=ids)
            logits = out.logits[0, -1, :]
            top1 = int(logits.argmax().item())
        ok = (top1 == target_id)
        correct += int(ok)
        decoded = tok.decode([top1])
        detail.append({"prefix": prefix, "expected": expected,
                        "decoded_top1": decoded, "ok": ok})
    return correct / max(len(items), 1), detail


class BoltHookCtx:
    """Context manager: when entering, runs the bolt's text-only forward
    so the Engram hook fires. Implementation: we directly call the bolt's
    forward through a thin wrapper.
    """
    def __init__(self, bolt):
        self.bolt = bolt
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def score_bolt(bolt, tok, items):
    """Use the bolt as the forward — text-only items have modality_ids
    all set to MODALITY_TEXT, so the bolt's perc_emb is unused; the only
    extra effect is the Engram's text-side residual at the attached layer.
    """
    correct = 0; detail = []
    for prefix, expected in items:
        ids = tok.encode(prefix, return_tensors="pt").to(DEVICE)
        exp_ids = tok.encode(expected, add_special_tokens=False)
        if not exp_ids: continue
        target_id = exp_ids[0]
        modality_ids = torch.full_like(ids, MODALITY_TEXT)
        with torch.no_grad():
            logits = bolt(ids, modality_ids)
            last = logits[0, -1, :]
            top1 = int(last.argmax().item())
        ok = (top1 == target_id)
        correct += int(ok)
        decoded = tok.decode([top1])
        detail.append({"prefix": prefix, "expected": expected,
                        "decoded_top1": decoded, "ok": ok})
    return correct / max(len(items), 1), detail


def main():
    print("=" * 70)
    print("Propositional control suite — does the bolt-on Engram hurt text recall?")
    print("=" * 70)
    print(f"  items: {len(ITEMS)}  (short factual / commonsense completions)")

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    # Baseline: bare frozen Qwen
    print("\n[baseline] frozen Qwen, no bolt")
    base_acc, base_detail = score_top1(qwen, tok, ITEMS, hook_callable=None)
    print(f"  base top-1 = {base_acc:.3f}  ({sum(d['ok'] for d in base_detail)}/{len(base_detail)})")

    # Bolt with no pretraining (random init)
    print("\n[bolt-untrained] Qwen + bolt with random Engram + perc_emb")
    bolt_u = QwenEngramBolt(qwen, tok, V_vis=64, V_aud=64, engram_attach_layer=24).to(DEVICE)
    bolt_u.install_hook()
    bu_acc, bu_detail = score_bolt(bolt_u, tok, ITEMS)
    print(f"  untrained-bolt top-1 = {bu_acc:.3f}  ({sum(d['ok'] for d in bu_detail)}/{len(bu_detail)})")
    bolt_u.remove_hook()

    # Bolt after generic-NTP pretraining on audio embeddings (typical config).
    # The Engram has nonzero text-side params; verify it still doesn't hurt.
    print("\n[bolt-pretrained] Qwen + bolt after generic-NTP pretrain")
    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri_large.npz")
    tr_emb, tr_pid, _, _ = split_by_identity(aud['emb'], aud['pid'])
    apply_fn = fit_naive_rq(tr_emb, n_levels=1, k_per=32)
    bolt_p = QwenEngramBolt(qwen, tok, V_vis=32, V_aud=32, engram_attach_layer=24).to(DEVICE)
    bolt_p.install_hook()
    from engram_module_mm import MODALITY_AUDIO
    pretrain_generic(bolt_p, tr_emb, tr_pid, apply_fn, MODALITY_AUDIO, tok,
                     n_steps=400, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)
    bp_acc, bp_detail = score_bolt(bolt_p, tok, ITEMS)
    print(f"  pretrained-bolt top-1 = {bp_acc:.3f}  ({sum(d['ok'] for d in bp_detail)}/{len(bp_detail)})")
    bolt_p.remove_hook()

    summary = {
        "n_items": len(ITEMS),
        "baseline_qwen": base_acc,
        "bolt_untrained": bu_acc,
        "bolt_pretrained": bp_acc,
        "delta_untrained": bu_acc - base_acc,
        "delta_pretrained": bp_acc - base_acc,
    }
    out = Path("/home/ubuntu/multimodal-user-memory/results/propositional_control.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n[done] {out}")

    # Per-item disagreements
    diffs = []
    for b, u, p in zip(base_detail, bu_detail, bp_detail):
        if b["ok"] != u["ok"] or b["ok"] != p["ok"]:
            diffs.append({
                "prefix": b["prefix"],
                "expected": b["expected"],
                "base": b["decoded_top1"],
                "untrained": u["decoded_top1"],
                "pretrained": p["decoded_top1"],
            })

    print("\n" + "=" * 70)
    print("HEADLINE")
    print("=" * 70)
    print(f"  baseline Qwen (no bolt):     {base_acc:.3f}")
    print(f"  + bolt (untrained Engram):   {bu_acc:.3f}  Δ={bu_acc - base_acc:+.3f}")
    print(f"  + bolt (generic-NTP trained):{bp_acc:.3f}  Δ={bp_acc - base_acc:+.3f}")
    print(f"\n  Interpretation:")
    if abs(bu_acc - base_acc) <= 0.05 and abs(bp_acc - base_acc) <= 0.05:
        print("  Bolt-on is non-invasive on text recall (≤5% drift in both conditions).")
        print("  Original framing claim 'no regression on propositional control' SUPPORTED.")
    else:
        print(f"  Drift > 5% in at least one condition. Disagreements: {len(diffs)} items.")
        for d in diffs[:10]:
            print(f"    {d}")


if __name__ == "__main__":
    sys.exit(main())
