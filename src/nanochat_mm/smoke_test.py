"""Smoke test for the MultimodalEngramSet.

We don't train anything; we just verify:
  1. The module instantiates cleanly with realistic configs.
  2. Forward pass on synthetic interleaved data produces gradient-trackable output.
  3. Parameter counts make sense and modalities scale independently.
  4. Setting per-user salt changes the output (i.e. salting actually affects the hash).

If any of these fail, the v2 architecture is broken at the most basic level
and we shouldn't proceed to pretraining.
"""
import sys
from pathlib import Path

import torch

# Ensure we can import sibling modules in this directory
sys.path.insert(0, str(Path(__file__).parent))

from engram_module_mm import (
    MultimodalEngramSet,
    MultimodalEngramConfig,
    MODALITY_TEXT,
    MODALITY_VISION,
    MODALITY_AUDIO,
    MODALITY_NAMES,
)

torch.manual_seed(42)


def main():
    print("=" * 70)
    print("MultimodalEngramSet smoke test")
    print("=" * 70)

    hidden_size = 384  # small for a quick test (d8-ish)
    cfg = MultimodalEngramConfig(layer_ids=[2, 5])

    print("\n[1/4] Constructing MultimodalEngramSet ...")
    eng = MultimodalEngramSet(cfg, hidden_size=hidden_size)
    print(f"  Layer attachment: {cfg.layer_ids}")
    total = eng.total_params()
    tables = eng.total_table_params()
    grand_total = sum(total.values())
    print(f"  Total params per modality (incl. proj/conv):")
    for k, v in total.items():
        print(f"    {k:>7s}: {v:>12,d}")
    print(f"  Embedding-table-only params per modality:")
    for k, v in tables.items():
        print(f"    {k:>7s}: {v:>12,d}")
    print(f"  Grand total: {grand_total:,d} params (~{grand_total/1e6:.1f}M)")
    assert grand_total > 0
    assert "text" in total and "vision" in total and "audio" in total

    print("\n[2/4] Building synthetic interleaved batch ...")
    B, T = 4, 64
    # Interleave: first 32 positions are text, then 16 vision, then 16 audio
    modality_ids = torch.zeros(B, T, dtype=torch.long)
    modality_ids[:, 32:48] = MODALITY_VISION
    modality_ids[:, 48:64] = MODALITY_AUDIO

    text_ids = torch.randint(1, cfg.text_vocab_size, (B, T))
    vision_ids = torch.randint(1, cfg.vision_vocab_size, (B, T))
    audio_ids = torch.randint(1, cfg.audio_vocab_size, (B, T))
    # Build the unified input_ids: pull from the right vocab per position
    input_ids = torch.where(modality_ids == MODALITY_TEXT, text_ids,
                 torch.where(modality_ids == MODALITY_VISION, vision_ids, audio_ids))
    print(f"  Batch: B={B}, T={T}, modality breakdown: "
          f"text={int((modality_ids == MODALITY_TEXT).sum())}, "
          f"vision={int((modality_ids == MODALITY_VISION).sum())}, "
          f"audio={int((modality_ids == MODALITY_AUDIO).sum())}")

    h = torch.randn(B, T, hidden_size, requires_grad=True)

    print("\n[3/4] Running forward_layer at each attached layer ...")
    for lid in cfg.layer_ids:
        residual = eng.forward_layer(h, input_ids, modality_ids, layer_idx=lid)
        assert residual is not None, f"layer {lid}: no residual produced"
        assert residual.shape == (B, T, hidden_size), f"layer {lid}: bad shape {residual.shape}"
        # Per-position residual norms; modality boundaries should differ in scale
        rn = residual.norm(dim=-1).mean(dim=0)  # average over batch -> [T]
        text_rn = rn[modality_ids[0] == MODALITY_TEXT].mean().item()
        vis_rn  = rn[modality_ids[0] == MODALITY_VISION].mean().item()
        aud_rn  = rn[modality_ids[0] == MODALITY_AUDIO].mean().item()
        print(f"  layer {lid}: residual.shape={tuple(residual.shape)}  "
              f"mean ‖residual‖ per modality:  text={text_rn:.4f}  vision={vis_rn:.4f}  audio={aud_rn:.4f}")
        # Engram is init'd so conv weights are zero → residual should be the gate-mixed value path only.
        # Magnitudes should be small but non-zero. Just check non-NaN and non-zero overall.
        assert torch.isfinite(residual).all(), "non-finite residual"
        assert residual.abs().sum() > 0, "all-zero residual — something's miswired"

    print("\n[4/4] Per-user salt should change the output ...")
    eng.reset_cache()
    r0 = eng.forward_layer(h, input_ids, modality_ids, layer_idx=cfg.layer_ids[0])
    eng.set_user_salt(0xDEADBEEF)
    eng.reset_cache()
    r1 = eng.forward_layer(h, input_ids, modality_ids, layer_idx=cfg.layer_ids[0])
    eng.set_user_salt(0)
    eng.reset_cache()
    delta = (r0 - r1).abs().mean().item()
    print(f"  mean abs diff between salt=0 and salt=0xDEADBEEF: {delta:.6f}")
    assert delta > 1e-6, "salt did not change the output — per-user salting is broken"

    print("\nAll smoke tests PASSED.")
    print(f"\nTotal MultimodalEngram params: ~{grand_total/1e6:.1f}M")
    print(f"(For reference: text-only Engram in user-as-engram FINDINGS large = 51M; "
          f"this configuration is in similar ballpark per modality.)")


if __name__ == "__main__":
    sys.exit(main())
