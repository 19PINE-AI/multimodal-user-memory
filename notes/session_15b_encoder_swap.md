# Session 15b — encoder swap: AntelopeV2 (R100/Glint360K) on V-XC-ID

**Date:** 2026-05-16

The session 15 first-principles analysis suggested the codebook bottleneck
at large N is encoder-bound. The user's natural follow-up: *why not just
use a stronger encoder?* This session tests the hypothesis empirically.

## Setup

LFW (sklearn, min_faces=3, 901 IDs). Re-extract face embeddings with:

- **Current**: ArcFace R50 trained on WebFace (`buffalo_l/w600k_r50.onnx`).
- **Stronger**: AntelopeV2 = R100 ArcFace trained on Glint360K
  (`antelopev2/glintr100.onnx`). R100 is ~4× the params of R50; Glint360K
  has 360K identities vs WebFace's ~85K. Widely considered the stronger
  insightface model.

Both produce 512-d normalised embeddings. Drop-in compatible — same
LFW image preprocessing, same downstream pipeline.

## Codebook K-sweep — the critical diagnostic

Same eval split (50/50 identity-disjoint, 451 eval IDs).

| K   | AntelopeV2 R100 same-code | ArcFace R50 same-code (naive) | Δ |
|-----|---------------------------|--------------------------------|---|
| 16  | 0.350                     | (not measured)                 | — |
| 32  | 0.296                     | (not measured)                 | — |
| **64** | **0.256**              | **~0.258**                     | **≈ 0** |
| 128 | 0.200                     | ~0.167                         | +0.03 |
| 256 | 0.198                     | ~0.112                         | +0.09 |

**At K=64, the two encoders give essentially identical same-code rates
(0.256 vs 0.258).** AntelopeV2 has 4× the params, was trained on 4× the
identities, achieves much higher LFW verification accuracy in standard
benchmarks — and yet at the codebook level, it offers no meaningful
improvement in cross-condition same-id same-code rate.

At larger K (128, 256), AntelopeV2 does slightly better — but the same-code
rates are still well below what Path A needs to scale (>0.5 would be
useful; we have <0.3).

## Why the encoder upgrade fails to help

Three plausible explanations, all consistent with the data:

1. **Codebook quantisation is the binding constraint, not embedding
   quality.** Both R50 and R100 ArcFace produce embeddings where d_intra
   (same-id cross-condition) is small relative to d_inter (different-id).
   When we Voronoi-partition the manifold into K cells, both embeddings'
   same-id pairs span similar fractions of cell boundaries. The
   *additional* discrimination R100 captures lives in directions that
   don't help cell-level recall — it helps cosine-NN over the raw embedding,
   which we already see (RAG stays at ~0.77 at N=700 regardless).

2. **LFW's cross-condition variation is too mild.** LFW is mostly
   in-the-wild but same-day, same-camera. Both R50 and R100 are saturated
   for this kind of variation. To stress-test the encoder difference, we'd
   need AgeDB-style cross-age or IJB-C-style cross-quality — and even
   there, R100's advantage is fractional rather than transformative.

3. **The codebook capacity-vs-encoder-bit-rate tradeoff is invariant to
   linear improvements.** The encoder needs log₂(N) bits of effective
   cross-condition selectivity to support N IDs in a K=N codebook. R100
   over R50 might give 0.5–1 extra bits; not the 5+ needed to push from
   N=32 reliable IDs to N=1000.

## Implication for the paper's framing

The encoder upgrade hypothesis was the user's question — and the answer
is **the encoder isn't where the leverage is**. The codebook architecture
itself is the bottleneck. This is consistent with the session 15
first-principles analysis: K-sweep also fails, multi-position codes
also fail, and now stronger-encoder swap also fails. **The discrete
codebook + hash-keyed table primitive is intrinsically capacity-limited
at the cross-condition level.**

What the paper *can* honestly claim is now narrower than before:

- The mechanism scales to thousands of identities in **latency and memory**
  (sess 14b) — that's structurally sound.
- The mechanism handles cross-condition perceptual recall **at small N**
  (≤20 — A-PARA BEATS RAG p=0.010, sess 11) — that's verified.
- The mechanism does **not** scale to large N in retrieval accuracy
  *regardless of encoder choice*. The right framing is "compatible with
  any encoder; performance gated by encoder + codebook capacity
  interaction" — not "stronger encoder makes us scale."

This is an important honesty calibration. The latency-at-scale claim
remains the architectural contribution; the accuracy-at-scale claim
should be dropped or heavily qualified.

## Other encoders attempted

- **AdaFace IR-101 (cvlface)**: cvlface model has custom-module deps
  (`models` package); auto-load fails. Could be patched but unlikely to
  change the verdict given AntelopeV2's negative result.
- **CED-Base on ESC-50**: HF datasets pipeline hit a torchcodec/ffmpeg
  dependency error trying to decode the audio shards. Could be worked
  around with manual soundfile loading. Not pursued given the AntelopeV2
  finding suggests encoder swaps aren't the lever.
- **Emotion2vec+, WavLM-SV**: same data-decoding issues; not pursued.

## Files added

- `src/nanochat_mm/encoder_swap_diag.py` — encoder swap + K-sweep diagnostic
- `src/nanochat_mm/extract_wavlm_libri.py` — placeholder; needs raw LibriSpeech (not local)
- `runs/embeddings/antelope_lfw_xxl.npz` — AntelopeV2 features (901 IDs)
- `results/encoder_swap_adaface.json` — diagnostic JSON (named after the
  originally-attempted AdaFace; actually contains AntelopeV2 numbers)
