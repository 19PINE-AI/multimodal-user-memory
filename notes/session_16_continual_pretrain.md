# Session 16 — continual-pretraining-scale test

**Date:** 2026-05-16

User's hypothesis: pretrained LMs and DeepSeek-Engram demonstrate that
parametric memory scales given enough pretraining compute. Path A's
400–10k-step training is too little to fairly test the primitive's
ceiling. Continual-pretraining-scale (~100k steps, larger train pool,
augmentation) should push the system substantially further.

This session does exactly that and reports an interesting result.

## Setup

- **Training pool expanded**: LFW min_faces=2 → 1680 IDs (up from 901),
  combined with AgeDB 500 → **2180 unique identities, 7871 samples**.
- **Codebook**: K=1024 (matches scale of expected N=1000+).
- **Pretrain**: 100,000 STE co-pretrain steps (10× the previous heavy
  pretrain). Embedding-space Gaussian augmentation σ=0.02 (matched to
  natural intra-id std on unit-sphere ArcFace embeddings) synthesises
  cross-condition variants at each step.
- **Compute**: ~2.7 hours on this workstation (96 ms/step).

## Result

| N | RAG | Path A | code-match | **frac-code** | ratio |
|---|---|---|---|---|---|
| 20 | 0.800 | 0.250 | 0.424 | **0.550** | 0.31 |
| 100 | 0.780 | 0.083 | 0.157 | **0.530** | 0.11 |
| 300 | 0.734 | 0.070 | 0.130 | **0.538** | 0.10 |
| 700 | 0.762 | 0.067 | 0.129 | **0.525** | 0.09 |

vs the prior 10k-step K=512 heavy-pretrain baseline at N=100:
  - frac-code-match: 0.32 → **0.53** (+0.21, +65% relative)
  - code-match retrieval: 0.51 → **0.16** (-0.35)
  - retr@1: 0.163 → **0.083** (worse net)

## The non-obvious finding

**Continual pretraining DID lift the codebook side substantially.** The
fraction-code-match — the rate at which a cross-condition query and its
registered sample quantise to the same code — climbed from ~0.32 to
~0.53. That's the codebook-quality axis the user predicted should
improve with more pretraining + larger pool + augmentation, and it did.

**But the gate side collapsed**, and the two effects roughly cancel in
net retr@1. The gate's mechanism term (code-match retrieval =
"given the codes match, does the gate produce the right marker")
dropped from 0.51 → 0.16 at N=100.

The mechanism: with K=1024 the Engram has 16× more table rows than K=64.
At fixed pretrain compute, each row receives much less training signal.
The gate's discrimination capacity is amortised across more parameters
than the data can support. So the gate "learns less well per row" even
though the codebook itself is now better.

retr@1 = frac_code_match × code_match_retr is bounded by the *product*.
You can't get a high product by lifting one term while the other
collapses.

## What this means for the framing

Two structural conclusions:

1. **The codebook-architecture ceiling I diagnosed in sess-15 was
   incomplete.** Augmentation + larger data + larger K DOES improve
   codebook same-code rate substantially. The "K-vs-N tradeoff with no
   escape" pessimism was too strong. The user was right that compute
   can move it.

2. **But there's a second ceiling — gate capacity vs compute.** At
   K=1024, the gate has too many rows to train solidly with 100k steps
   of compute. Either (a) more compute to fill them, (b) smaller K so
   each row sees more signal, or (c) sparser/structured Engram so
   capacity is amortised better.

For a true DeepSeek-Engram-style result, you need both terms high
simultaneously. With our compute budget that means either:
- More compute (1M+ steps), OR
- A different K-vs-data balance (e.g., K=256 with the same 100k steps
  would give each row 4× more samples)

## Concrete follow-ups (ranked by expected return)

1. **K=256 with the same 100k-step pretrain + augmentation.** Each row
   gets 4× more training signal than K=1024. If gate term recovers to
   ~0.5 while frac-code stays at ~0.4, retr@1 ≈ 0.20 at N=100 — a
   genuine improvement.

2. **K=512, 200k steps.** Doubles the compute, holds the K-data balance.
   Should let both terms climb together.

3. **Curriculum: start K=64 for the gate's first 30k steps, then expand
   to K=512 with the gate weights inherited.** Lets the gate establish
   on a manageable address space first, then scales.

4. **LM-side LoRA.** As before, but combined with continual pretrain.

## Files added

- `src/extract_lfw_xxxl.py` — relaxed LFW extraction (2180 combined IDs)
- `src/nanochat_mm/pathA_continual_pretrain.py` — 100k-step STE with
  embedding-space augmentation
- `runs/embeddings/arcface_face_xxxl.npz` — 2180-ID combined face pool
- `results/pathA_continual_K1024_steps100000_aug0.02_seed42.json`
