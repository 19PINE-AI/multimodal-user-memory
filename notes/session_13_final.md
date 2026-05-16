# Session 13 final — top-K verification + long-form co-pretraining

**Date:** 2026-05-16

Session 12 ended on three negative-result lift attempts. The two follow-ups
this session — top-K multi-seed verification (A) and long-form co-pretraining
(B) — produced one decisive new headline.

## A. A-SCN top-K=3 multi-seed (5 seeds)

The session-12 bright spot (single-seed A-SCN N=5 top-K=3 = 0.560 vs
naive 0.432) verified across seeds:

| N  | top-K mean ± std | single-code multi-seed (sess 11) | Δ |
|----|------------------|----------------------------------|---|
| 5  | **0.520 ± 0.044**| 0.432 ± 0.082                    | **+0.088** |
| 10 | 0.260 ± 0.042    | 0.276 ± 0.113                    | −0.016 |
| 20 | 0.100 ± 0.000    | 0.210 ± 0.042                    | −0.110 |

Welch's t at N=5: t ≈ 2.1, p ≈ 0.04. Bright spot retained — top-K helps
at N=5 where the codebook miss rate dominates; hurts at N=20 where gate
collisions from 3 markers per identity dominate. Path A still 52% of
RAG=1.000 at N=5.

## B. Long-form Path A + STE co-pretraining (5000 steps)

Scaled the existing STE co-training machinery from 400–600 steps to 5000
steps. The codebook + Engram + perc_emb co-train via STE; LM stays frozen.

### A-PARA K=32 (single-seed)

| N | naive baseline | co-pretrain | Δ |
|---|----------------|-------------|---|
| 5 | 0.650 | 0.450 | **−0.200** |
| 10| 0.425 | 0.425 | 0.000 |
| 20| 0.300 | 0.388 | **+0.088** |

Mixed. The longer-trained codebook shifts the small-N regime away from
the marker-friendly configuration; helps the larger-N capacity slightly.
Net: not a win on A-PARA.

### A-XR-ID K=32 (5 seeds × 5000 steps each)

**This is the headline result of the session.**

| N  | co-pretrain mean ± std (5 seeds) | single-code multi-seed (sess 11) | Δ |
|----|----------------------------------|----------------------------------|---|
| **5**  | **0.632 ± 0.047** | **0.440 ± 0.123** | **+0.192** |
| 10 | 0.368 ± 0.016 | 0.276 ± 0.127 | +0.092 |
| 20 | 0.210 ± 0.032 | 0.166 ± 0.054 | +0.044 |

Per-seed N=5: [0.68, 0.60, 0.64, 0.56, 0.68]. **Range 0.56–0.68; std 0.047.**

Welch's t at N=5: t ≈ 3.21, df ≈ 6, one-sided **p ≈ 0.009** —
**HIGHLY SIGNIFICANT.** At N=10 the lift is +0.092 (p ≈ 0.09, marginal).

Code-match retrieval at N=5: 0.94–1.00 (mechanism is essentially perfect
on matched codes); fraction-code-match: 48–60% (codebook miss rate now
substantially reduced). The longer co-training **simultaneously** improves
both the mechanism term (gate quality) and the codebook (STE settles into
LM-useful centroid positions), which is why N=5 lifts and the std drops
dramatically (0.123 → 0.047).

vs RAG=1.000: Path A is now **63% of RAG** at A-XR-ID N=5 (was 44%). Still
below RAG but the largest, statistically-verified lift we have on any
non-A-PARA modality.

## Combined framing update

| Cell | Status | Path A | RAG | Δ | Statistical |
|------|--------|--------|-----|---|-------------|
| A-PARA N=10 K=16 (10 seeds) | BEATS RAG | 0.480 ± 0.062 | 0.425 | +0.055 | **p=0.010** |
| A-XR-ID N=5 K=32 5k steps (5 seeds) | LIFT vs naive | 0.632 ± 0.047 | 1.000 | +0.192 over naive Path A | **p=0.009** vs naive |
| A-SCN N=5 K=32 top-K=3 (5 seeds) | LIFT vs naive | 0.520 ± 0.044 | 1.000 | +0.088 over naive Path A | p=0.04 vs naive |
| All other cells | trail RAG | various | — | — | — |

The paper now has:
  - **One statistically-significant BEATS-RAG cell** (A-PARA N=10).
  - **Two statistically-significant lifts over the naive Path A baseline**
    (A-XR-ID N=5 with co-pretrain, A-SCN N=5 with top-K). Neither beats
    RAG but both establish that the Path A mechanism has real headroom
    once the right recipe (codebook quality / co-training depth) is
    applied.
  - **V-XC-ID N=20** lift +0.16 over naive Path A (sess 11 multi-seed).
  - **V-STY** documented encoder-limited.

This is a much stronger relaxed framing than session 11 had alone.

## Path forward

1. **Multi-seed A-PARA co-pretrain** (parallel-track of A-XR-ID): the
   single-seed A-PARA co-pretrain was mixed; multi-seed might surface
   a different equilibrium.
2. **A-SCN + co-pretrain combination**: top-K helps A-SCN at N=5, co-pretrain
   helps A-XR-ID at N=5; combine them on A-SCN to push further?
3. **Paper writing**: numbers are now sufficient for a defensible draft.

## Files added this session

- `src/nanochat_mm/pathA_topk_multiseed.py` — multi-seed top-K runner
- `src/nanochat_mm/pathA_copretrain_long.py` — long-form co-pretrain (5000 steps)
- `results/pathA_topk3_multiseed_a-scn_K32.json`
- `results/pathA_copretrain_a-para_K32_steps5000.json`
- `results/pathA_copretrain_a-xr-id_K32_steps5000.json` (single seed; overwritten by last seed of multi-seed run)
- `results/pathA_copretrain_a-xr-id_K32_steps5000_multiseed.json` — full 5-seed aggregate
