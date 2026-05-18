# Perceptual Engram — Final Results Summary (AttentionMemory pivot)

**Date:** 2026-05-17 (session 18)
**Architecture:** Continuous attention memory bolted on frozen Qwen2.5-3B-Instruct.
**Trainable params:** ~8M (W_q, W_o, out_gain, log_inv_temp, perceptual projections).
**Frozen params:** ~3.1B (Qwen LM + encoder).

## Headline: 3 multi-seed-verified BEATS-RAG cells (p<0.05)

| Cell | n | RAG ceiling | AttMem mean ± std | t-stat | p-val |
|---|---:|---:|---:|---:|---:|
| **V-XC-ID-XXXL N=10** (2180 face IDs) | 4 | 0.933 | **0.992 ± 0.014** | 8.39 | **0.006** |
| **V-STY-CLIP N=5** (painter style) | 5 | 0.400 | **0.640 ± 0.116** | 4.13 | **0.015** |
| **V-STY-CLIP N=10** (painter style) | 5 | 0.400 | **0.460 ± 0.025** | 4.81 | **0.009** |

AttMem is the first parametric perceptual memory mechanism to multi-seed-verify a BEATS-RAG result at the >500-ID scale (Path A's BEATS-RAG was at A-PARA N=10 with 84 IDs).

## Full PerceptMem v0.2 scorecard

| Sub-modality | N | RAG | AttMem | ratio | vs Path A |
|---|--:|---:|---:|---:|---:|
| A-XR-ID (LibriSpeech ECAPA, 30 IDs) | 5  | 1.00 | 0.87 | 0.87 | — |
|                                       | 10 | 1.00 | 0.90 | 0.90 | 2.8× |
|                                       | 20 | 1.00 | 0.90 | 0.90 | — |
| A-SCN (ESC-50 AST, 50 IDs)           | 5  | 1.00 | 1.00 | 1.00 | — |
|                                       | 10 | 0.93 | 0.83 | 0.89 | 2.1× |
|                                       | 20 | 0.87 | 0.73 | 0.85 | — |
| A-PARA (wav2vec spk×emo, 168 IDs, n=5) | 5  | 0.80 | 0.73 ± 0.00 | 0.92 | — |
|                                          | 10 | 0.47 | 0.44 ± 0.04 | 0.94 | 0.98× (parity) |
|                                          | 20 | 0.40 | 0.39 ± 0.02 | 0.97 | — |
|                                          | 50 | 0.29 | 0.21 ± 0.01 | 0.74 | — |
| V-XC-ID-XXXL (ArcFace, 2180 IDs, n=4) | 5  | 0.93 | 0.93 ± 0.00 | 1.00 | — |
|                                         | 10 | 0.93 | **0.99 ± 0.01** | **1.07** | **~10×** |
|                                         | 20 | 0.80 | 0.81 ± 0.01 | 1.01 | — |
|                                         | 50 | 0.77 | 0.73 ± 0.00 | 0.95 | — |
|                                         | 100 | 0.78 | 0.74 ± 0.01 | 0.94 | ~10× |
|                                         | 300 | 0.73 | 0.64 ± 0.00 | 0.87 | ~9× |
|                                         | 700 | 0.76 | 0.63 ± 0.00 | 0.83 | ~9× |
|                                         | 1000 | 0.77 | 0.59 (n=1) | 0.77 | ~8× |
| V-STY-CLIP (painter style, 30 IDs, n=5) | 5  | 0.40 | **0.64 ± 0.12** | **1.60** | **2.4×** |
|                                            | 10 | 0.40 | **0.46 ± 0.03** | **1.15** | — |
|                                            | 20 | 0.33 | 0.23 ± 0.01 | 0.69 | — |

## Latency (Qwen2.5-3B, H100-class GPU)

| N | AttMem query | AttMem batch-insert | RAG-with-context | Path A insertion (per id) |
|--:|------:|------:|---:|---:|
|    10 | 14.9 ms | 0.25 ms | 20.7 ms | ~1000 ms |
|   100 | 14.6 ms | 0.51 ms | 67.2 ms | ~1000 ms |
|  1000 | 15.8 ms | 0.52 ms | **823 ms** | ~1000 ms |
| 10000 | 16.6 ms | 0.69 ms | **OOM (>32k context)** | ~1000 ms |

**AttMem query is flat ~15 ms regardless of N.** RAG-with-context is **52× slower** at N=1000 and OOMs at N=10000.

vs Path A insertion: AttMem batch-insert of 1000 ids is 0.52 ms; Path A is ~1,000,000 ms (16 minutes). **~2,000,000× speedup at N=1000.**

## Architectural validations

| Validation | Result |
|---|---|
| Propositional control (no text regression) | **PASS** — hook-no-op byte-perfect; bolt.forward() preserves top-1 8/8 across 8 text prompts |
| Per-modality bank independence (zero-shot) | **PASS** — vision retr@1 0.77, audio retr@1 0.93, cross-modal leak <7% |
| Pretraining convergence | 5K steps for A-PARA / A-XR-ID / A-SCN / V-STY; 12K for V-XC-ID-XXXL |
| Curriculum bank_size fix at large N | N=700 lifted 0.20 → 0.63 with bs ∈ [64, 1024] uniform |
| Tied vs untied embeddings | Qwen2.5-3B tied → use `input_embedding[marker]`; Qwen2.5-7B untied → must use `lm_head.weight[marker]` (auto-detected in code) |

## LM-size scaling (Qwen 3B → 7B at V-XC-ID-XXXL, 12K steps)

| N | 3B (n=3) | 7B (lm_head fix) |
|--:|---:|---:|
|  10 | 0.992 | 1.000 |
|  20 | 0.808 | 0.833 |
|  50 | 0.733 | 0.720 |
| 100 | 0.742 | 0.737 |
| 300 | 0.637 | 0.624 |
| 700 | 0.629 | 0.582 |
| 1000 | 0.594 | 0.497 |

7B BEATS 3B at N=10/20; lags at N>=300. Final loss 4.94 (7B) vs 3.35 (3B):
7B isn't converged in the same 12K-step budget; needs more compute for fair comparison.

## What's left for camera-ready

1. ~~V-STY-CLIP multi-seed verification~~ ✓ Done (BEATS p<0.05)
2. ~~V-XC-ID-XXXL at N=1000~~ ✓ Done (77% of ceiling)
3. ~~Latency benchmark~~ ✓ Done
4. ~~Propositional control~~ ✓ Done
5. Head-to-head vs Online-PVLM, MyVLM (require external code; future work)
6. Qwen3-VL full eval (architecture wired; needs GPU loading retest)
7. Larger audio pool (current ecapa_libri_large has only 58 speakers; would need VoxCeleb-2)
8. Paper prose writing
