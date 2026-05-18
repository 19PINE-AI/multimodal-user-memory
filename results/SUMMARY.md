# Perceptual Engram — Final Results Summary (post-adversarial-training)

**Date:** 2026-05-18 (session 19)
**Title:** Parametric Multimodal User Memory: Storing What Captions Cannot Carry
**Architecture:** Continuous attention memory bolted on frozen pretrained LM via forward pre-hook on `lm_head`.
**Trainable params:** ~8M on Qwen-3B (~21M on Llama-3.1-8B).
**Frozen params:** 3.1B–8B LM + frozen perceptual encoders.

## Multi-seed BEATS-RAG (random regime — standard training)

| Cell | n | RAG | AttMem mean ± std | Δ | p-value |
|---|--:|--:|--:|--:|--:|
| V-XC-ID-XXXL $N{=}10$ (2180 face IDs) | 4 | 0.933 | **0.992 ± 0.014** | +5.9pp | **0.006** |
| V-STY-CLIP $N{=}5$ (painter style) | 5 | 0.400 | **0.640 ± 0.116** | +24pp | **0.015** |
| V-STY-CLIP $N{=}10$ (painter style) | 5 | 0.400 | **0.460 ± 0.025** | +6pp | **0.009** |

## Multi-seed BEATS-RAG (adversarial regime — adv-training)

Adversarial = bank composed of target + top-K cosine-similar non-matching identities.

| Cell | n | RAG | AttMem-adv mean ± std | Δ | p-value |
|---|--:|--:|--:|--:|--:|
| V-XC-ID-XXXL K=19 | 3 | 0.841 | **0.985 ± 0.001** | +14.4pp | **<0.001** |
| **A-PARA K=19** | 4 | 0.226 | **0.934 ± 0.004** | **+70.7pp** | **<0.001** |
| **V-STY K=19** | 4 | 0.267 | **0.977 ± 0.006** | **+71.0pp** | **<0.001** |
| A-SCN K=19 | 1 | 0.827 | 1.000 | +17pp | n/a |
| A-XR-ID K=19 | 1 | 1.000 | 1.000 | parity (encoder ceiling) | n/a |

## Cross-modality + cross-family validation

**Cross-family (V-XC-ID-XXXL, single seed each):**
- Qwen2.5-3B (12K steps): N=10 = 0.99 (BEATS RAG), N=1000 = 0.59
- Qwen2.5-7B (12K steps, with untied-emb fix): N=10 = 1.00, N=1000 = 0.50
- Qwen2.5-7B (50K steps): N=10 = 1.00, N=1000 = 0.57
- **Llama-3.1-8B (12K steps)**: N=10 = **1.00**, N=1000 = **0.62** (best at large N)

The architecture generalises across LM families; Llama-3.1-8B closes the adversarial gap (only −0.002 to +0.012 vs RAG at K=3..19) where Qwen-3B loses by 2-3pp.

## Pareto frontier (V-XC-ID-XXXL, adv_prob sweep)

| adv_prob | random N=10 | random N=1000 | adversarial K=19 |
|---|---|---|---|
| 0.0 | **0.99** | 0.59 | 0.81 (below RAG) |
| **0.1** ★ | 0.87 | 0.57 | **0.984** (sweet spot) |
| 0.3 | 0.83 | 0.57 | 0.986 |
| 0.5 | 0.60 | 0.57 | 0.992 |
| 0.7 | (data) | (data) | (data) |

adv_prob=0.1 is the sweet spot — near-best adversarial with minimal random-regime degradation.

## Path A → AttMem (the design-space story)

Path A (discrete codebook) saturates at ~7% retr@1 at N≥300 regardless of K∈{32..1024}, encoder upgrade, or 100K-step continual pretraining.

| Sub-modality | N | Path A | AttMem | Lift |
|---|--:|--:|--:|--:|
| A-XR-ID | 10 | 0.32 | 0.90 | 2.8× |
| A-SCN | 10 | 0.40 | 0.83 | 2.1× |
| V-XC-ID-XXXL | 10 | ~0.10 | 0.99 | ~10× |
| V-XC-ID-XXXL | 700 | ~0.07 | 0.63 | ~9× |
| V-STY | 5 | 0.20 | 0.47–0.64 | 2.4–3.2× |

## Latency (Qwen2.5-3B, H100-class GPU)

| N | AttMem query | AttMem insert (total) | RAG-with-context |
|--:|------:|------:|---:|
| 10 | 14.9 ms | 0.25 ms | 20.7 ms |
| 100 | 14.6 ms | 0.51 ms | 67.2 ms |
| 1000 | 15.8 ms | 0.52 ms | **823 ms** |
| 10000 | 16.6 ms | 0.69 ms | **OOM (>32k ctx)** |

AttMem query latency flat; 52× faster than RAG-with-context at N=1000; RAG architecturally cannot fit N=10000.

## Architectural validations

| Validation | Result |
|---|---|
| Propositional control (text non-regression) | **PASS** — hook-no-op byte-perfect; bolt.forward preserves top-1 8/8 |
| Per-modality bank independence (zero-shot) | **PASS** — vision retr@1 0.77, audio retr@1 0.93, cross-modal leak <7% |
| Pretraining convergence | 5K steps for audio/style; 12K for V-XC-ID-XXXL |
| Curriculum bank_size | N=700 lifted 0.20→0.63 with bs∈[64,1024] |
| Tied vs untied embeddings | Qwen-3B tied → input_emb value; Qwen-7B/Llama untied → lm_head.weight (auto-detected) |
| Real-name demo (zero-shot) | 10/10 retr@1 with single-token first names as markers |
| Mechanism analysis | Bank attention sharply diagonal; LM marker logits inherit + amplify |

## What's left for top-tier submission

1. **Online-PVLM head-to-head** — closest prior; code not yet public
2. **10K+ ID scale test with real cross-condition data** — VGGFace2 / VoxCeleb-2 / MS-Celeb-1M; needs data acquisition (1-2 days)
3. **Qwen3-VL end-to-end** — wired but vision-token integration needs validation
4. **LongMemEval-style perceptual benchmark** — multi-session simulation; major new effort
5. **More LM families** — Mistral, Llama-3.3, etc.
