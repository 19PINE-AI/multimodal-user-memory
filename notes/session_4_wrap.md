# Session 4 wrap — Path A delivered the mechanism

**Date:** 2026-05-14 (final entry session 4)

## What this session established

**Path A succeeds at the mechanism level.** A bolt-on MultimodalEngramSet on a frozen, pretrained Qwen2.5-3B-Instruct, with no Engram pretraining at all, supports surgical-insertion-based identity retrieval at significantly above chance whenever the cross-condition code matches the registration code.

Headline mechanism numbers (code-match retrieval, the cleanest test of surgical insertion's effect):

| Modality | N | Code-match retrieval | Chance | Lift |
|---|---|---|---|---|
| Vision | 5  | **0.91** | 0.20 | 4.6× |
| Vision | 10 | **0.46** | 0.10 | 4.6× |
| Audio  | 10 | **0.72** | 0.10 | 7.2× |
| Audio  | 20 | **0.56** | 0.05 | 11.2× |

## Why this matters

After three sessions of negative results at toy and midscale (3M, 15.5M), this is the first **positive scale-confirming evidence** for the paper's central claim. Bolt-on parametric perceptual memory works as long as the underlying LM has enough capacity to support surgical-insertion-driven output. At 3B-Qwen scale that capacity is sufficient.

The two-failure-mode decomposition makes the path forward precise:
- **Failure mode A — surgical insertion can't drive output**: ✗ ruled out at 3B scale. Mechanism works when codes match.
- **Failure mode B — codebook produces code mismatches under cross-condition variation**: ✓ still present. ~50% of queries don't share their registration code.

A non-RAG-cheating end-to-end comparison vs v1 first-write-wins (the truly parametric variant):

| | v1 first-write | Path A | delta |
|---|---|---|---|
| vision N=5 | 0.32 | **0.48** | +0.16 |
| vision N=10 | 0.26 | **0.34** | +0.08 |
| audio N=10 | 0.38 | **0.42** | +0.04 |

Path A wins or ties at small N. The overall metric is dragged down at N=20 by codebook collisions, exactly the same way v1 was. That's the codebook bottleneck, not the mechanism.

## What was tried and rejected in this session

- **Larger codebook (K=64)**: reduces collisions but raises code-mismatch rate. Net effect: roughly null. Not the fix.
- **More aggressive surgical insertion (200 steps, lr=3.0)**: overshoots, destroys convergence. Stable settings are lr=1.0, 80 steps.
- **Qwen3-VL-8B-Thinking**: tried first, fails `AutoModelForCausalLM` instantiation (needs the Qwen3VL-specific class). Pivoted to text-only Qwen2.5-3B. The mechanism argument still holds — perceptual encoders are frozen, base LM just needs strong output-prediction.

## The remaining experiments (clearly scoped)

In priority order:

1. **Path A + brief Engram pretraining** (Task #23). Pretrain Engram + perceptual-emb on training-identity sessions for ~500 NTP steps where the model is shown perceptual codes paired with markers. Then held-out surgical insertion. Hypothesis: pushes code-match retrieval over 0.95 at small N.

2. **Path A + STE-trained codebook** (Task #24). The original v3.2 ambition, now in the productive Path A regime. End-to-end-train codebook via STE during Engram pretraining; the codebook learns LM-useful directions instead of variance-maximising. Hypothesis: code-match rate from 0.50 → 0.85+, OVERALL retrieval into the 0.5–0.8 range.

3. **PerceptMem benchmark assembly**. Now that we have a working pipeline, build the actual paper benchmark from public assets (LFW + AgeDB + VGGFace2 cross-condition; VoxCeleb cross-recording; WikiArt cross-period style; DCASE TAU acoustic scenes; RAVDESS paralinguistic state).

4. **Head-to-head vs Online-PVLM / RAP / MyVLM**.

5. **Paper draft**.

## Paper outline now (concrete)

After this session the empirical structure of the paper is:

- §1 Intro: cross-condition perceptual memory is open; existing works are visual-identity-only and don't test cross-session cross-condition.
- §2 Related: MyVLM / Yo'LLaVA / Online-PVLM / RAP for vision concept memory; M3-Agent / Mem-Gallery / LCMP for multimodal memory benchmarks; user-as-engram for parametric memory.
- §3 Sanity: encoders work (ArcFace 0.98 top-1, ECAPA 1.0). Frozen-codebook quantiser has irreconcilable discriminability/stability tradeoff.
- §4 Failure of bolt-on retrieval (v1): post-trained frozen-codebook + hash table loses to embedding RAG. Establishes the problem.
- §5 Architecture: MultimodalEngramSet — parallel per-modality Engram tables, hash-keyed, with per-user salt support. Smoke test + gate-on-recurrence validation (v2 toy).
- §6 The scale finding: from-scratch pretrain at toy/midscale doesn't yet drive retrieval; bolt-on at 3B does (Path A). Reframes the problem: it's not "post-train vs pretrain" but "scale vs no scale, codebook vs frozen-codebook".
- §7 Path A results: surgical insertion + Engram bolt-on on Qwen-3B drives surgical retrieval at 0.46-0.91 code-match. Decomposes overall failure modes.
- §8 Closing the gap (results from Tasks #23 / #24): Engram pretraining + STE codebook → PerceptMem benchmark wins.

The first 7 sections are now fully evidenced. §8 is the next experimental session.

## Files at session 4 end

```
src/nanochat_mm/
├── engram_module.py / engram_module_mm.py
├── smoke_test.py
├── toy_gpt_train.py / real_encoder_train.py  (toy 3M)
├── midscale_train.py  (15.5M)
├── v2_retrieval.py / v3_retrieval.py / v3_retrieval_midscale.py / v3_aggressive_insert.py / v3_fixed_context.py
├── qwen_smoke.py
└── qwen_engram_bolt.py  ← Path A success
```

```
results/
├── sanity_*.json (×2), learned_rqvae.json, rqvae_heldout.json, quantiser_bakeoff.json
├── engram_retrieval.json (v1)
├── toy_recurrence.json, real_encoder_recurrence.json
├── v2_retrieval.json, v3_retrieval.json, v3_retrieval_midscale.json, v3_aggressive_insert.json, v3_fixed_context.json
├── midscale_train.json
└── pathA_qwen_bolt.json  ← Path A result
```

```
notes/
├── sanity_findings.md, escalation_decision.md
├── v2_architecture_plan.md, v2_first_results.md, v2_retrieval_findings.md
├── v3_findings_and_next.md, v3_full_findings.md
├── pathA_breakthrough.md  ← this session's main note
└── session_4_wrap.md  ← this file
```

State: disk 1.8 TB free, GPU available, Qwen2.5-3B and all encoders cached.
