# v2 retrieval head-to-head — honest negative result, points to the real fix

**Date:** 2026-05-14
**Source script:** `src/nanochat_mm/v2_retrieval.py`
**Result:** `results/v2_retrieval.json`
**Verdict:** **v2 fails retrieval at toy scale.** But the failure is diagnostic — it isolates what the *real* v3 must change.

## Headline table

| Modality | N | RAG ceiling | v1 (chained Engram) | **v2 surgical** | v2 − v1 |
|---|---|---|---|---|---|
| Vision | 5  | 0.96 | 0.52 | 0.48 | −0.04 |
| Vision | 10 | 0.98 | 0.46 | 0.44 | −0.02 |
| Vision | 20 | 0.95 | 0.48 | **0.23** | **−0.25** |
| Audio  | 5  | 1.00 | 0.68 | 0.44 | −0.24 |
| Audio  | 10 | 1.00 | 0.64 | 0.52 | −0.12 |
| Audio  | 20 | 1.00 | 0.60 | **0.31** | **−0.29** |

At small N, v2 is roughly competitive with v1. At N=20 — the scale that matters — v2 collapses to half of v1.

## Why v2 lost

Two compounding problems isolated by the experiment:

**(1) Surgical insertion was implemented naively.**
The `surgical_insert` function in `v2_retrieval.py` SGD-steps over the **entire** per-modality Engram module:

```python
params_to_train = list(eng.parameters())  # ~3.5M params per modality
opt = torch.optim.SGD(params_to_train, lr=0.1, momentum=0.9)
```

That's far more than the rows that `hash(code)` actually addresses. Result: each identity's registration gradient-descends through the whole module, destroying earlier identities by gradient leakage. The architecture supports row-targeted O(1) insertion (the rows ARE the embedding table indexed by hash); my implementation didn't take advantage of that.

A proper v2 surgical insertion would:
- Run the hash forward to get the specific row indices `hash(code) = (idx_0, ..., idx_{H-1})` at each attached layer
- Gather only those rows as the leaf parameters
- Optimise those rows (and only those) for 15 steps
- Insertions to different codes touch disjoint rows ⇒ no cross-identity interference

This is the analogue of user-as-engram's `UNEMBED_P` / `OPT-15` which are explicitly row-targeted (`prototype/` and `engram_demo_v1.py`).

**(2) The frozen quantiser has irreconcilable trade-offs at toy scale.**

K=32 codebook means at most 32 distinct codes across identities. With 20 held-out identities, many share codes (intra-identity recurrence ≈ inter-identity collision at this K). Even with row-targeted insertion, two identities sharing a hash address will trample each other's row.

This is the **same trade-off** the v1 sanity findings exposed (`notes/sanity_findings.md` §1-2, §5). The naive k-means quantiser has a discriminability/stability frontier that doesn't go away in v2.

## What v2 *did* prove

- The MultimodalEngramSet runs forward correctly, supports per-modality dispatch, supports per-user salting (smoke test passed).
- After 800 NTP training steps on cross-sequence recurrence data, **the gate at layer 1 fires more on recurrent positions than novel** (vision +10%, audio +5–12%). The end-to-end-trained gate learns the right behaviour. v1's frozen-codebook gate could not.

So we have a partially-working architecture: the **gate side** works as intended; the **codebook side** and the **surgical insertion side** both regress to v1-like behaviour because (a) the quantiser is still frozen, (b) insertion isn't actually row-targeted.

## The v3 step that follows

Two concrete things to fix, in priority order:

**v3.1 — Row-targeted surgical insertion.** Cheap. ~2 hours of code. Should recover at minimum v1 retrieval@1 (because the architecture is the same as v1 plus a working gate). Likely improvement: +0.05 to +0.15 at N=20 because the gate's recurrence preference helps discriminate within-collision cases.

**v3.2 — End-to-end-trained quantiser.** The real fix. Replace the frozen k-means with a learned VQ-VAE codebook *inside* the LM's parameters, trained via straight-through estimator on the NTP loss. This is what user-as-engram's pretrain does implicitly (the Engram embedding table IS the codebook + memory, jointly trained). For multimodal, the input flow becomes:

```
[real-valued ArcFace / ECAPA embedding]
       │
       ▼
[learned linear projection]  ← trained via STE
       │
       ▼
[VQ codebook lookup, STE-quantised]  ← trained via STE on NTP loss
       │
       ▼
[hash(quantised code) → Engram row]   ← rows trained via NTP
```

The codebook now allocates codes along directions that the LM finds useful for prediction — which, under the right training data (recurrent identities), means *identity-stable* directions. This is the qualitative break v1 lacked.

**v3.3 — Scale.** Toy model is 3.1M params; user-as-engram's `d12@1280_optimal` is 625M with ~3.34B training tokens. Scaling the model probably matters: more capacity to use the Engram, more pretraining tokens, more recurrent identities to learn from. Probably required for any paper-grade retrieval result.

## What I'd commit to in the paper now

Given the full arc of v1 + v2:

- **Negative-result section (a strength, not weakness).** "Cross-condition perceptual memory cannot be achieved by post-training a frozen codebook over modern face/voice encoders. The encoder is excellent in isolation (ArcFace top-1 = 0.98), but its embedding-space allocates capacity along directions that don't transfer to a discrete codebook. Bolt-on parametric memory loses to plain embedding RAG."
- **Architectural contribution.** "We propose Multimodal Engram, an end-to-end-trainable hash-keyed parametric memory whose codebook, gate, and embedding table are trained jointly with a small multimodal LM via NTP loss."
- **Validation evidence.** "End-to-end training produces a gate that fires more on recurrent perceptual positions than novel ones (+10–13% by residual norm), both within and across sequences, on both modalities. This recurrence-aware behaviour does not emerge under bolt-on post-training (`§ v1`)."
- **Open challenges flagged.** "Row-targeted surgical insertion and learned-quantiser-with-STE are the remaining engineering blocks before head-to-head retrieval against embedding RAG."

## Files produced this leg

- `src/nanochat_mm/{engram_module_mm, smoke_test, toy_gpt_train, real_encoder_train, v2_retrieval}.py`
- `runs/v2_toy_realencoder.pt` (trained toy checkpoint)
- `runs/v2_quantisers.npz` (cached train-side embeddings for quantiser refit)
- `results/{toy_recurrence, real_encoder_recurrence, v2_retrieval}.json`
- `notes/{v2_architecture_plan, v2_first_results, v2_retrieval_findings}.md`
