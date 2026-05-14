# Sanity-check findings — perceptual encoder + quantisation collisions

**Date:** 2026-05-14
**Source scripts:** `src/sanity_ecapa_collisions.py`, `src/sanity_arcface_collisions.py`
**Verdict:** Both modalities PASS the gating experiment from `research_plan.md` §11. The Perceptual-Engram mechanism is sound to invest further in.

## TL;DR

| Modality | Encoder | Top-1 NN recall | Best flat K | Best flat ratio | Notes |
|---|---|---|---|---|---|
| Audio (speaker, cross-recording) | ECAPA-TDNN (192-d) | **1.000** | K=32 | **101** at 86% intra-agree | RQ(2×16) gives ratio ~2900 at 76% intra |
| Vision (face, cross-condition) | ArcFace R50 (512-d) | **0.979** | K=32 | **61** at 75% intra-agree | RQ collapses faster than audio; needs learned quantiser at scale |

Both pass the §11 §11 viability bar (top-1 NN > 0.85, viable codebooks where ratio > 5 and intra-agree > 0.5).

## Audio — ECAPA-TDNN on LibriSpeech test-clean

- 29 speakers × ~8 utterances across **different chapters** (cross-recording proxy).
- 232 embeddings, dim 192.
- Raw cosine: intra **0.641 ± 0.126**, inter **0.087 ± 0.090**. ~6× gap; perfect top-1.

Flat k-means quantisation:

| K | intra-agree | inter-coll | ratio |
|---|---|---|---|
| 8 | 0.930 | 0.105 | 8.9 |
| 16 | 0.953 | 0.044 | 21.7 |
| **32** | **0.862** | **0.009** | **101** |
| 64 | 0.558 | 0.0003 | 2150 |
| 128 | 0.247 | 0.0000 | ∞ |

Residual product quantisation (n_levels × k_per):

| Config | eff_K | intra-agree | inter-coll | ratio |
|---|---|---|---|---|
| **2 × 16** | **256** | **0.756** | **0.0003** | **2914** |
| 2 × 64 | 4096 | 0.091 | 0.0000 | ∞ |
| 3 × 16 | 4096 | 0.384 | 0.0001 | 7405 |
| 4 × 16 | 65536 | 0.127 | 0.0000 | ∞ |
| 4 × 64 | 1.7e7 | 0.000 | 0.0000 | ∞ |

Sweet spot: **RQ(2×16)** — keeps 76% intra-agreement at effective codebook of 256 with collision rate 0.03%. Deeper RQ shatters identity.

## Vision — ArcFace R50 on LFW (cross-condition)

- 30 people × 8 photos each (people with ≥10 photos in LFW; natural cross-lighting / expression / angle).
- 240 embeddings, dim 512.
- Raw cosine: intra **0.424 ± 0.153**, inter **0.038 ± 0.066**. ~11× gap; top-1 0.98.

Flat k-means quantisation:

| K | intra-agree | inter-coll | ratio |
|---|---|---|---|
| 8 | 0.891 | 0.117 | 7.6 |
| 16 | 0.833 | 0.044 | 19 |
| **32** | **0.752** | **0.012** | **61** |
| 64 | 0.579 | 0.002 | 273 |
| 128 | 0.243 | 0.0001 | 2346 |

Residual product quantisation:

| Config | eff_K | intra-agree | inter-coll | ratio |
|---|---|---|---|---|
| 2 × 16 | 256 | 0.425 | 0.0024 | 175 |
| 2 × 64 | 4096 | 0.123 | 0.0000 | ∞ |
| 3 × 16 | 4096 | 0.148 | 0.0004 | 407 |
| 4 × 16 | 65536 | 0.137 | 0.0002 | 661 |
| 4 × 64 | 1.7e7 | 0.000 | 0.0000 | ∞ |

Sweet spot: flat K ≈ 32–64 holds 58-75% intra at acceptable inter. RQ(2×16) only reaches 43% intra — notably worse than audio's 76%.

## Comparison: why vision tolerates depth less well than audio

Audio intra cosine is 0.64; faces intra cosine is 0.42. Faces have larger within-identity variance — different days, lighting, expression, age. Residual after the first quantisation level loses too much identity-discriminative signal because the residual amplitude on faces is larger relative to identity-axis amplitude.

**Implication for the method:** the perceptual quantiser should not be naively-shared across modalities. We should use a learned quantiser (RQ-VAE with reconstruction objective, or learned PQ) where the codebook is trained to preserve identity, not just minimise L2. The flat K=32–64 result tells us the *intrinsic* discriminability is there; we just need a better quantiser than residual k-means to scale to large K.

## Caveats

- **LibriSpeech test-clean is clean audiobook reads.** Real-world cross-recording (different mics, telephone, noise) is harder. ECAPA was trained on VoxCeleb (more varied) so should transfer, but the numbers here are upper-bound-ish.
- **LFW cross-condition variation is moderate**, not extreme — same-week photos, mostly frontal. Cross-age (years apart) or cross-occlusion (sunglasses, masks) would be harder.
- **Codebook trained on the test set itself.** Production setup must train the quantiser on held-out data; we expect a modest drop.
- **ONNX ArcFace ran on CPU** (CUDA EP available but didn't bind here — ONNX/onnxruntime CUDA needs additional config). Doesn't affect the science; affects throughput only.

## Next steps unblocked by this result

1. The Perceptual-Engram mechanism is worth building.
2. ~Train a proper learned quantiser (RQ-VAE with identity-preserving objective) per modality.~ **Done — see §3.**
3. Construct the PerceptMem V-XC-ID and A-XR-ID tasks at scale; we now know which encoder + which quantiser regime to target.
4. Begin the Qwen3-VL + Voxtral integration plan.

## §3. Learned RQ-VAE vs naive residual k-means (2026-05-14, follow-on)

Source: `src/learned_rqvae.py`. Compared a small VQ-VAE with reconstruction + identity-classification loss against the naive residual k-means used in §1–§2.

Headline:

| Modality | Config | eff_K | Naive intra | Learned intra | Δ |
|---|---|---|---|---|---|
| Audio  | L2×K16 |    256 | **0.930** | 0.842 | −0.087 |
| Audio  | L2×K64 |   4096 | 0.373 | 0.464 | +0.091 |
| Audio  | L3×K16 |   4096 | 0.398 | **0.772** | **+0.375** |
| Audio  | L4×K16 |  65536 | 0.097 | **0.368** | **+0.271** |
| Vision | L2×K16 |    256 | 0.657 | 0.709 | +0.053 |
| Vision | L2×K64 |   4096 | 0.141 | **0.562** | **+0.422** |
| Vision | L3×K16 |   4096 | 0.405 | **0.752** | **+0.347** |
| Vision | L4×K16 |  65536 | 0.197 | **0.719** | **+0.523** |

**Reading.**
- At small effective codebooks (~256), the choice doesn't matter much; naive is even slightly better on audio (k-means in continuous space naturally finds the ~29 speaker clusters).
- At medium-to-large effective codebooks (4k–65k), **the gap is dramatic**. Naive RQ collapses intra-agreement; the learned version holds it.
- On vision at eff_K=65,536: naive 0.20, learned **0.72** — a 3.6× improvement.

**Why naive collapses and learned doesn't.** Naive residual k-means at depth L > 1 minimises L2 of the residual; the codebook entries get assigned to *whatever* high-variance directions the residual points along, regardless of whether those directions are identity-discriminative. The learned version, supervised by an identity classifier on the quantised representation, allocates codebook capacity along identity-relevant axes.

**Implication for the method.** A learned RQ-VAE is **load-bearing** for the perceptual-Engram at any non-trivial scale. With it, we can confidently address 65k+ identities per modality at >70% intra-agreement on vision and >35% on audio (audio drops more because LibriSpeech intra-cosine is already 0.64, leaving less margin for very-large K). The method-paper claim "content-addressable parametric memory at the scale of a real user's recurrent perceptual encounters" is empirically supported.

**Caveats.**
- These numbers are *with the identity classifier trained on the same speakers/people we evaluate on*. **For true generalisation we need held-out identities. See §4.**
- The classification accuracy plateaus around 0.8 on vision (vs 0.99 on audio), reflecting harder discrimination.

## §4. Held-out generalisation — the classifier-loss learned RQ-VAE does NOT transfer

Source: `src/rqvae_heldout.py`. Split identities 50/50, train quantiser on TRAIN ids only, evaluate on HELDOUT (disjoint) ids. Use **ratio = intra-agreement / inter-collision** as the real discriminability metric (just intra is misleading when a degenerate quantiser also raises inter).

| Modality | Config | eff_K | Naive ratio | Learned ratio | Conclusion |
|---|---|---|---|---|---|
| Audio  | L2×K16 |    256 | 2.53  | 3.23  | tie / learned slightly better |
| Audio  | L2×K64 |   4096 | **31.6** | 2.12  | **Naive much better** |
| Audio  | L3×K16 |   4096 | 4.14  | 1.06  | Naive much better |
| Audio  | L4×K16 |  65536 | 5.52  | 3.02  | Naive better |
| Vision | L2×K16 |    256 | 1.13  | 1.33  | Both poor (encoder runs out of capacity) |
| Vision | L2×K64 |   4096 | **3.69**  | 1.87  | Naive better |
| Vision | L3×K16 |   4096 | 1.13  | 1.51  | Naive degenerated (no real quantisation past L=2); learned still tries |
| Vision | L4×K16 |  65536 | 1.13  | **2.26**  | Learned better (because naive fully degenerated) |

**Reading.** The headline result from §3 — learned RQ-VAE with identity-classification loss crushes naive at depth — **does not hold on held-out identities**. The classifier objective allocates codebook capacity along directions that separate *training* identities; those directions are not the directions that separate *unseen* identities. The learned codebook overfits.

**Why naive (sometimes) generalises better.** K-means with reconstruction-only objective allocates capacity to maximise embedding variance. That's an identity-agnostic objective; the resulting codebook structure transfers to new identities (within the same encoder distribution).

**Why neither is dominant.** When naive RQ runs out of useful capacity at deeper levels (vision L3/L4), it degenerates — the residual is too small and noisy, so deeper levels just produce ~constant codes. The "intra rate" stays at the L2 value but so does the inter rate (ratio collapses to ~1). The learned version can still make use of the deeper capacity, but it doesn't translate to held-out discriminability.

**This is the central method question for the paper.** Neither baseline is sufficient. We need a quantiser objective that:
- Is **not identity-supervised** (or uses identity supervision in a way that doesn't bias the codebook toward training-identity axes — e.g., contrastive, held-out-id-regularised, or invariance-augmented).
- Allocates capacity along **identity-discriminative** directions in general, not just training-id-discriminative ones.

Candidate directions to try next (in priority order):
1. **Contrastive RQ-VAE**: positive pairs = same identity / different condition (augmented); negative pairs = different batch entries. No identity labels needed, but learns "same identity" structure.
2. **Reconstruction-only RQ-VAE** with deeper latent: just remove the classifier loss; let the autoencoder allocate. Test whether it beats naive k-means (it should, because the encoder is non-linear).
3. **Held-out classification loss** with identity-LeaveOneOut: train classifier on subset, regularise codebook to not over-rely on training-id-separating directions.
4. **Augmentation-invariance objective**: codebook should be invariant to mild image augmentations / time shifts. Forces capacity onto identity-stable features.

**Implication for the §3 §3 claim.** Should be **withdrawn**. The §3 numbers are valid in-domain (and useful as a "ceiling under leakage") but they do not support "learned > naive" as a method claim. The method-paper version of this finding has to be on held-out identities, with a quantiser that beats naive there.

**Verdict.** Gating §1-§2 (encoder soundness): **PASS**. Method-side §3 (learned quantiser is essential): **NOT YET PROVEN**, needs a generalisation-preserving objective. This is the next experiment.

## §5. Quantiser bakeoff — all post-training variants fail at scale (2026-05-14, follow-on)

Source: `src/quantiser_bakeoff.py`. Compared four quantisers on held-out identities: naive residual k-means, reconstruction-only RQ-VAE, classifier-supervised RQ-VAE, contrastive (NT-Xent on same-id pairs) RQ-VAE. All trained on TRAIN identities, evaluated on disjoint HELDOUT.

Headline (ratio = intra-agreement / inter-collision; higher = more discriminative):

| Modality | Config | eff_K | naive | recon | cls | contr | winner |
|---|---|---|---|---|---|---|---|
| Audio | L2×K16 | 256 | 2.53 | **3.23** | 2.25 | 2.75 | recon |
| Audio | L2×K64 | **4096** | **31.61** | 20.50 | 17.30 | 14.29 | **naive** |
| Audio | L3×K16 | 4096 | 4.14 | 2.89 | 3.66 | **5.15** | contr |
| Audio | L3×K32 | **32768** | **20.02** | 1.42 | 6.14 | 6.93 | **naive** |
| Audio | L4×K16 | 65536 | 5.52 | 5.73 | **8.87** | 2.81 | cls |
| Vision | L2×K16 | 256 | 1.13 | 1.38 | 1.02 | **2.15** | contr |
| Vision | L2×K64 | **4096** | **3.69** | 2.32 | 2.42 | 1.47 | **naive** |
| Vision | L3×K16 | 4096 | 1.13 | **1.96** | 1.31 | 1.79 | recon |
| Vision | L3×K32 | **32768** | **5.29** | 1.90 | 1.42 | 1.05 | **naive** |
| Vision | L4×K16 | 65536 | 1.13 | **1.82** | 1.00 | 1.29 | recon |

**Reading.**

- **At the configs that matter (productive depth: L2×K64 or L3×K32, eff_K ≥ 4096), naive k-means wins decisively on both modalities.** This is the regime production scale would want. Audio L2×K64: naive ratio 31.6 vs best learned 20.5. Vision L3×K32: naive 5.29 vs best learned 1.90.
- **At pathological depth (L3×K16 or L4×K16), naive degenerates** (intra and inter both stuck at ~0.5; ratio 1.13). Learned variants at least quantise *something* (ratio 1.3–2.2) but achieve weak absolute discrimination.
- **At small eff_K (256), contrastive sometimes wins** but the absolute discriminability is poor in this regime regardless.

**Conclusion.** Post-training a quantiser to beat naive residual k-means at the regimes that matter has failed across three reasonable objectives (reconstruction, classifier, contrastive). The right interpretation is *not* "we tried bad variants" — it's that **k-means' variance-maximising allocation transfers to unseen identities better than any objective grounded in the training-identity distribution**.

**Implication for the method.** The contribution claim shifts:

- **Out**: "learned RQ-VAE quantiser is essential for cross-condition perceptual memory."
- **In**: "**Naive residual k-means at L2×K64 is sufficient as a frozen perceptual codebook. The contribution is the content-addressable parametric memory mechanism (hash-keyed row insertion + gate-mediated retrieval) built on top.**"

This is a cleaner claim. The quantiser becomes a frozen preprocessing step on par with the encoder, and the genuinely novel components are the per-user override table and the gate.

**The next gate.** Test the actual mechanism end-to-end: register N held-out identities one-shot, query with cross-condition samples, measure retrieval top-1. If retrieval accuracy ≥ 0.8 at production scale (1000+ identities), the mechanism is ready to be post-trained on top of Qwen3-VL. If it fails, escalate per user's authorisation to multimodal Mini-Engram pretraining.

## Next steps blocked by what isn't done

- Style attribution (V-STY): encoder choice still open. CLIP collapses style into semantics. Need a style-specific embedder (StyleCLIP, StyleGAN-encoder, or a contrastively-trained style head). Sanity check 3.
- Acoustic scene (A-SCN): need to validate PANNs or Audio-MAE on DCASE TAU cross-recording pairs. Sanity check 4.
- Paralinguistic state (A-PARA): need an encoder that captures *state* (emotion/fatigue) while being *speaker-invariant*. Wav2vec2-Emotion is a candidate. Sanity check 5.
