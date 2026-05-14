# Escalation decision — from post-training to from-scratch pretraining

**Date:** 2026-05-14 (same session as the sanity checks and bakeoff)
**Trigger:** Post-training results in `notes/sanity_findings.md` §5 and `results/engram_retrieval.json`.
**Authorisation:** User said: *"If you find all these methods yield negative result, you may go back to user-as-engram methods by pretraining your own models."*

## What we tried (all post-training, all failed at production scale)

1. **Naive residual k-means quantiser + Engram table.** Best vision retrieval@1 = 0.48 vs embedding-RAG baseline 0.96. Massive hash collisions; chained disambiguation just falls back to embedding NN.
2. **Reconstruction-only RQ-VAE quantiser.** Held-out ratio worse than naive at all productive depths.
3. **Classifier-supervised RQ-VAE.** Overfits to training-identity axes; doesn't transfer.
4. **Contrastive NT-Xent RQ-VAE.** Best at small eff_K (256), loses at scale.

## Why post-training failed (the fundamental issue)

In a registration-query setup, the registered embedding has ONE code, and queries from the same identity under different conditions have codes that match the registered one only **40-70% of the time** with any frozen codebook fit to ArcFace/ECAPA embeddings. The other 30-60% of queries either:
- Land in a slot that has no registered identity (miss), or
- Land in a slot that has a *different* identity registered (collision).

The §2 sanity check measured **intra-PAIR agreement** across all same-identity pairs; the retrieval task is asymmetric (single registration, many queries), and the asymmetry exposes the codebook's instability under cross-condition variation.

A frozen codebook cannot fix this. The codebook's allocation is determined by embedding-space variance (k-means) or by training-identity discrimination (supervised), neither of which is the right signal. The right signal is **the LM's downstream objective of using stored memory** — and that signal is only available during joint LM + Engram training.

This is the architectural lesson DeepSeek-Engram embodies in the text setting and that user-as-engram reproduces nano-scale: **the codebook, the gate, and the memory table must be trained together with the LM**. Bolt-on post-training is insufficient.

## What we keep from this work

- **Encoder soundness** (sanity §1, §2): ArcFace and ECAPA-TDNN are validated as invariance-preserving feature extractors. We use them frozen.
- **Embedding RAG baseline numbers**: 0.96 vision / 1.0 audio at N=5–20. These set the win condition for from-scratch pretraining.
- **Identity-disjoint evaluation protocol**: register-query with held-out identities is the right task. Reused as-is.
- **Honest failure of the post-training path**: this is genuinely a paper-worthy negative result. It's what motivates the from-scratch architecture in the paper's intro. Without this experiment we'd be hand-waving; with it we have a clean argument.

## What we need to build (the from-scratch escalation)

We follow the user-as-engram nano-scale-reproduction template. Adapt to multimodal:

### Architecture: Mini-Multimodal-Engram

A small (≤625M scaling-params) decoder-only multimodal LM with the following components, **all trained from scratch jointly**:

```
input modalities (text / vision / audio)
       │
       ▼
[frozen perceptual encoders]: ArcFace, ECAPA-TDNN, …  produce identity-bearing embeddings
       │
       ▼
[learned quantiser]: small VQ head per modality, codebook entries trained end-to-end
       │
       ▼
[Engram embedding table]: hashed N-gram (text) + hashed perceptual-code (vision/audio)
       │      ▲
       │      │ ← surgical insertion at inference (per user)
       ▼      │
[Engram gate] decides whether to fire on (context, key)
       │
       ▼
[decoder-only transformer] combines hidden state + Engram lookup output → next-token prediction
```

The crucial difference from the post-training attempt: the **quantiser, gate, and table are part of the LM's computational graph during pretraining**. The next-token-prediction loss flows back into the codebook through the gate's straight-through estimator, training the codebook to allocate codes that the LM finds useful for prediction.

### Pretraining corpus

Following user-as-engram's ClimbMix-400B template, but multimodal:
- **Text**: a slice of FineWeb / ClimbMix (~1-3B tokens).
- **Vision-text**: MMC4 interleaved or LAION-style image-text pairs (~500M-1B tokens).
- **Audio-text**: paired data — LibriSpeech transcripts (~960h), WavCaps, AudioCaps (~500M-1B audio-equivalent tokens).
- **Recurrence corpus** (critical for the gate): video / podcast / show data where the same identity appears across multiple sessions. Candidates: VoxCeleb (speaker recurrence in interviews), Ego4D (visual identity recurrence in egocentric), VoxConverse (multi-speaker conversations).

Estimated total: 3-7B multimodal tokens. Targets Karpathy 12 t/p for the scaling-params budget, same heuristic as user-as-engram FINDINGS.

### Hardware and timeline

Single Blackwell RTX PRO 6000 (102GB). Based on user-as-engram precedent:
- `engram_d12_w1280_optimal` (625M params, 3.34B tokens, ~10.5h)
- A multimodal version will run slower (more compute per token due to encoder passes) and need more tokens (multimodal data is more diverse).
- Realistic estimate: **3-7 days for the pretrain**, after corpus preparation. Corpus prep itself: **2-3 days**.

### Phase plan (revised v2 of `research_plan.md` §9)

| Phase | Days | Output |
|---|---|---|
| 0a. Sanity checks 1-2 + post-training bakeoff (done) | 1 | This document |
| 0b. **Adopt naive k-means as the codebook for post-training comparison baselines** | 0.5 | Frozen-codebook baseline numbers (already in `results/engram_retrieval.json`) |
| 1. Multimodal corpus pipeline | 3-5 | Tokeniser + interleaved-format dataset on disk (1.8TB free, comfortable) |
| 2. Architecture: extend nanochat (`~/user-as-engram/nanochat`) with vision/audio token streams | 3-5 | `mini_multimodal_engram` module |
| 3. Mini-Engram pretrain @ d12 | 3-5 | First multimodal Engram checkpoint |
| 4. Sanity: zero-shot text generation OK; per-modality recurrence learned (qualitative) | 1 | Smoke test passing |
| 5. PerceptMem benchmark construction | 3 | V-XC-ID + A-XR-ID + ≥1 of {V-STY, A-SCN, A-PARA} |
| 6. Surgical insertion experiments at scale | 3 | Headline retrieval@1 vs embedding-RAG on PerceptMem |
| 7. Comparison vs Mem0, M3-Agent-style, MyVLM-style, Online-PVLM, RAP, TAME | 5 | Full results table |
| 8. Paper writing | ~14 | Submission |

Total: ~6 weeks engineering + ~2 weeks writing ≈ 2 months end-to-end. Faster than the v1 plan's 4 months because we now know the answer: post-training is the wrong substrate.

### The win condition

PerceptMem cross-condition identity retrieval@1, vs embedding-RAG baseline (which post-training Engram failed to beat at 0.96):

- **Pretrained Mini-Multimodal-Engram with surgical insertion** ≥ embedding-RAG retrieval@1 at N=1000 identities, AND uses **zero context tokens** at query time (the parametric memory advantage), AND insertion ≤ 1s per identity (matching user-as-engram's surgical insertion budget).

If we can hit this, the paper writes itself. If we can't even *match* embedding-RAG with the pretrained version, the deeper question is whether perceptual content is parametrically storable at all — and that's a paper too, just a different one.

### Risks specific to from-scratch pretraining

1. **Multimodal pretrain at this scale is harder than text-only.** Vision-language interleaved data is messier than ClimbMix; audio-text alignment is harder still. We may underperform the corpus quality bar and the model's multimodal capability will lag.
2. **Recurrence corpus may be too small.** The gate's "fire on recurrent identity" behaviour requires *long-range* recurrence in training data. VoxCeleb has tens of seconds per interview clip but the recurrence across episodes of a podcast or seasons of a show is what we really need.
3. **Compute budget overrun.** A 625M-param multimodal pretrain may need 10B+ tokens to be coherent, pushing into 1-week wall-clock territory.
4. **The post-training failure is a fundamental result.** If even the pretrained version doesn't beat embedding-RAG at perceptual retrieval, the conclusion may be that parametric perceptual memory is not better than retrieval for the cases we tested. That's still a paper, but a defensive one.

### Immediate next actions

1. Start corpus pipeline. Decide on tokeniser (reuse nanochat RustBPE 32768 from user-as-engram) and visual/audio tokeniser. Encodec for audio is already cached; Cosmos-Tokenizer-DI8x8 for vision is cached.
2. Sketch the multimodal-engram architecture as a fork of user-as-engram's `nanochat`. Add vision-token + audio-token streams; route to Engram table via either modality-specific hash spaces or a unified hash space with modality-tag bytes.
3. Reach out to authors of M3-Agent, Mem-Gallery, RAP for benchmark engagement *before* spending the pretrain budget — make sure the benchmark direction we'd be optimising for is one those communities will engage with.
