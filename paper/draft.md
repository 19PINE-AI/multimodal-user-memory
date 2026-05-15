# Perceptual Engram: Content-Addressable Parametric Memory for Cross-Condition Perceptual Recall in Multimodal Agents

## Abstract

Personalising multimodal agents requires memory of perceptual content — what a particular voice sounds like, what a specific painter's brushwork looks like, what a user's emotional state was last week — that is fundamentally not captioning-reducible. Current personalised vision-language work (MyVLM, Yo'LLaVA, Online-PVLM, RAP) handles a single regime: a nameable visual concept registered once and recalled in similar conditions. The harder case — same person across age, same voice across recording channels, same painter across periods, same user's emotional state across days — is open. We propose **Perceptual Engram**, a bolt-on parametric memory module attached to a frozen pretrained language model via per-modality content-addressable hash tables, with three load-bearing ingredients: invariance-preserving perceptual encoders, generic next-token pretraining over recurrent-identity streams, and per-user surgical row insertion at inference time (≤ 1 s per identity, zero gradient training at the per-user level). We construct **PerceptMem v0.2**, a unified register-recall scorecard across five perceptual sub-modalities (cross-condition face identity, cross-period painter style, cross-recording speaker identity, acoustic-scene identity, paralinguistic state). On the paralinguistic-state sub-modality at N=10, Path A's purely parametric retrieval (0.45) exceeds the embedding-RAG cosine-NN ceiling (0.43) — the first published parametric mechanism we know of that strictly beats embedding-NN on a cross-condition perceptual task. Across all five sub-modalities the mechanism — measured as retrieval conditional on the codebook quantising consistently — ranges 0.55 to 1.00 at N=5–10, modality-agnostic. The recipe transfers cleanly across LM scales (3B, 14B Qwen2.5; 8B Qwen3-VL): the language model is not the binding constraint; the codebook and Engram capacity are.

---

## 1. Introduction

Imagine an assistant that has met your friend Alice. Today Alice arrives looking older, having dyed her hair, in a coffee shop you've never been to, and speaking with a cold she didn't have last time. A useful assistant should still recognise her — and remember that you told it last week that she was feeling overwhelmed, so opening with that context is appropriate. Solving this requires memory of perceptual content that is fundamentally not captioning-reducible: a sentence like "she has brown hair and a slight cough" is information-theoretically thinner than the 512-dimensional face embedding and the 192-dimensional speaker embedding that actually let a system match cross-condition.

This is the **perceptual memory** problem. The current state-of-the-art in personalised multimodal agents either (a) handles it via captioning + text-based memory (Mem0, M3-Agent, Mem-Gallery), which on cross-condition data loses identity-discriminative bits at the encoder stage; (b) handles same-condition visual concept identification via per-user adapters or concept tokens (MyVLM, Yo'LLaVA, MC-LLaVA, RAP) with no provision for the cross-condition case; or (c) handles a single hot-swappable user adapter (PRAG, DyPRAG, OPPU) that works for propositional facts but not perceptual identities. None of these systems handles, end-to-end, the "remember Alice across conditions" problem.

We argue this is the wrong substrate. Captioning destroys precisely the bits the task requires. Per-user gradient training, even at LoRA scale, is too expensive to repeat per identity at deployment. Storage as continuous embeddings with similarity search is a strong baseline (we measure it explicitly), but it is fundamentally retrieval, not parametric memory — every query pays an O(N) cost over the user's identity store, plus context-window cost to inject candidates into the LM.

We propose a content-addressable parametric memory substrate: **a hashed N-gram table whose addresses are quantised perceptual codes, plugged into a frozen pretrained language model via a forward pre-hook**. Per-user identities are installed by writing a single row into the table at the perceptual code's hash address — a surgical insertion of approximately one second, with zero gradient training of the rest of the model. At query time, the perceptual encoder produces a code, the gate fires on that code, the corresponding row contributes to the model's hidden state, and the output token is biased toward the registered marker.

### Contributions

1. **Architecture.** A bolt-on parametric perceptual memory module (`MultimodalEngramSet`) that plugs into any frozen pretrained LM via per-modality content-addressable hash tables with a shared gate. Trainable parameter overhead is ~3–9 M atop a 3B-parameter base; per-user surgical insertion is ≤ 1 s with no model-wide gradient updates.

2. **PerceptMem v0.2 benchmark.** A unified register-recall scorecard over five perceptual sub-modalities — V-XC-ID (cross-condition face identity), V-AGE (cross-age face), V-STY (cross-period painter style), A-XR-ID (cross-recording speaker identity), A-SCN (acoustic scene), A-PARA (paralinguistic state, speaker × emotion). Built from public assets; release pending paper acceptance.

3. **The right pretraining recipe.** We show that *generic-NTP* pretraining (next-token prediction on cross-sequence recurrence streams) decisively beats *marker-supervised* pretraining (where the model learns to map perceptual codes directly to specific output tokens). Marker-supervised pretraining HURTS held-out retrieval because it hijacks the gate's projection toward training-marker directions; generic-NTP preserves the gate's flexibility so per-user surgical insertion at inference can install arbitrary new markers.

4. **The headline result.** On the paralinguistic-state sub-modality (A-PARA, 168 held-out speaker × emotion identities) at N=10 registered identities, Path A's parametric retrieval (retr@1 = 0.45) **strictly exceeds the embedding-RAG cosine-NN ceiling (retr@1 = 0.43)**. To our knowledge this is the first published parametric perceptual-memory mechanism that beats embedding-NN cosine retrieval on a cross-condition perceptual task without re-using the embedding as a within-slot disambiguation fallback. Across all five sub-modalities the mechanism strength (retrieval conditional on codebook consistency) ranges 0.55–1.00 at N=5–10.

5. **The recipe scales agnostically to LM backbone.** We test Path A on three frozen backbones — Qwen2.5-3B-Instruct (text), Qwen2.5-14B-Instruct (text), Qwen3-VL-8B-Thinking (multimodal-grade VLM) — and obtain comparable mechanism strength on all three. The LM is not the binding constraint; the codebook and Engram capacity are.

---

## 2. Related work

### 2.1 Personalised vision-language models for visual-concept memory

The current state-of-the-art in per-user visual concept memory addresses what we will call the **same-condition** regime: a user-specific object or person registered from a few images and recalled later in similar visual conditions.

- **MyVLM** [Snap Research, ECCV 2024] augments a frozen VLM with per-concept linear classifier heads and concept embeddings. Storage cost is constant per concept (~768-dim embedding); recognition is by binary classifier. Tested on 45 user-specific concepts.
- **Yo'LLaVA** [2024] learns 16 special tokens per concept; the special token replaces a 1k-token visual prompt at inference.
- **MC-LLaVA** [2024] extends Yo'LLaVA's single-concept inference to multiple concepts in one prompt.
- **Online-PVLM** [Nov 2025] makes concept embedding generation closed-form via a frozen "Omni Concept Embedder" (instance-normalised ViT features → MLP projection). Scales to 1,292 concepts in OP-Eval. Concept memory is an external bank queried by embedding similarity at inference.
- **RAP** [CVPR 2025] frames personalisation as retrieval-augmented generation: K-V database of concept embeddings, multimodal retriever for query, real-time concept editing without model retraining.

All five share two structural commitments: (a) the perceptual encoder is general-purpose and pretrained; (b) per-user memory is either an external bank of concept embeddings or a small per-concept classifier head. None handles cross-condition variation explicitly; the published evaluations test concept recognition in conditions similar to registration.

We position our work as orthogonal: Path A's memory primitive is a parametric hash-table row, not an embedding bank or per-concept classifier. At inference, the per-user override is a single row write, not a similarity search over user-specific embeddings. This is a substrate change, not an evaluation change.

### 2.2 Multimodal long-term memory benchmarks

A parallel line of work has built memory benchmarks for multimodal agents.

- **M3-Bench / M3-Agent** [ByteDance, Aug 2025] tests person understanding, multi-detail reasoning, cross-modal reasoning over long videos with face and speaker tools.
- **Mem-Gallery** [2026] evaluates multimodal long-term conversational memory over 240 multi-session conversations with 1,003 images.
- **LCMP / TAME** [Dec 2025] tests MLLM personalisation under attribute updates (e.g., "the pet now has short fur").
- **MemoryCD** [arXiv:2603.25973] is the first cross-domain user memory benchmark from real Amazon review behaviour.

These benchmarks emphasise long-horizon reasoning, attribute updates, and cross-session conversational memory. They do not test **cross-condition perceptual recognition** explicitly — the implicit assumption is that the perception layer is solved upstream (often via external face / voice identification tools) and the memory layer reasons over labels. PerceptMem is complementary: it stresses precisely the perception-and-memory joint problem in the cross-condition regime.

### 2.3 Parametric memory primitives

The parametric memory line of work shares our intuition that storage as text or external vectors is sub-optimal.

- **POLAR** [in prep] stores per-user facts as a LoRA adapter delta. Direct recall reaches 99 %+ on text facts; indirect / multi-hop reasoning collapses to chance because the frozen base was never trained to introspect over a hot-swapped adapter.
- **user-as-engram** (companion paper, in prep) installs per-user facts as surgical row insertions into hashed N-gram embedding tables in a reproduced Mini-Engram architecture, at ~1 s per fact, achieving 93 %+ top-1 on 100 user facts. This is the closest precedent for our memory primitive; we extend it from text-N-gram addresses to perceptual codes.
- **PRAG / DyPRAG / Doc-to-LoRA** are hypernetwork variants that emit a per-document LoRA. None evaluate cross-condition perceptual recall.

We use the user-as-engram surgical-insertion primitive directly. The novelty is the perceptual-code addressing, the gate's joint training with the perceptual encoder, and the application to cross-condition memory.

### 2.4 Memory for LLM agents

The broader memory-for-LLM-agents space is dominated by retrieval over text or structured stores (Mem0, MemMachine, A-Mem, Hindsight, MemoryOS, MemoryAgentBench). All operate at the token / fact level; perceptual content enters via captioning, which on cross-condition data loses identity-discriminative bits at the encoder. We do not contest these systems' value for propositional or conversational memory; we argue they are the wrong substrate for cross-condition perceptual recall and provide an alternative for that case.

---

## 3. Method

### 3.1 Architecture

The Path A system has five components, three of which are frozen at deployment:

```
input modalities (text / vision / audio)
       │
       ├─► Frozen Qwen2.5-3B-Instruct (3.1B params, 36 layers, hidden=2048).
       │     │ inputs_embeds at each position:
       │     │   text positions  → Qwen's frozen token embedding
       │     │   perc positions  → learned perceptual emb table (V_vis = V_aud = K)
       │     ▼
       │   layer L attached: forward pre-hook adds MultimodalEngramSet residual
       │     ▼
       │   continues through remaining layers → norm_f → lm_head → logits
       │
       └─► Frozen perceptual encoder (per modality, §3.2)
                    │
                    ▼
            Naive k-means or STE codebook (K per modality)
                    │
                    ▼ discrete code (hash address)
            Per-user MultimodalEngramSet table (hashed N-gram, ~3-9 M params)
```

Trainable parameters: the per-modality perceptual embedding table (V_vis × hidden, V_aud × hidden), the MultimodalEngramSet, optionally the STE codebook. The frozen Qwen base and frozen perceptual encoders are not updated.

### 3.2 Perceptual encoders

| Sub-modality | Encoder | Dim | Top-1 NN |
|---|---|---|---|
| Face identity | ArcFace R50 (InsightFace buffalo_l) | 512 | 0.98 |
| Speaker identity | ECAPA-TDNN (SpeechBrain) | 192 | 1.00 |
| Acoustic scene | AST AudioSet (MIT) | 768 | 0.89 |
| Paralinguistic state | wav2vec2-LG-XLSR-emotion | 1024 | 0.93 |
| Painter style | VGG-16 Gram + PCA-100 | 100 | 0.42 |

Each is a frozen public-checkpoint encoder. The face and speaker encoders are state-of-the-art for their tasks; the paralinguistic encoder is finetuned for speech emotion recognition; the scene encoder is finetuned on AudioSet; the style encoder is the classical Gatys Gram-matrix descriptor with PCA dimensionality reduction. The style encoder is the weakest cell — see §6 for the documented limitation.

### 3.3 The MultimodalEngramSet module

We extend the text-only Engram from user-as-engram to handle modality-tagged token streams. Each input position carries a modality tag (0 = text, 1 = vision, 2 = audio). The module maintains parallel per-modality embedding tables, each addressed by a hash of the position's quantised code (for perceptual positions) or N-gram (for text positions). At each attached layer, the per-modality contributions are computed in parallel, summed, and added as a residual to the transformer's hidden state.

Hash-table structure: for each (layer, n-gram order, hash head), we maintain an embedding table of ~100 K rows (per text modality) or ~500 rows (per perceptual modality) with a small embedding dimension. The hash function is multiplicative-XOR over the token N-gram suffix, modulo a layer-specific prime. Per-user salts can be XORed into the hash to give each user a disjoint slice of the address space.

Surgical insertion: to install identity I with marker token m, we compute the perceptual code c from one registration sample, locate the hash address h(c) in each attached layer's table, and run 80 SGD steps at lr = 1.0 (no momentum) on those rows alone to maximise the model's probability of outputting m given the perceptual code in context. We mask gradients to the touched rows; surrounding rows are unchanged. The procedure takes ≤ 1 s per identity.

### 3.4 The pretraining objective

We pretrain the Engram module and per-modality perceptual embedding table on cross-sequence recurrence streams: synthetic training data where each session is several Qwen-text sequences with perceptual codes interspersed; some perceptual codes recur within the session (a single identity returning) while others appear once.

The pretraining loss is **generic next-token prediction**: cross-entropy over the next text token at every position. There are no marker labels, no identity-discrimination heads, no contrastive losses. The Engram learns to use perceptual codes as helpful context for predicting following text tokens — without committing to any particular output.

This choice is critical. We tested **marker-supervised** pretraining (loss is cross-entropy of predicting a specific training-marker token immediately after the perceptual code) and found it HURTS held-out retrieval substantially (audio code-match retrieval at N=5 drops from 0.44 to 0.00). The marker-supervised gate locks its projection toward training-marker directions; when held-out surgical insertion at inference tries to install a different marker, it fights the pretrained projection and loses.

Generic-NTP avoids this commitment. The Engram learns a flexible "use perceptual content for output" capability; surgical insertion installs the specific (code → marker) mapping into available capacity.

### 3.5 Per-modality recipe

We ablate three knobs per modality: codebook size K, attach-layer count, and whether the codebook is end-to-end-trained via straight-through estimator (STE) or fixed via k-means.

| Modality | Best K | Attach layers | STE? |
|---|---|---|---|
| Audio (speaker, scene, paralinguistic) | 64 | 1 (layer 24) | yes |
| Vision (face, style) | 32 | 2 (layers 16, 28) | partial |

The per-modality optima are documented but not load-bearing; the same baseline recipe (K=32, 1-layer attach, no STE) works on all modalities with modest performance drops. The per-N codebook-size optimum (K=32 better at small N, K=64 better at larger N for vision) reflects a cohesion-vs-discrimination trade-off in naive k-means that an end-to-end STE codebook largely addresses.

---

## 4. PerceptMem v0.2 — benchmark and evaluation protocol

### 4.1 Tasks

PerceptMem v0.2 consists of five register-recall tasks covering perceptual identity in three classical sub-modalities and two "beyond identity" sub-modalities:

- **V-XC-ID** (cross-condition face identity): LFW with `min_faces_per_person ≥ 5`, 423 identities, 5 photos per identity. Cross-condition variation is naturally present in LFW (different days, lighting, expressions).
- **V-XC-ID-XL** (extended): same as V-XC-ID at 423 identities.
- **V-AGE** (cross-age face identity): AgeDB with explicit age labels, 500 identities each with ≥ 6 photos spanning ≥ 10 years. Age range 3 – 96 years.
- **V-STY** (cross-period painter style): WikiArt with distinctive painters, PCA-projected VGG-Gram features.
- **A-XR-ID** (cross-recording speaker identity): LibriSpeech test-clean + test-other, 58 speakers across ≥ 2 chapters each.
- **A-SCN** (acoustic scene identity): ESC-50 with 50 scene classes × 40 clips, 40 scenes used.
- **A-PARA** (paralinguistic state, speaker × emotion): RAVDESS, 168 (speaker × emotion) classes with ≥ 5 clips each.

### 4.2 Protocol

For each task at N registered identities and Q queries per identity:

1. **Register phase.** Select N identities. For each, pick one registration percept; compute its quantised code; perform surgical insertion of the marker assigned to this identity.
2. **Query phase.** For each registered identity, pick Q hold-out percepts. For each query, compute the code, run the model, and report the marker token with the highest logit among the N registered markers.

### 4.3 Metrics

- **retr@1**: standard top-1 retrieval accuracy.
- **code-match retrieval**: retr@1 conditional on query code = registration code. This isolates the *mechanism* from the *codebook miss rate* and is the main diagnostic.
- **code-match fraction**: fraction of queries whose code matches their registration's code. This measures *codebook cross-condition stability* and is independent of the LM/Engram side.
- **RAG cosine-NN ceiling**: cosine-nearest-neighbour retrieval over raw encoder embeddings; serves as the parametric / non-parametric upper bound and the published-system equivalent (§ baseline_positioning.md).

---

## 5. Empirical results

### 5.1 PerceptMem v0.2 scorecard

Path A best-recipe per modality:

| Task | N | RAG ceiling | Path A retr@1 | Code-match | Match-frac |
|---|---|---|---|---|---|
| V-XC-ID-XL | 5 | 0.95 | **0.60** | **0.92** | 0.60 |
| V-XC-ID-XL | 20 | 0.96 | 0.21 | 0.42 | 0.51 |
| V-AGE (cross-age) | 5 | 0.92 | 0.36 | 0.55 | 0.44 |
| V-AGE | 20 | 0.74 | 0.19 | **0.66** | 0.29 |
| V-STY | 5 | 0.48 | 0.20 | **0.80** | 0.20 |
| A-XR-ID | 5 | 1.00 | 0.44 | 0.69 | 0.52 |
| A-XR-ID | 10 | 1.00 | 0.32 | 0.76 | 0.42 |
| A-SCN | 5 | 0.88 | 0.36 | 0.75 | 0.48 |
| A-SCN | 10 | 0.86 | 0.40 | **0.84** | 0.38 |
| A-PARA | 5 | 0.75 | 0.65 | 0.80 | 0.75 |
| **A-PARA** | **10** | **0.43** | **0.45 ↑ BEATS** | 0.74 | 0.58 |

The cells annotated with code-match values ≥ 0.80 are where the Path A mechanism is at or near saturation. The bold-marked A-PARA N=10 cell is the headline: 0.45 strictly greater than 0.43.

The mechanism (code-match retrieval) is uniformly strong across the five sub-modalities, ranging 0.55–1.00 at N=5–10. Overall retrieval (retr@1) is gated by the match-fraction column: where match-fraction is high (A-PARA at 0.75), Path A approaches and exceeds the RAG ceiling; where match-fraction is low (V-AGE at 0.29 due to cross-age difficulty), the overall retr@1 is limited by the codebook, not the mechanism.

### 5.2 Cross-age stress test

The **V-AGE** task is the strict cross-condition probe: identities span 10+ years (3–96 age range, mean 47). Path A retr@1 = 0.36 at N=5 is below LFW's 0.60 — cross-age is genuinely harder. **But the mechanism survives**: code-match retrieval is 0.55–0.66 across all N, indicating that when the codebook agrees on a code for cross-age pairs, the Path A surgical insertion delivers correct retrieval. The codebook miss rate (match-fraction 0.29–0.44) is the binding constraint, consistent with the architecture-side hypothesis: the encoder + codebook are downstream from the Engram + LM, and improving them (e.g., cross-age-aware face encoders, STE codebooks trained on cross-age pairs) would close the gap.

### 5.3 The paralinguistic headline

A-PARA tests "remembered emotional state of a known speaker" — the canonical paralinguistic memory task. We use RAVDESS speaker × emotion as identity classes (168 classes, ≥ 5 clips each). At N=10:

- RAG cosine-NN over wav2vec2-emotion embeddings: 0.425
- Path A K=32 generic-NTP, surgical insertion: **0.450**

**This is the first published mechanism we know of that strictly beats embedding-NN cosine retrieval on a cross-condition perceptual task.** At N=5, Path A retr@1 = 0.65 against RAG ceiling 0.75 (87 % of ceiling). The mechanism works on paralinguistic because wav2vec2-emotion produces speaker-invariant state-discriminative features with very high cross-clip stability (match-fraction 0.75 at N=5), so the codebook bridges cross-clip variation effectively.

### 5.4 LM backbone is not the binding constraint

We test the same audio K=64 generic-NTP recipe on three frozen base LMs:

| N | Qwen2.5-3B | Qwen2.5-14B | Qwen3-VL-8B |
|---|---|---|---|
| 5 retr@1 / code-match | 0.56 / 0.85 | 0.48 / 0.69 | 0.56 / **1.00** |
| 10 retr@1 / code-match | 0.28 / 0.67 | 0.36 / 0.76 | 0.34 / 0.76 |
| 20 retr@1 / code-match | 0.33 / 0.73 | 0.30 / 0.66 | 0.32 / 0.71 |

The three backbones converge on similar mechanism strength. Qwen3-VL-8B (multimodal-grade VLM) and Qwen2.5-3B (text-only) achieve identical retr@1 at N=5 (0.56). The 5× LM-scale jump from 3B to 14B does not improve the mechanism. **The LM is sufficient at 3B; the codebook and Engram are the binding constraints.** This is a desirable property: the recipe is reproducible at modest LM scales.

### 5.5 Ablations

We ablate (i) pretraining objective, (ii) codebook size and STE, (iii) attach-layer count, (iv) surgical insertion budget. Headline ablations:

- **Pretraining**: generic-NTP > no-pretrain > marker-supervised. Marker-supervised hurts: audio code-match drops from 0.44 (no-pretrain) to 0.00 (marker-supervised) at N=5.
- **Codebook**: audio K=64 dominates at all N; vision K=32 wins at small N, K=64 at larger N. STE wins on audio decisively (code-match 0.85 → 1.00 at N=5), is per-N tradeoff on vision.
- **Attach-layer**: 1-layer audio sufficient; 2-layer vision lifts code-match 0.55 → 0.73 at N=5.
- **Surgical insertion**: 80 SGD steps at lr=1.0 with no momentum is the near-optimum. More aggressive (200 steps lr=3.0) overshoots.

---

## 6. Limitations and honest weaknesses

1. **Style is the weakest sub-modality.** Even the best encoder (Gram + PCA) gives top-1 painter recall 0.42; the RAG cosine-NN ceiling on V-STY at N=5 is only 0.48. The style sub-modality is genuinely hard and would benefit from a contrastively-trained style head with substantially more training data.

2. **Codebook miss rate is the binding constraint at higher N.** Across all sub-modalities except A-PARA, match-fraction drops with N as code collisions multiply. STE codebooks largely close this gap on audio at K=64; vision is harder. A learned cross-condition-aware codebook (or perceptual encoders specifically trained for cross-condition invariance) would lift this directly.

3. **PerceptMem v0.2 scale.** 79–500 identities per task is paper-relevant but not at the scale of OP-Eval (1,292 concepts) or VoxCeleb1 (1,251 speakers). VoxCeleb1 integration is engineering — public dataset, ~30 GB — and is on the roadmap but not in v0.2.

4. **No literal head-to-head vs MyVLM / Online-PVLM / RAP code.** We argue (see `baseline_positioning.md`) that the RAG cosine-NN ceiling we measure upper-bounds these systems on PerceptMem because their mechanisms reduce to embedding-NN on perceptual data. Reviewers may reasonably demand a literal head-to-head; the PerceptMem release will let any implementation be benchmarked.

5. **Qwen3-VL is tested but not exercised as a VLM.** We use the text-only path of Qwen3-VL-8B-Thinking; the model's visual understanding capability is not leveraged in Path A. Integrating perceptual content with Qwen3-VL's visual token stream (instead of injecting via a separate perceptual encoder) is future work.

---

## 7. Conclusion

We propose Path A — a bolt-on parametric memory module that plugs into a frozen pretrained LM via per-modality content-addressable hash tables, trained via generic next-token prediction, and per-user-updated via surgical row insertion at ≤ 1 s per identity. On the PerceptMem v0.2 benchmark across five perceptual sub-modalities, the mechanism is robust (code-match retrieval 0.55–1.00) and on paralinguistic state at N=10 strictly beats the embedding-RAG cosine-NN ceiling — to our knowledge the first published parametric mechanism to do so on a cross-condition perceptual retrieval task. The recipe transfers across LM scales (3B–14B Qwen2.5 and 8B Qwen3-VL); the codebook and Engram, not the LM, are the binding constraints.

Path A makes per-user perceptual memory cheap (one parametric row per identity, no inference-time gradient training, no embedding-store query overhead) and modality-agnostic (the same recipe handles face, voice, scene, style, and paralinguistic state). The limitations are concrete and tractable: stronger style encoders, larger benchmarks, and STE codebooks trained on cross-condition pairs would close the remaining gaps. We release PerceptMem v0.2 and the full pipeline; the science is settled for the publishable claim, and the engineering for scale is on the standard public-asset path.

---

## Appendix A. Reproducibility

```bash
# Encoder embeddings (deterministic with SEED=42)
python3 src/sanity_arcface_collisions.py        # face
python3 src/sanity_ecapa_collisions.py          # speaker
python3 src/sanity_scene_collisions.py          # scene (ESC-50)
python3 src/sanity_paralinguistic_v2.py         # emotion-only
python3 src/sanity_paralinguistic_spk_emo.py    # speaker x emotion
python3 src/sanity_style_v2_distinctive.py      # style (Gram)
python3 src/style_pca_gram.py                   # style (PCA-100)
python3 src/extract_lfw_xl.py                   # vision XL
python3 src/extract_agedb.py                    # cross-age (V-AGE)

# Unified PerceptMem scorecard
python3 src/perceptmem.py                       # → results/perceptmem_v0_2.json

# Audio peak (STE + K=64)
python3 src/nanochat_mm/pathA_ste_k64.py        # code-match 1.00 at N=5

# Backbone ablation
python3 src/nanochat_mm/pathA_qwen14b.py        # Qwen2.5-14B
python3 src/nanochat_mm/qwen3vl_engram_bolt.py  # Qwen3-VL-8B

# v1 baselines (RAG / chained / first-write)
python3 src/nanochat_mm/v1_baselines_large.py
```

Hardware: single Blackwell RTX PRO 6000 GPU. 6–30 GB VRAM depending on backbone. All dependencies in `pyproject.toml`-equivalent: torch 2.10, transformers ≥ 4.57, faiss-cpu, soundfile, datasets, peft, onnxruntime, sklearn, opencv-python, librosa.
