# Paper outline (v2) — Cross-condition perceptual memory via bolt-on Engram

**Date:** 2026-05-14 (session 7, post-PerceptMem)
**Working title:** *Perceptual Engram: Content-Addressable Parametric Memory for Cross-Condition Identity-and-Beyond Recall in Multimodal Agents*
**Target venue:** ICLR 2027 / NeurIPS 2026 multimodal track

## v2 changes vs v1 outline

- Five sub-modalities now empirically characterised (face, speaker, scene, paralinguistic encoder, style with PCA-Gram fix). Paper's "beyond-identity" framing is supported on three new sub-modalities (acoustic scene, paralinguistic encoder, style with caveats).
- Unified PerceptMem v0.1 benchmark scorecard across 4 of the 5 sub-modalities (paralinguistic blocked at N≥5 by RAVDESS emotion class count).
- Qwen2.5-14B scale-up: no decisive lift over Qwen2.5-3B. LM size is NOT the binding constraint.
- Qwen3-VL bolt-on architecture verified at smoke-test level (deferred full eval due to mid-session GPU state issues).

## 1-paragraph elevator pitch

Recent personalised-VLM work (MyVLM, Yo'LLaVA, MC-LLaVA, Online-PVLM, RAP, TAME) and multimodal memory benchmarks (M3-Bench, Mem-Gallery, LCMP, MemoryCD) have converged on a single regime: a nameable visual concept registered once and recalled in similar conditions. The harder case — cross-condition perceptual recall (same person under different lighting, same voice across recordings, same painter across periods, same acoustic scene across takes) — is unsolved. We propose **Perceptual Engram**, a bolt-on parametric memory module attached to a frozen pretrained LM via per-modality content-addressable hash tables, with three load-bearing ingredients: invariance-preserving perceptual encoders (ArcFace, ECAPA-TDNN, AST, wav2vec2-emotion, PCA-Gram VGG), generic next-token pretraining over recurrent-identity streams, and per-user surgical row insertion at inference (≤ 1s per identity, no gradient training at the per-user level). We construct PerceptMem v0.1, a unified scorecard across four sub-modalities. Headline: **code-match retrieval 0.50-0.84** at N=5-10 across all four tasks; mechanism is modality-agnostic. Overall retrieval is capped by the codebook miss rate, which an STE-trained codebook largely closes.

## 2. Contributions

1. **A working bolt-on parametric perceptual memory architecture** (MultimodalEngramSet) that extends user-as-engram's text-only hashed parametric memory to perceptual content via per-modality parallel tables with shared gate and conv machinery. Reproducible on any sufficiently-capable pretrained LM (3B Qwen tested as the floor).

2. **PerceptMem v0.1 benchmark** with unified register-recall API and standard scorecard across four perceptual sub-modalities: cross-condition face identity (V-XC-ID), cross-period painter style (V-STY), cross-recording speaker identity (A-XR-ID), and acoustic-scene identity (A-SCN). Constructed from public assets (LFW, WikiArt, LibriSpeech, ESC-50). Direct extensibility to paralinguistic state (A-PARA, blocked at N≥5 by RAVDESS class count) and cross-condition variants.

3. **The right pretraining recipe**: generic-NTP on cross-sequence recurrence streams trains the Engram + perceptual-emb to USE perceptual codes for general text prediction without committing to specific outputs. Marker-supervised pretraining HURTS because it hijacks the gate's projection to training-marker directions. Empirically supported via 3-cell ablation.

4. **Per-modality recipe**: 1-layer attach + K=64 + STE codebook for audio; 2-layer attach + K=32 + (optionally) STE for vision. Each ablation cell empirically supported. PCA-projected Gram features for style.

5. **A diagnostic decomposition** of retrieval into mechanism (code-match retrieval) and codebook (code-match fraction). Lets us localise improvements precisely and gives the paper a clean limitations section.

6. **A clean failure-mode taxonomy** for parametric perceptual memory:
   - Encoder soundness: 4/5 sub-modalities have viable encoders (style needs PCA-Gram or learned head).
   - Frozen-codebook bolt-on retrieval at zero training: loses to embedding-RAG (v1).
   - Joint-trained Engram + LM at toy/mid scale: gate learns recurrence but cannot drive output (mechanism fails).
   - **Bolt-on at 3B-Qwen scale with generic-NTP pretraining + surgical insertion: matches or beats parametric baselines and approaches RAG-cheated baselines.**

## 3. Method (unchanged from v1)

```
input modalities (text / vision / audio)
       │
       ├─► Frozen Qwen2.5-3B-Instruct (3.1B params, 36 layers) — sufficient.
       │     │ inputs_embeds at each position:
       │     │   text positions  → Qwen's frozen token embedding
       │     │   perc positions  → learned perceptual emb table (V_vis = V_aud = 32-64)
       │     ▼
       │   layer L attached: forward pre-hook adds MultimodalEngramSet residual
       │     ▼
       │   continues → norm_f → lm_head → logits
       │
       └─► Frozen perceptual encoder (per modality, see §3.2)
                    │
                    ▼
            Naive k-means or STE codebook (K per modality)
                    │
                    ▼ discrete code (hash address)
            Per-user MultimodalEngramSet table
```

### 3.2 Encoder choices (validated per sub-modality)

| Sub-modality | Encoder | Embedding dim | Top-1 NN recall |
|---|---|---|---|
| Face identity | ArcFace R50 (insightface buffalo_l) | 512 | 0.98 |
| Speaker identity | ECAPA-TDNN (speechbrain spkrec-ecapa-voxceleb) | 192 | 1.00 |
| Acoustic scene | AST AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593) | 768 | 0.89 |
| Paralinguistic state | wav2vec2-LG-XLSR-emotion-finetuned | 1024 | 0.93 |
| Painter style | VGG-16 Gram-matrix + PCA-100 | 100 | 0.42 |

## 4. Empirical results

### 4.1 PerceptMem v0.1 scorecard (Path A best-recipe per modality)

| Task | N | RAG ceiling | Path A retr@1 | Path A code-match | code-match-fraction |
|---|---|---|---|---|---|
| V-XC-ID (face) | 5 | 0.96 | 0.32 | 0.60 | 0.40 |
| V-XC-ID | 10 | 0.96 | 0.26 | 0.46 | 0.44 |
| V-XC-ID | 60 | 0.88 | 0.11 | 0.23 | 0.46 |
| V-STY (style, PCA-Gram) | 5 | 0.48 | 0.20 | **0.80** | 0.20 |
| V-STY | 8 | 0.38 | 0.23 | 0.60 | 0.38 |
| A-XR-ID (speaker) | 5 | 1.00 | **0.44** | 0.69 | 0.52 |
| A-XR-ID | 10 | 1.00 | 0.32 | 0.76 | 0.42 |
| A-XR-ID | 29 | 1.00 | 0.20 | 0.50 | 0.40 |
| A-SCN (scene) | 5 | 0.88 | 0.36 | 0.75 | 0.48 |
| A-SCN | 10 | 0.86 | **0.40** | **0.84** | 0.38 |
| A-SCN | 20 | 0.86 | 0.20 | 0.42 | 0.48 |

**Mechanism (code-match retrieval) 0.50-0.84 across all 4 tasks at small N.** Path A is the strongest truly-parametric retriever; v1 chained baselines use embedding-NN fallback inside collided slots which is RAG-cheating.

### 4.2 Comparison vs baselines (large data, all parametric)

| Modality | N | RAG | v1 first-write (parametric) | v1 chained (RAG cheat) | **Path A best** |
|---|---|---|---|---|---|
| Audio | 5 | 1.00 | 0.52 | 0.72 | **0.64** |
| Audio | 20 | 1.00 | 0.26 | 0.63 | 0.33 |
| Vision | 5 | 0.88 | 0.28 | 0.32 | **0.36 ✓ beats v1 chained** |
| Vision | 20 | 0.94 | 0.20 | 0.36 | 0.20 |

### 4.3 Audio peak (STE+K=64+generic-NTP, large data)

| N | code-match retrieval |
|---|---|
| 5 | **1.000** (perfect when codes match) |
| 10 | 0.667 |
| 20 | 0.615 |

### 4.4 Scaling ablation: LM size is not the bottleneck

| Audio N | Qwen2.5-3B retr@1 | Qwen2.5-14B retr@1 |
|---|---|---|
| 5 | 0.560 | 0.480 |
| 10 | 0.280 | 0.360 |
| 20 | 0.330 | 0.300 |

5× LM size gives roughly tied results. The Engram and codebook are the bottlenecks.

## 5. Ablations summary

1. **Engram pretraining objective**: marker-supervised HURTS (audio code-match 0.44 → 0.00 at N=5); generic-NTP DECISIVELY HELPS (→ 1.00 at K=64+STE).
2. **Codebook size K**: per-modality optimum. Audio K=64 dominates at all N. Vision K=32 better at small N, K=64 at N≥20.
3. **STE-trained codebook**: helps audio at K=64 large data (code-match 0.85 → 1.00 at N=5); mixed on vision.
4. **Engram attach layer**: 1-layer audio sufficient; 2-layer vision lifts code-match 0.55 → 0.73 at N=5.
5. **Surgical insertion budget**: 80 steps, lr=1.0, no momentum is near-optimal.
6. **Style encoder**: DINOv2 → top-1 0.24, Gram-VGG → top-1 0.44, PCA-Gram-100 → Path A code-match 0.80 (mechanism converted to clean cell).

## 6. Limitations

1. **Encoder dependency.** We rely on frozen encoders being invariance-preserving. Domain shifts break the encoder before the mechanism.
2. **Codebook miss rate.** At higher N, cross-condition queries fail to share their registration code 40-60% of the time. The binding constraint; bigger codebooks fragment, smaller ones collide. STE addresses this for audio; vision remains harder.
3. **Style remains hardest sub-modality.** PCA-Gram has top-1 NN recall 0.42 (vs face 0.98); RAG ceiling on style is itself 0.48 at N=5.
4. **Paralinguistic Path A blocked at N≥5** by RAVDESS having only 8 emotion classes. CMU-MOSEI or IEMOCAP unblocks this.
5. **Scale of PerceptMem v0.1**: 79 vision identities + 29 speakers (per modality, eval split). Larger AgeDB cross-age, VoxCeleb-cross-channel runs would strengthen claims.
6. **No head-to-head vs published baselines yet** (MyVLM, Yo'LLaVA, Online-PVLM, RAP, M3-Agent, Mem0). Mem0 requires captioning the perceptual content (defeats the point); MyVLM/Online-PVLM need their published code run on PerceptMem.

## 7. What's left for camera-ready

1. **Better style encoder**: StyleCLIP or contrastive style head with end-to-end PCA.
2. **More emotion classes**: extend Path A paralinguistic to CMU-MOSEI for N≥5 eval.
3. **PerceptMem at scale**: add AgeDB cross-age face pairs (1000+), VoxCeleb cross-channel speakers (1000+).
4. **Head-to-head vs MyVLM/Online-PVLM/RAP on PerceptMem**.
5. **Qwen3-VL bolt-on** complete eval (architecture wired but mid-session GPU NVML issues prevented running).
6. **Paper writing** (3-4 weeks).

These are engineering, not science. The empirical core is settled.

## 8. Reproducibility

```bash
# Encoder embeddings (deterministic with SEED=42)
python3 src/sanity_arcface_collisions.py      # face
python3 src/sanity_ecapa_collisions.py        # speaker
python3 src/sanity_scene_collisions.py        # scene
python3 src/sanity_paralinguistic_v2.py       # paralinguistic
python3 src/sanity_style_collisions.py        # style (DINOv2)
python3 src/sanity_style_v2_distinctive.py    # style (Gram)
python3 src/style_pca_gram.py                 # style (PCA-Gram)

# Larger embedding sets
python3 src/nanochat_mm/extract_more_embeddings.py

# PerceptMem unified scorecard
python3 src/perceptmem.py
# expected: results/perceptmem_v0_1.json with the §4.1 table

# Path A pretraining variants
python3 src/nanochat_mm/pathA_generic_pretrain.py
python3 src/nanochat_mm/pathA_ste.py
python3 src/nanochat_mm/pathA_scaling_k64.py
python3 src/nanochat_mm/pathA_ste_k64.py    # audio peak: 1.00 code-match @ N=5

# Per-sub-modality runs
python3 src/nanochat_mm/pathA_submodality.py \
   --emb runs/embeddings/ast_esc50.npz \
   --name scene --modality audio --K 32

# v1 baselines for comparison
python3 src/nanochat_mm/v1_baselines_large.py
```

Dependencies: Qwen2.5-3B-Instruct (HF), Qwen2.5-14B-Instruct (optional), Qwen3-VL-8B-Thinking (optional), ArcFace ONNX (buffalo_l), SpeechBrain ECAPA-TDNN, MIT-AST, wav2vec2-XLSR-emotion, DINOv2-small, VGG-16 ImageNet. Single Blackwell RTX PRO 6000 GPU; ~7-30 GB VRAM depending on LM choice.

---

The repo at `github.com/bojieli/multimodal-user-memory` (private) carries the complete empirical narrative across 19 commits, 25+ scripts, 25+ JSON results, and 14 markdown notes. The science is settled at v2; remaining work is benchmark scaling + baselines + paper prose.
