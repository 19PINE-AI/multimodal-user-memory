# Perceptual Engram: Continuous Attention Memory for Cross-Condition Perceptual Recall in Multimodal Agents

**Authors:** [redacted for review]
**Date:** 2026-05-18

---

## Abstract

Personalised multimodal memory has converged on a single regime: a nameable visual concept is registered once and recalled later in similar conditions. The harder case — **cross-condition perceptual recall** — is unsolved. The same person under different lighting, the same voice across recordings, the same painter across periods, the same acoustic scene across takes, the same user's paralinguistic state — these targets resist naming as discrete concepts and are absent from existing personalisation benchmarks. We propose **Perceptual Engram**, a bolt-on continuous attention memory attached to a frozen pretrained language model via a forward pre-hook on `lm_head`, with three load-bearing ingredients: invariance-preserving perceptual encoders, a per-modality bank of (encoder embedding, marker input embedding) pairs, and pretrained kNN-LM-style logit injection trained for ≤ 12K steps with ~8M trainable parameters. We construct **PerceptMem v0.2**, a unified scorecard across five sub-modalities. At V-XC-ID N=10 on a 2180-ID face pool, AttentionMemory's purely parametric retrieval (0.989, n=3 seeds) **BEATS** the embedding-RAG cosine-NN ceiling (0.933) with p=0.038, and BEATS additionally at V-STY N=5 (p=0.015, 1.6× ratio) and V-STY N=10 (p=0.009). Across all five sub-modalities AttentionMemory reaches 83–117% of the encoder ceiling, while a discrete-codebook predecessor (Path A) capped at ~7% retr@1 at N≥300 regardless of codebook size, encoder upgrade, or 100K-step continual pretraining. Insertion is O(1) wall-clock (~0.5 ms for 1000 IDs combined) — 2,000,000× faster than the SGD-per-id baseline. Query latency is flat at 15 ms over N from 10 to 10000; RAG-with-LM-context is 52× slower at N=1000 and OOMs at N=10000 within Qwen's 32k context window. Top-1 next-token prediction is preserved byte-for-byte on text-only inputs, satisfying the no-regression-on-text-recall constraint.

---

## 1. Introduction

The personalisation literature for multimodal large language models has converged on a recognisable regime. A user registers a small set of nameable visual concepts — "this is Bibi", "this is my office mug" — and the system later answers questions or generates captions that refer to those concepts. The recent wave of work in this regime (MyVLM, Yo'LLaVA, MC-LLaVA, Online-PVLM, RAP, TAME) and the benchmarks that score it (Mem-Gallery, LCMP, MemoryCD) implicitly assume that the concept is *named*, that the registration condition and the recall condition are *perceptually similar*, and that the memory's job is essentially a lookup keyed by a stable concept label.

This regime is the easy slice of perceptual memory. The hard slice, and the one that matches how humans actually use perceptual memory, is content that **resists naming as a concept**:

- *Paralinguistic prosody of a known speaker*: today the user says "fine" in a way that doesn't sound like how they said "fine" last week — and recognising the *manner* is the recall task.
- *Style and authorship*: this brushwork is the same painter you showed me Tuesday, even though it's from a different period.
- *Acoustic scene*: this voice memo was recorded in the same room you were in two sessions ago.
- *Cross-condition perceptual reappearance*: the same person photographed under different lighting, different angle, different age, different microphone.

None of the existing systems address this hard slice, and no existing benchmark probes it. The published methods are either *vision-only* (every system we listed above), or they extract a textual entity label and offload the heavy lifting to a separate face/speaker recognition tool (M3-Agent), or they treat memory as a special-token vocabulary extension trained per concept (Yo'LLaVA). None of these architectures admit a *continuous* perceptual memory keyed by an invariance-preserving encoder.

The natural mechanism for continuous perceptual memory is a content-addressable parametric module: register a percept by writing its encoder embedding into a per-user table, recall by querying the table with a different sample of the same identity. This is the design we explore. We make three claims about the resulting system, all of which we verify empirically:

1. **Continuous attention memory beats discrete-codebook parametric memory** by 2–10× retr@1 across all five sub-modalities tested. We show this by direct comparison against a discrete-codebook predecessor (Path A) that we built and tuned to its ceiling over 16 sessions before pivoting.

2. **The mechanism, in 3 of 5 sub-modality × N cells we test, BEATS the embedding-RAG cosine-NN ceiling**, multi-seed verified at p < 0.05. This includes a face-identity task at N=10 over a 2180-ID pool, the largest scale at which any parametric perceptual memory has been shown to beat embedding RAG.

3. **The system is O(1) at insertion and effectively O(1) at query** in wall-clock terms. Registration of 1000 identities takes 0.5 ms (a single tensor concatenation); query latency is dominated by the LM forward (~15 ms) and is flat over N from 10 to 10000.

The remainder of the paper is structured as follows. §2 surveys the personalisation and memory landscape and isolates the gaps we address. §3 presents the architecture, including the four design choices that distinguish the working implementation from three earlier attempts that produced random output. §4 constructs PerceptMem v0.2 from public assets. §5 reports the full empirical evaluation: the BEATS-RAG cells, the Path A → AttMem improvement, the latency benchmark, and the no-text-regression validation. §6 discusses the design space, including why discrete codebooks saturate, why scaling the frozen LM doesn't trivially help, and what limits remain.

## 2. Related work and positioning

### 2.1 What is done — and where it stops

We summarise the relevant published work along three axes: target content type, modality coverage, and insertion mechanism.

**Per-user nameable visual concept identity.** MyVLM (ECCV 2024) trains a per-concept binary classifier head plus a concept embedding in feature space, with per-concept gradient updates. Yo'LLaVA (2024) extends this with a special-token approach (~16 tokens per concept). MC-LLaVA (2024) handles multi-concept inference. RAP (CVPR 2025) introduces a K-V retrieval database with a multimodal retriever for real-time concept editing. Online-PVLM (Nov 2025) is the closest prior: it provides train-free closed-form visual concept insertion at scale (OP-Eval at 1,292 concepts), via frozen Omni Concept Embedder → instance norm → mean pool → MLP project, with a concept memory bank for cross-session recall.

All five systems are **vision-only** and target **nameable concept identity** under near-identical conditions.

**Multimodal memory benchmarks.** Mem-Gallery (2026) contains 240 conversations and 1,003 images; it explicitly excludes audio and does not test perceptual variation across sessions. LCMP + TAME (KDD 2026) tests **attribute** updates ("the dog had a haircut") on 30 GPT-Image-1-generated concepts. A-MBER and MemEmo (2026) test the *use* of memory for affective interpretation, but memory storage is propositional (dialogue history), not perceptual. ID-LoRA (Mar 2026) handles identity-driven audio-visual generation, not memory. SpeakerLM (Aug 2025) does speaker diarisation but is not framed as agent memory.

None of these benchmarks probe **cross-condition perceptual variation** of the kind we target.

### 2.2 What's open

Stripping away what's done, four genuine gaps remain:

**G1.** Audio is essentially unaddressed by the personalised-MLLM line. Audio personalisation exists (SpeakerLM, voice cloning) but as separate engineering, not as agent memory primitives the LM can query.

**G2.** Beyond-identity perceptual memory: paralinguistic state, style and authorship, acoustic scene, prosodic pattern, cross-condition reappearance.

**G3.** Cross-condition perceptual variation in benchmarks. LCMP probes attribute variation, not perceptual variation. The question "would the system match this percept to its stored identity?" is structurally unexamined.

**G4.** A unified mechanism for vision + audio perceptual memory. The two modalities are addressed by disjoint communities and their methods do not compose.

### 2.3 Our positioning

We are the first to (a) handle audio in this framework, (b) target non-identity perceptual content (style, prosody, scene) and cross-condition reappearance via a single mechanism, (c) provide a benchmark — PerceptMem v0.2 — that scores both axes, (d) demonstrate parametric-memory BEATS RAG at the 2180-ID scale with p < 0.05 multi-seed.

Existing personalised multimodal memory works for nameable visual concept identity under near-identical conditions; we extend it to audio entirely, to non-identity perceptual qualities, and to cross-condition reappearance, using a unified content-addressable parametric mechanism.

## 3. Method — Continuous Attention Memory

### 3.1 Architecture overview

```
input modalities (text / vision / audio)
       │
       ├─► Frozen Qwen2.5-3B-Instruct (3.1B params, 36 layers, hidden 2048)
       │     │ inputs_embeds at each position:
       │     │   text positions  → Qwen's frozen token embedding
       │     │   perc positions  → learned per-modality projection (vis_proj / aud_proj)
       │     ▼
       │   ... 36 transformer layers ...
       │     ▼
       │   model.norm (final RMSNorm)
       │     ▼
       │   ← FORWARD PRE-HOOK on lm_head — adds:
       │   ←   residual = out_gain · W_o ( softmax(q·K^T · inv_temp) · V )
       │     ▼
       │   lm_head → logits
       │
       └─► Frozen perceptual encoder (per modality)
                    │
                    ▼ encoder_emb (e.g. 512-d ArcFace; 192-d ECAPA; 1024-d wav2vec)
            Register: bank.append((encoder_emb, marker_token_id))   — O(1) wall-clock
            Query:    bank cross-attention → residual at perc positions
```

The bank stores `(key, value)` pairs where the key is the L2-normalised encoder embedding and the value is the LM's value-side embedding for a marker token. For models with tied input/output embeddings (Qwen2.5-3B), the value is `input_embedding[marker]`; for untied models (Qwen2.5-7B), it is `lm_head.weight[marker]` (auto-detected). This choice ensures the residual addition pre-`lm_head` produces a clean per-marker logit boost via `lm_head[marker] · value = ||·||²` rather than a cross-product of two unrelated vectors.

### 3.2 Trainable parameters

| Parameter | Shape | Init | Role |
|---|---|---|---|
| `W_o` | H × H (2048×2048 for 3B) | I (identity) | Output projection of retrieved value |
| `out_gain` | scalar | 8.0 | Multiplies residual; dominates LM's natural-negative marker logit |
| `log_inv_temp` | scalar | log(20) | Inverse softmax temperature |
| `vis_proj` | H × D_vis (2048×512) | normal | LM-input projection for perceptual positions |
| `aud_proj` | H × D_aud (2048×192) | normal | (same, audio) |
| `W_q` | D × H | random | Unused at inference; legacy slot |

Total: ~8M parameters on top of 3.1B frozen LM. Optimiser: AdamW lr=3e-4, weight decay 0.01.

### 3.3 Pretraining recipe

Each step:
1. Sample bank size `bs` uniformly from `[bs_min, bs_max]` (curriculum; we use `[64, 1024]` for V-XC-ID-XXXL).
2. Sample `bs` IDs from the training pool; for each, one sample as the registration key, one different sample as the cross-condition query.
3. Assign markers 30001..30001+bs-1.
4. Insert all bank rows.
5. Forward through the LM with text prefix "You see [perc]" and modality_ids tagged.
6. Loss: `cross_entropy(logits[:, -1, :], target=marker_for_query_id)`.

5K–12K steps converge for all five sub-modalities (vs Path A's 100K).

### 3.4 The four critical design choices

The architecture above is the *fourth* iteration. Three earlier attempts produced random-level retr@1 (0.07 at N=5) despite plausible-looking loss curves. The four bug fixes that turned random output into BEATS-RAG were:

1. **No `sqrt(D)` divisor.** With L2-normalised keys, dividing softmax logits by sqrt(D) shrinks cosine-difference logits below 0.1 for D ≥ 512, producing near-uniform attention. We use `logits = (q · k) · inv_temp` directly.

2. **`log_inv_temp` init = log(20).** Gives sharp attention at zero-shot.

3. **Hook on `qwen.lm_head` pre-forward**, not on `qwen.model.layers[K]`. The residual reaches logits without dilution through additional transformer blocks and the final norm.

4. **`out_gain = 8.0` (learnable).** The natural logit boost from `marker_emb · marker_emb` is ~||emb||² ≈ 1.2 for Qwen, but unusual marker tokens (IDs 30001+) have a very-negative LM logit (-10 to -20). The gain scales the residual to dominate without per-token bias tables.

A fifth design choice — **curriculum bank_size** — is required only at large N. Fixed `bs=64` training causes a train/eval distribution shift that drops retr@1 from 0.63 to 0.20 at N=700.

[*Sections 4-6 elided in this draft. To be written: §4 PerceptMem construction; §5 full empirical results; §6 design-space discussion and limitations.*]

---

[*Working draft, 2026-05-18. Full results in `notes/session_18_attmem_validation.md` and `results/SUMMARY.md`.*]
