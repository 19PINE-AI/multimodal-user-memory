# Multimodal User Memory Beyond Nameable Concepts

**Working title:** *Perceptual Engram — Memory for the Unnameable: Cross-Session Recall of Style, Prosody, Scene, and Cross-Condition Identity in Multimodal Agents*

**Status:** v1 plan, 2026-05-14. Sibling research line to [[polar-research]], [[user-as-engram]], [[user-as-lora]], [[UserAsCode]].

**v1 changes vs v0:**
- Pivot from "train multimodal Mini-Engram from scratch" to "**post-train a perceptual-Engram module on top of a frozen open multimodal base** (Qwen3-VL + Voxtral)." DeepSeek-Engram is not open; user-as-engram's pretrain-from-scratch path is not necessary for our contribution. LLaVA-style stage-1-only training of the new gate/module is sufficient and reproducible by anyone with a VLM checkpoint.
- Section 11 (gating experiments) is now mostly **complete**. Both modalities pass; see `notes/sanity_findings.md`. RQ depth needs care on vision (intra-agreement collapses faster than on audio); learned quantiser is the path forward.

**Tagline:** *Personalised VLMs solved "remember my cat." Memory of "the way she says 'fine'" is open.*

---

## 1. One-paragraph pitch

Current personalised multimodal memory has converged on a single regime: a nameable visual concept (a specific pet, a specific person, a specific object) is registered once and recalled later in similar conditions. MyVLM, Yo'LLaVA, MC-LLaVA, Online-PVLM, RAP, and TAME all live in this regime, as do the benchmarks that score them (Mem-Gallery, LCMP, MemoryCD). This regime is the easy slice of perceptual memory. The hard slice — and the one that matches how humans actually use perceptual memory — is content that **resists naming as a concept**: paralinguistic prosody, acoustic scene, vocal-identity-under-state-shift, style and brushwork attribution, handwriting and gait, cross-condition reappearance of the same identity under different lighting / angle / age / recording quality. None of the existing methods cover this, and no existing benchmark probes it. We construct the benchmark and propose **Perceptual Engram**, a unified content-addressable parametric memory keyed by invariance-preserving perceptual codes, with O(1) surgical insertion borrowed from [[user-as-engram]].

---

## 2. Novelty positioning — what is and isn't done

This is the critical section. The space turned out to be denser than first inspection suggested. The novelty claim has to be honest about what each prior method already provides.

### 2.1 What is already done (we cannot claim as contribution)

| Capability | Representative system | Year | Notes |
|---|---|---|---|
| Per-user visual concept identity (45 objects/people) | MyVLM | ECCV 2024 | Concept head (binary classifier per concept) + concept embedding in feature space. Per-concept gradient training. Single-concept inference. |
| Special-token visual concept registration (~16 tok/concept) | Yo'LLaVA | 2024 | Trained-per-concept, slow convergence, negative-sample dependent. |
| Multi-concept visual personalisation | MC-LLaVA | 2024 | Extension; resolves MyVLM's single-concept-per-inference. |
| Train-free O(1) closed-form visual concept insertion at scale (1,292 concepts in OP-Eval) | **Online-PVLM** | Nov 2025 | This is the closest prior. Frozen Omni Concept Embedder → instance norm → mean pool → MLP project. Concept memory bank for cross-session. Vision-only. |
| Retrieval-augmented visual personalisation | RAP (CVPR 2025) | 2025 | K-V database, multimodal retriever, real-time concept editing. Vision-only. |
| Multi-session multimodal memory benchmark (vision-language) | Mem-Gallery | 2026 | 240 conversations, 1,003 images, 1,711 QA. Tests visual-centric search/reasoning, conflict resolution. **Explicitly excludes audio.** **Does not test perceptual variation across sessions.** |
| Attribute-update personalisation benchmark | LCMP + TAME (KDD 2026) | Dec 2025 | 30 concepts, GPT-Image-1-generated variants, short-term vs long-term attribute changes. Tests semantic attribute updates, not perceptual condition variation. Vision-only. |
| Affective interpretation grounded in memory | A-MBER, MemEmo | 2026 | Tests model's *use* of memory for inferring user affect. Dialogue history. Not perceptual storage. |
| Identity-driven audio-visual generation | ID-LoRA | Mar 2026 | Generation, not memory. |
| Speaker registration in audio-LLM | SpeakerLM | Aug 2025 | Diarisation + recognition; not framed as agent memory across sessions. |

The space we cannot claim: **train-free, scalable, cross-session visual concept-identity memory.** Online-PVLM is the right point of reference here, not MyVLM.

### 2.2 What is open (our contribution territory)

Strip away what's done and four genuine gaps remain:

**G1. Audio modality is essentially unaddressed by the personalised-MLLM line.** MyVLM, Yo'LLaVA, MC-LLaVA, Online-PVLM, RAP, TAME, Mem-Gallery, LCMP — every system and benchmark above is vision-only. Audio personalisation exists (SpeakerLM, voice cloning) but as separate engineering for transcription / TTS, not as agent memory primitives that the LM can query.

**G2. Beyond-identity perceptual memory.** All existing personalisation targets **nameable concept identity** ("my cat Bibi"). It does not target:
- *Paralinguistic state of a known speaker* — e.g. recognising that the user's "fine" today sounds different from how they said "fine" last week.
- *Style and authorship* — recognising a painter's brushwork across sessions, a director's framing, your friend's handwriting.
- *Acoustic scene* — recognising "you're in the same room you recorded audio in two sessions ago."
- *Cross-condition perceptual reappearance* — same person under different lighting / different angle / different age / recording on a different microphone.

**G3. Cross-condition perceptual variation in benchmarks.** LCMP tests *attribute* variation (the dog had a haircut, now described as "with short fur"). It does not test *perceptual* variation (the same dog photographed in dim light at a different angle). The visual side of the question "would the system still match this percept to its stored identity?" is structurally unexamined.

**G4. Unified vision + audio perceptual memory.** Even if all the above were solved separately, there is no unified mechanism for "the agent's memory of perceptual experience." The two modalities are addressed by disjoint communities.

### 2.3 The novelty pitch in one sentence

*Existing personalised multimodal memory works for nameable visual concept identity under near-identical conditions; we extend it to audio entirely, to non-identity perceptual qualities (style, prosody, scene), and to cross-condition reappearance, using a unified content-addressable parametric mechanism.*

---

## 3. Target cases (the concrete things the system must handle)

Two senses only. From the working list previously discussed, we pick a tractable subset that covers the structural categories.

### 3.1 Visual axis

| Sub-case | Concrete example | What's hard |
|---|---|---|
| Cross-condition identity | Same person photographed at the party (session N) and in a coffee shop with different lighting (session N+k) | Need invariance to lighting / angle / age / capture device |
| Style / authorship | "Is this painter the same as the one you showed me Tuesday?" — early vs late period of the same artist | Style is a manifold, not a label |
| Cross-condition scene reappearance | Recognise the agent has been in this room before, from a corner view it hasn't seen | Scene memory under viewpoint change |
| Visual aesthetic preference | "You've shown me 30 photos you liked; would you like this one?" | Continuous preference over an aesthetic manifold |

### 3.2 Audio axis

| Sub-case | Concrete example | What's hard |
|---|---|---|
| Cross-recording speaker identity | Same speaker on different microphones / different rooms / different days | Need invariance to channel and room |
| Paralinguistic state recall | "Does the user sound more tired today than last week?" — comparison against the user's own baseline | Memory of *the user's own voice baseline* under state shift |
| Acoustic scene recall | "Is this the same room I recorded audio in two sessions ago?" | Room acoustic fingerprint |
| Prosodic pattern recall | "When the user says 'fine' like that, what did they mean last time?" | Memory of speaker-specific prosodic pattern |
| Musical taste / style memory | "You've shown me 40 tracks; how would this one rank?" | Continuous preference over an audio manifold |

We will scope the first paper to a subset — likely (visual cross-condition identity, style attribution, cross-recording speaker identity, paralinguistic state, acoustic scene). Aesthetic / taste memory is a stretch goal.

---

## 4. Benchmark — **PerceptMem**

The contribution is the benchmark plus the method. The benchmark stands on its own.

### 4.1 Construction principles

1. **Cross-session by construction.** Stimuli are partitioned into *registration* (the agent encounters the percept in early sessions, with some textual context — "this is Alice", "this is the conference room") and *probe* (the agent encounters a perceptually-different but identity-equivalent stimulus in a later session). The agent must produce a recall judgement.

2. **Compose from public assets, do not collect.** Risk-reduce by reusing established datasets. Novelty is in the *task framing*, not the media:
   - Faces under condition variation: CelebA-HQ + AgeDB (cross-age), or VGGFace2 (cross-condition pairs).
   - Voices across recordings: VoxCeleb (per-speaker across years and conditions).
   - Painter style: WikiArt by artist with early/late period split.
   - Acoustic scene: DCASE TAU Urban Acoustic Scenes (same scene class, different recordings).
   - Paralinguistic state: RAVDESS / IEMOCAP for emotion variants of the same speaker; CMU-MOSEI for multimodal affect.

3. **Memory-system framing, not perception-system framing.** The benchmark calls into a memory API: `register(percept, label, session_id)` and `recall(percept) → label or null`. We score the memory system, not the perceptual encoder. Each baseline is allowed any encoder; this is what isolates the *memory* contribution.

4. **Hold the memory system to scale.** We test at 100, 500, and 1000+ registered identities/styles/scenes per user, matching Online-PVLM's scale claim. Performance degradation under scale is part of the score.

### 4.2 Task suite

| Task | Construction | Metric |
|---|---|---|
| **V-XC-ID**: Cross-condition face identity | VGGFace2 / AgeDB. Register *k* photos per identity in session 1; probe in session N with photo under different lighting / angle / age. | Top-1 / top-5 |
| **V-STY**: Painter style attribution across periods | WikiArt. Register a painter via early-period works; probe with late-period works. Distractor pool = 9 other registered painters. | Top-1 |
| **V-SCN**: Cross-view scene reappearance | Matterport3D rendered views. Register room from view A; probe from view B. | Same-room AUC |
| **A-XR-ID**: Cross-recording speaker identity | VoxCeleb. Register speaker from recording R1 (year Y1, channel C1); probe from R2 (Y2, C2). | EER (equal error rate) |
| **A-PARA**: Paralinguistic state comparison vs. user's own baseline | RAVDESS / IEMOCAP. Register speaker-neutral baseline; probe with same speaker in different emotion. Ask "is this the same speaker, in what state?" | Speaker top-1 + state accuracy |
| **A-SCN**: Acoustic scene reappearance | DCASE TAU. Register scene from one recording; probe from a different recording of same scene class. | Same-scene AUC |

We also include a **propositional control** suite (LongMemEval-style text-only QA over the registration dialogues) to ensure no method regresses on textual recall while solving perceptual recall — a real risk for parametric methods.

### 4.3 Robustness probes

- **Distractor scaling**: 100 / 500 / 1000+ registered entities.
- **Adversarial perceptual neighbours**: probes drawn from the same identity vs. a *similar* identity (sibling, similar-breed dog, similar acoustic scene).
- **Cross-session interference**: register many entities; probe entities registered ≥ 50 sessions earlier to check forgetting / decay characteristics.

---

## 5. Method — **Perceptual Engram**

### 5.1 Architecture (revised — LLaVA-style post-training, no from-scratch pretraining)

The method is a **perceptual memory module bolted on a frozen open multimodal base**, trained LLaVA-stage-1-style. The base provides multimodal understanding; the module provides per-user perceptual memory.

```
[image / audio]
       │
       ├─► Frozen VLM/A-LM backbone (Qwen3-VL-8B-Thinking + Voxtral-Mini-3B)
       │     │ produces text-aware multimodal tokens & generates responses
       │     ▼
       │   answer
       │
       └─► Frozen perceptual encoder (ArcFace / ECAPA-TDNN / style head)
                          │
                          ▼
                   Learned quantiser (RQ-VAE with identity-preserving objective)
                          │
                          ▼ discrete code = hash address
                   Per-user Perceptual-Engram table
                          │
                          ▼ row payload (text-token sequence)
                   Learned gate / cross-attention → injected into VLM context
```

Five components, of which only two are trained, and neither is the base:

1. **Frozen multimodal base.** Qwen3-VL-8B-Thinking (vision, already in HF cache) + Voxtral-Mini-3B (audio, already cached) routed by modality. Both are reused as-is; no LoRA on the base in v1.

2. **Frozen perceptual encoders.** Off-the-shelf, validated by `notes/sanity_findings.md`:
   - Faces: ArcFace R50 (`buffalo_l/w600k_r50.onnx`, top-1 LFW recall 0.98).
   - Speakers: ECAPA-TDNN (`spkrec-ecapa-voxceleb`, top-1 LibriSpeech cross-recording recall 1.0).
   - Acoustic scene, painter style, paralinguistic state: encoder TBD per sanity check 3-5.

3. **Learned quantiser** (the first thing we train). RQ-VAE per modality, trained on held-out unlabeled corpora with a reconstruction + identity-preservation loss. The sanity-check finding is that naive residual k-means collapses identity past depth ~2 on vision; a learned quantiser with identity supervision should do better. Quantiser training: ~hours on single GPU per modality. The output is a small frozen codec.

4. **Per-user Perceptual-Engram table.** A hash-keyed parametric memory inherited from [[user-as-engram]]: per-user override table, surgical row insertion via UNEMBED_P / OPT-15. The key change is that the address is the perceptual code (not an N-gram). Per-user row insertion remains O(1), no gradient at insertion. The table's value space is a learned embedding that, when retrieved, is injected into the VLM context as a "soft token."

5. **Learned gate** (the second thing we train, and the load-bearing piece). A small router that decides whether to consult the Perceptual-Engram given the current multimodal context, and merges the retrieved soft-token into the VLM. Three candidate implementations:
   - **(5.a) Prefix-token injection** (simplest): the retrieved row's payload is prepended to the text prompt as a soft prompt. Trained on (image/audio + caption) where the caption references a recurring identity.
   - **(5.b) Cross-attention gate**: small cross-attention head between the VLM's text stream and the retrieved row. Higher capacity, more trainable parameters.
   - **(5.c) Adapter-LoRA gate**: a tiny per-base LoRA gated by the perceptual code. Closest to user-as-engram's "Engram fires LoRA-style updates" intuition.

We start with (5.a) for v1 because it's least invasive to the frozen base and validates the *mechanism*. If it underperforms, (5.b) is the natural escalation.

### 5.2 What we actually train

| Component | Trainable params | Data | Budget |
|---|---|---|---|
| Frozen base (Qwen3-VL, Voxtral) | 0 | — | 0 |
| Frozen perceptual encoders | 0 | — | 0 |
| Per-modality quantiser (RQ-VAE) | ~5–20M | ~10–50h unlabeled face/voice/scene per modality | ~hours on 1 GPU per modality |
| Gate (path 5.a → 5.b) | ~5–50M | ~100k–1M (multimodal stream + recurring identity) examples synthesised from public datasets | ~1–3 days on 1 GPU |

**No base pretraining. No full fine-tuning of any LM.** This is the entire training budget.

### 5.2 Why this should beat existing baselines

- **vs. text RAG** (Mem0, MemMachine): captions destroy the perceptual signal. Cross-condition recall is impossible from "a man with brown hair."
- **vs. caption-then-text-memory** (M3-Agent-style): same problem — identity is offloaded to external tools and the memory layer never integrates it.
- **vs. embedding-based RAG over face / voice DBs**: this is the strongest baseline. Perceptual-Engram should match or slightly beat it on raw recall and *win on scale* (O(1) hash lookup vs. O(N) cosine search) and *win on integration* (the LM has direct parametric access via the gate, not via tool-call indirection).
- **vs. Online-PVLM** (the closest prior): we extend to audio entirely; we extend to non-identity perceptual content (style, prosody, scene); we are content-addressable rather than feature-space-injected, which gives a cleaner scaling story; and crucially we run on a benchmark Online-PVLM has not been evaluated on. The fight on visual-identity-only at OP-Eval scale will be close and we should not over-claim there — it is plausible Online-PVLM ties or beats us on V-XC-ID, and that's a fine outcome as long as we win on the audio and beyond-identity axes.
- **vs. MyVLM / Yo'LLaVA**: O(1) insertion vs. their per-concept gradient training. Scale beyond their 45-concept ceiling.

### 5.3 Win conditions

- **Primary**: significant gain on A-XR-ID, A-PARA, A-SCN, V-STY (the axes where no prior method exists).
- **Parity acceptable**: V-XC-ID against Online-PVLM (both methods are train-free at scale; we're not claiming the visual identity slice).
- **No regression**: propositional control suite — perceptual-Engram must not hurt text recall.

---

## 6. System / implementation (revised)

- **Vision base**: Qwen3-VL-8B-Thinking — already cached at `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking`. Supports thinking traces (useful for memory-conditional reasoning).
- **Audio base**: Voxtral-Mini-3B — already cached. Alternative: Qwen2-Audio (not yet cached). Routed by modality detection at the input layer.
- **Face encoder**: ArcFace R50 ONNX (`buffalo_l/w600k_r50.onnx`, 167MB, already downloaded to `~/.insightface/models/`).
- **Speaker encoder**: SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`, ~30MB, cached after first sanity check).
- **Quantiser**: RQ-VAE with 2 levels × 32–64 codes per level (initial; tune per modality). Sanity-finding-derived sweet spot ranges from `notes/sanity_findings.md`. Identity-preservation loss added on top of reconstruction.
- **Engram table**: borrow per-user override table primitive from [[user-as-engram]] `prototype/` directory (multi-tenant override design; ~1s surgical insertion).
- **Gate** (initial path 5.a): soft-prompt prefix injection. ~5M params.
- **Hardware**: single Blackwell RTX PRO 6000 (102GB). All training (quantiser + gate) fits in single-GPU budget; multimodal pretraining not required.

### Caveat (resolved 2026-05-14)

- Disk: was 32GB free at start; freed 1.8TB by deleting prior project run checkpoints (authorised by user). Now 1.8TB available.
- GPU: CUDA works; `nvidia-smi` NVML warning is cosmetic. ONNX `CUDAExecutionProvider` not auto-bound — affects throughput only.
- `insightface` package had a stale `mpl_toolkits.mplot3d` import path; patched in `__init__.py` to skip the broken `app`/`thirdparty` submodules (reversible).

---

## 7. Evaluation protocol

### 7.1 Baselines

Run all baselines on PerceptMem at matched encoder budget where applicable:

1. **No memory** (raw MLLM, lower bound)
2. **Text RAG** (Mem0 default): captions of percepts indexed by sentence-encoder
3. **M3-Agent-style** (entity-centric graph with external face/speaker tools)
4. **Embedding RAG** (face/voice embedding + cosine retrieval, FAISS index)
5. **MyVLM / Yo'LLaVA / MC-LLaVA** on visual sub-tasks (where applicable; not all support cross-session)
6. **Online-PVLM** on V-XC-ID / V-STY (head-to-head)
7. **RAP** on V-XC-ID / V-STY (head-to-head)
8. **TAME / LCMP** baseline on V-XC-ID (note: LCMP is attribute-shift, may not transfer cleanly; framed as a probe of "do attribute-shift methods handle perceptual shift")

### 7.2 Metrics

Primary metric per task as listed in §4.2. Aggregate score: macro-average across the six perceptual tasks. Token cost: per-query context tokens consumed (parametric methods should win). Latency: wall-clock retrieval / insertion.

### 7.3 Engagement risk

Get at least one author from Online-PVLM, RAP, or M3-Agent to run their system on PerceptMem before submission. The diagnostic table is more credible if competitors' own implementations are used rather than our re-implementations.

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Online-PVLM has unpublished audio extension imminent | Medium | The visual-identity axis is theirs to claim; we explicitly do not contest it. Lead with audio + beyond-identity. |
| Quantiser collisions destroy cross-condition recall | Medium | ArcFace and ECAPA-TDNN are designed for this; RQ-VAE preserves neighbour structure. Validate on held-out cross-condition pairs *before* committing to the full pipeline. |
| Gate retraining (path 5.1.b) doesn't converge in budget | High | Use routing layer (path 5.1.a) for v1. Gate retraining becomes the v2 / journal extension. |
| Style / aesthetic axis is too subjective for a reproducible benchmark | Medium | Score against the human-curated WikiArt labels for V-STY; for aesthetic preference treat it as a stretch goal not in the headline. |
| Reviewers conflate this with Online-PVLM / MyVLM | High | The novelty positioning section (§2) is load-bearing. State it clearly in intro. The audio axis and the cross-condition framing are the load-bearing points. |
| The audio results are good but vision results don't win | Medium | Reposition as an audio-personalisation paper. Audio alone is a clean contribution; LCMP / Mem-Gallery have explicitly excluded it. |

---

## 9. Timeline (revised under LLaVA-style training; faster than v0)

| Phase | Weeks | Deliverable | Status |
|---|---|---|---|
| 0. Sanity checks (audio + face quantisation) | 0.5 | `notes/sanity_findings.md` | **Done 2026-05-14** |
| 1. Sanity checks (scene + style + paralinguistic) | 1 | extended findings | Next |
| 2. Learned quantiser training (per modality) | 2 | RQ-VAE checkpoints; codebook stats | |
| 3. Benchmark construction (V-XC-ID, A-XR-ID, A-SCN first) | 2 | PerceptMem v0.1 | |
| 4. Baselines wired (text RAG, embedding RAG, M3-Agent-style) | 2 | diagnostic table on PerceptMem v0.1 | |
| 5. Perceptual-Engram gate training (path 5.a) | 2 | end-to-end first numbers | |
| 6. Remaining tasks (V-STY, A-PARA) | 2 | PerceptMem v1.0 | |
| 7. Head-to-head: Online-PVLM, RAP, MyVLM, Yo'LLaVA | 3 | full results table | |
| 8. Paper writing | 3 | submission | |

Total: ~17.5 weeks (≈4 months) end-to-end vs v0's 5 months. The savings come from skipping pretraining.

---

## 10. Relation to the existing research line

This paper extends the parametric-memory program from the other repos along the modality axis:

- [[polar-research]] — POLAR's orthogonal LoRA decomposition. Sibling. Stays propositional.
- [[user-as-engram]] — hashed-table parametric memory with surgical insertion. **This paper's direct substrate.** We adopt the insertion mechanism and the multi-tenant override-table design; we generalise the address from text-N-gram to perceptual-code.
- [[UserAsCode]] — executable code as memory. Orthogonal. Handles propositional and verifiable content; this paper handles perceptual and non-verifiable content. They compose: a complete personal-memory system would route propositional facts to UserAsCode and perceptual facts to Perceptual Engram.

The unification narrative across the five papers: *the right substrate for user memory is parametric and content-addressable; the addressing scheme should match the content type — N-grams for propositional facts ([[user-as-engram]]), code symbols for verifiable facts ([[UserAsCode]]), perceptual codes for unnameable perceptual content (this paper).*

---

## 11. Sanity-check results (resolved)

**Both gating experiments PASS.** Full numbers in `notes/sanity_findings.md`. Headline:

| Modality | Top-1 NN recall | Best K (flat) | intra | inter | ratio |
|---|---|---|---|---|---|
| Audio (ECAPA, LibriSpeech cross-chapter) | 1.000 | 32 | 0.86 | 0.009 | 101 |
| Vision (ArcFace, LFW cross-condition)   | 0.979 | 32 | 0.75 | 0.012 | 61 |

Mechanism is viable. The follow-on engineering question is **how to train an identity-preserving quantiser** that scales to large effective K without collapsing intra-identity agreement — naive RQ collapses past depth 2 on vision. This is the central thing to validate in phase 2.

## 12. Immediate next steps

1. **Sanity check 3 (style)**: validate a style-discriminative encoder (StyleGAN-encoder features, or a CLIP-Style-Sim head) on WikiArt cross-period pairs.
2. **Sanity check 4 (acoustic scene)**: PANNs CNN14 or Audio-MAE on DCASE TAU cross-recording pairs.
3. **Sanity check 5 (paralinguistic state)**: wav2vec2-Emotion or HuBERT-prosody on RAVDESS — needs to be speaker-invariant.
4. **Learned RQ-VAE training**: train a small RQ-VAE per modality with reconstruction + identity-preservation loss, replace naive k-means.
5. **Per-user Engram table prototype**: port the override-table code from `~/user-as-engram/refs/engram_demo_v1.py` and `prototype/`, replace N-gram hash key with perceptual code.
6. **Begin Qwen3-VL + Voxtral wiring**: load both, verify multimodal inference works end-to-end before adding the Engram module.
7. **Outreach**: contact authors of Online-PVLM, RAP, TAME to gauge willingness to run on PerceptMem before submission.
