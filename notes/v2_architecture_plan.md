# v2 architecture plan — Multimodal Mini-Engram (from-scratch pretrain)

**Status:** Plan for the next session, after this session's post-training failure.
**Source of architecture:** `~/user-as-engram/nanochat/nanochat/engram_module.py` (420 LoC, ports DeepSeek-Engram to nanochat).
**Source of training stack:** `~/user-as-engram/nanochat/scripts/engram_pretrain.py` (253 LoC).

## Architectural insight

Reading `engram_module.py`, the Engram has five components stacked into the transformer:

1. **Tokenizer compression**: surjection from base vocab → canonical (NFKC + lowercase + space norm) vocab. **Reused as-is for text.**
2. **Multi-head hashing per N-gram order**: each N-gram (sizes 2..max) is hashed with `n_head_per_ngram` heads into a per-order embedding table of size `engram_vocab_per_ngram`. **Reusable; key change is the hash input.**
3. **Per-order embedding tables**: ~100k entries × multiple heads, total ~256-dim per N-gram-order. **Reusable; we add modality-specific tables in parallel.**
4. **Context-aware sigmoid gate**: decides which order/head fires given the current backbone hidden state. **Reusable; the gate input is hidden state, modality-agnostic.**
5. **Depthwise causal short conv + residual add**: temporal smoothing + residual. **Reusable as-is.**

The Engram module's central abstraction is *hashed embedding lookup keyed by a token-stream pattern*. The "token stream" can be text N-grams (as in DeepSeek-Engram) or **perceptual code tuples interleaved into the input** (our extension).

## Multimodal extension — the key design choice

Two paths to interleave perceptual content:

**Path A: separate Engram tables per modality, all gated by the same hidden state.**
- Visual percept → ArcFace embedding → quantiser → code tuple → "visual N-gram" → visual Engram table.
- Audio percept → ECAPA-TDNN → quantiser → code tuple → "audio N-gram" → audio Engram table.
- Text → normal text Engram table (unchanged).
- All three tables added residually into the backbone.

**Path B: unified modality-tagged vocabulary.**
- All discrete codes (text BPE tokens, visual VQ codes, audio VQ codes) share a single token vocabulary with modality-tag bytes.
- Single Engram table over the combined vocabulary.
- Closer to Chameleon's "any-mode" design.

Path A is simpler to implement and easier to ablate per modality. Path B is more architecturally pure but harder to train. **Start with Path A** and ablate to Path B if Path A underperforms.

## Concrete code plan

Files to create under `/home/ubuntu/multimodal-user-memory/src/nanochat_mm/` (forked from user-as-engram's nanochat):

```
src/nanochat_mm/
├── perceptual_encoder.py   # frozen ArcFace + ECAPA-TDNN wrappers, batched
├── perceptual_quantiser.py # the learned VQ-head per modality, trained jointly during pretrain
├── engram_module_mm.py     # extension of engram_module.py with per-modality tables
├── multimodal_dataloader.py # interleaves text + perceptual streams
├── multimodal_gpt.py       # GPT backbone with multimodal Engram wired in
└── pretrain_mm.py          # main training script (fork of engram_pretrain.py)
```

The key trick in `multimodal_dataloader.py`: each training example is a sequence of (modality_tag, token_id) pairs. For text it's just (TEXT, bpe_id). For images it's (IMAGE, code_tuple_id). For audio (AUDIO, code_tuple_id). The Engram module sees the modality tag and routes hashing to the correct table.

## What replaces `extract_or_load_*_embeddings` and frozen quantiser

In v1 (post-training) the quantiser was frozen (k-means or learned-but-frozen). In v2, **the quantiser is trained jointly with the LM via straight-through estimator on the codebook lookup**. The LM's NTP loss flows back into the codebook through the gate's STE, training codes to be useful for LM prediction.

This is the key qualitative difference vs v1's failure. In v1 the codebook had to be perceptually-stable in isolation; in v2 it has to be *useful to the LM*, which is a much richer training signal that naturally produces stability where stability is what the LM uses.

## Pretraining corpus (revised, with concrete sources)

| Stream | Source | Size | Notes |
|---|---|---|---|
| Text base | NVIDIA ClimbMix slice via Karpathy mirror | ~1-3B tokens | Reuse the corpus user-as-engram used |
| Image-caption | LAION-400M subset or MMC4 | ~500M paired tokens | Filter to faces / scenes where ArcFace activates |
| Audio-caption | WavCaps + AudioCaps + LibriSpeech (already local) | ~500M paired tokens | Speech-heavy for ECAPA to be meaningful |
| **Recurrence stream (critical)** | YT-Temporal-1B narrations / VoxConverse multi-speaker / Ego4D narration | ~500M paired tokens | This is what teaches the gate to fire on recurring identities |

Total: ~3-5B tokens. d12 scaling-params budget at Karpathy 12 t/p ≈ 4-5B tokens.

The **recurrence stream is novel** and not part of user-as-engram's setup. Without it, the gate won't learn to fire on perceptual recurrence — it'll just learn to memorise per-shot percepts. We need data where the SAME face / SAME voice appears across long-range context.

## Compute budget revised

Single Blackwell RTX PRO 6000, bf16 (FP8 slower on sm120 per FINDINGS §6.1):

- **Phase 0** (corpus prep, tokeniser, quantiser-bootstrap): 3-5 days
- **Phase 1** (Mini-Multimodal-Engram d12 @ ~4B multimodal tokens): 4-7 days (multimodal slower than text-only per token due to encoder passes)
- **Phase 2** (PerceptMem benchmark construction): 3 days
- **Phase 3** (surgical insertion experiments + baselines): 1 week
- **Phase 4** (paper writing): 2-3 weeks

Total: ~6-8 weeks engineering + 2-3 writing.

## Concrete first-task list (next session, in order)

1. **Decide vision and audio VQ tokeniser**:
   - Vision: Cosmos-Tokenizer-DI8x8 (cached) gives 8×8 spatial codes per image. Test on LFW.
   - Audio: Encodec 24kHz (cached) for raw audio, or quantise ECAPA-TDNN output directly. Lean toward quantising ECAPA output for identity-focused use case.
2. **Fork nanochat into `~/multimodal-user-memory/src/nanochat_mm/`**.
3. **Modify `engram_module.py` → `engram_module_mm.py`** with parallel modality-specific tables and a modality-tag-aware gate.
4. **Build a small toy interleaved dataset** (LibriSpeech utterances + their transcripts + speaker tags) and run a 100M-token pretrain to validate the pipeline end-to-end.
5. **Only after that**: scale up to the full multimodal corpus.

## Risks and de-risking

| Risk | Probability | Mitigation |
|---|---|---|
| Multimodal corpus quality issues | Medium | Start with audio-text only (LibriSpeech + speaker tags); add vision after audio works |
| Gate doesn't learn to fire on perceptual recurrence | Medium | Audit gate firing rates per modality at intermediate checkpoints; add explicit per-modality firing-rate regularisation if needed |
| End-to-end quantiser collapses (low codebook usage) | Medium | EMA codebook updates + dead-code reinit (standard tricks from VQ-VAE / SoundStream) |
| Compute overrun beyond 7 days | High | Define a "minimum viable" Mini-Engram at d8@512 (~138M params, ~50min in FINDINGS) as the first checkpoint; only scale up once that works |
| Pretrained model is *also* worse than embedding RAG | **Plausible** | The defensive paper: "perceptual content may not be parametrically storable better than retrieval; here is the analysis." Still publishable, just less exciting |

## What we won't do

- Fine-tune Qwen3-VL (rejected: this session's bakeoff showed post-training the quantiser is the wrong mechanism; full LM post-training is even more constrained).
- Build a hybrid (parametric for some queries, RAG for others) before validating each separately.
- Try every quantiser objective. We've ruled out classifier, reconstruction-only, contrastive on frozen-codebook. End-to-end-trained codebook is the substantively-different mechanism left to try.
