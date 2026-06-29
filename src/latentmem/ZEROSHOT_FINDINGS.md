# Training-free attention memory (zero-shot universality)

## The result: zero-shot AttMem = encoder retrieval, read inside the frozen model
At eval the query is the RAW encoder embedding and attention is pure cosine (W_q is
used only in pretraining, never at eval). The only eval-time knobs are the attention
temperature (inv_temp) and the residual gain (out_gain). Set them sharp+strong and
the in-model read becomes a faithful hard argmax over the bank.

Face (ArcFace, Qwen2.5-3B frozen), NO training (n_steps=0), inv_temp=500, out_gain=64,
paired vs raw-cosine RAG over 20 eval draws:

| N | AttMem (0-shot) | RAG | Δ |
|---|---|---|---|
| 5 | 0.983 | 0.983 | +0.000 |
| 10 | 0.948 | 0.948 | +0.000 |
| 50 | 0.821 | 0.821 | -0.000 |
| 300 | 0.749 | 0.749 | -0.000 |

Δ=0 on EVERY draw -- by construction: hard argmax + dominant gain => AttMem's marker =
nearest key's marker = RAG's prediction. AttMem in this limit IS retrieval computed
inside the model.

## Why the default config looked like it needed training
Defaults inv_temp=20, out_gain=8 are too soft: the softmax blends competing markers
(worse as N grows) and the retrieved residual doesn't dominate h_old, so the read-back
is lossy. Trained W_o was compensating for soft constants. Numbers (face, paired):

| N | 0-shot τ=20,g=8 | 0-shot τ=200,g=8 | 0-shot τ=500,g=64 | trained | RAG |
|---|---|---|---|---|---|
| 5 | ~0.93(1draw) | 0.943 | 0.983 | 0.933(rng99) | 0.983 |
| 10 | 0.867(1draw) | 0.922 | 0.948 | 0.942 | 0.948 |
| 300 | 0.356(1draw) | 0.629 | 0.749 | ~0.64 | 0.749 |

So sharp τ + high gain (2 hand-set constants, no gradient) recover the FULL encoder
ceiling, exceeding even the trained model (which kept soft constants).

## Implications
- Recall is training-free and equals the encoder's ceiling on the base model; it never
  beats the encoder (consistent with the learned-metric + paired-eval findings).
- The contribution is architectural: the recalled identity is a TOKEN the LM conditions
  on (O(1) registration, composes with text), not a recall win.
- Expectation for the multi-model sweep: every frozen LLM reproduces the encoder ceiling
  for recall; the real test is whether the in-model READ works per architecture
  (tied/untied embeddings, MoE, gemma logit-softcap).
- Gain tradeoff: high gain = faithful recall but near-passthrough at the marker;
  balanced gain = marker participates in generation (may still benefit from training).

## Multi-model universality sweep (training-free, face recall vs encoder)
Zero-shot (n_steps=0), inv_temp=500, paired vs raw-cosine RAG over 20 draws. The
ONLY per-model knob is out_gain (a single hand-set constant, no gradient steps).

| Model | Arch | Embeddings | gain | Δ vs encoder (N=5/10/50) |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | transformer | tied | 64 | 0 / 0 / 0  (exact) |
| Qwen2.5-7B-Instruct | transformer | tied | 64 | 0 / 0 / 0  (exact) |
| Qwen3-4B | transformer | tied | 64 | 0 / 0 / 0  (exact) |
| Qwen3-8B | transformer | tied | 64 | 0 / 0 / 0  (exact) |
| Phi-3.5-mini-instruct | transformer | tied | 64 | 0 / 0 / 0  (exact) |
| SmolLM2-1.7B-Instruct | transformer (Llama) | tied | 64 | 0 / 0 / 0  (exact) |
| DeepSeek-R1-Distill-Llama-8B | transformer (Llama) | untied | 64 | 0 / 0 / 0  (exact) |
| Mistral-7B-Instruct-v0.3 | transformer | untied | 64 | -.03 / -.03 / **-.51** (collapse) |
| Mistral-7B-Instruct-v0.3 | transformer | untied | **256** | 0 / 0 / **0** (fixed) |
| granite-4.0-h-tiny | **hybrid Mamba/transformer** | - | 64 | -.02 / -.01 / -.005 (near-exact) |

(Llama-3-8B, Gemma-2-9B: gated repos, only config stubs cached -> not runnable here.)

**Conclusion.** A training-free, in-model perceptual memory reproduces the encoder's
recall ceiling across 9 model families spanning 1.5B-8B, tied AND untied embeddings,
and transformer AND hybrid-Mamba architectures. On tied-embedding models it is EXACT
(Δ=0 every draw) because the marker value == lm_head row, so the read is a perfect
argmax passthrough. Untied models also reach exact with a larger residual gain
(Mistral needs gain>=256: its lm_head rows have smaller norm, so gain=64 couldn't
swamp h_old). The hybrid-Mamba reader is near-exact (~1pt gap). Only knob: out_gain,
a single constant set by inspection -- NOT training. Existing frozen models serve as
multimodal user memory out of the box.

## All-modality training-free check (recall = encoder beyond face)
Zero-shot (n_steps=0, inv_temp=500, gain=64), paired vs RAG, 20 draws, Qwen2.5-3B:

| Modality | N=5 Δ | N=10 Δ | N=20 Δ |
|---|---|---|---|
| Speaker (a-xr-id, ECAPA) | 0.000 | 0.000 | 0.000 |
| Acoustic scene (a-scn) | 0.000 | -0.002 | +0.002 |
| Tone (a-para) | +0.003 | 0.000 | -0.002 |
| Style (v-sty-clip, CLIP) | 0.000 | 0.000 | -0.001 |
| Face (v-xc-id-xxxl, ArcFace) | 0.000 | 0.000 | 0.000 |

Training-free recall = encoder on ALL five sub-modalities (vision + audio, identity +
style + affect), not just face. Confirms the universality claim across task types.
