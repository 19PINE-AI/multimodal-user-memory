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
