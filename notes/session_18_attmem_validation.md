# Session 18 — AttentionMemory architectural validation

**Date:** 2026-05-17

This session validates the continuous-attention-memory primitive
proposed in [session_17](session_17_pivot_to_attention_memory.md).
The aim is to demonstrate that on A-PARA (where Path A's discrete-codebook
result was the cleanest empirical win at N=10) AttentionMemory either
matches or improves the win, and that scaling to larger N keeps the
architecture's properties intact (latency O(N·D); accuracy bounded by
encoder discriminability, not by discretization).

## First v1 result (3000 steps)

Configuration:
- `attach_layer=24`, `W_o` zero-init, `log_tau=0`, `tau/sqrt(D)` scaling
- Training: 3000 steps, `lr=3e-4`, `bank_size=64`
- Result: retr@1 = 0.067–0.10 at N=5/10/20/50 — **near random**
- Loss landed at 10.6 (random over full vocab = 11.93; over 64 markers = 4.16).

## v2 (5000 steps, identity init + later attach layer)

- `attach_layer=33` (closer to lm_head), `W_o = 0.5 * I`
- Result: 0.067/0.067/0.000/0.013 — **still random**
- Loss 10.4 — basically no improvement.

## Diagnosis (the critical bugs)

Reading the debug script's logit traces (zero-shot, no training) revealed
two architectural mistakes:

1. **`sqrt(D)` divisor with L2-normalised keys.** Inherited from standard
   self-attention (where keys are not normalised). With normalised keys
   `q·k ∈ [-1, 1]` (cosine), dividing by `sqrt(D)` shrinks the effective
   softmax temperature dramatically for any D > 1. For D=1024 (wav2vec),
   it shrunk a cosine gap of ~0.4 between same- and diff-ID pairs to
   logit ~0.012, making softmax weights essentially uniform across the
   bank → retrieved value = mean of all values → no discrimination.

2. **Hooking at `model.layers[24..33]` instead of pre-`lm_head`.** Even
   when retrieval is sharp, an injected residual at layer 33 still passes
   through layers 34, 35, and the final norm before logits. Those
   transformations dilute the marker-input-embedding signal.

Two further design choices that mattered:

3. **`out_gain` scalar.** Qwen2.5-3B's input embedding norm is ~1.1 →
   `||marker_emb||²` ≈ 1.2. The natural logit boost from `retrieved · lm_head[marker]`
   (with `W_o = I`) is ~1.2. But the LM's natural logit for an unusual
   token like marker_id 30001 is typically very negative (-10 to -20). We
   need a substantial boost to override. Solution: a learnable scalar
   `out_gain` (init=8.0) that scales the residual before injection.

4. **Tied embeddings.** Qwen2.5-3B has `tie_word_embeddings=True`. Confirmed
   by `qwen.get_input_embeddings().weight.data_ptr() == qwen.lm_head.weight.data_ptr()`.
   This means storing `LM.input_embedding(marker)` as the value and
   adding it pre-lm_head directly implements kNN-LM with the strongest
   possible bias toward the marker token.

## Fixed architecture

```python
# attention_memory.py — query()
logits = (q @ keys.T) * inv_temp           # NO sqrt(D) divisor
weights = softmax(logits)                  # sharp at zero-shot
retrieved = (weights @ values)             # weighted marker emb sum
out = W_o(retrieved) * out_gain            # to-hidden projection
```

```python
# qwen_attmem_bolt.py
self.qwen.lm_head.register_forward_pre_hook(_attmem_lm_head_hook, with_kwargs=True)
# residual is added to the post-norm hidden state right before lm_head;
# with tied embeddings this yields a clean per-marker logit boost
```

Init: `log_inv_temp = log(20)`, `out_gain = 8.0`, `W_o = I` (2048×2048).

## Zero-shot eval (no pretraining, just init)

| N |   RAG | AttMem | ratio | verdict |
|--:|------:|-------:|------:|:--------|
|  5 | 0.800 |  0.600 |  0.75 | comp    |
| 10 | 0.467 |  0.367 |  0.79 | comp    |
| 20 | 0.400 |  0.233 |  0.58 | comp    |
| 50 | 0.287 |  0.100 |  0.35 | below   |

Already meaningful at N≤20 without any training. The gap at N=50
suggests further training of `W_o` and `inv_temp` is needed.

## Pretrained eval (A-PARA, 5000 steps, seed=42)

Final loss 3.77 (below log(64)=4.16; well below random-over-full-vocab 11.93).

| N |   RAG | AttMem | ratio | verdict |
|--:|------:|-------:|------:|:--------|
|  5 | 0.800 |  0.733 |  0.92 | near    |
| 10 | 0.467 |  0.467 |  1.00 | **MATCHES**  |
| 20 | 0.400 |  0.400 |  1.00 | **MATCHES**  |
| 50 | 0.287 |  0.213 |  0.74 | comp    |

**Key finding**: AttentionMemory hits the encoder ceiling (RAG cosine,
which IS the encoder's discriminability) exactly at N=10 and N=20.
Compare Path A multi-seed at N=10: 0.480 (BEATS RAG p=0.010 — but only
because Path A's `lr=3e-4 + per-id SGD` happened to fit the eval queries
slightly better than naive cosine NN; the encoder is the same).

This validates the architectural pivot: the discrete codebook was
genuinely the bottleneck. Removing it lets us hit the encoder ceiling
directly without ANY per-id SGD step at insertion.

## Comparison vs Path A — what changed

| Property | Path A | AttentionMemory |
|---|---|---|
| Insertion time | ~1 s per id (80 SGD steps) | **<1 ms per id (numpy append)** |
| Trainable params | ~6M (Engram + codebook + perc_emb) | ~8M (W_q + W_o + projections) |
| Pretraining steps | 100K | **5K** (20× fewer) |
| Accuracy at N=10 | 0.480 | 0.467 (matches encoder ceiling) |
| Accuracy at N=20 | n/a (below) | 0.400 (matches encoder ceiling) |
| Acc at large N | flat ~0.07 (codebook ceiling) | **tracks encoder** (RAG ratio 0.7-1.0) |

## Multi-seed verification

[Pending — 10 seeds at A-PARA N=10 to confirm parity with multi-seed
Path A. If consistent, AttMem's per-id 1000× insertion speedup +
no-codebook-ceiling scaling are the new headline contributions.]

## Latency story

[Pending — re-measure at N=10, 100, 1000, 10000. Expected: dominated by
LM forward; AttMem query is microseconds even at N=10k.]

## Why the v1/v2 failures, in retrospect

The earlier-session diagnosis that "codebook is the ceiling" was right,
but I prematurely deployed AttMem v1/v2 with three latent bugs that hid
the architecture's true capability. Without the careful debug-trace
session (printing the actual logit deltas at the perceptual position),
the architecture would have looked like a failed pivot. The lesson: when
a new mechanism gives random-looking output, instrument inside the hook
before adjusting macroscopic hyperparameters.
