# Session 17 — pivot to continuous attention memory

**Date:** 2026-05-17

After 17 sessions of characterising the discrete-codebook architecture's
ceiling, we pivot to continuous attention memory. The user's
observation: discrete codebooks were inherited from DeepSeek-Engram's
text-N-gram-hash mechanism (where input *is* discrete). For continuous
perceptual content the natural primitive is attention over a memory
bank, not a hash table over a quantised codebook.

## What we keep from Path A

Everything that worked is encoder-agnostic and addressing-agnostic:

- **Bolt-on framework**: frozen pretrained LM + forward pre-hook at one
  attached layer that adds a residual into the hidden state.
- **Surgical insertion philosophy**: per-identity row writes; no per-id
  gradient training of the rest of the model.
- **Multi-modal**: vision and audio handled by per-modality submodules.
- **Pretrained recall recipe**: generic-NTP-style cross-recurrence
  pretraining of the small bolt-on parameters.
- **Eval harness**: PerceptMem scorecard, the multi-seed verification
  infrastructure, the latency benchmark, RAG-comparison code.

## What we drop

- **Codebook discretization** (k-means / STE quantiser): replaced with
  direct use of the continuous encoder embedding as a memory bank key.
- **N-gram hash on perceptual code**: replaced with cross-attention
  between an LM-side query and the bank keys.
- **Per-row Engram tables**: replaced with a growing append-only memory
  bank of (key, value) pairs.
- **Per-id SGD insertion (80 steps)**: replaced with O(1) row append
  (no gradient step needed at insertion time).

## New primitive — `AttentionMemory`

```
key_i   = encoder(reg_sample_i)             ∈ R^D    # frozen encoder output, L2-normalised
value_i = LM.input_embedding(marker_token_i) ∈ R^H   # the LM's existing input embedding
```

At the attached LM layer, the hook computes:

```
query  = W_q · hidden_state_at_attached_pos   ∈ R^D     # learned projection (small)
logits = (keys @ query) / sqrt(D)              ∈ R^N
weights = softmax(logits / tau)                ∈ R^N    # tau is a learned temperature
retrieved = weights @ values                   ∈ R^H
hidden_state ← hidden_state + W_o · retrieved          # learned output projection
```

**Trainable parameters**: just `W_q` (H × D), `W_o` (H × H), and `tau`
— roughly a few hundred K params. Everything else (LM, encoder, bank
keys, bank values) is frozen at insertion / use time.

**Insertion**: literally one numpy append for the bank arrays. No SGD.

**Pretraining**: train W_q, W_o, tau on synthetic recall tasks. Build
random (key, value) banks of size 5–500 from the training pool; run
forward; loss = NTP loss biased toward the marker token. Frozen
encoder, frozen LM.

## Expected properties (predictions to test)

| Property | Path A (discrete) | AttentionMemory (this) | RAG-with-context |
|---|---|---|---|
| Per-query compute | O(1) hash lookup | O(N·D) matmul | O((T+N)² · D) attention |
| Per-id storage | O(K) amortised | O(D + H) per id | O(D + T_tokens) per id |
| Insertion time | ~1 s (80 SGD steps) | **~ms (numpy append)** | ms (append embedding) |
| Pretraining target | Engram + codebook + perc_emb (~6M params) | **W_q, W_o, tau (~200K params)** | none |
| Accuracy ceiling | codebook same-code rate | **encoder discriminability** | encoder discriminability |
| LM integration | residual via hook | **residual via hook** | prompt context |

The single thing AttentionMemory gives up vs Path A is O(1) query.
For N=10,000 with D=512, that's a 5 MFLOP matmul — microseconds,
dwarfed by the 21 ms LM forward. The latency story barely changes
(Path A vs AttentionMemory both microseconds beyond the LM forward; RAG
with context grows quadratically in T).

## Specific hypotheses

1. **At N=10 on A-PARA**: AttentionMemory should match-or-beat Path A's
   BEATS-RAG result (0.48 vs RAG 0.42, p=0.010). The mechanism is
   cleaner (no codebook miss), so the margin may grow.

2. **At N=100–1000 on V-XC-ID**: AttentionMemory's accuracy should
   track RAG closely — both bounded by encoder cross-condition
   discriminability, not by an extra discretization step. Expected
   retr@1 0.6–0.8 at N=700, vs Path A's 0.05–0.07 at the same N.

3. **Latency at N=10k**: AttentionMemory query ≈ Path A query (21 ms)
   + a small constant (microseconds for the matmul). Still 30× faster
   than naive RAG-with-LM-context.

4. **Pretraining converges faster** because there are 30× fewer
   trainable parameters. Maybe 1–5K steps instead of 100K.

5. **Insertion is genuinely O(1) in wall-clock**: append + done.
   Path A's "O(1) in N" was technically true but the constant was
   ~1 s of SGD per identity. AttentionMemory's constant is microseconds.

## Implementation plan

```
src/nanochat_mm/attention_memory.py      # the new primitive (~150 LoC)
src/nanochat_mm/qwen_attmem_bolt.py      # bolt-on wrapper (~150 LoC, mostly copy)
src/nanochat_mm/attmem_pretrain.py       # pretraining recipe
src/nanochat_mm/attmem_eval.py           # eval at multiple N (re-use PerceptMem code)
```

Estimated effort: 2–3 days of coding + experiments. The existing
infrastructure (encoder embeddings, train/eval split, RAG ceiling
function, multi-seed harness, scale-eval harness) all carries over
directly.

## Codebook becomes ablation, not the headline

For the paper, the discrete-codebook results from sessions 5–16 become
the *motivation* section's ablation: "Here is what a discrete-codebook
parametric memory primitive achieves; here is why its codebook
discretization caps recall at large N regardless of K tuning, encoder
quality, or pretraining compute; here is the natural successor that
removes the bottleneck while keeping all the bolt-on framework's
desirable properties."

That's a *stronger* contribution than "we built a codebook and it
works at small N" — it's a principled exploration of the design space.
