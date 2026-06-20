# Architecture — the mechanism

The whole memory is a table of rows, one per registered perception. This document
describes the continuous attention memory primitive (`src/nanochat_mm/attention_memory.py`),
how it reads, the four bug-fixes that made it work, and the structural design rule
that says when it helps.

## A perception is a row in a bank

Row *i* holds:

- **key** `k_i` — the L2-normalized embedding the modality's frozen, off-the-shelf
  encoder produces for the perception (ArcFace for a face, ECAPA-TDNN for a
  voiceprint, mid-layer CLIP for a painter style). Lives in *encoder space*, e.g.
  R^512 for ArcFace.
- **value** `v_i` — the language model's **own** input embedding for a marker
  token assigned to that identity (e.g. `<id_11>`). Lives in *model hidden space*,
  e.g. R^2048 for Qwen2.5-3B.

Storing the model's *native* vector as the value (rather than an arbitrary learned
one) is deliberate: it makes the eventual logit boost for the correct marker clean
and self-reinforcing rather than an accident of two unrelated vectors.

There is one bank per modality (vision and audio are separate banks). Registration
is a single tensor append — no SGD step:

```python
k = l2_normalize(encoder(perception))   # key   (encoder space)
v = model.input_embedding[marker]       # value (model's own space)
bank.K = torch.cat([bank.K, k[None]])   # O(1)
bank.V = torch.cat([bank.V, v[None]])
```

## Reading the memory is one step of attention

A forward pre-hook is attached at the frozen model's **output head**. When a
perception arrives at generation time, the model forms a query `q` from the hidden
state and computes:

```
w = softmax(β · qᵀK)        # attention over the bank's keys
r = wᵀV                     # weighted blend of the matching markers' values
h' = h + g · W_o · r        # residual added to the hidden state, just before logits
```

- `β` — a **learned sharpness** (do NOT divide by √D; see below).
- `g` — a **learned gain**.
- `W_o` — a learned projection, initialized to identity.

In words: the perception softly looks up the bank, pulls back a blend of the
matching markers' model-side vectors, and nudges the next-token prediction toward
them. The blend is the entire mechanism — about **8M trainable parameters** over
the frozen model (the primitive itself — `W_q`, `W_o`, `τ`/`β`, `g` — is ~200k;
the 8M figure includes the per-modality query projections used in the full bolt).

## Getting it to work was four bug-fixes, not four ideas

The architecture above is the fourth iteration; three earlier versions produced
random output despite healthy-looking loss curves. The fixes are unglamorous and
are where the time went:

1. **Do not divide the attention logits by √D.** With normalized keys it flattens
   every cosine difference into near-uniform attention.
2. **Initialize the sharpness β high**, so attention is decisive from the first
   step.
3. **Attach the hook at the output head**, not at an intermediate layer, so the
   residual reaches the logits undiluted.
4. **Give the gain a large learnable value**, so the nudge actually overcomes the
   model's built-in reluctance to emit unusual marker tokens.

A fifth choice — **varying the bank size during training** (curriculum) — is not a
correctness fix but is essential for scaling to large memories: fixed-size
training collapses from 0.63 to 0.20 recall at 700 identities due to a train/test
size mismatch.

## Why register, rather than fine-tune

The economics are the point:

- **Insertion:** O(1) wall-clock `torch.cat` — ~0.5 ms to add a thousand rows.
- **Recall:** constant-time (~15 ms) regardless of bank size — the lookup is a
  tiny matmul dwarfed by the model's own forward pass.
- **The alternative** (feeding registered perceptions in as context) grows
  linearly per query and runs out of memory past the context window. Our
  mechanism is **52× faster at 1,000 identities** and is the only one of the two
  still functioning at 10,000.

Memory a companion updates after every conversation has to be this cheap to write.

## The key/value-orthogonality design rule

The cleanest finding came from running the memory inside a vision-language model
(Qwen2.5-VL) that sees raw face images through its own visual front-end:

- With an **external** face encoder (ArcFace) as the key → reproduces the win,
  perfect recall at 10 identities.
- With the VLM's **own** vision tokens as the key → cannot beat retrieval at all.

The difference is geometric: the VLM's vision tokens already live in the model's
hidden space, so the key and the value point the same way and the model's
representation offers no *new* direction to discriminate along. The rule:

> **The memory's key must be orthogonal to the model's value space for the second
> ruler to be sharper than the first.**

This is why a purpose-built encoder (ArcFace, ECAPA, CLIP) beats a general-purpose
one here.

## Portability

The mechanism is not tied to one model. It transfers from Qwen2.5-3B to 7B and to
Llama-3.1-8B (which, same recipe, beats retrieval and edges out Qwen-3B at large
memories), needing only an automatic adjustment for whether a model ties its input
and output embeddings. It is recipe-sensitive: Mistral-7B did not converge in the
same budget and likely needs gentler training.

## See also

- [`BENCHMARK.md`](BENCHMARK.md) — what we measure it on.
- [`RESULTS.md`](RESULTS.md) — the numbers.
- Paper §3 (`paper/body.tex`) and Appendix B (anatomy of a single recall).
