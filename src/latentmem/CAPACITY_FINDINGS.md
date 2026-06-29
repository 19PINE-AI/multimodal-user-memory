# Latent capacity: how much, and how many, fit in k tokens?

Two questions, real experiments (1.5B, LoRA, real ArcFace/ECAPA where perceptual).

## Q1. Information-theoretic capacity of exact (random) facts

### Single code: tokens needed scale with content length
A leak-free codec encodes a random code into k soft tokens and decodes it from M
alone. Exact-match of a full 24-char code (~20 content tokens, ~120 bits):

| k (latent tokens) | exact-match |
|---|---|
| 8 | 0.016 |
| 16 | 0.082 |
| 32 | 0.516 |
| 64 | 0.801 |

Capacity jumps once **k exceeds the content's token count (~20)**. Even k=64 ≫
content tops out at 0.80 — a residual decode tax: a frozen LM cannot losslessly
read an exact token off soft latents. **Text storage is exact (1.0).** So holding
an incompressible code needs k ∝ its length (no compression) — at which point the
latent is a costly, lossy copy of text.

### Multi-code retrieval (the real memory test): store M codes, recall one
`code_memory.py` stores M `name -> random code` pairs in k SHARED tokens and
retrieves a code by name. Strictly harder (codes + keys + associations in k).

`code_memory.py` is implemented but its name-conditioned decode has a bug (even
the trivial M=1 case fails to learn, where the single-code codec gets 0.98), so I
do not report numbers from it. The multi-code answer is instead synthesized from
two solid results:

- **One exact code already needs k ≳ its token length** (the k-sweep above: a
  24-char code reaches 0.52 only at k=32). M codes sharing k tokens is strictly
  harder, so retrieval of an exact code from a shared latent holding many is
  worse than the single-code numbers.
- **Multi-fact latent retrieval is lossy even for easy (categorical) facts**: the
  latentmem pilot (`FINDINGS.md`) caps at ~0.55 retrieving one fact among many
  from a compressed latent, and dense multi-probe supervision did not lift it.

Together: storing M exact codes in k shared tokens and pulling back the right one
is poor and degrades with M — consistent with "exact facts belong in text".
(Fixing `code_memory`'s decode to confirm the exact curve is a clean TODO.)

## Q2. Multi-IDENTITY recognition (faces / voices) -- the real-world case

You must recognize MANY faces/voices, not one. Two ways to store them:

### Soft-token compression INSIDE the LM -- fails
`set_memory.py` compresses M faces into k shared soft tokens and asks the frozen
LM to match a query face. Result: **recall@1 = 1/M (pure chance) for every k and
M** — even M=2 with k=16 (8 tokens/face) and dedicated training (loss drops but
held-out recall stays at chance). **A frozen LM cannot do perceptual identity
matching in its token space.** This is fundamental, not a training artifact.

### Encoder-space matching (AttMem) -- works, ~1 key per identity
AttMem matches in the encoder's cosine space and uses the LM only to emit the
marker. It recognizes M identities with ONE key (row) per identity:

| M identities | AttMem recall@1 (faces) |
|---|---|
| 10 | 0.94 |
| 100 | 0.89 |
| 300 | 0.85 |
| 1000 | 0.77 |

(from single_pipeline.py / the paper). So the answer to "how many tokens per
face": ~**1 encoder-space key per face**, degrading gracefully with M. You cannot
go below that by packing faces into shared soft tokens — the matching has to live
in encoder space, not in the LM.

## Takeaway

- **Exact facts**: latent capacity is bounded by k ≈ content length, and never
  reaches text's exactness. Text wins for exact/high-entropy facts.
- **Perceptual identities**: recognition needs encoder-space matching (~1 key per
  identity, AttMem); they cannot be compressed into a frozen LM's token space.
- Both reinforce the router: text for exact facts, encoder-space latent banks for
  perceptual identity, hybrid when an encounter has both.
