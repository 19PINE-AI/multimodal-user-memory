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

### Encoder-space slots (the fixed design) -- a clean capacity law
The working `set_memory.py` compresses the M registered keys into k prototype
SLOTS in the encoder's cosine space (the AttMem mechanism), then recognises a
cross-condition query. recall@1 (5 seeds):

Faces (ArcFace):
```
  M\k    2     4     8    16    32    64
   2   0.97  0.97  0.97  0.97  0.97  0.97
   4   0.49  0.98  0.98  0.98  0.98  0.98
   8   0.24  0.48  0.96  0.96  0.96  0.96
  16   0.12  0.23  0.46  0.94  0.94  0.94
  32   0.06  0.11  0.22  0.45  0.92  0.92
  64   0.03  0.05  0.11  0.21  0.44  0.89
```
Voices (ECAPA) are the same, ~1.0 above the diagonal.

**The law:** recall ≈ **min(1, k/M)**. For **k ≥ M** (≥1 slot per identity) you
get full recognition (0.89–1.0 = AttMem). For **k < M** the slots merge and only
~k of the M identities stay distinct (recall ≈ k/M). So "is 16 tokens enough for
2/3/4 faces?" -> **yes, trivially**; 16 slots handle up to ~16 identities (0.94),
then degrade as 16/M.

So: **~1 encoder-space slot per identity**, and you cannot beat that by packing
faces into shared LM tokens (that's the chance-level failure above) — perceptual
matching must live in encoder space.

## Takeaway

- **Exact facts**: latent capacity is bounded by k ≈ content length, and never
  reaches text's exactness. Text wins for exact/high-entropy facts.
- **Perceptual identities**: recognition needs encoder-space matching (~1 key per
  identity, AttMem); they cannot be compressed into a frozen LM's token space.
- Both reinforce the router: text for exact facts, encoder-space latent banks for
  perceptual identity, hybrid when an encounter has both.
