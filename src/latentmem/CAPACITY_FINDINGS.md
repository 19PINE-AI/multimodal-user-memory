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

`code_memory.py` stores M `name -> random code` pairs in k=16 SHARED tokens and
retrieves one by name (autoregressive LM-completion `M ; "name: " -> code`, with
token-consistent encoding -- the doc is encoded from the same ids used as the
decode target; the earlier failure was a BPE context-tokenization mismatch).
Retrieval exact-match (6-char codes, k=16):

| M codes | exact-match |
|---|---|
| 1 | 0.87 |
| 2 | 0.06 |
| 4 | 0.00 |
| 8 | 0.00 |
| 16 | 0.00 |

**A shared latent holds ONE exact code (0.87) but collapses the moment you store
2+ and must retrieve the right one by name** (0.87 -> 0.06 from M=1 -> M=2, then
~0). Note k=16 > content tokens at M=2, so this is not raw capacity -- it is
content-based associative retrieval of exact content, which a compressed latent
does very poorly. A text store does the same lookup trivially and exactly. This
matches the latentmem multi-probe result (~0.55 even for easy categorical facts)
and is far worse for exact codes.

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
