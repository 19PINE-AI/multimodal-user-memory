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
~0).

**More k does NOT rescue it** (M x k sweep, exact-match):

| M \ k | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 2 | 0.06 | 0.11 | 0.10 | 0.04 |
| 4 | 0.00 | 0.01 | 0.01 | 0.00 |

Exact-match sits at the floor across all k (k=128 >> the content of 2-4 short
codes, and no better than k=32). So multi-code retrieval is **retrieval-brittle,
not capacity-limited** -- the bottleneck is content-based binding/lookup (match a
name to its exact code among several), which a compressed latent cannot do at any
k. A text store does the lookup trivially and exactly.

**The sharp contrast (the headline of this study):**
- **Perceptual identity is a CAPACITY question** -- set_memory recall = min(1, k/M);
  more slots linearly hold more identities, graceful degradation. Latent scales.
- **Exact-fact retrieval is a BINDING question** -- code_memory collapses at M>=2
  and adding latent tokens does not help at all. Latent fails at any k.

So latent memory scales for perceptual recognition (~1 slot/identity) but cannot
do exact-fact associative retrieval regardless of budget -> exact facts need a
text store. This is the mechanistic core of the router.

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

### Is min(1,k/M) fundamental, or just k-means? (Exp 5)
`learned_compressor.py` replaces k-means slots with slots learned by gradient
descent (self-supervised on noise-augmented queries; never sees the eval query).
recall@1, 5 seeds, ArcFace LFW-XL:

| M | k | min(1,k/M) | k-means | learned_hard | learned_soft |
|---|---|---|---|---|---|
| 8 | 4 | 0.500 | 0.500 | 0.500 | 0.950 |
| 16 | 8 | 0.500 | 0.487 | 0.475 | 0.912 |
| 32 | 8 | 0.250 | 0.225 | 0.244 | 0.756 |
| 32 | 16 | 0.500 | 0.456 | 0.481 | 0.906 |
| 64 | 16 | 0.250 | 0.219 | 0.237 | 0.778 |

**learned_hard ≈ k-means ≈ min(1,k/M).** The law is FUNDAMENTAL: k hard slots can
keep at most k of M identities separable (pigeonhole), and learning the slots does
not beat clustering them. `learned_soft` (k slots + an M×k soft code per identity)
shatters the law (0.95 at M=8,k=4) but pays **O(M·k) storage** — no longer
compression to k tokens; at O(M) storage you may as well keep the M keys (= AttMem
with M slots = RAG). So min(1,k/M) is the real capacity frontier for a fixed
k-slot budget, and k-means already sits on it.

## Takeaway

- **Exact facts**: latent capacity is bounded by k ≈ content length, and never
  reaches text's exactness. Text wins for exact/high-entropy facts.
- **Perceptual identities**: recognition needs encoder-space matching (~1 key per
  identity, AttMem); they cannot be compressed into a frozen LM's token space.
- Both reinforce the router: text for exact facts, encoder-space latent banks for
  perceptual identity, hybrid when an encounter has both.

## Capacity law across ALL modalities (grid standard)
set_memory.py per modality (5 seeds, 20 sets), recall@1 of M identities compressed
into k prototype slots. The diagonal (k=M, one slot per identity) is the encoder's
own recall at M; below it, slots merge.

| modality | encoder diag (k=M, M=2..max) | mean|recall - min(1,k/M)| | normalized* |
|---|---|---|---|
| face (ArcFace) | 0.98..0.88 | 0.036 | 0.002 |
| voice (ECAPA) | 1.00..0.99 | 0.003 | 0.001 |
| acoustic (AST) | 0.95..0.76 | 0.092 | 0.012 |
| style (CLIP) | 0.76..0.20 | 0.337 | 0.070 |
| tone (wav2vec) | 0.94..0.26 | 0.191 | 0.125 |
*normalized = recall(M,k)/recall(M,k>=M) vs min(1,k/M)

**Refined law: recall(M,k) ~ min(1, k/M) * C(M)**, where C(M) is the encoder's own
recall at M registrations. The slot-compression factor min(1,k/M) holds across ALL
five modalities (normalized error <=0.012 for the strong-encoder ones, <=0.13 for the
weak style/tone encoders). The absolute ceiling C(M) is ~1 for strong encoders
(face/voice/acoustic) and lower for the weaker style/tone encoders -- consistent with
recall = encoder everywhere. min(1,k/M) is the special case C(M)=1.
