# Reviewer-grade baselines (experiments requested for the paper)

## Exp 1 -- Learned-metric baseline (the make-or-break one)

AttMem at eval ranks by cosine over the RAW encoder embeddings; it never learns a
similarity. So: does a similarity learned on the same encoder features (same
train identities, same 50/50 split, same recall@1 protocol) match AttMem?

Faces (ArcFace, 1090 train / 1090 eval ids), recall@1:

| N | raw (RAG) | LDA | whiten | AttMem |
|---|---|---|---|---|
| 5 | 0.933 | **1.000** | 0.933 | 0.933 |
| 10 | 0.933 | 0.967 | 0.933 | **1.000** |
| 50 | 0.773 | 0.707 | 0.773 | 0.733 |
| 300 | 0.734 | 0.667 | 0.744 | 0.641 |
| 1000 | 0.767 | 0.724 | **0.776** | (n/a) |

Painter style (CLIP-mid), recall@1 (AttMem = single seed 42, not the 5-seed 0.64):

| N | raw | LDA | whiten | AttMem(s42) |
|---|---|---|---|---|
| 5 | 0.400 | 0.200 | **0.600** | 0.467 |
| 10 | 0.400 | 0.333 | 0.367 | 0.467 |

**Finding.** A *learned metric on the same encoder* matches or beats AttMem's
small-N gain. Faces: LDA = AttMem at small N (LDA 1.0 vs AttMem 0.93 at N=5;
AttMem 1.0 vs LDA 0.97 at N=10 -- they trade, neither dominates). Style: a plain
within-class whitening hits 0.60 at N=5, above raw 0.40 and above this seed's
AttMem. So the "sharper ruler" is essentially **metric learning**: the gain over
*raw* cosine is real, but it is not unique to the parametric memory -- a
closed-form LDA or whitening on the encoder captures it.

At scale (N>=300) raw/whitened cosine beats BOTH AttMem and LDA -- the small-N
advantage of any learned re-encoding does not generalise to large held-out pools.

**Implication for the paper.** Qualify "beats embedding retrieval": AttMem beats
*raw* cosine, but a learned metric on the encoder matches it, and at scale raw
cosine wins. AttMem's distinct value is therefore NOT recall@1 superiority but
the in-model parametric memory: O(1) registration, composition with text memory,
and the capacity laws (Fig 7). The recall@1 headline should be reframed around
that, with a learned-metric column added to the scorecard.

Caveats: the contrastive linear head I trained underperformed (weak
instantiation; LDA/whiten are the strong learned baselines). recall@1 here uses
the paper's single eval draw (rng 99); a multi-seed eval would tighten the
numbers but not the qualitative conclusion.
