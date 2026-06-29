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
instantiation; LDA/whiten are the strong learned baselines). recall@1 above uses
the paper's single eval draw (rng 99).

## Exp 4 -- Statistical rigor: the eval draw is high-variance (the bigger problem)

The paper's significance tests vary only the *training* seed and compare AttMem
against retrieval at a SINGLE eval draw (`np.random.default_rng(99)` -- the
registration photo + query sampling is fixed). That draw turns out to be
high-variance. Re-running raw/LDA/whiten across 20 eval draws (registration/query
resampling, same seed-42 split):

Faces, recall@1 mean±std over 20 draws:

| N | raw | lda | whiten |
|---|---|---|---|
| 5 | 0.983±0.037 | 0.987±0.035 | 0.983±0.037 |
| 10 | 0.953±0.042 | 0.930±0.047 | 0.955±0.039 |
| 300 | 0.747±0.017 | 0.693±0.016 | 0.751±0.015 |

Style, recall@1 mean±std over 20 draws:

| N | raw | whiten | attmem(s42,1draw) |
|---|---|---|---|
| 5 | 0.480±0.109 | 0.487±0.126 | 0.467 |
| 10 | 0.422±0.093 | 0.410±0.107 | 0.467 |

**The headline retrieval values were lucky low draws.** Face N=10 retrieval is
0.953±0.042 across draws, NOT the 0.933 the paper's p=0.006 test treats as a fixed
constant; style N=5 retrieval is 0.480±0.109, not 0.40. The eval-draw std (±0.04
face, ±0.11 style) is comparable to or larger than the claimed AttMem gains, so
the single-draw significance overstates certainty.

The correct test pairs AttMem and retrieval on the SAME draws and tests the
per-draw difference. `evaluate_paired` does this; result below.

### The headline recall wins do NOT survive paired evaluation
4 training seeds (42-45) x 20 eval draws, AttMem and RAG on IDENTICAL samples:

| Cell | Paper headline | Paired AttMem | Paired RAG | Δ | paired p | verdict |
|---|---|---|---|---|---|---|
| Face N=10 | +5.9pp p=0.006 | 0.942±0.035 | 0.948±0.033 | -0.006 | 0.002 | BEHIND |
| Style N=5 | +24pp p=0.015 | 0.476±0.116 | 0.473±0.102 | +0.002 | 0.77 | TIE |
| Style N=10 | +6pp p=0.009 | 0.357±0.081 | 0.428±0.074 | -0.072 | <.001 | BEHIND |

**Root cause (worse than a lucky draw): a sampling mismatch.** The paper's AttMem
came from `evaluate()`, which shuffles each id's samples TWICE (separate reg and
query loops); the RAG baseline came from `embedding_rag_ceiling()`, which shuffles
ONCE (reg=idxs[0], queries=idxs[1:]). So AttMem and RAG were scored on DIFFERENT
query samples. Validation: at draw 99, `evaluate_paired` (both on the same sample)
gives AttMem=RAG=1.0 for face N=10 -- the 0.992-vs-0.933 gap was the sampling
artifact, not a real win. The tiny paired diff-std (~0.019 face) confirms AttMem's
ranking tracks raw cosine almost exactly, because its keys ARE the encoder
embeddings.

**Conclusion: on random banks AttMem has NO recall advantage over raw cosine.**
The three headline "random" cells are parity-or-behind. The paper's recall story
must move to (a) the adversarial/look-alike regime (training the bank to expect
look-alikes -- a capability RAG lacks; +14 to +71pp single-draw, paired test in
flight) and (b) the in-model + capacity-law contributions, which need no recall
win at all.

## Exp 4b -- Do the adversarial wins survive? (mechanism + shuffle check)

Single-position paired adversarial (target always bank slot 0, marker 30001), 20 draws:
- Face K=19:  AttMem 0.972 vs RAG 0.852  Δ+0.120  p<.0001
- Style K=19: AttMem 0.965 vs RAG 0.258  Δ+0.707  p<.0001
- Tone K=19:  AttMem 0.954 vs RAG 0.216  Δ+0.738  p<.0001

**Mechanistic red flag.** Full forward: marker logit(m) = lm_head[m]·h_old +
gain*Σ_j w_j (lm_head[m]·W_o(v_j)), with v_j = slot j's marker embedding and w_j =
encoder-cosine attention. The blended values carry NO per-key identity, so W_o
cannot recover the target when a look-alike has higher cosine -- AttMem's attention
weights the look-alike higher too. Therefore AttMem CANNOT exceed encoder cosine on
adversarial banks. The only constant is that the target always wears marker 30001 at
slot 0, and lm_head[30001]·h_old gives it a baseline boost (gain=8 doesn't swamp
h_old). Prediction: the win is a first-slot/first-marker artifact and collapses when
the target slot is randomised (ATTMEM_ADV_SHUFFLE=1). Shuffle check running.

If confirmed: ALL recall-superiority claims (random AND adversarial) are artifacts.
The honest paper = recall equals the encoder ceiling everywhere (training-free,
in-model), no recall win ever. This actually STRENGTHENS the "training-free, works
out of the box" thesis: training never helped recall; the apparent training wins
were eval artifacts.
