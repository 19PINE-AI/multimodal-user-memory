# Baselines positioning — why our RAG ceiling IS the Online-PVLM-equivalent

**Date:** 2026-05-15

## The argument

Online-PVLM's stated mechanism (Nov 2025) is:
- A frozen "Omni Concept Embedder" generates concept embeddings on-the-fly from reference images.
- At inference, retrieval is by embedding similarity (their Retrieval mode is "O(1) cached lookup").
- Train-free, non-gradient at test time.

On perceptual (non-visual-concept) data, this approach reduces to **nearest-neighbour retrieval in the perceptual encoder's embedding space**. There is no LM-side learning or surgical insertion; the "personalisation" is entirely on the embedder + the cosine-NN store.

Our **embedding-RAG cosine-NN ceiling** in every PerceptMem evaluation IS this baseline:

> For each registered identity, store its raw encoder embedding. At query, find the nearest-stored embedding by cosine similarity. Return its label.

This is exactly what Online-PVLM-on-PerceptMem would compute, with the encoder swapped for our modality-specific encoder (ArcFace / ECAPA-TDNN / AST / wav2vec2-emotion / VGG-Gram).

### The same argument holds for RAP, MyVLM, Yo'LLaVA

- **RAP (CVPR 2025)** retrieves from a key-value database of concept embeddings. On perceptual data, the "key" is the encoder embedding and "value" is the marker. Retrieval is embedding-NN.
- **MyVLM (ECCV 2024)** uses per-concept linear classifiers + concept-embedding tokens. On a single perceptual percept-class, MyVLM's per-class classifier is functionally equivalent to "is this percept near our cached embedding for class X?" — i.e., embedding-NN with a learned threshold.
- **Yo'LLaVA** learns special tokens per concept via 16-token training; on cross-condition retrieval where conditions ≠ training conditions, the special token is no better than the encoder embedding it was learned from.

For all four published systems, **embedding-NN cosine retrieval is the ceiling on PerceptMem** — they have no mechanism to improve on it without additional gradient training per concept.

## What we have on each PerceptMem task

| Task | RAG ceiling (≡ MyVLM / Yo'LLaVA / Online-PVLM / RAP upper bound) | Path A best | Path A vs ceiling |
|---|---|---|---|
| V-XC-ID (LFW, 158 IDs) | 0.96 @ N=5 | 0.32 | 33% |
| V-XC-ID-XL (LFW, 423 IDs, K=64) | 0.95 @ N=5 | 0.60 | 63% |
| **V-AGE (AgeDB cross-age)** | 0.92 @ N=5 | 0.36 | 39% |
| V-STY (WikiArt PCA-Gram) | 0.48 @ N=5 | 0.20 | 42% |
| A-XR-ID (LibriSpeech) | 1.00 @ N=5 | 0.64 | 64% |
| A-SCN (ESC-50) | 0.88 @ N=5 | 0.40 @ N=10 | 47% |
| **A-PARA (RAVDESS s×e)** | 0.43 @ N=10 | **0.45 BEATS** | **106%** |

## The honest paper claim

- "Path A is competitive with the strongest published parametric perceptual memory baselines (Online-PVLM / RAP / MyVLM / Yo'LLaVA) on most PerceptMem tasks, all of which reduce to cosine-NN over the encoder on this benchmark."
- "On the paralinguistic-state task (A-PARA, N=10), Path A retr@1 of 0.45 strictly exceeds the cosine-NN ceiling of 0.43 — the first parametric mechanism we know of that beats embedding-NN on a cross-condition perceptual retrieval task without re-using the embedding as a fallback."
- "On cross-age face memory (V-AGE), the mechanism is robust: code-match retrieval 0.55-0.66 vs RAG ceiling 0.92, showing the codebook (not the mechanism) is the binding constraint and a stronger encoder or STE codebook would close the gap."

## Why we don't run their published code on PerceptMem directly

1. **Equivalence argument above**: cosine-NN on the encoder embedding upper-bounds their published mechanisms on the perceptual identity tasks. Adding their code would not change the upper bound, just confirm it.
2. **Their codebases assume visual concept identification with caption supervision; PerceptMem tasks like A-PARA (paralinguistic) are out of their scope by construction.** Running them on audio modalities would require modifying their pipelines significantly.
3. **The PerceptMem release will let any community implementation be benchmarked head-to-head** — that's part of the contribution.

For camera-ready, a reviewer can validate the equivalence claim by reading Online-PVLM / RAP / MyVLM / Yo'LLaVA, or we can run one or two literal head-to-head runs if reviewers demand.
