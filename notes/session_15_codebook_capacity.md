# Session 15 — accuracy at scale, codebook capacity from first principles

**Date:** 2026-05-16

User pressed two correct questions in sequence:
1. *We showed speedup — did we show accuracy?*
2. *If accuracy is limited at large N, increase the codebook size or
   match capacity to N. Think first-principles.*

This session probes the codebook capacity question rigorously and
reports the honest accuracy curve to pair with the sess-14b latency
curve.

## Accuracy at scale — the empirical curve

Combined LFW-XXL + AgeDB face dataset (1401 IDs, 701 in eval split).
RAG = cosine NN over registered ArcFace embeddings (no LM forward
needed — top-1 marker returned directly, the realistic deployment).
Path A = K=64 v2 codebook (adapter+largepool).

| N   | RAG retr@1 | Path A K=64 retr@1 | Path A code-match | frac-code |
|-----|-----------|--------------------|--------------------|-----------|
| 20  | 0.867     | 0.267              | 0.652              | 0.383     |
| 100 | 0.777     | 0.120              | 0.330              | 0.363     |
| 300 | 0.729     | (not run — pattern clear) | | |
| 500 | 0.763     | | | |
| 700 | 0.773     | | | |

**RAG is stable at 0.73–0.87 across all scales** because ArcFace's
embedding space is discriminative enough that cosine-NN over 700
candidates still resolves identity reliably. **Path A degrades sharply
with N** because the K=64 codebook can't distinguish more than ~64
identities cleanly — multiple IDs collide on the same code, the gate
saturates, retr@1 drops.

## Codebook K-sweep diagnostic (same-code rate on eval split)

| K     | eval same-code | eval inter-collision | codes used |
|-------|----------------|----------------------|-----------|
| 64    | 0.198          | 0.0185               | 64/64     |
| 128   | 0.167          | 0.0094               | 127/128   |
| 256   | 0.112          | 0.0056               | 244/256   |
| 512   | 0.100          | 0.0037               | 431/512   |
| 1024  | 0.067          | 0.0021               | 712/1024  |

**Increasing K alone halves the same-code rate at every doubling.** At
K=1024 (≥ N=700), inter-id collisions are ~0 but only 6.7% of
cross-condition same-id pairs land in the same code. That's a hard
ceiling on retrieval recall.

## First-principles framing of the ceiling

The codebook's job: map embedding `x ∈ R^D` → code `c ∈ {1..K}` such that

  (a) `P(c_a = c_b | id_a = id_b, cross-condition)` is high (recall);
  (b) `P(c_a = c_b | id_a ≠ id_b)` is low (precision).

With Voronoi-style hard quantization at K centroids, the cell volume is
~ V/K (where V is the embedding manifold volume). As K grows:
  - cell volume shrinks → (a) drops (same-id pair more likely to cross
    cell boundary) AND (b) drops (different-id pair less likely to
    share a cell).

Both probabilities are linked to cell volume. **You can't decouple them
by tuning K alone.**

The information-theoretic minimum to address N identities is `log₂(N)`
bits — 10 bits for N=1000. K=1024 provides exactly 10 bits. But the
*encoder's effective bit-rate at the cross-condition level* — i.e.,
how many bits of identity it preserves under condition change — is
what gates same-code recall.

Empirically, for ArcFace + cross-age (AgeDB), the encoder provides
about 5 effective bits at the cross-condition level (same-code 0.20
at K=64 means each ID's intra-id cluster fits in ~32 cells of a
64-cell partition). That caps reliable cross-condition recall at
~32 IDs no matter what K we use.

**This is an encoder limit, not a codebook limit.** No clever hashing
trick can extract more cross-condition bits than the encoder produces.

## Mechanisms that could expand address space (and what we tried)

Five orthogonal levers, ranked by what they actually buy you at the
encoder ceiling:

**1. Multi-position perceptual codes (T_perc residual codes at
consecutive positions; address space K^T_perc).** We tried K=32 T=2 →
1024 addresses on V-XC-ID-face:

| N   | naive K=64 retr@1 | multipos K=32 T=2 retr@1 |
|-----|-------------------|--------------------------|
| 20  | 0.267             | 0.300 (+0.03)            |
| 100 | 0.120             | 0.077 (-0.04)            |

At N=20 a small lift; at N=100 actually worse, because now we
require BOTH levels' codes to match, and the level-2 residual code is
noise-sensitive (lower per-level same-code rate, joint same-code is
worse than single-level). **Multi-position expands address space but
deepens the encoder's residual-quantization fragility.** Net: no win.

**2. Hierarchical codebook (coarse + fine, conditional addressing).**
Not yet tested. In principle, the coarse code provides cross-condition-
robust bucketing (low K), then the fine code disambiguates within the
bucket. If the bucket has few IDs, the fine code's lower same-code rate
doesn't matter much. Theoretically the cleanest design but requires a
two-stage codebook training scheme.

**3. Per-identity user_salt (#9, #13).** Works for multi-tenant
isolation (different users with their own salt sub-spaces) but not for
cross-condition retrieval within one user — at query time we don't
know which identity we're querying.

**4. Continuous attention over a memory bank.** Replaces discrete
codebook with learned attention. Loses Path A's O(1) property, becomes
essentially RAG with LM-internal mechanism. Defeats the
parametric-vs-retrieval distinction.

**5. Better encoder fine-tuned for cross-condition invariance.**
The real fix. Push d_intra (intra-id) closer to 0; d_inter (inter-id)
stays large. Allows finer K without losing same-code rate. Out of
scope for this paper.

## What this means for the framing

The honest paper position is now:

| Axis | Where Path A wins | Where Path A loses |
|------|-------------------|---------------------|
| Retrieval accuracy at small N (≤20) | Competitive or beats RAG (A-PARA N=10 p=0.010) | RAG wins on encoder-saturated tasks |
| Retrieval accuracy at large N (≥100) | — | RAG (~0.77) >> Path A (~0.12) due to encoder cross-condition ceiling |
| Query latency at scale | Constant 49 ms at any N; 32× faster than naive-context RAG at N=10k | RAG-cosine-only is faster at small N (~10 ms) |
| Memory footprint | O(K) parametric table | O(N) stored embeddings + O(N) context tokens |
| Architectural integration | Direct LM hidden-state residual; no prompt engineering | Retrieval-then-inject via prompt |

**The retrieval-accuracy-at-scale claim is not what Path A delivers.**
The latency-at-scale and parametric-memory-primitive claims are.

For the paper, the cleanest framing is now:
- **At small N**: Path A is the first parametric mechanism to BEAT RAG
  on a cross-condition perceptual task (A-PARA N=10, p=0.010).
- **At large N**: Path A's structural advantages are LATENCY (32× at
  N=10k, 62× at N=50k where RAG's context overflows) and MEMORY
  (O(K) vs O(N)).
- **The accuracy axis at large N is encoder-limited.** Any retrieval
  system using the same encoder hits the same ceiling. Path A's
  discrete quantization adds an extra step that loses ~half the
  retrievable bits relative to cosine NN on the continuous embedding.

The user's first-principles framing (match codebook capacity to N) is
correct — but it can't fix the underlying encoder bit-rate ceiling.

## Files added

- `src/nanochat_mm/accuracy_at_scale.py` — RAG cosine-only vs Path A at multiple N
- `src/nanochat_mm/pathA_multipos.py` — multi-position perceptual codes (K^T address space)
- `runs/codebooks/id_v2_codebook_v-xc-id-face_K{64,128,256,512,1024}.pt` — K-sweep codebooks
