# Session 12 final — post-reboot completion of the lift experiments

**Date:** 2026-05-16

Continuation of session 12's interrupted runs. After workstation reboot
restored CUDA, the queued experiments completed. Final verdict on the
three lift attempts launched in response to session 11's deflation:

## Track 1 — stable k-means restart-and-select (run pre-outage)

HURT eval same-code on all data-limited modalities (−0.05 to −0.07).
Train-metric selection overfits when only 25–84 train identities are
available. Documented dead-end; single-seed codebooks remain operational.

## Track 4 — WikiArt-XXL synthetic-data substitute (run pre-outage)

Codebook lift: +0.016 same-code rate (clip K=16 0.301 → xxl K=16 0.317).
Path A end-to-end (run post-reboot): **no improvement** over the
50-painter baseline:

| N  | xxl retr@1 | xxl RAG | naive clip-K=32 retr@1 |
|----|-----------|---------|------------------------|
| 5  | 0.240     | 0.360   | 0.240                  |
| 10 | 0.200     | 0.400   | 0.18                   |
| 20 | 0.220     | 0.380   | 0.10                   |

**The bottleneck on V-STY is the encoder, not the painter pool.**
Doubling painters (50 → 128) does not lift cross-period same-style
clustering. SD-generated WikiArt would face the same encoder ceiling.

## Track 2 — top-K codebook insertion (run post-reboot)

The most theoretically motivated intervention: write each identity's
marker at the top-3 nearest codes (not just argmin). Diagnostically:
the fraction-code-match factor lifted dramatically across all three
modalities (e.g. A-XR-ID 65% → 84%) — top-K *is* fixing the codebook
miss rate.

But the **mechanism term collapsed** (code-match retrieval):

| Modality | N | top-K=3 retr@1 | single-code retr@1 (multi-seed mean) | Δ |
|----------|---|----------------|--------------------------------------|---|
| A-XR-ID  | 5 | 0.440          | 0.440                                | 0.00 |
| A-XR-ID  | 10| 0.260          | 0.276                                | −0.02 |
| **A-XR-ID** | **20** | **0.070** | **0.166**                       | **−0.10** |
| A-SCN    | 5 | **0.560**      | **0.432**                            | **+0.13** ↑ |
| A-SCN    | 10| 0.280          | 0.276                                | 0.00 |
| A-SCN    | 20| 0.100          | 0.210                                | −0.11 |
| V-XC-ID  | 5 | 0.500          | 0.640                                | −0.14 |
| V-XC-ID  | 10| 0.500          | 0.565                                | −0.07 |
| V-XC-ID  | 20| 0.362          | 0.367                                | 0.00 |

**Mechanism explanation:** writing the same marker at K codes per identity
creates Engram-row inter-identity collisions. With N=20 identities and
K=32 codes, top-K=3 inserts 60 markers into 32 addresses — codes overlap,
the gate cannot disambiguate. The codebook miss rate dropped, the gate
saturation rate rose, and they cancel (or worsen) at larger N.

**One bright spot:** A-SCN at N=5 lifted from 0.432 → 0.560 (+0.128).
Path A at A-SCN N=5 is now 56% of RAG (1.000) — best non-A-PARA result.
But the multi-seed test was n=1 (single-seed run). Need 5-seed
verification to claim this robustly.

## Net verdict on the lift attempts

All three orthogonal interventions failed to convert the trailing-RAG
cells into BEATS-RAG. The original framing's "significant gain on all
four primary axes" reading remains unsupported under strict statistical
verification.

Path forward — three options for the paper:

1. **Accept the relaxed framing**: A-PARA at N=10 K=16 is the
   statistically verified BEATS-RAG cell (p=0.010, 9/10 seeds). The
   other three axes are first-published parametric mechanisms in their
   respective regimes, competitive with the cosine-NN RAG baseline.
   This is what session 11 already documented as the honest position.

2. **Co-pretrain DeepSeek-Engram-style** (the recipe the architecture
   inherits but we never ran). ~5 days on the workstation. The
   mechanism term is already 0.5–1.0 on most cells — co-training would
   tighten it but not change the codebook miss rate (which is encoder-
   bound on V-STY and data-bound on A-XR-ID / A-SCN).

3. **Hybrid Path A + RAG** as the deployed system. Mathematical
   guarantee: retr@1 ≥ max(Path A, RAG) per query. Easy half-day
   experiment. Honest framing: "Path A as the parametric primitive,
   RAG as the codebook-miss fallback." Reviewer-defensible without
   compromising the parametric story.

My recommendation: option 1 with option 3 as a closing system-level
section. Session 11's results are the strongest defensible paper as-is;
the lift attempts in this session were honest negatives that we should
note in the limitations but not chase further.

## Files added this session

- `runs/embeddings/clip_mid_wikiart_xxl.npz` — 128 × 30 = 3840 samples
- `runs/codebooks/id_v2_codebook_v-sty-xxl_K{16,32}.pt` — XXL codebooks
- `runs/codebooks/stable_codebook_{a-xr-id,a-scn,v-xc-id}_K*.pt`
- `results/id_codebook_v2_v-sty-xxl_K16.json` — XXL codebook diagnostic
- `results/pathA_idcb_v-sty-xxl_K16.json` — V-STY Path A on XXL pool
- `results/pathA_idcb_topk3_a-xr-id_K32.json` — top-K speaker
- `results/pathA_idcb_topk3_a-scn_K32.json` — top-K scene
- `results/pathA_idcb_topk3_v-xc-id_K64.json` — top-K face
