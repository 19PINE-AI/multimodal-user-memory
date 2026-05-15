# Session 12 partial — GPU outage interrupted the lift experiments

**Date:** 2026-05-15

Session 11 honestly diagnosed the deflated cells (A-XR-ID +0.16 at N=5,
A-SCN +0.03, V-XC-ID +0.16 at N=20; only A-PARA p=0.010 BEATS-RAG). Session 12
launched three orthogonal interventions to close that gap. Two completed,
one validated the codebook-quality intervention's limits, and the final
GPU-bound runs were interrupted by a workstation hardware/driver outage.

## What completed

### Stable k-means restart-and-select (#22)

Implementation: `stable_kmeans.py`. Run k-means with N=20 random inits per
modality; pick centroids that maximise `train_same_code − 0.3*inter_id_collision`
on training pairs.

Result — eval same-code rate change vs single-seed (seed=42):

| Modality | K  | single-seed eval | stable eval | Δ |
|----------|----|------------------|-------------|---|
| A-XR-ID  | 32 | 0.487            | 0.414       | −0.074 |
| A-SCN    | 32 | 0.444            | 0.393       | −0.051 |
| V-XC-ID  | 64 | 0.258            | 0.261       | +0.004 |

**HURT on data-limited modalities.** Selection by train-set metric overfits;
the chosen "best" centroids don't generalise. Same Online-PVLM-trap (#15)
applied to model selection.

Diagnostic value: the inter-seed variance is NOT random noise; different
k-means inits capture genuinely different valid clusterings, but the best
training fit ≠ the best eval generalisation when train identities are scarce.
This rules out cheap restart-and-select as a path; we still use the single-seed
codebook downstream.

### WikiArt-XXL feature extraction (#23)

Implementation: `extract_wikiart_xxl.py`. Pulled 30 works each from 128
painters of `huggan/wikiart` — encoded with the same CLIP-mid (concat layers
3/6/9) used by `clip_mid_wikiart.npz`. Output: 3840 samples × 2304-d.

The user asked specifically about Stable Diffusion synthetic data. The
publicly-available SD-WikiArt-style datasets don't exist on HF (searched
across `wikiart`, `artist style`, `painting style`, `stylealigned`,
`sd-concepts`). Real WikiArt at 128 painters addresses the same
data-scarcity bottleneck as SD synthesis would: cross-condition same-painter
pairs at 30 works each vs the existing 50×8 = 400 sample pool. Going to
3840 samples is 9.6× more data without any synthesis step.

Codebook v2 sweep on `v-sty-xxl` at K=16:

| Variant | eval same-code | inter-collision | codes used |
|---------|----------------|-----------------|-----------|
| naive K=16 (best) | 0.317 | 0.090 | 15/16 |
| adapter α=0.5    | 0.136 | 0.082 | 16/16 |
| adapter α=0.2    | 0.170 | 0.074 | 16/16 |
| K=8 naive (sweep) | 0.399 | 0.143 | 8/8 |
| K=32 naive (sweep)| 0.204 | 0.048 | 22/32 |

vs prior `clip_mid_wikiart` (50 painters):

|                  | clip 50 painters | xxl 128 painters | Δ |
|------------------|------------------|------------------|---|
| K=16 naive ev_same | 0.301           | 0.317            | +0.016 |
| K=8 naive ev_same  | 0.384           | 0.399            | +0.015 |

**Marginal lift only.** Doubling the painter pool did NOT proportionally
improve cross-condition same-code rate. **The encoder is the binding
constraint on V-STY**, not the data pool size. Confirms session-8's
documented limitation; the SD-synthetic-data path probably wouldn't help
either because the bottleneck is what CLIP can represent about cross-period
style, not how many painter examples we throw at it.

### Top-K codebook addressing (#22)

Implementation: `pathA_topk_run.py`. Surgical insertion writes the marker at
the top-K=3 nearest centroids per identity; query stays argmin-based.
Theoretical ~3× lift on the fraction-code-match factor (the codebook miss
rate that gates Path A's retr@1).

**Status: code complete, runtime interrupted by GPU outage.**

## What blocked

Around 18:30 UTC the workstation GPU entered a hardware/driver error state:

```
nvidia-smi: ERR!ERR! ECC: N/A   72629 MiB used    "No running processes found"
python: torch.cuda.is_available() == False
```

Other processes still hold 72 GB of allocations but nvidia-smi cannot read
process tables (NVML communication failed). Python's CUDA init fails.
The GPU needs a driver reset (root access) to recover; my bolt processes
were silently running on CPU, hanging in pretrain because Qwen-3B's text
forward is 10–100× slower without CUDA.

Three runs interrupted: Path A on `v-sty-xxl`, top-K on `a-xr-id`, top-K
on `a-scn`. None produced numbers.

## What we now know

1. **Stable k-means is not the answer.** Train-metric selection overfits
   small training pools across the audio modalities. Already documented
   as a dead-end; not used downstream.
2. **More real WikiArt data did not lift V-STY's codebook quality
   meaningfully.** +0.016 same-code rate per painter-pool doubling.
   The bottleneck on V-STY is encoder representational power, not
   training-data count. The SD-synthetic-data path that the user proposed
   is unlikely to lift V-STY for the same reason: the encoder defines
   the ceiling, and CLIP-mid features ceiling at ~0.4 cross-period
   same-style rate even with abundant painters.
3. **Top-K insertion remains untested.** This is the most promising
   intervention still in flight; will run once GPU is restored.

## Files added this session

- `src/nanochat_mm/stable_kmeans.py` — implemented; documented as
  not-effective; not used downstream.
- `src/nanochat_mm/pathA_topk_run.py` — implemented; awaiting GPU.
- `src/nanochat_mm/extract_wikiart_xxl.py` — completed; 128-painter
  WikiArt corpus extracted and encoded.
- `runs/embeddings/clip_mid_wikiart_xxl.npz` — 3840 × 2304 features.
- `runs/codebooks/id_v2_codebook_v-sty-xxl_K{16,32}.pt` — codebooks on
  the new pool.
- `results/id_codebook_v2_v-sty-xxl_K16.json` — diagnostic.

## Resume conditions

Once GPU recovers (driver reset or workstation reboot), the queued runs
are: top-K Path A on A-XR-ID K=32, A-SCN K=32, V-XC-ID K=64, plus V-STY
Path A on the v-sty-xxl K=16 codebook. Each takes ~5 minutes; total
~25 minutes if launched serially. Scripts and codebooks are ready —
no additional preparation required.
