# Reproducing the results

The system is about 200 lines of new code over a frozen-model `transformers`
stack. This document gives the exact commands behind every headline number.

## Environment

- Python 3.10+
- An H100-class GPU for training (a face run is ~17 min at 12K steps; audio/style
  runs ~5–10 min at 5K steps).

```bash
pip install torch transformers numpy scipy scikit-learn speechbrain
```

There is no pinned `requirements.txt`; the stack is standard PyTorch +
`transformers`. The frozen backbone defaults to Qwen2.5-3B and is overridable via
the `ATTMEM_MODEL_ID` environment variable.

## Data & encoders

PerceptMem uses standard datasets read through fixed, off-the-shelf encoders. The
training/eval scripts read **cached embeddings** (`.npz` files under `runs/`,
gitignored). Regenerate them with the extraction scripts before running:

| Sub-modality | Dataset | Encoder | Extraction script |
|---|---|---|---|
| Face | LFW + AgeDB | ArcFace | `src/extract_lfw_*.py`, `src/extract_agedb.py` |
| Painter style | WikiArt | mid-layer CLIP | `src/nanochat_mm/extract_wikiart_xxl.py` |
| Speaker | LibriSpeech | ECAPA-TDNN | `src/extract_voxceleb1.py`, `src/nanochat_mm/extract_wavlm_libri.py` |
| Acoustic scene | ESC-50 | AST | `src/extract_esc50_full.py` |
| Tone of voice | RAVDESS | wav2vec2 emotion | `src/nanochat_mm/extract_more_embeddings.py` |

The encoder feature `.npz` filenames each mode expects are listed in
`MODE_PATHS` in `src/nanochat_mm/attmem_train_and_eval.py`.

## Main entry point

```
python3 attmem_train_and_eval.py <mode> <n_steps> <seed> [bank_size_max] [adv_prob]
```

- `mode` — `v-xc-id-xxxl` (face), `v-sty-clip` (style), `a-xr-id` (speaker),
  `a-scn` (acoustic scene), `a-para` (tone of voice).
- `n_steps` — training steps (0 = untrained / encoder-ceiling probe).
- `seed` — RNG seed.
- `bank_size_max` — curriculum max bank size (0 = fixed bank_size 64).
- `adv_prob` — fraction of training steps that mix in hard look-alike banks
  (0 = random regime only).

Every run logs a JSON scorecard to `results/`.

## The headline commands

Run from `src/nanochat_mm/`.

```bash
# --- Random-regime wins (multi-seed, p<0.05) ---
for s in 42 43 44 47; do
  python3 attmem_train_and_eval.py v-xc-id-xxxl 12000 $s 1024
done
for s in 42 43 44 45 46; do
  python3 attmem_train_and_eval.py v-sty-clip 5000 $s
done

# --- Look-alike regime (multi-seed, p<0.001) ---
for s in 49 50 51; do
  python3 attmem_train_and_eval.py v-xc-id-xxxl 12000 $s 1024 0.3
done
for s in 42 43 44 45; do
  python3 attmem_train_and_eval.py a-para     5000 $s 0 0.3
  python3 attmem_train_and_eval.py v-sty-clip 5000 $s 0 0.3
  python3 attmem_train_and_eval.py a-scn      5000 $s 0 0.3
done

# --- Cross-family, VLM, latency, text non-regression ---
ATTMEM_MODEL_ID="NousResearch/Meta-Llama-3.1-8B-Instruct" \
  python3 attmem_train_and_eval.py v-xc-id-xxxl 12000 42 1024 0.3
python3 attmem_vl_train.py 3000 42 0          # memory inside Qwen2.5-VL
python3 attmem_latency_benchmark.py           # O(1) insertion / constant-time recall
python3 attmem_propositional_control.py       # text non-regression (byte-identical)
```

## Aggregation & statistics

Per-cell aggregation and the multi-seed *t*-tests are in
`src/nanochat_mm/stat_tests.py`. Each reported *p*-value is a two-sided one-sample
*t*-test of the seed-level AttMem recalls against the deterministic retrieval
value; no family-wise correction is applied, so *p*-values are read individually.

## See also

- [`RESULTS.md`](RESULTS.md) — the resulting tables.
- [`CODE_MAP.md`](CODE_MAP.md) — what each script does.
- Paper Appendix E (`paper/body.tex`) — the reproducibility appendix.
