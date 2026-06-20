# Code map

Where everything lives, and what the key scripts do.

## Top level

| Path | What it is |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `body.tex`), figures (`figs/`), built `main.pdf`, `build.sh`, `refs.bib`. |
| `site/` | React + Vite + Tailwind companion website, kept 1:1 with the paper. `npm install && npm run dev`. |
| `src/` | All experiment code. |
| `results/` | JSON scorecard from every run (committed). Filenames encode mode/steps/seed/config. |
| `notes/` | Session-by-session research log — the real chronology of how the project evolved. |
| `runs/` | Cached encoder embeddings & weights (gitignored, regeneratable). |
| `data/` | Local datasets (gitignored). |
| `research_plan.md`, `paper_outline*.md`, `paper_draft_v1.md` | Historical planning docs (v1 era, pre-pivot). |

## `src/` — top level

| File | Purpose |
|---|---|
| `perceptmem.py` | The PerceptMem benchmark eval surface — unified register/recall across the five sub-modalities. |
| `sanity_*.py` | Encoder gating sanity checks (ECAPA, ArcFace, CLIP style, scene, paralinguistic) — the early experiments that proved each encoder separates identities cross-condition. |
| `extract_*.py` | Dataset embedding extraction (LFW, AgeDB, VoxCeleb, ESC-50). |
| `learned_rqvae.py`, `quantiser_bakeoff.py`, `style_*.py`, `robustness_probes.py` | Quantiser / encoder-head experiments from the codebook era. |

## `src/nanochat_mm/` — the mechanism and experiments

The ★ files are the ones to read first.

| File | Purpose |
|---|---|
| ★ `attention_memory.py` | **The continuous attention memory primitive** (post-pivot). The bank, the attention read, O(1) insertion. |
| ★ `attmem_train_and_eval.py` | **Main train/eval entry point.** Args: `mode n_steps seed [bank_size_max] [adv_prob]`. `MODE_PATHS` maps modes → encoder feature files. |
| `qwen_attmem_bolt.py` | Bolts the attention memory onto a frozen Qwen via a forward pre-hook at the output head. |
| `attmem_vl_train.py`, `attmem_vl_eval.py`, `attmem_vl_arcface.py` | Memory inside Qwen2.5-VL — the key/value-orthogonality experiments. |
| `attmem_latency_benchmark.py`, `latency_*.py` | O(1) insertion / constant-time recall vs. context-stuffing (the 52× number). |
| `attmem_propositional_control.py` | Text non-regression: byte-for-byte identical next-token predictions on plain prompts. |
| `attmem_mixed_modal.py` | Cross-modal leakage check (vision + audio banks stay independent). |
| `attmem_mechanism_analysis.py` | The diagonal-vs-cosine analysis behind the mechanism figure (0.98 vs 0.46). |
| `attmem_scale_imdbface.py` | Scaling to large face pools. |
| `attmem_named_demo.py`, `attmem_demo.py` | Minimal runnable demos. |
| `stat_tests.py` | Multi-seed aggregation and the *t*-tests behind the reported *p*-values. |
| `extract_*.py` | Encoder feature extraction (WikiArt CLIP, WavLM/LibriSpeech, more embeddings). |
| `pathA_*.py` | **The discrete-codebook predecessor** — generic-NTP pretraining, STE codebook, continual pretraining, scaling, multi-seed. Kept as the argument for the design that replaced it. |
| `engram_module*.py`, `id_*codebook*.py`, `stable_kmeans.py` | Codebook / Engram internals from the Path A era. |
| `myvlm_baseline.py`, `online_pvlm_baseline.py`, `myvlm_style_baseline.py` | Prior-work baselines (Yo'LLaVA/MyVLM, Online-PVLM). |
| `qwen3vl_*.py`, `qwen_*_bolt.py`, `qwen_smoke.py` | Backbone wiring & smoke tests across Qwen variants. |
| `v1_*.py`, `v2_*.py`, `v3_*.py`, `toy_gpt_train.py`, `midscale_train.py` | Earlier architecture generations (v1 frozen-codebook bolt-on → v2/v3 → pivot). |
| `forgetting_probe*.py`, `salt_isolation*.py` | Interference / isolation probes. |

## `notes/` — the research chronology

The session notes are the ground truth for *how* the project got here. Highlights:

- `sanity_findings.md` — the encoder gating checks that started it all.
- `session_17_pivot_to_attention_memory.md` — **the pivot** from discrete codebook
  to continuous attention.
- `session_18_attmem_validation.md` — validation of the post-pivot mechanism.
- `pathA_headline.md`, `pathA_breakthrough.md`, `pathA_ste_findings.md` — the
  codebook era's best results and why it ultimately capped at ~7%.
- `baseline_positioning.md`, `escalation_decision.md` — positioning vs. prior work.

## Reading order for a newcomer

1. [`../README.md`](../README.md) — the idea and headline numbers.
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) + `src/nanochat_mm/attention_memory.py`.
3. [`BENCHMARK.md`](BENCHMARK.md) + `src/perceptmem.py`.
4. [`REPRODUCE.md`](REPRODUCE.md) + `src/nanochat_mm/attmem_train_and_eval.py`.
5. `paper/body.tex` for the full argument; `notes/` for the chronology.
