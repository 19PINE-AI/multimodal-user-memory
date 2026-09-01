# Code map

## Top level

| Path | Purpose |
|---|---|
| `paper/` | Submitted LaTeX source, bibliography, figures, build script, and local PDF. |
| `site/` | React + Vite publication page. Paper metadata is centralized in `site/src/paper.js`. |
| `src/` | Benchmark, encoder extraction, memory, grounding, and agent experiments. |
| `results/` | Committed aggregate outputs and chronological experiment logs. |
| `docs/` | Architecture, benchmark, result, and reproduction guides. |
| `notes/` | Research chronology, including abandoned alternatives and pivots. |
| `runs/` | Local checkpoints and embedding caches; gitignored. |
| `data/` | Local datasets; gitignored. |

## Final system: read these first

| File | Purpose |
|---|---|
| `src/nanochat_mm/attention_memory.py` | Per-modality key/value banks and the continuous attention read. |
| `src/nanochat_mm/qwen_attmem_bolt.py` | Installs the memory at the frozen model's output head. |
| `src/nanochat_mm/attmem_train_and_eval.py` | Main zero-step paired evaluation; also retains the earlier learned-read path when `n_steps > 0`. |
| `src/nanochat_mm/eval_agentic_production.py` | End-to-end visual grounding, face alignment, and identity recognition. |
| `src/nanochat_mm/eval_agentic_audio.py` | Multi-speaker temporal grounding and ECAPA recognition. |
| `src/nanochat_mm/eval_agentic_paintings.py` | Painting grounding and cross-work style recognition. |
| `src/nanochat_mm/composition_eval.py` | In-model face-marker-to-text-fact composition. |
| `src/nanochat_mm/agent_benchmark.py` | Multi-session agent evaluation with identity, fact recall, and stranger rejection. |

## Benchmark and encoder preparation

| File group | Purpose |
|---|---|
| `src/perceptmem.py` | Original unified five-sub-modality evaluation surface. |
| `src/extract_*.py` | LFW, AgeDB, VoxCeleb, and ESC-50 embedding extraction. |
| `src/nanochat_mm/extract_*.py` | WikiArt, LibriSpeech, and additional modality caches. |
| `src/nanochat_mm/text_baseline*.py` | Caption-and-search baselines for vision, audio, and style. |
| `src/latentmem/cross_domain_benchmark.py` | Final multi-domain breadth evaluation. |
| `src/latentmem/openset_verification.py` | Verification and stranger-rejection metrics. |

## Controls and analysis

| File | Purpose |
|---|---|
| `attmem_vl_eval.py`, `attmem_vl_arcface.py` | Compare VLM-native identity keys with external ArcFace keys. |
| `attmem_mixed_modal.py`, `eval_av_fusion.py` | Modality isolation and audio/visual fusion controls. |
| `attmem_propositional_control.py` | Verifies that an inactive perceptual bank leaves text-only predictions unchanged. |
| `attmem_mechanism_analysis.py` | Attention, cosine, and marker-logit analysis for the mechanism figure. |
| `attmem_latency_benchmark.py`, `latency_*.py`, `measure_cost.py` | Registration, recall, and context-cost measurements. |
| `myvlm_baseline.py`, `online_pvlm_baseline.py`, `myvlm_style_baseline.py` | Reimplementations of per-concept baselines. |
| `src/latentmem/learned_metric_baseline.py` | LDA/whitening comparison on identical encoder features. |

Paths in this table are relative to `src/nanochat_mm/` unless shown from the
repository root.

## Historical research paths

- `pathA_*.py`, `engram_module*.py`, and `id_*codebook*.py` implement the
  discrete-codebook predecessor. It is retained because its failure motivates
  the continuous design.
- `v1_*.py`, `v2_*.py`, `v3_*.py`, and the `src/latentmem/` training scripts
  record earlier architecture generations and capacity experiments.
- Nonzero-step and adversarial-training outputs in `results/` belong to the
  earlier learned-read phase. The final published headline uses the sharp,
  training-free paired protocol described in [RESULTS.md](RESULTS.md).

## Suggested reading order

1. [README](../README.md)
2. [Architecture](ARCHITECTURE.md)
3. `src/nanochat_mm/attention_memory.py`
4. [Benchmark](BENCHMARK.md)
5. [Results](RESULTS.md)
6. [Reproduction](REPRODUCE.md)
7. `paper/body.tex` for the full argument and appendices
