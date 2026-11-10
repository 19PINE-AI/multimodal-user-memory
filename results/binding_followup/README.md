# Binding follow-up: is the exact-fact latent collapse a training-length artifact?

The paper's original exact-fact experiment (`src/latentmem/code_memory.py`, Figure
`fig_capacity` right) found that `k=16` latent tokens hold one six-character code
(exact match 0.87) but collapse at two (0.06). Those runs were trained for 3,000 steps
on one seed. This follow-up reruns that experiment with longer training and a
per-item-slot variant. It supports the revised discussion, the new Appendix
"Binding exact facts in a latent memory" (`app:binding`), Figure `fig_binding` and
Tables `tab:binding` / `tab:bindanalysis` in the paper.

**Findings (seed 42 / seed 43 unless noted).**

- **The collapse reproduces at the original budget.** On 3 seeds at 3,000 steps,
  two-code recall is 0.000 / 0.037 / 0.000.
- **The collapse is mostly undertraining.** Trained past the plateau, two-code
  recall reaches 0.555 / 0.400 at LoRA rank 32 and 0.715 / 0.711 at rank 16. At
  rank 32, seed 42 first passes 0.5 at 16,500 steps; seed 43 never does (best 0.41
  in 26,500 steps).
- **Storage is not the bottleneck.** A linear slot probe reads both codes out of a
  two-code latent at 0.79–0.90 per-character accuracy, against 0.92–0.94 for one code.
- **Many errors are binding errors.** 14–16% of answers at rank 32 and 26–28% at
  rank 16 are the *other* stored code, intact (a quarter to a third of errors at
  rank 32; 88–96% at rank 16).
- **Rank 16 beats rank 32 on both seeds**, in fewer steps: three codes 0.311 / 0.410
  vs 0.135 / 0.230. Both rank-16 two-code latents split into item-specific tokens
  (cos(L_AB, L_BA) 0.32 / 0.49); the rank-32 latents stay mixed (0.81–0.88). A
  replicated observation over two seeds, not a law.
- **Per-item slots help but do not guarantee binding.** A hard per-line write mask
  reaches 0.893 / 0.998 on two codes and 0.992 / 0.639 on three. In the seed-43
  three-code run the latent is cleanly slotted, yet 31% of answers are another
  stored code, intact: the reader failed to map names to slots. The mask splits the
  16 latent queries into one group per line, and group `j` may attend only to the
  tokens of line `j`. **It needs the line boundaries when the memory is written.**

## Caveats

- **Seeds.** The 3,000-step rerun has 3 seeds (42/43/44); every trained-longer,
  rank-16 and per-line-mask arm has 2 (42/43). The seed spread is large (e.g. mask,
  three codes: 0.992 vs 0.639).
- **Rank.** The seed-42 rank-16 runs used an earlier harness revision (loss stop
  rule, 15k-step cap, no `lora_rank` field in meta, read as 16); the seed-43 ones
  used the current harness with the same 15k cap. All rank-16 runs hit the cap.
- **Lower bounds.** `c6_m2_s42_conv` (accuracy rule), `*_lossrule`,
  `c6_m2_s42_role_conv` (40k cap, still rising) and all rank-16 runs (15k cap) are
  lower bounds.
- **Two evaluations.** In-training curves use 256 documents (eval seed 7). Final
  numbers use 512 fresh documents (eval seed 999). They differ by up to ~0.07 for the
  same checkpoint.
- **Scope.** Not tested: more than three items, free-form text for the mask, or a
  `k` sweep at longer training (the original "more tokens do not help" sweep is
  3,000 steps only). A code-length grid (2/4/8 characters × M=1/2/3, seed 42) was
  still running when these results were committed and is not included.
- **Notes on the study report.** `STUDY_REPORT.md` is the study report as of seed
  43; its header lists the places where it differs from the JSONs (seed-43
  "queried" values in its wrong-code table are teacher-forced, not greedy; plateau
  ~8k is 7,500 in the 500-step curve).

## Files

| path | contents |
|---|---|
| `runs/*.json` | one file per trained model, written by `t3_binding.py analyze --full`: `meta` (config, stop step and rule, in-training `curve`), `lm` (teacher-forced and greedy recall per `M`), `probes`, `interference` |
| `summary.csv` | one row per run with the main metrics (generated) |
| `curves.csv` | every run's in-training curve, every 500 steps (generated) |
| `PAPER_NUMBERS.md` | every binding number in the paper, recomputed from the JSONs, with file and field (generated) |
| `STUDY_REPORT.md` | the original study report, verbatim (it calls the per-line mask `--role` and the study "T3") |
| `fig_binding.png`, `fig_capacity.png` | previews of the paper figures |

Code is in `src/latentmem/binding_followup/`:

- `t3_binding.py` is the harness. It imports `CodeMemory` from
  `src/latentmem/code_memory.py` unchanged. `--role` turns on the per-line mask.
- `summarize.py` builds the tables above.
- `run_all.sh` lists every training and analysis command.
- `needs_train.py`, `aggregate.py` and `peek.py` are the study's helper scripts.
- `historical/` holds the job queues and the stop-rule patch exactly as they ran.

The only edits to the harness relative to the study copy are path handling. Outputs
default to `results/binding_followup/runs` and checkpoints to `runs/binding_followup`
(gitignored). The 14 GB GPU cap is now opt-in via `BINDING_GPU_FRACTION`.

## Paper number → run → command

Environment: torch 2.10 + cu128, transformers 4.57.6, peft 0.21.2, model
`Qwen/Qwen2.5-1.5B-Instruct`. Run from the repo root. Training is deterministic
given the seed. Each `train` writes `runs/binding_followup/<tag>.pt`, and
`analyze --ckpt runs/binding_followup/<tag>.pt --full` writes `runs/<tag>.json`.
`H=src/latentmem/binding_followup/t3_binding.py`.

| paper number | value | run JSON (`runs/`) | field | train command (`python $H train --chars 6 ...`) |
|---|---|---|---|---|
| 3k-step rerun, M=1/2/3, seeds 42/43/44 | Table `tab:binding` row 2 | `c6_m{1,2,3}_s{42,43,44}_paper.json` | `lm.M.tf_exact` | `--mmax M --seed S --steps 3000 --suffix _paper` |
| trained longer, r32, M=1 (s42 / s43) | 0.982 / 0.992 | `c6_m1_s{42,43}_conv.json` | `lm.1.tf_exact` | `--mmax 1 --seed S --conv --steps 15000 --suffix _conv` |
| trained longer, r32, M=2 | 0.555 (26k) / 0.400 (26.5k) | `c6_m2_s{42,43}_conv.json` | `lm.2.tf_exact` | `--mmax 2 --seed S --conv --steps 40000 --suffix _conv` (s42 ran the accuracy-only "v7" stop; see note) |
| same s42 model, loss stop at 18.5k | 0.541 | `c6_m2_s42_conv_lossrule.json` | `lm.2.tf_exact` | earlier revision, loss rule |
| first in-training M=2 recall >= 0.5 | 16,500 / never | `c6_m2_s{42,43}_conv.json` | `meta.curve[].tf_M2` | same runs |
| trained longer, r32, M=3 | 0.135 (24.5k) / 0.230 (19.5k) | `c6_m3_s{42,43}_conv.json` | `lm.3.tf_exact` | `--mmax 3 --seed S --conv --steps 40000 --suffix _conv` |
| M=3 s42 under other stop rules | 0.129 / 0.117 | `c6_m3_s42_conv_lossrule.json`, `c6_m3_s42_conv_v7stop.json` | `lm.3.tf_exact` | earlier revisions (loss / v7 rule) |
| trained longer, r16, M=1/2/3, s42 | 0.996 / 0.715 / 0.311 | `c6_m{1,2,3}_s42_conv.r16.json` | `lm.M.tf_exact` | earlier revision, loss rule, 15k cap (current-code equivalent: next row's command with `--seed 42`) |
| trained longer, r16, M=1/2/3, s43 | 0.992 / 0.711 / 0.410 | `c6_m{1,2,3}_s43_conv_r16.json` | `lm.M.tf_exact` | `--mmax M --seed 43 --conv --steps 15000 --lora_rank 16 --suffix _conv_r16` |
| per-line mask, M=2 | 0.893 (40k cap) / 0.998 (24k) | `c6_m2_s{42,43}_role_conv.json` | `lm.2.tf_exact` | `--mmax 2 --seed S --conv --steps 40000 --suffix _conv --role` |
| per-line mask, M=3 | 0.992 / 0.639 (both 40k) | `c6_m3_s{42,43}_role_conv.json` | `lm.3.tf_exact` | `--mmax 3 --seed S --conv --steps 40000 --suffix _conv --role` |
| per-line mask, M=3: other / any stored code | 0.000 / 0.309; 0.992 / 0.953 | `c6_m3_s{42,43}_role_conv.json` | `lm.3.greedy_other_code_exact`, `greedy_any_stored_code` | `analyze --full` |
| mask, accuracy-only stop at 9k (s42) | 0.010 | `c6_m2_s42_role_conv_v7stop.json` | `lm.2.tf_exact` | v7 rule (see note) |
| mask, r16, 3k steps (s42) | 0.000 | `c6_m2_s42_role.r16.json` | `lm.2.tf_exact` | `--mmax 2 --seed 42 --steps 3000 --lora_rank 16 --role` (earlier revision) |
| answers: queried / other / any stored (Table `tab:bindanalysis`) | see table | `c6_m2_s42_{paper,conv,conv.r16,role_conv}.json`, `c6_m2_s43_{paper,conv,conv_r16,role_conv}.json` | `lm.2.greedy_exact`, `greedy_other_code_exact`, `greedy_any_stored_code` | `analyze --full` |
| slot probe line 1 / line 2 | see table | same | `probes.2.slot{0,1}_linear.char_acc` | `analyze --full` |
| one-code slot probe | 0.92 / 0.94 | `c6_m1_s{42,43}_conv.json` | `probes.1.slot0_linear.char_acc` | `analyze --full` |
| cos(L_AB, L_BA), R², per-token structure (Fig. `fig_binding`c) | see table | same | `interference.*` | `analyze --full` |

The paper figures come from `paper/figs/generate_figs.py` (`fig_capacity`,
`fig_binding`), which read `runs/*.json` directly. `PAPER_NUMBERS.md` lists every
remaining quoted number (swap shares, step counts, untrained-head probe) with its
field.

**Stop rules.** The current harness implements the "v8" rule for `--conv` runs with
M ≥ 2. It stops when top-M recall has not gained 0.01 in 4,000 steps **and** the
training loss has not improved by more than 3% in 4,000 steps, never before step
8,000, then anneals over 1,000 steps. To regenerate a v7 (accuracy-only) run, drop
`and step - best_step >= 4000` from the stop condition (`historical/patch8.py` shows
the change). One-code runs use the loss rule: no improvement of more than 3% in
5,000 steps, step ≥ 8,000.

Cost: each run fits in 14 GB of GPU memory. On a shared RTX PRO 6000 the 40k-step
masked three-code run took about 1.9 h and the 26k-step two-code run about 1 h
(from the training logs). `summarize.py` and the figures are CPU-only.
