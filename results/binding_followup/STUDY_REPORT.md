> Verbatim copy of the study report behind the binding follow-up (internal name "T3"; the per-line mask is called `--role` / "role-keyed" here). Paths refer to the study machine; see `README.md` for repo paths. One correction: two-code recall first passes 0.5 at 16,500 steps in the 500-step curve, not ≈18,000 (see `PAPER_NUMBERS.md`).

# T3 — Is AttMem's "binding-limited" latent memory a training-length artifact?

Status: **seed-42 report.** It is updated when seed 43 lands (the queue was paused for T12; see §7).
Host: `rtx-pro` (RTX PRO 6000 96 GB, shared; T3 was capped at 14 GB with `torch.cuda.set_per_process_memory_fraction(14/96)`).
Paper: *Parametric Multimodal User Memory* (arXiv:2608.28609), repo `19PINE-AI/multimodal-user-memory` @ `da749db`.

## TL;DR

- **The paper's two-code collapse (0.87 → 0.06) is undertraining.** The paper trained the binding experiment for **3,000 steps** (cosine to 0, one seed). I reproduced that protocol exactly on 3 seeds and got two-code exact match of **0.000 / 0.037 / 0.000**. The same model and hyperparameters, trained past the plateau, reach **0.555** (LoRA r32, the paper's rank) and **0.715** (r16) on two codes. Two-code recall first passes **50% at ≈18,000 steps** (r32, seed 42).
- **Both codes are stored early, and binding is what is slow.** Probes read both codes out of the 2-code latent at 0.79–0.90 char accuracy, against 0.92–0.93 for a single code. Most errors return **the other stored code, intact** (16% r32, 28% r16).
- **Whether a model binds tracks whether its latent separates the items.** In the r16 run (0.715), specific latent tokens became pure copies of item A or of item B (cos 0.99–1.00 with one item, ≤0.06 with the other); cos(AB,BA) = 0.32. In the r32 run (0.555) and in every shared-latent M=3 run, every token stays a ~0.6/0.6 mix of both items (cos(AB,BA) 0.81–0.88). Those models are stuck at 0.13–0.31 for three codes.
- **Hard per-item slots remove the problem at the same k=16.** A hard attention mask gives each line its own group of latent queries (`--role`, §5). It reaches **0.893 at M=2** and **0.992 at M=3**, with near-zero wrong-code answers. It needs line boundaries at write time.
- **Implication for the paper.** "Exact facts are binding-limited at any budget" does not hold as stated. A shared continuous latent *can* bind 2–3 exact codes. It learns this slowly and imperfectly unless it develops (or is given) per-item slots. That is the same "one slot per item" law the paper reports for perceptual identity. The router conclusion (text store for exact facts) may still be right on cost grounds, but not because binding is impossible.

## 1. Setup, and how it matches the paper

The paper's binding numbers come from `src/latentmem/code_memory.py`, run by `code_memory_retry.sh` (per-M sweep) and `codemem_ksweep.sh` (k-sweep). I read the settings from the scripts and from the archived logs (`/archive/papers/multimodal-user-memory/results/codemem_perm_M{1,2,4}.log`, `codemem_Mk_M2_k{16,128}.log`). Paper text: `paper/body.tex` l.324, Fig. `fig_capacity` (right panel).

| setting | paper (code_memory.py + retry script) | this study (`t3_binding.py`) |
|---|---|---|
| LM | Qwen2.5-1.5B-Instruct, frozen | same |
| write head | 16 learned queries, 8-head cross-attn over layer-14 (= L/2) hidden states, LayerNorms, output scale | **same class** (`code_memory.CodeMemory`, imported unchanged) |
| reader | LoRA on q/k/v/o, **r=32**, alpha=2r (18.23M trainable) | r=32 (main); r=16 for the rank comparison (§6) |
| latent size k | 16 | 16 |
| task | doc of M lines `Name: CODE`; query `Name: ` → CODE | same `make_doc` (token-consistent), plus line spans |
| codes | 6 chars from a 32-symbol alphabet (`ABCDEFGHJKLMNPQRSTUVWXYZ23456789`) | same; grid adds 2/4/8 chars |
| training mix | per-M model, each batch draws M' ~ U{1..M} | same |
| optimizer | AdamW lr 2e-4, batch 8, grad-clip 1.0 | same |
| **schedule / length** | **3,000 steps**, warmup 150, **cosine to 0** | *paper arm*: identical. *conv arm*: constant 2e-4 after warmup, stop rule (below), then a 1,000-step linear anneal |
| seeds | 1 (42) | paper arm 42/43/44; conv/role arms 42 (43 queued) |
| eval | teacher-forced exact match of the code span, 512 fresh docs | same metric (`tf_exact`), plus greedy decoding, wrong-code rate, char accuracy, probes, interference |

The paper's numbers: M=1 **0.869**, M=2 **0.0625**, M=4/8/16 0.00 (k=16). The k-sweep at M=2 gave 0.06 / 0.11 / 0.10 / 0.04 for k=16/32/64/128. All runs were 3,000 steps. The paper's own M=2 log shows the model solving *one-code* documents at 0.81 by step 2,700, but two-code recall was never logged during training.

**Stop rules used in the conv arm** (all M≥2 conv runs are capped at 40k steps; M=1 at 15k):
- *loss rule* (M=1 always; early M≥2 runs, labeled `_lossrule`): stop when the 500-step window loss has not improved by more than 3% for 5,000 steps, and step ≥ 8,000.
- *v7 accuracy rule* (labeled `_v7stop`): stop when top-M teacher-forced exact (256 queries, every 500 steps) has not improved by ≥ 0.01 for 4,000 steps, and step ≥ 8,000.
- *v8 rule* (current, set by the lead): stop only when **both** the v7 accuracy condition and the loss condition (no >3% gain over 4,000 steps) hold. It was introduced because v7 cut role-keyed M=2 off at 8k, inside its initial plateau, and cut M=3 off at 18k while the loss was still falling.

## 2. Training length is the difference

### 2a. The paper protocol, reproduced (3,000 steps, cosine, r32) — `*_paper`

| model (M_train) | seed | tf exact on M=1 docs | M=2 docs | M=3 docs |
|---|---|---|---|---|
| M=1 | 42 / 43 / 44 | 0.648 / 0.395 / 0.951 | – | – |
| M=2 | 42 / 43 / 44 | 0.000 / 0.594 / 0.000 | **0.000 / 0.037 / 0.000** | – |
| M=3 | 42 / 43 / 44 | 0.893 / 0.000 / 0.898 | 0.086 / 0.000 / 0.111 | 0.027 / 0.000 / 0.014 |

This reproduces the collapse: two-code recall is 0–4% (0–11% counting M=3 models on 2-code docs), against the paper's 6%. M=1 recall varies 0.40–0.95 across seeds at 3,000 steps. That spread is itself a sign that 3,000 steps is in the middle of a phase transition, not at convergence. Three of the nine 3k-step models had not yet left the initial loss plateau (0.000 on every M).

### 2b. Trained past the plateau — shared latent, r32, seed 42 (`c6_m2_s42_conv`)

Teacher-forced exact during training (256 held-out docs, eval seed 7):

| step | 2k | 3k | 4k | 6k | 8k | 10k | 12k | 14k | 16k | **18k** | 21k | 24k | 25k | 26k (annealed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M=2 docs | 0.00 | 0.08 | 0.18 | 0.29 | 0.29 | 0.36 | 0.40 | 0.45 | 0.49 | **0.51** | 0.58 | 0.56 | 0.52 | 0.63 |
| M=1 docs | 0.00 | 0.84 | 0.91 | 0.95 | 0.95 | 0.97 | 0.99 | 0.96 | 0.98 | 0.97 | 0.98 | 1.00 | 0.98 | 1.00 |

- The initial loss plateau ends at ≈2.5k steps. One-code recall saturates by ≈6k. Two-code recall climbs slowly and noisily, with dips at 17k / 20k / 23k (0.31–0.40) under constant lr. It **first passes 0.5 at step 18,000**, 6× the paper's budget. The v7 accuracy rule stopped training at 25k (best 0.578 at 21k), followed by the anneal.
- Final eval (512 fresh docs, eval seed 999): **tf exact 0.555** on M=2, 0.990 on M=1.
- The same model stopped by the loss rule at 18.5k (`c6_m2_s42_conv_lossrule`) gives 0.541. **That is a lower bound.**
- The r16 run (§6) reaches **0.715** at a fixed 15k steps.

I have the 2-code curve for one seed only. The ≥3-seed curves the lead asked for are queued (seed 43 conv/role; see §7). The 3-seed evidence so far is the paper-protocol arm, where all three seeds are at 0–4% at 3k steps.

## 3. Probes, interference and binding

**Wrong-code rate (greedy decoding, M=2 docs).** These are the fractions of answers that are exactly the queried code, exactly the *other* stored code, or any stored code.

| run | exact (queried) | exact (other code) | any stored code | char acc queried / other |
|---|---|---|---|---|
| paper protocol r32 s42 | 0.000 | 0.000 | 0.000 | 0.05 / 0.04 |
| conv r32 s42 | 0.555 | **0.156** | 0.711 | 0.70 / 0.31 |
| conv r16 s42 | 0.711 | **0.277** | **0.988** | 0.72 / 0.31 |
| role r32 s42 | 0.895 | 0.061 | 0.955 | 0.92 / 0.10 |

In the r16 shared run, 98.8% of answers are one of the two stored codes, intact. The remaining errors are pure binding errors (right content, wrong key).

**Probes on the latent** (trained on 10.8k docs, tested on 1.5k fresh docs; chance char acc 0.031; shuffled-label control 0.027–0.034 everywhere):

| run (M docs) | slot probe, linear: char acc (full-code exact) per line | slot probe, MLP | untrained write head, MLP | name-conditioned probe, linear / MLP |
|---|---|---|---|---|
| conv r32, M=1 | 0.92 (0.64) | 0.87 | 0.46 | 0.92 / 0.86 |
| paper r32 s42, M=2 | 0.34 (0.00), 0.45 (0.00) | 0.15, 0.15 | 0.24, 0.23 | 0.27 / 0.06 |
| conv r32, M=2 | 0.79 (0.29), 0.83 (0.37) | 0.67, 0.71 | 0.24, 0.23 | 0.44 / 0.40 |
| conv r16, M=2 | 0.90 (0.54), 0.90 (0.57) | 0.82, 0.84 | 0.25, 0.22 | 0.42 / 0.37 |
| role r32, M=2 | 0.81 (0.29), 0.80 (0.29) | 0.72, 0.73 | 0.41, 0.34 | 0.36 / 0.27 |
| conv r32, M=3 | 0.74, 0.55, 0.58 | 0.57, 0.36, 0.39 | 0.17, 0.11, 0.14 | 0.31 / 0.26 |
| role r32, M=3 | 0.86, 0.89, 0.92 | 0.81, 0.82, 0.86 | 0.38, 0.32, 0.31 | 0.28 / 0.22 |

- The *slot* probe decodes the code on line i. It shows both codes are present in trained 2-code latents at close to the single-code level (0.79–0.90 vs 0.92). So **storage is not the bottleneck**, which is outcome (a) in the plan.
- The *name-conditioned* probe (latent ⊕ one-hot of the queried name, among 66 names) **fails even where the LM reads perfectly** (role M=3: probe 0.22–0.28, LM 0.99). A probe trained on 10.8k examples cannot learn the name→slot lookup. Read it as a weak probe, not as evidence about binding.

**Interference** (latents centered on the mean latent; 400 (A,B) pairs; per sample, least squares L_AB ≈ α·L_A + β·L_B):

| run | cos(AB,A) | cos(AB,B) | R² (α, β) | R² unrelated-pair control | **cos(AB,BA)** |
|---|---|---|---|---|---|
| paper protocol r32 s42 (3k) | 0.59 | 0.61 | 0.88 (0.42, 0.43) | 0.54 | **0.93** |
| conv r32 s42 | 0.60 | 0.57 | 0.70 (0.64, 0.61) | 0.02 | **0.81** |
| conv r16 s42 | 0.52 | 0.52 | 0.54 (0.52, 0.52) | 0.02 | **0.32** |
| conv r32 s42, M=3 model | 0.65 | 0.63 | 0.81 (0.68, 0.66) | 0.02 | **0.88** |
| role r32 s42, M=2 | 0.50 | 0.42 | 0.43 (0.53, 0.44) | 0.04 | **−0.03** |
| role r32 s42, M=3 model | 0.45 | 0.51 | 0.47 (0.45, 0.52) | 0.01 | **−0.00** |

Per latent token: cosine of token t of L_AB with token t of L_A (first row) and of L_B (second row), for tokens 0–15:

```
paper r32 M2   A: .62 .59 .56 .58 .57 .56 .58 .63 .58 .59 .62 .57 .56 .57 .60 .67
               B: .55 .59 .62 .61 .62 .61 .63 .58 .64 .62 .61 .63 .63 .62 .61 .56
conv  r32 M2   A: .66 .69 .65 .45 .56 .48 .61 .59 .59 .62 .61 .62 .65 .52 .56 .64
               B: .62 .58 .52 .56 .55 .61 .49 .60 .53 .58 .54 .59 .57 .53 .62 .44
conv  r16 M2   A: .42 .61 .02 .38 .62 .18 .67 .99 .01 .01 .99 .72 .99 .05 1.0 .62
               B: .64 .61 .95 .29 .51 .82 .61 .04 .92 .93 .06 .52 .02 .95 .03 .60
role  r32 M2   A: 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 -.02 -.01 -.04 -.02 -.02 -.02 -.01 -.00
               B: .01 .01 .01 .01 .01 .01 .01 .01 .86 .87 .94 .84 .75 .85 .86 .89
conv  r32 M3   A: .53 .59 .65 .67 .68 .69 .69 .67 .60 .53 .60 .62 .52 .65 .61 .41
               B: .55 .64 .67 .61 .64 .61 .63 .63 .59 .54 .49 .62 .61 .62 .58 .53
```

- At 3k steps the two-item latent is close to a symmetric linear superposition: R² 0.88, α≈β≈0.42, and swapping the line order barely changes it (0.93).
- The r16 shared model, the best shared-latent model, *spontaneously learned per-item slots*. Tokens 7, 10, 12 and 14 copy item A; tokens 2, 8, 9 and 13 copy item B (≥0.92 with one item, ≤0.06 with the other). That makes the latent order-dependent (cos(AB,BA) 0.32), and its errors are almost all clean swaps.
- The r32 shared model and the M=3 shared models never separated the items. Every token stays a ~0.6/0.6 mix (superposition), and they bind worst.
- The role-keyed latent is slotted by construction. Its first-line group is *identical* to L_A (cos 1.00), because the causal LM's hidden states for line 1 do not depend on line 2.

Relation to the superposition debate (Cammarata vs Gorton/Gao): here, keeping several items superposed in shared directions is exactly the failure mode, and a model succeeds roughly to the degree its latent separates items into dedicated units. That fits Gao/Gorton (superposition is real and costly), and it also fits Cammarata's observation that trained networks prefer clean, single-item units when they can form them. I'd call this suggestive, not decisive: it is one seed per configuration.

## 4. M=3 under the three stop rules (shared latent, r32, seed 42)

| run | stop rule | stop step | tf exact M=3 docs | M=2 docs | M=1 docs | any stored code (M=3) | wrong-code (M=3) |
|---|---|---|---|---|---|---|---|
| `c6_m3_s42_conv_lossrule` | loss | 25,500 | 0.129 | 0.367 | 0.984 | 0.21 | 0.08 |
| `c6_m3_s42_conv_v7stop` | v7 accuracy | 19,000 | 0.117 | 0.309 | 0.986 | 0.20 | 0.09 |
| `c6_m3_s42_conv` | v8 accuracy+loss | 24,500 | **0.135** | 0.320 | 0.984 | 0.21 | 0.08 |
| `c6_m3_s42_conv.r16` | loss (r16, cap 15k) | 15,000 (cap) | 0.311 | 0.559 | 0.998 | 0.45 | 0.14 |

The three r32 runs agree within noise (0.12–0.135), so the stop rule is not what limits M=3 at r32. M=3 under the paper protocol was 0.014–0.027. During training, tf_M3 had reached 0.105 by 14k and was flat to 23.5k.

## 5. The `--role` variant (hard per-line slots)

**Definition.** `Mem.encode(..., role=True)` in `t3_binding.py`. The k=16 latent queries are split into M contiguous groups with `np.array_split(arange(16), M)` (8/8 for M=2; 6/5/5 for M=3). Cross-attention is given a hard boolean `attn_mask` so that **group j can attend only to the tokens of line j** (`Name: CODE\n`), and to nothing else in the document. Everything else is unchanged: the same write-head parameters, LoRA reader, loss and data. **It requires the line boundaries (item spans) at write time**, which the shared-latent design does not. Each latent group is therefore a function of exactly one item: a hard slot-per-item memory at the same k.

| run | stop | tf exact M=1 / M=2 / M=3 docs | wrong-code (top M) | any stored code |
|---|---|---|---|---|
| `c6_m2_s42_role_conv` (M_train=2) | 40k cap (still rising) | 0.992 / **0.893** / – | 0.061 | 0.955 |
| `c6_m3_s42_role_conv` (M_train=3) | 40k cap | 1.000 / 0.998 / **0.992** | 0.000 | 0.992 |
| `c6_m2_s42_role_conv_v7stop` | v7 rule @ 9k | 0.465 / 0.010 / – | 0.01 | 0.02 |
| r16 role, paper protocol (3k, `c6_m2_s42_role.r16`) | 3k | 0.000 / 0.000 | – | – |

Training curves (tf exact, 256 docs):

```
role M=2:  step 1k–7k: loss 4.5→3.6, M1=M2=0.00 (initial plateau, ~8k steps)
           8k 0.18/0.00  10k 0.48/0.02  12k 0.68/0.18  16k 0.90/0.38  20k 0.94/0.45
           24k 0.90/0.57  28k 0.98/0.73  36k 0.97/0.78  40k 0.99/0.88   (M1/M2)
role M=3:  1k 0/0/0  2k 0.87/0.40/0.20  4k 0.97/0.72/0.49  6k 0.99/0.77/0.54
           12k 0.98/0.98/0.83  20k 0.98/0.98/0.90  28k 0.98/1.00/0.93  40k 1.00/1.00/1.00   (M1/M2/M3)
```

- **Role M=2 has a long initial plateau.** Loss stays at ~4 and recall at 0 until ≈8k steps. That is why the v7 accuracy-only rule stopped it at 9k with 0.01, and why the v8 rule exists. After escaping, it climbs to 0.88 and is still rising at the 40k cap, so 0.893 is a lower bound.
- **Role M=3 left the plateau by 2k**, passed 0.5 on three codes at ≈5k, and reached 0.99. With hard slots, three codes are bound almost perfectly at k=16, which is the regime where the shared latent manages 0.13.
- Seed 42 only. Why M=2 plateaued ~4× longer than M=3 is unexplained by one seed (the M=3 mix includes M'=1,2,3 docs, which may give a denser signal).

## 6. Rank 16 vs rank 32 (one seed; seed-43 r16 runs queued)

| condition (seed 42) | r16 | r32 |
|---|---|---|
| M=1, conv (15k cap, loss rule) | 0.996 | 0.982 |
| M=2 shared, conv | **0.715** (15k cap, loss rule, never fired) | **0.555** (26k, v7 rule; 0.541 @ 18.5k loss rule) |
| M=3 shared, conv | **0.311** (15k cap, loss rule) | **0.135** (24.5k, v8; 0.117–0.129 other rules) |
| M=1, paper protocol (3k) | 0.805 | 0.648 |
| M=2, paper protocol (3k) | 0.000 | 0.000 |

**Protocol differences (must be read with the table):**
- The r16 runs came from an earlier harness revision (`ckpt/old_r16/*.pt`; their meta has no `lora_rank` field, and `load()` treats a missing field as 16). They used the loss stop rule (min 8k, patience 5k) with a **15k cap**, and both M≥2 runs hit the cap. They were evaluated every 1,000 steps on 128 queries; evaluation does not touch the training RNG, so the trajectories are unaffected.
- The r32 runs used the v7/v8 rules with a 40k cap and trained longer (24.5–26k steps). So r16 did better *with less training*.
- Mechanistically, r16 M=2 developed item-specific latent tokens and r32 M=2 did not (§3). The M=3 r16 run did *not* develop clean slots either (all tokens ~0.35–0.69 mixed), yet scored 2.3× higher than r32.

**Caveat:** this is one seed per rank. Slot formation looks like a symmetry-breaking event, which could easily be seed-dependent, so I'm not claiming a rank effect. The seed-43 r16 runs (`c6_m{1,2,3}_s43_conv_r16`) were queued for this comparison, against seed-43 r32 runs that are also still queued, and have not run yet.

## 7. Caveats

1. **Single seed** for every conv and role result (seed 42). The paper-protocol arm has 3 seeds. Seed 43 (conv M=1/2/3, role M=2/3, r16 M=1/2/3) is queued in `driver9.sh`, which the lead paused for T12. This report will be updated when it runs.
2. **Lower bounds.** `*_lossrule` runs, `c6_m2_s42_conv` (v7 rule, run under the earlier harness revision), and `c6_m2_s42_role_conv` (40k cap, still rising) are all lower bounds on what longer training reaches.
3. **Different eval sets.** In-training numbers (eval seed 7, 256 docs) and final numbers (eval seed 999, 512 docs) differ by up to ~0.07 at M=2 (0.63 vs 0.555 for the same checkpoint). Tables use the final eval unless marked as curves.
4. The name-conditioned probe is too weak to decide anything (§3). Binding conclusions rest on the LM's own wrong-code rate and the per-token structure.
5. Constant lr 2e-4 gives noisy M=2 curves (dips of up to 0.25). An lr schedule tuned for long training might converge higher or faster. I did not tune; I kept the paper's lr.
6. The code-length grid (chars 2/4/8 × M=1/2/3, seed 42) has not run yet.
7. My reproduction of the paper protocol at M=1 (0.40–0.95) brackets the paper's 0.869. The paper's single-seed numbers are within the seed spread I observe.

## 8. Reproducing every number

Environment (rtx-pro): `cd ~/clean-tests/t3 && . ../.venv/bin/activate` (torch 2.10+cu128, transformers 4.57.6, peft 0.21.2). The harness imports `code_memory.py` from the paper repo checkout at `~/clean-tests/multimodal-user-memory/src/latentmem` (`da749db`). Model: `Qwen/Qwen2.5-1.5B-Instruct` from the HF cache. A copy of the harness and drivers is in `results/T3/code/`. Each command below writes `ckpt/<tag>.pt` (trainable weights + meta; runs since the v7 rule also include the AdamW state), then `analyze` writes `results/T3/<tag>.json`.

```bash
# paper protocol (3k steps, cosine, LoRA r32), 3 seeds  -> §2a
for S in 42 43 44; do for M in 1 2 3; do
  python t3_binding.py train --chars 6 --mmax $M --seed $S --steps 3000 --suffix _paper
  python t3_binding.py analyze --ckpt ckpt/c6_m${M}_s${S}_paper.pt --full
done; done
# converged shared latent, r32 (current code = v8 stop rule)  -> §2b, §3, §4
python t3_binding.py train --chars 6 --mmax 1 --seed 42 --conv --steps 15000 --suffix _conv   # loss rule
python t3_binding.py train --chars 6 --mmax 2 --seed 42 --conv --steps 40000 --suffix _conv
python t3_binding.py train --chars 6 --mmax 3 --seed 42 --conv --steps 40000 --suffix _conv
# role-keyed slots  -> §5
python t3_binding.py train --chars 6 --mmax 2 --seed 42 --conv --steps 40000 --suffix _conv --role
python t3_binding.py train --chars 6 --mmax 3 --seed 42 --conv --steps 40000 --suffix _conv --role
# rank 16  -> §6 (current-code equivalent; see note)
python t3_binding.py train --chars 6 --mmax 2 --seed 42 --conv --steps 15000 --lora_rank 16 --suffix _conv_r16
# analysis for any checkpoint
python t3_binding.py analyze --ckpt ckpt/<tag>.pt --full
python peek.py "<glob>.json"      # one-line summary per result file
```

| number in this report | result JSON (`~/clean-tests/results/T3/`) | checkpoint (`~/clean-tests/t3/`) | stop rule used |
|---|---|---|---|
| paper protocol, all cells | `c6_m{1,2,3}_s{42,43,44}_paper.json` | `ckpt/c6_m{M}_s{S}_paper.pt` | fixed 3k cosine |
| M=2 shared r32 = 0.555 | `c6_m2_s42_conv.json` | `ckpt/c6_m2_s42_conv.pt` | v7 accuracy |
| M=2 shared r32 lower bound 0.541 | `c6_m2_s42_conv_lossrule.json` | `ckpt/c6_m2_s42_conv_lossrule.pt` | loss |
| M=3 shared r32 = 0.135 | `c6_m3_s42_conv.json` | `ckpt/c6_m3_s42_conv.pt` | v8 |
| M=3 v7 / loss variants | `c6_m3_s42_conv_v7stop.json`, `..._lossrule.json` | `ckpt/c6_m3_s42_conv_{v7stop,lossrule}.pt` | v7 / loss |
| role M=2 = 0.893 | `c6_m2_s42_role_conv.json` | `ckpt/c6_m2_s42_role_conv.pt` | 40k cap |
| role M=2 v7 stop (9k) | `c6_m2_s42_role_conv_v7stop.json` | `ckpt/c6_m2_s42_role_conv_v7stop.pt` | v7 |
| role M=3 = 0.992 | `c6_m3_s42_role_conv.json` | `ckpt/c6_m3_s42_role_conv.pt` | 40k cap |
| M=1 conv r32 = 0.982 | `c6_m1_s42_conv.json` | `ckpt/c6_m1_s42_conv.pt` | loss (cap) |
| all r16 numbers | `c6_m{1,2,3}_s42_conv.r16.json`, `c6_m{1,2}_s42.r16.json`, `c6_m2_s42_role.r16.json` | `ckpt/old_r16/<same tag>.pt` | loss (15k cap) / 3k cosine |
| training curves | the `meta.curve` field inside each checkpoint; `logs/train_<tag>.log` | – | – |

Notes:
- Training is deterministic given the seed. The loss-rule and v7 reruns of `c6_m2_s42_conv` reproduced identical curves to 17k steps.
- The current harness implements the **v8** rule for all `--conv` M≥2 runs. To regenerate a v7-rule run exactly, drop `and step - best_step >= 4000` from the stop condition (see `code/patch8.py`).
- The r16 runs used an earlier revision with a loss rule for M≥2. Their checkpoints are the reference artifacts.
- Probe/interference settings: `analyze --full` trains slot / name-conditioned probes on 12k docs (10% validation for early stopping, best epoch reported), tests on 1.5k fresh docs, and runs an untrained-head control and a shuffled-label control. Interference uses 400 (A,B) pairs and 1,000-doc mean-centering.
