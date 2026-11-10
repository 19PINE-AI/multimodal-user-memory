> Verbatim copy of the study report behind the binding follow-up (internal name "T3"; the per-line mask is called `--role` / "role-keyed" here), updated for seed 43. Paths refer to the study machine; see `README.md` for repo paths. Notes from checking it against `runs/*.json` (see `PAPER_NUMBERS.md`): (1) in the wrong-code table, the seed-43 "exact (queried)" values for conv r32 / r16 (0.400 / 0.711) are teacher-forced; the greedy values are 0.393 / 0.705, and the paper's Table `tab:bindanalysis` uses greedy. (2) "One-code recall saturates within ~4–6k steps in every run" holds for the two-code models (≤4k); the three-code r32 models reach 0.9 on one-code documents at 7k (s42) and 8.5k (s43). (3) The role M=2 s42 plateau ("~8k") is 7,500 steps in the 500-step curve. The code-length grid mentioned below was still running when this copy was made and is not included.

# T3 — Is AttMem's "binding-limited" latent memory a training-length artifact?

Status: **seeds 42 and 43 complete** for the paper-protocol, converged shared-latent, role-keyed, and rank-16 arms (paper protocol also has seed 44). The code-length grid (chars 2/4/8 × M=1/2/3, seed 42) is running and will be appended. The previous seed-42-only version is kept as `REPORT_seed42_only.md`.
Host: `rtx-pro` (RTX PRO 6000 96 GB, shared; T3 was capped at 14 GB with `torch.cuda.set_per_process_memory_fraction(14/96)`).
Paper: *Parametric Multimodal User Memory* (arXiv:2608.28609), repo `19PINE-AI/multimodal-user-memory` @ `da749db`.

## TL;DR

- **The paper's two-code collapse (0.87 → 0.06) is undertraining.** The paper trained the binding experiment for **3,000 steps** (cosine to 0, one seed). Reproducing that protocol exactly on seeds 42/43/44 gives two-code exact match of **0.000 / 0.037 / 0.000**. The same model and hyperparameters, trained past the plateau, reach two-code recall of **0.555 / 0.400** (LoRA r32, the paper's rank; seeds 42 / 43) and **0.715 / 0.711** (r16).
- **Both codes are stored, and binding is what is slow.** Slot probes read both codes out of trained 2-code latents at 0.79–0.90 char accuracy, against 0.92–0.94 for a single code. A large share of errors return **the other stored code, intact** (14–16% at r32, 26–28% at r16).
- **Success tracks whether the latent separates items, and that replicates on both seeds.** In both r16 M=2 runs, specific latent tokens became near-pure copies of item A or of item B (cos ≥0.82, up to 1.00, with one item; ≤0.1 with the other); cos(AB,BA) = 0.32 / 0.49. In both r32 M=2 runs and every r32 M=3 run, every token stays a ~0.6/0.6 mix (cos(AB,BA) 0.81–0.88), and recall is lower on both seeds.
- **Rank 16 beats rank 32 on both seeds, with less training:** M=2 0.715 / 0.711 vs 0.555 / 0.400; M=3 0.311 / 0.410 vs 0.135 / 0.230 (r16 capped at 15k steps; r32 trained 19.5–26.5k).
- **Hard per-item slots (`--role`) help a lot but are not sufficient.** M=2 reaches **0.893 / 0.998**; M=3 reaches **0.992 / 0.639**. In the seed-43 M=3 run the latent is cleanly slotted, yet 31% of answers are the wrong stored code: the reader can still fail to bind name→slot.
- **Implication for the paper.** "Exact facts are binding-limited ... a latent memory cannot solve [this] at any practical budget" does not hold as stated. A shared continuous latent *can* bind 2–3 exact codes. It learns this slowly and seed-dependently, and does best when its items end up in separate latent units. The paper's router (text store for exact facts) may still be the right engineering choice on cost and reliability, but not because binding is impossible.

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
| seeds | 1 (42) | paper arm 42/43/44; conv, role and r16 arms 42/43 |
| eval | teacher-forced exact match of the code span, 512 fresh docs | same metric (`tf_exact`), plus greedy decoding, wrong-code rate, char accuracy, probes, interference |

The paper's numbers: M=1 **0.869**, M=2 **0.0625**, M=4/8/16 0.00 (k=16). The k-sweep at M=2 gave 0.06 / 0.11 / 0.10 / 0.04 for k=16/32/64/128. All runs were 3,000 steps. The paper's own M=2 log shows the model solving *one-code* documents at 0.81 by step 2,700, but two-code recall was never logged during training.

**Stop rules used in the conv arm** (M≥2 runs are capped at 40k steps; M=1 at 15k):
- *loss rule* (M=1 always; early seed-42 M≥2 runs, labeled `_lossrule`; all r16 runs): stop when the 500-step window loss has not improved by more than 3% for 5,000 steps, and step ≥ 8,000.
- *v7 accuracy rule* (seed-42 `c6_m2_s42_conv`, and runs labeled `_v7stop`): stop when top-M teacher-forced exact (256 queries, every 500 steps) has not improved by ≥ 0.01 for 4,000 steps, and step ≥ 8,000.
- *v8 rule* (set by the lead; all seed-43 r32 runs and seed-42 M=3 and role runs): stop only when **both** the v7 accuracy condition and the loss condition (no >3% gain over 4,000 steps) hold. It was introduced because v7 cut role-keyed M=2 off at 8k, inside its initial plateau, and cut M=3 off at 18k while the loss was still falling.
- r16 seed-43 runs ran under the v8 code but hit their 15k cap; r16 seed-42 runs ran under the loss rule and hit the same cap.

## 2. Training length is the difference

### 2a. The paper protocol, reproduced (3,000 steps, cosine, r32) — `*_paper`

| model (M_train) | seed | tf exact on M=1 docs | M=2 docs | M=3 docs |
|---|---|---|---|---|
| M=1 | 42 / 43 / 44 | 0.648 / 0.395 / 0.951 | – | – |
| M=2 | 42 / 43 / 44 | 0.000 / 0.594 / 0.000 | **0.000 / 0.037 / 0.000** | – |
| M=3 | 42 / 43 / 44 | 0.893 / 0.000 / 0.898 | 0.086 / 0.000 / 0.111 | 0.027 / 0.000 / 0.014 |

This reproduces the collapse: two-code recall is 0–4% (0–11% counting M=3 models on 2-code docs), against the paper's 6%. M=1 recall varies 0.40–0.95 across seeds at 3,000 steps. That spread is itself a sign that 3,000 steps is in the middle of a phase transition, not at convergence. Three of the nine 3k-step models had not yet left the initial loss plateau (0.000 on every M).

### 2b. Trained past the plateau

Final eval (512 fresh docs, eval seed 999), teacher-forced exact on M-code docs:

| condition | seed 42 | seed 43 | stop (s42 / s43) | first step tf ≥ 0.5 in training (s42 / s43) |
|---|---|---|---|---|
| M=2 shared, r32 | **0.555** | **0.400** | 26k (v7) / 26.5k (v8) | 16.5k / never (best 0.41) |
| M=2 shared, r16 | **0.715** | **0.711** | 15k cap / 15k cap | 7k / 8k |
| M=3 shared, r32 | 0.135 | 0.230 | 24.5k (v8) / 19.5k (v8) | never / never |
| M=3 shared, r16 | 0.311 | 0.410 | 15k cap / 15k cap | never / never |
| M=1, r32 | 0.982 | 0.992 | 15k cap | – |
| M=1, r16 | 0.996 | 0.992 | 15k cap | – |

M=2 curves (teacher-forced exact on M=2 docs, 256 held-out docs, eval seed 7; "first ≥ 0.5" uses 500-step resolution):

```
step:            2k    3k    4k    6k    9k   12k   15k   18k   21k   24k   end(annealed)
r32 s42:        .00   .08   .18   .29   .33   .40   .43   .51   .58   .56   .63 @26k
r32 s43:        .02   .03   .07   .08   .16   .22   .27   .26   .30   .28   .40 @26.5k
r16 s43:        .01   .08   .14   .33   .59   .63   .72    –     –     –    (15k cap)
```

- In every run, one-code recall saturates (≥0.9) within ~4–6k steps. Two-code recall climbs slowly and noisily under constant lr. At r32 it takes **≥16.5k steps (5.5× the paper's budget)** to touch 0.5 on seed 42, and seed 43 never reaches it within 26.5k. At r16 it passes 0.5 at **7–8k steps** on both seeds.
- **Lower bound:** `c6_m2_s42_conv_lossrule` (the same seed-42 r32 model stopped by the loss rule at 18.5k) gives 0.541.

## 3. Probes, interference and binding

**Wrong-code rate (greedy decoding, M=2 docs).** These are the fractions of answers that are exactly the queried code, exactly the *other* stored code, or any stored code (seed 42 / seed 43).

| run | exact (queried) | exact (other code) | any stored code |
|---|---|---|---|
| paper protocol r32 | 0.000 / 0.037 | 0.000 / 0.029 | 0.000 / 0.070 |
| conv r32 | 0.555 / 0.400 | **0.156 / 0.143** | 0.711 / 0.535 |
| conv r16 | 0.711 / 0.711 | **0.277 / 0.260** | **0.988 / 0.965** |
| role r32 (M_train=2) | 0.895 / 0.998 | 0.061 / 0.000 | 0.955 / 0.998 |

At r16, 96–99% of answers are one of the two stored codes, intact. The remaining errors are pure binding errors (right content, wrong key).

**Probes on the latent** (trained on 10.8k docs, tested on 1.5k fresh docs; chance char acc 0.031; shuffled-label control 0.027–0.035 everywhere). Seed-42 values; seed 43 is shown where it differs materially.

| run (M docs) | slot probe, linear: char acc (full-code exact) per line | slot probe, MLP | untrained write head, MLP | name-conditioned probe, linear / MLP |
|---|---|---|---|---|
| conv r32, M=1 | 0.92 (0.64) [s43 0.94] | 0.87 | 0.46 | 0.92 / 0.86 |
| paper r32, M=2 | 0.34 (0.00), 0.45 (0.00) | 0.15, 0.15 | 0.24, 0.23 | 0.27 / 0.06 |
| conv r32, M=2 | 0.79 (0.29), 0.83 (0.37) [s43 0.85, 0.87] | 0.67, 0.71 | 0.24, 0.23 | 0.44 / 0.40 |
| conv r16, M=2 | 0.90 (0.54), 0.90 (0.57) [s43 0.86, 0.88] | 0.82, 0.84 | 0.25, 0.22 | 0.42 / 0.37 |
| role r32, M=2 | 0.81 (0.29), 0.80 (0.29) [s43 0.89, 0.92] | 0.72, 0.73 | 0.41, 0.34 | 0.36 / 0.27 |
| conv r32, M=3 | 0.74, 0.55, 0.58 [s43 0.63, 0.36, 0.49] | 0.57, 0.36, 0.39 | 0.17, 0.11, 0.14 | 0.31 / 0.26 |
| role r32, M=3 | 0.86, 0.89, 0.92 [s43 0.87, 0.87, 0.89] | 0.81, 0.82, 0.86 | 0.38, 0.32, 0.31 | 0.28 / 0.22 |

- The *slot* probe decodes the code on line i. It shows that the codes are present in trained 2-code latents at close to the single-code level, so **storage is not the bottleneck at M=2**. At M=3 (shared, r32), lines 2–3 are less decodable (0.36–0.58): some content is lost there.
- The *name-conditioned* probe (latent ⊕ one-hot of the queried name, among 66 names) **fails even where the LM reads perfectly** (role M=3 s42: probe 0.22–0.28, LM 0.99). A probe trained on 10.8k examples cannot learn the name→slot lookup. Read it as a weak probe, not as evidence about binding.

**Interference** (latents centered on the mean latent; 400 (A,B) pairs; per sample, least squares L_AB ≈ α·L_A + β·L_B; seed 42 / seed 43):

| run | R² (L_AB ≈ αL_A+βL_B) | R² unrelated-pair control | **cos(AB,BA)** | M=2 exact |
|---|---|---|---|---|
| paper protocol r32 (3k) | 0.88 / 0.76 | 0.54 / – | **0.93 / 0.79** | 0.000 / 0.037 |
| conv r32 | 0.70 / 0.79 | 0.02 / – | **0.81 / 0.83** | 0.555 / 0.400 |
| conv r16 | 0.54 / 0.61 | 0.02 / – | **0.32 / 0.49** | 0.715 / 0.711 |
| conv r32, M=3 model | 0.81 / 0.74 | 0.02 / – | **0.88 / 0.86** | (M=3: 0.135 / 0.230) |
| conv r16, M=3 model | – / 0.67 | – | **– / 0.71** | (M=3: 0.311 / 0.410) |
| role r32, M=2 model | 0.43 / 0.47 | 0.04 / – | **−0.03 / 0.00** | 0.893 / 0.998 |
| role r32, M=3 model | 0.47 / 0.46 | 0.01 / – | **−0.00 / −0.01** | (M=3: 0.992 / 0.639) |

Per latent token: cosine of token t of L_AB with token t of L_A (first row) and of L_B (second row), for tokens 0–15:

```
conv r32 M2 s42  A: .66 .69 .65 .45 .56 .48 .61 .59 .59 .62 .61 .62 .65 .52 .56 .64
                 B: .62 .58 .52 .56 .55 .61 .49 .60 .53 .58 .54 .59 .57 .53 .62 .44
conv r32 M2 s43  A: .67 .65 .66 .67 .68 .67 .53 .65 .63 .66 .66 .61 .61 .60 .59 .64
                 B: .63 .61 .62 .60 .60 .61 .60 .63 .60 .63 .56 .65 .61 .66 .64 .33
conv r16 M2 s42  A: .42 .61 .02 .38 .62 .18 .67 .99 .01 .01 .99 .72 .99 .05 1.0 .62
                 B: .64 .61 .95 .29 .51 .82 .61 .04 .92 .93 .06 .52 .02 .95 .03 .60
conv r16 M2 s43  A: .01 1.0 .03 .69 .10 .88 .29 .69 .49 .79 .59 .59 .99 .69 .69 .40
                 B: .84 .02 .91 .43 .82 .25 .72 .46 .67 .28 .65 .67 .05 .57 .55 .29
conv r16 M3 s43  A: .51 .57 .51 .61 .64 .37 .63 .95 .28 .75 .39 .52 .64 .15 .79 .64
                 B: .63 .58 .50 .62 .62 .81 .58 .12 .83 .51 .74 .58 .44 .82 .34 .66
role r32 M2 s43  A: 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 .01 -.01 .01 .00 .00 .01 -.01 .00
                 B: .02 -.00 .01 .01 .00 .01 .01 .01 .93 .95 .93 .92 .93 .95 .87 .96
```

- At 3k steps the two-item latent is close to a symmetric linear superposition: R² 0.76–0.88, and swapping the line order barely changes it (0.79–0.93).
- **Both r16 M=2 models spontaneously learned item-specific latent tokens.** On s42, tokens 7, 10, 12 and 14 copy A and tokens 2, 8, 9 and 13 copy B. On s43, tokens 1 and 12 copy A and tokens 0, 2 and 4 copy B. That makes the latent order-dependent (cos(AB,BA) 0.32 / 0.49). The r16 M=3 s43 model is partially slotted (token 7 → A 0.95, B 0.12) and is the best shared-latent M=3 model (0.410).
- **Both r32 M=2 models, and all r32 M=3 models, kept every token a ~0.6/0.6 mix.** They are order-symmetric (0.81–0.88) and bind worst.
- The role-keyed latent is slotted by construction. Its first-line group is *identical* to L_A (cos 1.00), because the causal LM's hidden states for line 1 do not depend on line 2.

Relation to the superposition debate (Cammarata vs Gorton/Gao): keeping several items superposed in shared directions is the failure mode, and on both seeds a model succeeds roughly to the degree its latent separates items into dedicated units. That fits Gao/Gorton (superposition is real and costly). It also fits Cammarata's observation that trained networks form clean, single-item units when they can. But see §5: separated units are not *sufficient*, since a reader can still mis-bind clean slots.

## 4. M=3 under the three stop rules (shared latent, r32)

| run | stop rule | stop step | tf exact M=3 docs | M=2 docs | M=1 docs | any stored code (M=3) | wrong-code (M=3) |
|---|---|---|---|---|---|---|---|
| `c6_m3_s42_conv_lossrule` | loss | 25,500 | 0.129 | 0.367 | 0.984 | 0.21 | 0.08 |
| `c6_m3_s42_conv_v7stop` | v7 accuracy | 19,000 | 0.117 | 0.309 | 0.986 | 0.20 | 0.09 |
| `c6_m3_s42_conv` | v8 accuracy+loss | 24,500 | **0.135** | 0.320 | 0.984 | 0.21 | 0.08 |
| `c6_m3_s43_conv` | v8 accuracy+loss | 19,500 | **0.230** | 0.424 | 0.971 | 0.33 | 0.09 |

The three seed-42 runs agree within noise (0.12–0.135), so the stop rule is not what limits M=3 at r32. Seed 43 is higher (0.230) but still far from M=2 levels. For comparison, M=3 under the paper protocol was 0.000–0.027, and r16 reached 0.311 / 0.410 within its 15k cap.

## 5. The `--role` variant (hard per-line slots)

**Definition.** `Mem.encode(..., role=True)` in `t3_binding.py`. The k=16 latent queries are split into M contiguous groups with `np.array_split(arange(16), M)` (8/8 for M=2; 6/5/5 for M=3). Cross-attention is given a hard boolean `attn_mask` so that **group j can attend only to the tokens of line j** (`Name: CODE\n`), and to nothing else in the document. Everything else is unchanged: the same write-head parameters, LoRA reader (r32), loss and data. **It requires the line boundaries (item spans) at write time**, which the shared-latent design does not. Each latent group is therefore a function of exactly one item: a hard slot-per-item memory at the same k.

| run | stop | tf exact M=1 / M=2 / M=3 docs | wrong-code (top M) | any stored code | first tf ≥ 0.5 |
|---|---|---|---|---|---|
| role M=2, s42 | 40k cap (still rising) | 0.992 / **0.893** / – | 0.061 | 0.955 | 21.5k |
| role M=2, s43 | 24k (v8) | 0.996 / **0.998** / – | 0.000 | 0.998 | 4k |
| role M=3, s42 | 40k cap | 1.000 / 0.998 / **0.992** | 0.000 | 0.992 | 4.5k |
| role M=3, s43 | 40k (v8 fired at 39k) | 0.994 / 0.709 / **0.639** | **0.309** | 0.953 | 16k |
| role M=2 s42, v7 stop (`_v7stop`) | 9k | 0.465 / 0.010 / – | 0.01 | 0.02 | – |

Training curves (tf exact, 256 docs):

```
role M=2 s42:  1k–7k: loss 4.5→3.6, M1=M2=0.00 (initial plateau ~8k steps)
               8k .18/.00  12k .68/.18  16k .90/.38  20k .94/.45  24k .90/.57  28k .98/.73  40k .99/.88   (M1/M2)
role M=2 s43:  1k–3k plateau; 4k .94/.58  6k .99/.76  9k .97/.97  18k 1.0/.99                          (M1/M2)
role M=3 s42:  2k .87/.40/.20  4k .97/.72/.49  12k .98/.98/.83  20k .98/.98/.90  40k 1.0/1.0/1.0        (M1/M2/M3)
role M=3 s43:  3k .65/.17/.06  6k .91/.55/.36  12k .96/.66/.48  21k .99/.67/.57  33k .99/.71/.59  39k .99/.74/.52
```

- **The initial loss plateau varies a lot by seed:** about 8k steps for role M=2 s42, 3k for s43, and under 2k for role M=3 s42. That variation is why the v7 accuracy-only rule stopped role M=2 s42 at 9k with 0.01, and why the v8 rule exists.
- **Three of four role runs bind (near-)perfectly.** Role M=3 s43 is the exception (0.639). Its latent is cleanly slotted (group A ≡ L_A at cos 1.00; group B 0.88–0.96 with L_B), yet 31% of answers are the *other stored code*, intact, and 95% are some stored code. The reader LoRA failed to learn the name→slot association for this seed, and the curve had flattened at 0.5–0.6 for 25k steps. So **hard slots fix storage separation but do not guarantee read-side binding**.

## 6. Rank 16 vs rank 32 (two seeds)

| condition | r16 (s42 / s43) | r32 (s42 / s43) |
|---|---|---|
| M=1, conv | 0.996 / 0.992 | 0.982 / 0.992 |
| **M=2 shared, conv** | **0.715 / 0.711** | **0.555 / 0.400** |
| **M=3 shared, conv** | **0.311 / 0.410** | **0.135 / 0.230** |
| M=2 wrong-code (other stored code) | 0.277 / 0.260 | 0.156 / 0.143 |
| M=2 cos(AB,BA) | 0.32 / 0.49 | 0.81 / 0.83 |
| training steps | 15k / 15k (cap) | 26k / 26.5k (M=2); 24.5k / 19.5k (M=3) |
| M=1 / M=2, paper protocol (3k), s42 only | 0.805 / 0.000 | 0.648 / 0.000 |

**Protocol differences (must be read with the table):**
- **r16 seed 42** came from an earlier harness revision (`ckpt/old_r16/*.pt`; their meta has no `lora_rank` field, and `load()` treats a missing field as 16). It used the loss stop rule with a **15k cap**, and both M≥2 runs hit the cap. It was evaluated every 1,000 steps on 128 queries; evaluation does not touch the training RNG, so the trajectories are unaffected.
- **r16 seed 43** (`*_s43_conv_r16`) used the current harness (`--lora_rank 16 --conv --steps 15000`, v8 code), and all three runs hit the 15k cap.
- **r32 runs** used the v7/v8 rules with a 40k cap and trained 19.5–26.5k steps. **So r16 did better on both seeds with less training.**
- The two r32 seed-42 runs and the r32 seed-43 runs used different stop rules (v7 vs v8). §4 shows the rule changes M=3 by ≤0.02 on seed 42.

**Reading:** the rank effect now replicates on two seeds at both M=2 and M=3, and it lines up with the mechanism in §3 (r16 forms item-specific latent tokens on both seeds; r32 does not). Two seeds is still a small sample, and I have not tested intermediate ranks or r64. A plausible but untested explanation is that a lower-rank reader puts more pressure on the write head to separate items. Treat this as a replicated observation, not an established rank law.

## 7. Caveats

1. **Two seeds** for conv, role and r16 (42/43); three for the paper protocol. The seed-to-seed spread is large: role M=3 gives 0.992 vs 0.639, and r32 M=2 gives 0.555 vs 0.400.
2. **Lower bounds.** `*_lossrule` runs, `c6_m2_s42_conv` (v7 rule), `c6_m2_s42_role_conv` (40k cap, still rising) and all r16 runs (15k cap; r16 M=2 s42 was still rising at 0.805 in-training) are lower bounds on what longer training reaches.
3. **Different eval sets.** In-training curves (eval seed 7, 256 docs) and final numbers (eval seed 999, 512 docs) differ by up to ~0.07. Tables use the final eval unless marked as curves.
4. The name-conditioned probe is too weak to decide anything (§3). Binding conclusions rest on the LM's own wrong-code rate and the per-token latent structure.
5. Constant lr 2e-4 gives noisy M=2 curves (dips of up to 0.25). An lr schedule tuned for long training might converge higher or faster. I did not tune; I kept the paper's lr.
6. **The code-length grid** (chars 2/4/8 × M=1/2/3, seed 42, r32) is still running. Its results will be appended.
7. My paper-protocol M=1 numbers (0.40–0.95 across seeds) bracket the paper's 0.869. The paper's single-seed numbers sit within the seed spread I observe.

## 8. Reproducing every number

Environment (rtx-pro): `cd ~/clean-tests/t3 && . ../.venv/bin/activate` (torch 2.10+cu128, transformers 4.57.6, peft 0.21.2). The harness imports `code_memory.py` from the paper repo checkout at `~/clean-tests/multimodal-user-memory/src/latentmem` (`da749db`). Model: `Qwen/Qwen2.5-1.5B-Instruct` from the HF cache. A copy of the harness and drivers is in `results/T3/code/`. Each command below writes `ckpt/<tag>.pt` (trainable weights + meta; runs since the v7 rule also include the AdamW state), then `analyze` writes `results/T3/<tag>.json`.

```bash
# paper protocol (3k steps, cosine, LoRA r32), 3 seeds  -> §2a
for S in 42 43 44; do for M in 1 2 3; do
  python t3_binding.py train --chars 6 --mmax $M --seed $S --steps 3000 --suffix _paper
  python t3_binding.py analyze --ckpt ckpt/c6_m${M}_s${S}_paper.pt --full
done; done
# converged shared latent, r32 (current code = v8 stop rule)  -> §2b, §3, §4
for S in 42 43; do
  python t3_binding.py train --chars 6 --mmax 1 --seed $S --conv --steps 15000 --suffix _conv   # loss rule
  python t3_binding.py train --chars 6 --mmax 2 --seed $S --conv --steps 40000 --suffix _conv
  python t3_binding.py train --chars 6 --mmax 3 --seed $S --conv --steps 40000 --suffix _conv
# role-keyed slots  -> §5
  python t3_binding.py train --chars 6 --mmax 2 --seed $S --conv --steps 40000 --suffix _conv --role
  python t3_binding.py train --chars 6 --mmax 3 --seed $S --conv --steps 40000 --suffix _conv --role
# rank 16, 15k cap  -> §6
  for M in 1 2 3; do python t3_binding.py train --chars 6 --mmax $M --seed $S --conv --steps 15000 --lora_rank 16 --suffix _conv_r16; done
done
# analysis for any checkpoint
python t3_binding.py analyze --ckpt ckpt/<tag>.pt --full
python peek.py "<glob>.json"      # one-line summary per result file
```

| number in this report | result JSON (`~/clean-tests/results/T3/`) | checkpoint (`~/clean-tests/t3/`) | stop rule used |
|---|---|---|---|
| paper protocol, all cells | `c6_m{1,2,3}_s{42,43,44}_paper.json` | `ckpt/c6_m{M}_s{S}_paper.pt` | fixed 3k cosine |
| shared r32, M=1/2/3, s42 | `c6_m{1,2,3}_s42_conv.json` | `ckpt/c6_m{M}_s42_conv.pt` | loss (M=1) / v7 (M=2) / v8 (M=3) |
| shared r32, M=1/2/3, s43 | `c6_m{1,2,3}_s43_conv.json` | `ckpt/c6_m{M}_s43_conv.pt` | loss (M=1) / v8 |
| s42 lower-bound / alt-rule variants | `c6_m2_s42_conv_lossrule.json`, `c6_m3_s42_conv_{lossrule,v7stop}.json` | `ckpt/<same tag>.pt` | loss / v7 |
| role r32, M=2/3, s42 and s43 | `c6_m{2,3}_s{42,43}_role_conv.json` | `ckpt/c6_m{M}_s{S}_role_conv.pt` | v8 / 40k cap |
| role M=2 s42 v7 stop (9k) | `c6_m2_s42_role_conv_v7stop.json` | `ckpt/c6_m2_s42_role_conv_v7stop.pt` | v7 |
| r16, s42 | `c6_m{1,2,3}_s42_conv.r16.json` (+ 3k-step `c6_m{1,2}_s42.r16.json`, `c6_m2_s42_role.r16.json`) | `ckpt/old_r16/c6_m{M}_s42_conv.pt` etc. | loss, 15k cap |
| r16, s43 | `c6_m{1,2,3}_s43_conv_r16.json` | `ckpt/c6_m{M}_s43_conv_r16.pt` | v8 code, 15k cap |
| training curves | the `meta.curve` field inside each checkpoint; `logs/train_<tag>.log` | – | – |

Notes:
- Training is deterministic given the seed. The loss-rule and v7 reruns of `c6_m2_s42_conv` reproduced identical curves to 17k steps.
- The current harness implements the **v8** rule for all `--conv` M≥2 runs. To regenerate a v7-rule run exactly, drop `and step - best_step >= 4000` from the stop condition (see `code/patch8.py`). The r16 seed-42 runs used an earlier revision with a loss rule for M≥2. Their checkpoints are the reference artifacts.
- Probe/interference settings: `analyze --full` trains slot / name-conditioned probes on 12k docs (10% validation for early stopping, best epoch reported), tests on 1.5k fresh docs, and runs an untrained-head control and a shuffled-label control. Interference uses 400 (A,B) pairs and 1,000-doc mean-centering.
