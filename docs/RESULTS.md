# Results

All numbers are top-1 recall with the argmax restricted to registered markers.
"±" is sample standard deviation (ddof=1) across seeds; each *p* is a two-sided
one-sample *t*-test of the seed-level AttMem recalls against the deterministic
retrieval value, with *n* seeds as listed. No family-wise correction — read each
*p* individually.

## The seven significant cells (p < 0.05)

| Regime | Cell | n | Retrieval | AttMem (mean±std) | Δ | p |
|---|---|---|---|---|---|---|
| random | Face, N=10 | 4 | 0.933 | 0.992 ± 0.017 | +5.9pp | 0.006 |
| random | Style, N=5 | 5 | 0.400 | 0.640 ± 0.130 | +24pp | 0.015 |
| random | Style, N=10 | 5 | 0.400 | 0.460 ± 0.028 | +6pp | 0.009 |
| adversarial | Face, K=19 | 3 | 0.841 | 0.985 ± 0.001 | +14.4pp | <.001 |
| adversarial | Acoustic scene, K=19 | 4 | 0.827 | 1.000 ± 0.000 | +17.3pp | <.001 |
| adversarial | Tone of voice, K=19 | 4 | 0.226 | 0.934 ± 0.005 | +70.7pp | <.001 |
| adversarial | Style, K=19 | 4 | 0.267 | 0.977 ± 0.007 | +71.0pp | <.001 |

## The three regimes (when it helps)

1. **Encoder imperfect → training adds real signal.** On the face pool an
   untrained memory is below retrieval at every size; training pulls it above and
   the lift *grows* with the memory — a few points at N=10 to ~40 points at N=700.
2. **Encoder already perfect → training only adds noise.** On speaker identity the
   encoder hits 1.00 on its own; an untrained memory matches it exactly and a
   *trained* one is slightly worse. The scorecard reports this plainly.
3. **Inside a VLM → key/value orthogonality decides it** (see below).

## The look-alike regime in detail

Standard training (no look-alike exposure) **trails** retrieval on the confusable
banks where the encoder has headroom: −3.3pp on faces, −6.3pp on tone of voice,
−16.0pp on style; it ties on speaker (both 1.00). Adversarially-aware training
(mixing hard banks into 30% of steps) reverses this completely — the adversarial
rows above.

The random/adversarial trade-off is modality-dependent. On faces the frontier is
gradual (a 10% mix keeps 0.87 random while reaching 0.98 adversarial); on tone of
voice any look-alike mix sharply lowers random recall while adversarial saturates
at once. A larger model family (Llama-3.1-8B) recovers both at once.

## Vision-language model: key/value orthogonality

Inside Qwen2.5-VL-3B, same frozen model, only the key encoder changes:

| Configuration | N=10 | N=100 | N=1000 |
|---|---|---|---|
| Retrieval over ArcFace embeddings (encoder ceiling) | 0.933 | 0.780 | 0.767 |
| AttMem + external ArcFace keys (orthogonal to hidden space) | **1.000** | 0.710 | 0.494 |
| Retrieval over native vision tokens | 0.40 | 0.35 | — |
| AttMem + native vision tokens (co-located with hidden space) | 0.40 | 0.11 | — |

An external ArcFace key reproduces the win; the VLM's native vision tokens already
live in hidden space and only tie native-token retrieval — the wrong key
regardless of method.

## The discrete-codebook predecessor (Path A)

After 100K-step continual pretraining on the 2180-identity face pool, at 300
registered identities, recall is pinned near 0.07 **regardless of codebook size**:

| Codebook size K | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| Path A recall@1 | 0.057 | 0.064 | 0.070 | 0.070 |
| gate routes to correct code | 0.43 | 0.51 | 0.42 | 0.54 |
| embedding retrieval (same encoder) | 0.73 | 0.73 | 0.73 | 0.73 |

A small codebook packs many people into each cell (8.7% inter-identity collisions
at K=16); a large one shatters each identity across cells (same-person
cross-condition match rate drops 0.33 → 0.20). Net recall is squeezed from both
sides and never escapes ~0.07 — a ~10× gap from continuous attention. Even when
the gate routes a query to the right code, every other identity sharing that cell
is an equally good answer. This is the same information loss as captioning,
learned instead of written.

## Limitations (from the paper)

1. At very large memories the encoder's own ceiling reasserts itself — AttMem
   reaches 0.59 at 1,000 faces against the encoder's 0.77.
2. Real cross-condition data measured to ~2,000 identities; constant-time
   behaviour confirmed to 10,000, but recall at that scale awaits data.
3. Inside a VLM, native vision tokens only tie retrieval; an external,
   modality-specific encoder is preferred.
4. Portability is recipe-sensitive (Mistral-7B did not converge in budget).
5. The closest prior system, Online-PVLM, has released neither code nor
   checkpoints, so a head-to-head must wait.

## Raw data

Every run's JSON scorecard is committed under `results/`. Filenames encode the
configuration, e.g.
`attmem_v-sty-clip_steps5000_seed42_advp30.json` =
mode `v-sty-clip`, 5000 steps, seed 42, 30% adversarial mix.
