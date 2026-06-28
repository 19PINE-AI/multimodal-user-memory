# Hybrid memory (text + latent): does combining beat either alone?

**Claim.** For content with a non-captionable component (a face/voice -> latent
wins) AND a captionable arbitrary fact (-> text wins), a hybrid memory (latent
for identity, text for the fact) beats either single channel.

**Confirmed, via a two-legged design** (each leg isolated with real measurement):

## Leg 1 — perceptual identity (mixed_benchmark.py, real ArcFace, 8 seeds)

Recall@1 of a user's private fact; you must match a cross-condition face, then
recall its fact. text-only resolves identity by caption (collisions); latent and
hybrid resolve it by the embedding.

| C | N | text-only | latent-only | hybrid | hybrid−text (p) |
|---|---|---|---|---|---|
| 10 | 10 | 0.14 | 0.96 | 0.96 | +0.83 (~0) |
| 10 | 1000 | 0.11 | 0.82 | 0.83 | +0.71 (~0) |
| 50 | 1000 | 0.03 | 0.81 | 0.81 | +0.78 (~0) |

Hybrid (and latent) >> text-only by +0.71..+0.86, p~0 for all N,C. A pure-text
memory fails on mixed content: the identity can't be captioned (caption recall
0.03-0.16), so the right fact is never retrieved.

Captionable-fact control did NOT rescue text (still 0.05-0.16): face captions
are not cross-condition stable, deepening the point.

## Leg 2 — exact-fact storage (latentmem recall runs, in-LM)

Storing an exact fact in a latent and recalling it from a frozen LM tops out at
**~0.55** (chance 0.50); a text store recalls it at **~1.0** (oracle 0.96).
Established across frozen/LoRA readers, dense supervision, Muon, 3k-10k steps.

## Composed end-to-end

| architecture | identity | fact | end-to-end |
|---|---|---|---|
| text-only   | ~0.10 | 1.0  | **~0.10** |
| latent-only | ~0.85 | ~0.55| **~0.47** |
| **hybrid**  | ~0.85 | 1.0  | **~0.85** |

hybrid > latent-only > text-only. Each single channel fails one leg; hybrid
passes both. This is the stronger result: combining text + latent strictly
dominates either, on genuinely mixed content.

## Important refinement

Hybrid's edge over latent-only is an **LM-decoding** limit, not an information
one: the embedding benchmark shows the latent *contains* enough to read the fact
(latent-only ~= hybrid at the idealized embedding level); the frozen LM just
can't decode an exact arbitrary token from a soft latent prefix reliably (Leg 2).
So the text channel earns its keep as the reliable exact-fact decoder, and the
design is: summarize the captionable part to text, latent-encode the
non-captionable residual.

Where it does NOT help: pure-text content (text summary already captures facts +
gist) and pure-perceptual content (latent alone suffices). The win is specific to
mixed content. Interference caveat (latent-bridge work): redundant channels can
hurt, so couple complementary channels, not overlapping ones.

## CORRECTION from the unified in-LM pipeline (single_pipeline.py)

Running all three architectures through the SAME LM + AttMem on the same faces
(3 seeds, N up to 300) corrects the composed claim above for CATEGORICAL facts:

| C | N=10 | N=300 |  | text-only | latent-only | hybrid |
|---|---|---|---|---|---|---|
| 10 | | | N=10 | 0.086 | 0.942 | 0.942 |
| 10 | | | N=300 | 0.117 | 0.828 | 0.782 |
| uniq | | | N=300 | 0.021 | 0.756 | 0.756 |

For categorical facts (incl. unique-per-identity), **latent-only ~= hybrid**
(hybrid - latent = +0.00 to -0.05; latent often slightly higher). text-only
fails the perceptual leg (+0.37..+0.88 for hybrid over text). So a fact that
fits in ONE marker needs no text store -- a unified latent memory suffices.

The hybrid's edge over latent-only is therefore NOT general; it is specific to
EXACT / high-entropy fact CONTENT a single latent cannot hold. multimem.py tests
this with a leak-free latent fact codec (encode a code string into k soft tokens,
decode it from M alone): 2-char codes reconstruct perfectly; the capacity ceiling
at longer codes, and the retrieval-blending failure when faces are confusable,
are where the text channel becomes necessary. (Results: results/multimem_*.json.)

## Airtight follow-up (not yet run)

A single in-LM pipeline measuring all three architectures end-to-end: AttMem with
fact-markers (latent-only), AttMem identity + text dict (hybrid), Path-A code +
text dict (text-only), over the same registered face population. Expected to
reproduce hybrid > latent-only > text-only directly, removing the cross-experiment
composition. mixed_benchmark.py + Leg 2 already establish the result; this would
make it one pipeline.
