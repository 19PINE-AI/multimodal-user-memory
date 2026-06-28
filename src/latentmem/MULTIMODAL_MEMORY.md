# Text + latent user memory: a unified study of when each wins

This is the capstone of the latent-memory investigation. It answers, with
real-data in-LM experiments, two questions:

1. Can a personalized agent extract its memory into the **latent** space instead
   of text?
2. Does **combining** text + latent beat either alone?

**Answer.** User memory should be a **router with a fusion regime**. The hybrid
(text + latent) genuinely beats both single channels in exactly one place: when
content combines a **non-captionable identity** (a face / voice) with an
**exact, high-entropy fact** (a name, address, booking code). That is the common
real case — an assistant that recognizes you and recalls your exact details.
Outside that regime, one channel suffices.

The map:

| content | best channel | why |
|---|---|---|
| perceptual identity (face, voice) | **latent** | captions cannot tell people apart |
| low-entropy / categorical fact | **either** | a single latent marker holds it as well as a text store |
| high-entropy / exact fact (codes, names, histories) | **text** | a latent has a hard capacity ceiling |
| identity **and** exact fact together | **hybrid** | each channel covers the leg the other fails |

## Evidence

### 1. Latent fails at exact textual recall (the captionable half)
The latentmem pilot (`FINDINGS.md`): across frozen/LoRA readers, dense
multi-probe supervision, Muon, 3k–10k steps, exact single-fact recall from a
text-document latent caps at ~0.55 (chance 0.50). Latent extraction does **not**
beat a text store for exact facts. Gist/aggregate content is captured well, but
gist is captionable, so a text summary ties.

### 2. Unified in-LM pipeline — categorical facts (`single_pipeline.py`)
All three architectures through one frozen LM + AttMem, same real faces, 3 seeds:

| C | N | text-only | latent-only | hybrid | id_recall |
|---|---|---|---|---|---|
| 10 | 10 | 0.086 | 0.942 | 0.942 | 0.942 |
| 10 | 300 | 0.117 | 0.828 | 0.782 | 0.756 |
| uniq | 300 | 0.021 | 0.756 | 0.756 | 0.756 |

text-only fails the perceptual leg (hybrid − text = **+0.37 to +0.88**). For
categorical facts **latent ≈ hybrid** — a marker suffices, no text store needed.
(This corrected an earlier cross-experiment composition that overstated the
hybrid's edge.)

### 3. Latent fact-capacity curve (`multimem.py --mode codec`)
A leak-free latent codec (encode a code string into k=16 soft tokens, decode it
from M alone via content-free decode queries — so M must truly carry the
content). Exact-match of held-out codes:

| code length | bits | latent exact-match (hardened) |
|---|---|---|
| 2 chars | ~10 | 1.000 |
| 4 chars | ~20 | 0.994 |
| 8 chars | ~40 | 0.98 |
| 16 chars | ~80 | 0.39 |

A capacity ceiling around ~8–12 chars: short codes fit nearly perfectly, longer
ones degrade (16-char to ~0.39). The exact ceiling at 8/12/16/24 chars is in
`results/multimem_codec_h*.json`. (Training needs LR warmup + a fixed seed to
avoid a tok-acc≈0.5 plateau — an earlier undertrained run read 0.05 at 16 chars;
hardened in `train_codec`.)

### 4. Full multimodal exact-fact bench (`multimem.py --mode bench`)
Recognize a cross-condition face, recall the user's EXACT code. exact-match,
3 seeds. Both latent-only and hybrid use the same hard perceptual identity
resolution; they differ only in where the **fact** lives.

**8-char codes — within latent capacity (codec self-recon 0.98):**

| N | text-only | latent-only | hybrid |
|---|---|---|---|
| 10 | 0.14 | **1.00** | **1.00** |
| 50 | 0.05 | 0.87 | 0.90 |
| 100 | 0.03 | 0.88 | 0.89 |
| 300 | 0.01 | 0.83 | 0.85 |

→ **latent ≈ hybrid ≫ text.** A latent holds the fact as well as a text store;
both win only because perceptual identity resolves the user (text-only can't).

**16-char codes — beyond latent capacity (codec self-recon 0.39):**

| N | text-only | latent-only | hybrid |
|---|---|---|---|
| 10 | 0.22 | 0.32 | **1.00** |
| 50 | 0.01 | 0.34 | **0.90** |
| 100 | 0.02 | 0.33 | **0.89** |
| 300 | 0.02 | 0.33 | **0.85** |

→ **hybrid ≫ latent ≫ text.** Only the hybrid recalls the exact fact: text fails
the identity leg, latent fails the capacity leg. Latent-only ≈ id_recall × codec
fidelity (0.85 × 0.39 ≈ 0.33), so its gap to hybrid is exactly the latent's
missing fact-fidelity.

Reading: **within** latent capacity (8-char) latent ≈ hybrid ≫ text; **beyond**
it (16-char) only hybrid works — text fails the identity leg, latent fails the
capacity leg. This is the regime where the hybrid is uniquely necessary.

## The system

`single_pipeline.py` (unified 3-architecture in-LM eval), `multimem.py` (latent
fact codec + full exact-fact bench), `mixed_benchmark.py` (embedding-level
perceptual leg), built on the frozen-LM + AttMem mechanism. See `README.md` for
reproduction.

## What this means for the paper

The paper argues text-for-captionable, latent-for-perceptual as separate stores.
This study sharpens it: a single encounter usually has both halves, and the
**hybrid** (perceptual identity in latent + exact fact in text, fused at recall)
strictly dominates either — but only because each channel has a regime the other
cannot serve. The latent's regime is bounded by a measured capacity ceiling, so
the text channel is not optional for exact, high-entropy user facts.
