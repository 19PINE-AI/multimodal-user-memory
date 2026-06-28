# Latent user-memory pilot

A decisive, cheap test of one question:

> Can a small **learned write head** compress a user-memory document into *k*
> continuous vectors **M** such that a frozen LM, conditioned on **M alone**,
> answers as well as if it had the full document in context — and better than
> *k* tokens of text at the same budget?

This is the *captionable / textual* half of the memory thesis (the part that is
genuinely improvable by latent extraction). It deliberately uses a **text LM**
(`Qwen/Qwen2.5-3B-Instruct`), not a VLM: if latent extraction can't beat a text
summary here, the heavier VLM version isn't worth building yet. The perceptual
half stays with AttMem — they answer different halves of the problem.

## Mechanism

```
doc  --frozen LM--> hidden states --WriteHead (k query tokens, cross-attn)--> M [k×H]
[M ; probe]  --frozen LM-->  answer logits          (READ / student)
[doc ; probe] --frozen LM--> answer logits          (TEACHER / oracle, no grad)
```

Only the **WriteHead** trains (~a few M params). The LM is frozen; **M** is
injected as `inputs_embeds` — the same soft-token interface AttMem uses.

**Objective** (per step): `alpha·KL(teacher‖student) + beta·CE(student, gold)`.
The KL term is the principled one — it asks M to make the model *behave* as if
it saw the document (behavioral sufficiency), and needs no labels. CE grounds
the task.

## Data

`data.py` — self-contained synthetic user-memory (account-settings) docs with
two probe types, both with clean single-token yes/no gold labels:
- **gated**: answer = AND of `n_relevant` governing settings (integration depth).
- **recall**: value lookup of one setting (fine-grained single-fact recall).

Document length (`n_settings`) and depth (`n_relevant`) are both controllable,
so we can find where latent compression starts to break. Train and eval are
disjoint persona draws (generalization to unseen users).

## Run

```bash
# waits for GPU headroom, then sweeps the latent budget k ∈ {4,8,16,32}
NEED_GB=30 STEPS=3000 bash run_pilot.sh

# or a single config
python3 train.py --n_steps 3000 --k 16 --batch 8        # waits for GPU by default
python3 eval.py  ../../results/latentmem_k16_*.pt        # standalone eval
```

## What to read

Each eval prints four numbers on held-out users:

| metric | meaning |
|---|---|
| `full` | [doc ; probe] — oracle upper bound |
| `mem`  | [M ; probe] — **ours** (k latent vectors) |
| `text` | [trunc(doc,k) ; probe] — matched-budget text baseline |
| `none` | [probe] — no-memory lower bound (label prior ≈ 0.5) |
| `suff_kl` | mean KL(teacher‖student) — the sufficiency gap (→ 0 is the goal) |

**Win condition:** `mem` > `text` at equal budget, and `mem` approaching `full`
as k grows, with `suff_kl` shrinking. If `mem` ≤ `text`, latent extraction is
not buying anything over a text summary on this task — a clean negative.

> Note: the built-in `text` baseline is *truncation* (weak). The honest strong
> baseline is an LLM-written summary at the same token budget; that's the next
> add before drawing conclusions, and is flagged in `model.text_baseline_logits`.

---

# Multimodal user-memory system (text + latent, and when each wins)

The pilot above answered the textual half. The rest of this directory is the
full investigation of **how to combine text and latent memory**, with real-data,
in-LM evaluations. Read `FINDINGS.md` then `HYBRID_FINDINGS.md` for the
conclusions; the modules:

| module | what it does |
|---|---|
| `single_pipeline.py` | All three architectures (text / latent / hybrid) end-to-end through ONE frozen LM + AttMem on the SAME real faces. The apples-to-apples comparison. |
| `multimem.py` | The regime where hybrid genuinely wins: recognize a face, recall an EXACT private code. `--mode codec` measures latent fact-capacity; `--mode bench` runs the full 3-architecture exact-fact benchmark. |
| `mixed_benchmark.py` | Embedding-level perceptual-leg benchmark (caption vs latent identity) over real ArcFace faces, multi-seed/N/C. |
| `muon.py` | Single-device Muon optimizer (used in the recipe search). |

## Key findings (one paragraph)

Personalized user memory should be a **router**, not a single store. Perceptual
identity (faces/voices) must go **latent** — captions cannot tell people apart
(`single_pipeline`: text-only trails hybrid by +0.37..+0.88). The *fact* about a
recognized person should go **text** only when it is **high-entropy/exact**: a
latent holds short exact codes perfectly but has a hard capacity ceiling
(`multimem --mode codec`: 8-char 0.93 exact, 16-char 0.05). For low-entropy /
categorical facts a single latent marker suffices and latent ≈ hybrid (no text
needed). So the **hybrid genuinely beats both only when content combines a
non-captionable identity with an exact, high-entropy fact** — the common real
case (an assistant recognizing you and recalling your exact booking code).

## Reproduce

```bash
# Unified in-LM pipeline (categorical facts): text vs latent vs hybrid
python3 single_pipeline.py --train_steps 6000 --seeds 0 1 2 --ns 10 50 100 300 --cs 2 10 50 0

# Latent fact-capacity curve (can a latent hold an exact code of N chars?)
bash codec_sweep.sh           # -> results/multimem_codec_c{2,4,8,16}.json

# Full multimodal exact-fact bench (within vs beyond latent capacity)
bash bench_sweep.sh           # -> results/multimem_bench_c{8,16}.json
```
