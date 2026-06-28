# Latent user-memory pilot — findings

**Question.** Can a learned write head compress a user-memory document into k
continuous vectors M such that a frozen(ish) LM, reading M alone, answers as
well as with the full document in context — i.e. can the *captionable* half of
user memory go latent instead of text?

**Answer (this task): no, for exact factual content.** Across the full recipe
space, the latent memory tops out near chance on single-fact recall.

## Result (Qwen2.5-1.5B, k=16, held-out users)

Single-fact recall ("is setting X enabled?"), oracle (full context) = **0.963**,
chance/text-at-matched-budget = **0.500**:

| recipe | mem |
|---|---|
| frozen reader, 1 probe/doc | 0.498 |
| + LoRA reader (rank 16) | 0.529 |
| + dense 8-probe/doc supervision + LoRA | 0.555 |
| + dense supervision, frozen | 0.543 |

Gated AND-of-3 task (oracle 0.803, prior 0.490): bridge-informed AdamW recipe
plateaus at **mem≈0.59** and does **not** improve with steps (3k≈10k) — and the
recall result shows this 0.59 is a coarse aggregate ("how many settings are on"),
not retrievable per-fact memory.

Levers tried, none cracked exact-fact storage: bridge-style projection
(recon-dominant loss, mid-layer capture at 0.67 depth, LayerNorm-matched output);
Muon vs AdamW; LoRA on the read path (oracle kept clean via `disable_adapter`);
dense multi-probe supervision; 3k–10k steps.

## Why

Binding N discrete *(identity → value)* pairs into k continuous vectors and
retrieving an arbitrary one by content is a superposition/binding problem that
continuous compression handles poorly. Text has no such problem: each fact is a
discrete, addressable token sequence. This is the exact inverse of the perceptual
case (AttMem), where continuous encoder embeddings beat text because the
discriminating signal is sub-symbolic and *not* captionable.

The mechanism is sound, not buggy: the clean oracle scores 0.96, the text/none
baselines sit exactly at chance, and M does move logits on the gated task.

## Implication

The best design for multimodal user memory is a **router**, not a unified latent
extractor:

- captionable / factual content  → **text** memory + retrieval (latent loses here)
- perceptual content             → **parametric latent** encoder-banks (AttMem wins)

This is positive evidence for the paper's central thesis.

## Known limitation of this study / where a latent win is still plausible

The synthetic task is *exact* fact lookup — the adversarial worst case for
latent. The regime where ICAE/gist-style latent extraction does show gains is
*semantic / gist* recall (summary, paraphrase QA), where approximate sufficiency
suffices. A LoCoMo-style semantic task is the honest next test; expectations
should be modest given the binding result above.

Also untested by deliberate choice (low expected value vs cost): rank-32+ LoRA,
k=64, and 3B/7B readers. The flatness across frozen/LoRA/dense-supervision makes
a scale-driven reversal unlikely; the bridge paper itself (8B) only ties/edges
text where *reasoning* helps, never on exact recall.

## Reusable harness

`model.py` (write head + optional LoRA read path + clean-oracle context),
`muon.py`, `train.py` (KL-distillation + task-CE + curriculum + doc-recon +
multi-probe), `eval.py` (full/mem/text/none + sufficiency-KL), `data.py`
(controlled gated/recall/multi-probe data). One caveat: `recon_loss` currently
teacher-forces the real doc, so M is bypassed (leakage) — it needs gist-style
masking before the reconstruction term is meaningful.
