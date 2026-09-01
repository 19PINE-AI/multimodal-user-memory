# Parametric Multimodal User Memory

**Storing what captions cannot carry.**

[arXiv abstract](https://arxiv.org/abs/2608.28609) ·
[PDF](https://arxiv.org/pdf/2608.28609) ·
[Project website](https://01.me/research/multimodal-user-memory/)

**Bojie Li** (Pine AI) and **Noah Shi** (University of Washington)

## Abstract

A personalized agent needs a *user memory*: a persistent model of who its user
is. Today it is almost always *text* — transcripts and captions retrieved by
similarity. This serves the *captionable* half of a person ("my cat is named
Bibi"), but discards the *perceptual* half no caption can hold: how a voice
sounds, how a face reads across age and lighting, how tired someone sounds. We
measure this loss across five modalities: a strong caption-based re-identifier
recovers as little as 0.11 of a dedicated encoder's recall, collapsing toward
chance on non-nameable signals.

We instead **ground** perceptual memory in the model, decomposing recall into two
subproblems: a vision-language model grounds the referent in context (*what* and
*where*), and a dedicated encoder extracts an identity *key* (*who*), stored as
one inline token read by attention at generation with no external round-trip.
Neither suffices alone — the VLM identifies cross-age faces at only 0.54 recall
where a face encoder reaches 0.81, and an ungrounded encoder recognizes a
two-person-scene referent at 0.05 — yet together they reach correct-region oracle
(0.96), generalizing to multi-speaker audio and video. The recognition core is
*training-free*: it reproduces the encoder's recall on any frozen model at
O(1) registration cost. On **PerceptMem** (12 domains, 1,080 tasks) perceptual
identity is capacity-limited (recall ≈ min(1, k/M) of the encoder's ceiling)
while exact facts are binding-limited: identity belongs in a parametric bank,
facts in a text store. The two memories compose cleanly: an agent with both can
remember not only what its user said, but also what they are like.

## The idea

Perceptual recall is split into three jobs, each handled where the relevant
model is strongest:

1. **Ground the referent.** A vision-language or audio-language model resolves
   which person, object, or temporal span the user means in context.
2. **Identify it.** A frozen specialist encoder—such as ArcFace for faces,
   ECAPA-TDNN for speakers, or CLIP for painter style—turns the grounded region
   into a cross-condition identity key.
3. **Store and read it in-model.** The key is paired with one of the language
   model's own marker-token embeddings. Attention over the bank returns that
   marker inside the frozen model's forward pass, without captioning the
   perception or making an external retrieval round-trip.

Registration appends one key/value row and requires no per-user optimization:

```python
# Register identity 11 from one grounded perception—no gradient update.
k = l2_normalize(encoder(grounded_region))
v = model.input_embedding["<id_11>"]
bank.K = torch.cat([bank.K, k[None]])
bank.V = torch.cat([bank.V, v[None]])
```

## Main findings

- **Captions lose perceptual identity.** Across five modalities, caption-based
  re-identification falls to 0.11 of the dedicated encoder's recall on the least
  nameable signals.
- **Grounding is necessary.** Whole-scene encoding reaches only 0.05 recall on
  the two-person face task. Grounding the requested referent before encoding
  reaches the correct-region oracle at 0.96.
- **The encoder sets the ceiling.** The training-free in-model read reproduces
  the specialist encoder across ten frozen model families. Its role is to give
  the encoder a native, composable home inside the model—not to claim a new
  recognition metric.
- **PerceptMem broadens the test.** The final benchmark contains 12
  dataset/encoder domains and 1,080 tasks across face identity, speaker identity,
  acoustic scenes, painter style, and tone of voice.
- **Text and perceptual memory compose.** Perceptual identity belongs in the
  parametric bank; exact facts remain in a text store. The combined agent can
  recognize a returning user and retrieve what it knows about them.

See the [paper](https://arxiv.org/abs/2608.28609) for the full experiments and
the [results guide](docs/RESULTS.md) for a map from claims to committed outputs.

## Repository layout

```text
multimodal-user-memory/
├── paper/                  # LaTeX source, figures, and submitted PDF
├── site/                   # React + Vite companion website
├── src/
│   ├── perceptmem.py       # unified benchmark surface
│   └── nanochat_mm/        # memory, grounding, agent, and baseline experiments
├── results/                # committed experiment outputs
├── docs/                   # architecture, benchmark, results, and reproduction guides
├── notes/                  # chronological research record
├── runs/                   # local checkpoints/embeddings (gitignored)
└── data/                   # local datasets (gitignored)
```

Documentation:

- [Architecture](docs/ARCHITECTURE.md)
- [PerceptMem benchmark](docs/BENCHMARK.md)
- [Reproduction guide](docs/REPRODUCE.md)
- [Results guide](docs/RESULTS.md)
- [Code map](docs/CODE_MAP.md)

## Quick start

The experiments require Python 3.10+, PyTorch, Transformers, and the specialist
encoders for the modalities being evaluated. Dataset files, model checkpoints,
and cached embeddings are intentionally not included.

```bash
pip install torch transformers numpy scipy scikit-learn speechbrain

# Training-free paired recall on the face domain. Cached encoder embeddings
# must first be placed under runs/embeddings; see docs/REPRODUCE.md.
ATTMEM_INV_TEMP=300 ATTMEM_OUT_GAIN=64 ATTMEM_PAIRED_NS=10 \
  python3 src/nanochat_mm/attmem_train_and_eval.py v-xc-id-xxxl 0 42
```

The default backbone and dataset paths used by each experiment are documented
in the corresponding script. Many full evaluations require an H100-class GPU;
the paper and committed JSON outputs can be inspected without one.

## Citation

If you use this work, please cite the [arXiv paper](https://arxiv.org/abs/2608.28609).
Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@misc{li2026parametricmultimodalusermemory,
  title         = {Parametric Multimodal User Memory: Storing What Captions Cannot Carry},
  author        = {Bojie Li and Noah Shi},
  year          = {2026},
  eprint        = {2608.28609},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2608.28609}
}
```

## License and contact

Licensed under the [Apache License 2.0](LICENSE) — © 2026 Pine AI.

Contact: Bojie Li ([boj@19pine.ai](mailto:boj@19pine.ai)).
