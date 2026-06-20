# Parametric Multimodal User Memory

**Storing what captions cannot carry.**

An agent that personalizes to a user remembers two kinds of things. The
*captionable* half — "my cat is named Bibi", "I'm vegetarian" — survives being
written down and lives happily in a text vector store. The *perceptual* half —
how a face reads a year older, how a voice sounds tired today versus last week,
how a painter's brushwork looks across periods — does **not** survive captioning.
"A brown-haired man" does not tell two brown-haired men apart. The instant you
write a perception down, you throw away the signal that made it discriminative.

This repository is the research line for the missing half: a **parametric
multimodal memory** that stores a perception *as a perception* — a row in a small
per-modality bank bolted onto a frozen language model — and reads it back by
attention inside the model, with no caption in between.

> **Paper:** *Parametric Multimodal User Memory: Storing What Captions Cannot
> Carry* — Bojie Li, Pine AI (June 2026). Built PDF: [`paper/main.pdf`](paper/main.pdf).
> Companion site source: [`site/`](site/).

---

## The one-paragraph idea

Each registered identity is **one row**: a *key* (the L2-normalized embedding a
frozen, off-the-shelf encoder produces for the perception — ArcFace for a face,
ECAPA for a voice, CLIP for a style) paired with a *value* (the language model's
**own** input embedding for a marker token, e.g. `<id_11>`). At generation time
the current perception forms a query, softmax-attends over the bank just before
the output head, and adds a residual that nudges the next token toward the
matching marker. Registration is a single `torch.cat` — no per-user training.
Recall is constant-time (~15 ms) no matter how many identities are stored.

```python
# Register identity #11 from a single photo — no training, O(1):
k = l2_normalize(arcface(photo))        # key:   R^512  (encoder space)
v = model.input_embedding["<id_11>"]    # value: R^2048 (model's own space)
bank.K = torch.cat([bank.K, k[None]])
bank.V = torch.cat([bank.V, v[None]])
```

The mechanism (≈8M trainable params over a 3.1B frozen model) does not merely
*match* text-free embedding retrieval — it **beats** it, because it compares
perceptions in the language model's representation space, which is a sharper
ruler than the encoder's own cosine, exactly where the encoder's similarity is
imperfect.

## Headline results

On **PerceptMem**, our five-sub-modality benchmark (face, painter style, speaker,
acoustic scene, tone of voice — each chosen because text provably cannot carry
its discriminating signal):

| Regime | Cell | Retrieval | AttMem | Δ | p |
|---|---|---|---|---|---|
| random | Face, N=10 | 0.933 | **0.992** | +5.9pp | 0.006 |
| random | Style, N=5 | 0.400 | **0.640** | +24pp | 0.015 |
| adversarial | Tone of voice, K=19 | 0.226 | **0.934** | +70.7pp | <.001 |
| adversarial | Painter style, K=19 | 0.267 | **0.977** | +71.0pp | <.001 |
| adversarial | Acoustic scene, K=19 | 0.827 | **1.000** | +17.3pp | <.001 |

Training the memory to expect look-alikes (siblings, same-room recordings) beats
retrieval by **+14 to +71 points** on four of five sub-modalities. The win lands
exactly where the encoder is weakest; where the encoder is already perfect
(speaker), there is nothing to add and the memory ties. See
[`docs/RESULTS.md`](docs/RESULTS.md) for the full per-cell tables.

## Why not just quantize the perception into a code?

We spent sixteen development cycles on exactly that — a discrete-codebook
predecessor (**Path A**) that snaps each perception to one of *K* learned codes.
It caps at ~7% recall past a few hundred identities regardless of codebook size,
encoder, or training budget — a ~10× gap from continuous attention. Any
categorical bottleneck (a word, a code) discards the signal the encoder worked to
preserve; it is captioning, learned instead of written. Keeping the perception
continuous is the whole game. Path A lives in the repo as the cleanest possible
argument for the design that replaced it.

---

## Repository layout

```
multimodal-user-memory/
├── paper/                  # LaTeX source, figures, built PDF
│   ├── main.tex, body.tex  # the paper
│   ├── figs/               # figure PDFs
│   └── build.sh            # pdflatex + bibtex build
├── site/                   # React + Vite companion website (1:1 with paper)
├── src/
│   ├── perceptmem.py       # PerceptMem benchmark eval surface
│   ├── nanochat_mm/        # the mechanism + all experiment scripts
│   │   ├── attention_memory.py        # ★ the continuous attention memory primitive
│   │   ├── attmem_train_and_eval.py   # ★ main train/eval entry point
│   │   ├── attmem_vl_*.py             # memory inside Qwen2.5-VL
│   │   ├── attmem_latency_benchmark.py
│   │   ├── attmem_propositional_control.py  # text non-regression check
│   │   ├── pathA_*.py                 # the discrete-codebook predecessor
│   │   └── extract_*.py               # encoder feature extraction
│   ├── sanity_*.py         # encoder gating sanity checks
│   └── extract_*.py        # dataset embedding extraction
├── results/                # JSON outputs from every run (committed)
├── notes/                  # session-by-session research log & findings
├── runs/                   # cached weights / embeddings (gitignored)
├── data/                   # local datasets (gitignored)
└── docs/                   # documentation (start here)
```

See [`docs/`](docs/) for the full guide:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the mechanism in detail, the
  four bug-fixes that made it work, and the key/value-orthogonality design rule.
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — PerceptMem: the five sub-modalities,
  their encoders, and the register/recall interface.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md) — exact commands to reproduce every
  headline number, plus the data/encoder setup.
- [`docs/RESULTS.md`](docs/RESULTS.md) — full per-cell result tables.
- [`docs/CODE_MAP.md`](docs/CODE_MAP.md) — what every directory and key script does.

## Quick start

```bash
# Environment: Python 3.10+, PyTorch, transformers. An H100-class GPU for training.
pip install torch transformers numpy scipy scikit-learn speechbrain

# Run the main train/eval entry point (mode, n_steps, seed, [bank_size_max], [adv_prob]):
cd src/nanochat_mm
python3 attmem_train_and_eval.py v-sty-clip 5000 42        # painter style, random regime
python3 attmem_train_and_eval.py a-para 5000 42 0 0.3      # tone of voice, look-alike regime
```

Modes: `v-xc-id-xxxl` (face), `v-sty-clip` (painter style), `a-xr-id` (speaker),
`a-scn` (acoustic scene), `a-para` (tone of voice). Every run logs to
`results/`. A face run is ~17 min on an H100 (12K steps); audio/style ~5–10 min.
Reproducing perceptual data requires the standard datasets and encoders — see
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Where this sits

This is one half of a larger bet — that a personalized agent should keep its
memory *inside* the model rather than in an external retrieval index. The
companion line, *User as Engram* ([arXiv:2606.19172](https://arxiv.org/abs/2606.19172),
also under [19PINE-AI](https://github.com/19PINE-AI/user-as-engram)), makes the
same bet for the *captionable* half — writing per-user facts into a hash-keyed
parametric memory. This repo is the perceptual counterpart. The mechanisms differ
(hash-keyed N-gram rows for facts, continuous cross-attention over perceptual
banks for perceptions); the thesis — *store it in the model, not in an index* —
is shared.

## License & contact

Internal research, Pine AI. Contact: Bojie Li (`boj@19pine.ai`).
