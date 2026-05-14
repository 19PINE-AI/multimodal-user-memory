# Multimodal User Memory — Perceptual Engram

Research line: cross-session perceptual memory for multimodal agents, where the content (face / voice / style / scene / prosody) is **not naturally captionable**.

See [`research_plan.md`](research_plan.md) for the framing, related work survey, novelty positioning, and method/benchmark plan.

## Status

- Plan: **v1** (2026-05-14). LLaVA-style post-training on Qwen3-VL + Voxtral bases, not from-scratch pretraining.
- Sanity checks 1 & 2: **PASS**. ECAPA-TDNN cross-recording (ratio 101 at K=32), ArcFace cross-condition (ratio 61 at K=32). Details in [`notes/sanity_findings.md`](notes/sanity_findings.md).
- Next: learned RQ-VAE (replacing naive k-means), scene/style/paralinguistic encoder sanity checks, then Qwen3-VL + Voxtral wiring.

## Layout

```
multimodal-user-memory/
├── research_plan.md           # Main plan (v1)
├── README.md                  # This file
├── src/                       # All experiment code
│   ├── sanity_ecapa_collisions.py    # Audio gating experiment (passed)
│   ├── sanity_arcface_collisions.py  # Vision gating experiment (passed)
│   └── ...
├── notes/                     # Findings, designs, decisions
│   └── sanity_findings.md
├── results/                   # JSON outputs from experiments
│   ├── sanity_ecapa_collisions.json
│   └── sanity_arcface_collisions.json
├── runs/                      # Model checkpoints / cached weights
│   └── pretrained-ecapa/      # SpeechBrain ECAPA-TDNN (auto-downloaded)
├── scripts/                   # Orchestration shell scripts
├── refs/                      # Borrowed code from sibling repos (engram prototypes etc.)
└── data/                      # Local datasets (small)
```

## Reproducing the gating experiments

```bash
cd ~/multimodal-user-memory

# Audio (uses ~/data/LibriSpeech/test-clean)
python3 src/sanity_ecapa_collisions.py

# Vision (uses sklearn LFW; downloads ~250MB on first run)
python3 src/sanity_arcface_collisions.py
```

Outputs land in `results/`.

## Sibling research lines

- `~/polar-research` — orthogonal LoRA decomposition for agent post-training
- `~/user-as-engram` — surgical row insertion into hashed N-gram tables (text)
- `~/user-as-lora` — meta-trained base for learned introspection over per-user LoRAs
- `~/UserAsCode` — executable code as memory representation

This repo extends the line **along the modality axis**: each sibling stays text-based, this one targets perceptual content (the unnameable cases).
