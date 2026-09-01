# PerceptMem benchmark

**PerceptMem** measures cross-condition perceptual recall: register an identity
from one observation, then recognize it from a different observation recorded
under changed conditions. The benchmark evaluates memory rather than raw
perception by giving every method the same frozen specialist encoder.

## Register/recall contract

Every domain exposes the same conceptual interface:

```text
register(modality, marker, perception) -> append one identity row
recall(modality, perception)           -> predicted registered marker
```

One sample per identity is registered. Queries come from a different image,
recording, session, age, lighting condition, or artwork. Recall is recall@1 over
the registered markers.

The main comparison uses identical registrations and queries for the in-model
read and cosine nearest-neighbor retrieval. The marker assigned to each bank
slot is randomized on every draw so a fixed token bias cannot inflate recall.

## Five conceptual sub-modalities

| Sub-modality | Representative data / encoder | Signal text loses |
|---|---|---|
| Face across age and lighting | LFW + AgeDB / ArcFace | fine-grained facial identity |
| Painter style across works | WikiArt / CLIP mid-layer | brushwork and stylistic identity |
| Speaker across recordings | LibriSpeech + VoxCeleb / ECAPA-TDNN | timbre |
| Acoustic scene | ESC-50 / AST | the specific acoustic environment |
| Tone of voice | RAVDESS / wav2vec2 | person-relative affect and prosody |

For tone, an identity is a `(speaker, emotional state)` pair rather than an
utterance label.

## Final 12-domain evaluation

The submitted paper expands the five concepts into 12 dataset/encoder domains.
Each domain is evaluated at `N ∈ {10, 20, 40}` over 30 draws: 90 tasks per
domain and 1,080 tasks in total.

| Domain | Key encoder | Identity pool |
|---|---|---:|
| Face / AgeDB, cross-age | ArcFace | 500 |
| Face / LFW | ArcFace | 1,680 |
| Face / LFW | AntelopeV2 | 901 |
| Face / LFW + AgeDB | ArcFace | 2,180 |
| Speaker / LibriSpeech | ECAPA-TDNN | 58 |
| Speaker / VoxCeleb | ECAPA-TDNN | 40 |
| Acoustic scene / ESC-50 | AST | 50 |
| Painting style / WikiArt | CLIP mid-layer | 128 |
| Painting style / WikiArt | DINOv2 | 50 |
| Vocal tone / RAVDESS | wav2vec2 emotion | 168 |
| Face / AgeDB control | Qwen2.5-VL native tokens | 567 |
| Painting style / WikiArt | contrastive head | 26 |

The Qwen2.5-VL row is a deliberately weak key-space control, not a recommended
configuration.

## Baselines and controls

- **Random chance:** `1/N`.
- **Caption and search:** a strong vision/audio language model writes a
  re-identification description; a sentence encoder retrieves over the text.
- **Encoder cosine:** nearest-neighbor search over the same registered keys. It
  is the recognition ceiling the in-model read is designed to reproduce.
- **Whole-scene encoding:** tests why grounding is necessary.
- **Correct-region oracle:** specialist encoder on the known target crop/span.
- **VLM-native features:** tests whether the localizer can also serve as the
  identity encoder.
- **Per-concept and learned-metric baselines:** charitable implementations of
  classifier, projection, LDA, and whitening alternatives.

## Data policy

The repository contains code and aggregate result files, not datasets or
biometric embeddings. Obtain every dataset from its original distributor and
follow its license and intended-use terms. Generated `.npz` caches live under
`runs/embeddings/`, which is gitignored.

See the paper's Appendix “Detailed results” for the complete per-domain table
and [REPRODUCE.md](REPRODUCE.md) for the evaluation entry points.
