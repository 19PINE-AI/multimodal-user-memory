# Results guide

The canonical result is the published paper:
[arXiv:2608.28609](https://arxiv.org/abs/2608.28609). This file maps its main
claims to the committed outputs and distinguishes them from earlier exploratory
trained-memory experiments that remain in `results/` for research provenance.

## Main conclusions

1. **Text is a lossy perceptual-memory channel.** A caption-based
   re-identifier retains as little as 0.11 of a dedicated encoder's recall on
   non-nameable signals.
2. **Grounding and identification solve different problems.** On the
   two-person face task, whole-scene encoding reaches 0.05 recall; VLM grounding
   followed by alignment and ArcFace reaches the correct-region oracle at 0.96.
3. **The in-model read faithfully reproduces the encoder.** With the final
   sharp, training-free read, AttMem and cosine retrieval make the same decision
   on random and hard-distractor banks.
4. **The result is portable.** The encoder match holds across ten frozen model
   families and a five-model-by-five-modality grid.
5. **Perceptual and text memory compose.** Perceptual identity recalls a native
   marker; that marker can key exact facts in a text store.

## Paired recognition result

Both methods below use identical registrations and queries over 20 draws, with
the target marker slot randomized on each draw.

| Regime | Cell | Encoder cosine | AttMem | Difference |
|---|---|---:|---:|---:|
| random | Face, `N=10` | 0.948 | 0.948 ± 0.033 | 0.0 pp |
| random | Face, `N=100` | 0.792 | 0.792 ± 0.021 | 0.0 pp |
| random | Face, `N=300` | 0.749 | 0.749 ± 0.018 | 0.0 pp |
| random | Face, `N=1000` | 0.776 | 0.777 ± 0.008 | 0.0 pp |
| random | Style, `N=5` | 0.473 | 0.473 ± 0.104 | 0.0 pp |
| random | Style, `N=10` | 0.428 | 0.428 ± 0.076 | 0.0 pp |
| hard distractors | Face, `K=19` | 0.853 | 0.853 ± 0.007 | 0.0 pp |

This equality is intentional. The contribution is the grounded, in-model
container and its composition behavior; recognition quality belongs to the
chosen encoder.

## Other reported results

- **12-domain breadth:** 1,080 PerceptMem tasks at `N=10,20,40`. The in-model
  memory tracks each domain's encoder and exceeds chance in every domain.
- **Model universality:** every cell in the five-model-by-five-modality grid is
  within one recall point of its encoder; most are within 0.003.
- **Open set:** verification AUROC is 0.99 for voice and 0.97–0.99 for faces.
  Stranger rejection is 0.96–0.99 at 20 enrolled identities and 0.81 at the
  harder 80-user agent operating point.
- **Composition:** face-to-fact accuracy is 4–10× the face-withheld chance
  baseline and follows recognition accuracy times the model's in-context lookup
  reliability.
- **End-to-end agent:** at 80 enrolled users, identity and routed fact recall are
  both 0.86, stranger rejection is 0.81, and overall task success is 0.83.
- **Capacity router:** compressing `M` identities into `k` prototype slots
  follows `min(1, k/M) × C(M)`, where `C(M)` is the encoder ceiling. Exact facts
  instead fail from key/value binding interference and belong in a text store.

## Output map

| Claim or figure | Primary committed outputs |
|---|---|
| paired encoder match | `attmem_paired_*.json` |
| grounding faces | `agentic_prod_*.json`, `agentic_realistic_*.json` |
| grounding paintings | `agentic_paint_*.json` |
| grounding audio | `agentic_audio_*.json`, `conversation_grounding_*.json` |
| text-caption baselines | `text_baseline.json`, `text_baseline_audio.json`, `text_baseline_style.json` |
| cross-domain benchmark | `cross_domain.json` |
| in-model composition | `composition*.json`, `reasoning_eval.json` |
| open-set evaluation | `openset_verification.json` |
| end-to-end agent | `agent_benchmark.json` |
| latency and cost | `latency*.json`, `cost.json` |
| modality isolation | `attmem_mixed_modal.json`, `av_fusion.json` |

## Historical outputs

Files whose names include nonzero training steps or adversarial-training
probabilities record an earlier learned, softer read. Some of those experiments
appeared to beat raw cosine on selected cells, but the final paired protocol
identified marker-slot bias and under-sharpening as confounds. They are retained
to make the research path auditable; they are not the published headline.

When a table here and a historical JSON file differ, use the paper's paired
protocol and final training-free outputs.
