# PerceptMem — the benchmark

**PerceptMem** measures cross-condition perceptual recall honestly. It has five
sub-modalities, each chosen for exactly one reason: *the signal that discriminates
is provably destroyed by captioning.* They span vision and audio, identity and
style, the physical and the affective.

Defined in `src/perceptmem.py`; per-task pipelines in `src/nanochat_mm/`.

## The five sub-modalities

| Sub-modality | Mode | Source / encoder | Why text cannot carry it |
|---|---|---|---|
| Face across age & lighting | `v-xc-id-xxxl` | LFW + AgeDB / ArcFace | "brown-haired man" fits thousands |
| Painter style | `v-sty-clip` | WikiArt / mid-layer CLIP | no caption separates early vs. late Monet |
| Speaker across recordings | `a-xr-id` | LibriSpeech / ECAPA-TDNN | timbre is not transcribable |
| Acoustic scene | `a-scn` | ESC-50 / AST | "traffic noise" fits every street |
| Tone of voice | `a-para` | RAVDESS / wav2vec2 | "sounded tired" lacks a personal baseline |

The mode strings (left of `Source`) are the first argument to
`attmem_train_and_eval.py`. The full mode → data-file map is in
`src/nanochat_mm/attmem_train_and_eval.py` (`MODE_PATHS`).

## Design principles

- **Tests memory, not perception.** Every sub-modality has a fixed, off-the-shelf
  encoder that *any* method may use as black-box infrastructure. No one wins by
  having a better encoder.
- **Cross-session by construction.** A registered sample and its query come from
  *different* recordings / photos / periods.
- **Disjoint train/eval identities.** The identities used to train the memory
  never overlap with those it is evaluated on.
- **Minimal interface.** Just `register(modality, label, perception)` and
  `recall(modality, perception)`. For a memory of *N* identities, register one
  sample each, then issue cross-condition queries and ask the method to name the
  right one.
- **Tone of voice is a pair.** Each registered "identity" is a
  (speaker, emotional-state) pair, so recalling it means matching a paralinguistic
  state across separate utterances — impossible for a per-utterance caption
  without the user's own history.

## The baseline to beat: embedding retrieval

Throughout, the reference is **embedding retrieval** — the *same* encoder, cosine
nearest-neighbour over the registered keys. We sometimes call it the *encoder
ceiling*, because it is the best one can do with the encoder's own similarity. But
it is **not** a ceiling for AttMem, which measures similarity in the model's
representation space instead and can therefore exceed it.

The three doomed text routes PerceptMem rules out (paper §2):

1. **Caption-and-search** (Mem0, MemoryLLM) — the description is true and useless;
   it fits thousands of speakers.
2. **Recognize-and-label** (M3-Agent) — stores a bare label "speaker 47"; the
   perception never reaches the model.
3. **Per-concept training** (Yo'LLaVA, MyVLM) — works, but ~1s of gradient descent
   per identity and a concept-specific artifact, not a general memory.

## Two evaluation regimes

- **Random** — distractors drawn uniformly from the pool.
- **Adversarial** — distractors are the *K* most cosine-similar identities (the
  hardest look-alikes: siblings, same-room recordings). Activated by the
  `adv_prob` argument, which also mixes hard banks into a fraction of training.

The two regimes trade off; the shape of the trade-off is modality-dependent (see
[`RESULTS.md`](RESULTS.md)). A ~10% look-alike mix is the sweet spot for
random-population deployments; 30% suits known-adversarial settings.

## See also

- [`REPRODUCE.md`](REPRODUCE.md) — how to get the data and run it.
- [`RESULTS.md`](RESULTS.md) — what the numbers come out to.
- Paper §4 (`paper/body.tex`) — the full benchmark description.
