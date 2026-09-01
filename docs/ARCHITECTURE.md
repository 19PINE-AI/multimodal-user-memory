# Architecture

The final system separates perceptual recall into three jobs: **ground** the
referent, **identify** it, and **store/read** the resulting identity inside a
frozen language model.

## 1. Ground: resolve what the user means

Perceptions usually arrive in context rather than as clean crops. A user says
“remember her” while pointing into a group photo, refers to the painting on the
left, or asks about the speaker who just talked. A multimodal language model
resolves that referring expression to a visual bounding box or audio time span.

Where useful, a modality-specific detector refines the grounded region before
encoding. The face pipeline, for example, runs RetinaFace alignment inside the
VLM-selected region.

Grounding is not optional. Encoding an entire two-person scene gives the
specialist encoder no way to know which person the user meant and reaches only
0.05 recall in the paper's face experiment.

Relevant implementations:

- `src/nanochat_mm/eval_agentic_production.py` — face grounding, alignment, and
  identification.
- `src/nanochat_mm/eval_agentic_paintings.py` — visual grounding outside the
  face domain.
- `src/nanochat_mm/eval_agentic_audio.py` and
  `eval_conversation_grounding.py` — temporal grounding for speakers.

## 2. Identify: use a specialist encoder

The grounded region is mapped to an L2-normalized identity key by a frozen
encoder chosen for that perceptual domain:

| Perceptual signal | Encoder |
|---|---|
| Face identity | ArcFace |
| Speaker identity | ECAPA-TDNN |
| Painter style | CLIP mid-layer features |
| Acoustic scene | AST |
| Tone of voice | wav2vec2 emotion encoder |

The VLM is deliberately not asked to be the identity encoder. In the final
evaluation, Qwen2.5-VL native vision tokens reach 0.54 recall on cross-age
AgeDB faces at `N=20`, compared with 0.81 for ArcFace. The components are
complementary: the VLM is the localizer; the specialist encoder is the identity
metric.

## 3. Store and read: attention over marker-token values

Each registered identity occupies one row:

- key `k_i ∈ R^D`: the normalized specialist-encoder embedding;
- value `v_i ∈ R^H`: the frozen language model's own embedding for the marker
  token assigned to that identity.

For a query key `q`, the read at the output head is:

```text
w  = softmax(β qᵀK)
r  = wᵀV
h' = h + g Wₒr
```

`K` and `V` stack the registered rows, `β` is a fixed attention sharpness, `g`
is a fixed residual gain, and `Wₒ` is the identity in the final experiments.
The returned vector biases the next-token distribution toward the matching
marker. Recognition therefore happens during the frozen model's ordinary
forward pass instead of in an external retrieve-and-reprompt loop.

The final read is training-free. Its purpose is to reproduce the encoder's
nearest-neighbor decision as a native token, not to learn a better similarity
metric.

Core code:

- `src/nanochat_mm/attention_memory.py` — per-modality banks and attention read.
- `src/nanochat_mm/qwen_attmem_bolt.py` — frozen-model wrapper and output-head
  hook.
- `src/nanochat_mm/attmem_train_and_eval.py` — zero-step paired evaluation and
  the earlier trained-variant harness.

## Four details required for a faithful read

1. Do not divide L2-normalized cosine logits by `sqrt(D)`; doing so makes the
   attention nearly uniform.
2. Use a high `β` so the read approximates a hard argmax.
3. attach the residual at the output head so later layers do not dilute it.
4. Use sufficient gain `g` for the marker logit to dominate. Models with untied
   input/output embeddings require a larger gain than tied-embedding models.

With those settings, the in-model decision matches encoder cosine retrieval to
within 0.001 across the paired face sweep from `N=5` to `N=1000`.

## Registration, deletion, and isolation

Registration appends one `(key, value)` row and performs no gradient update.
Deletion removes that row; there is no fine-tuned identity artifact to unlearn.
Vision and audio use separate banks, and the mixed-modality control verifies
that populating one does not materially activate the other.

The rows contain biometric templates. A deployment should require explicit
consent, encrypt the bank or keep it on-device, support deletion, reject
unenrolled identities, and evaluate the selected encoder for demographic bias.
