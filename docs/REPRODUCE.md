# Reproducing the results

The repository contains research scripts rather than a packaged inference
library. Full reproduction requires the original datasets, specialist encoder
checkpoints, Hugging Face model downloads, and a CUDA GPU. Inspecting the paper,
site, source, and committed JSON outputs does not require those assets.

## Environment

- Python 3.10+
- PyTorch with CUDA for full model runs
- Transformers, NumPy, SciPy, scikit-learn, and SpeechBrain
- Dataset-specific packages used by the selected extraction/evaluation script

Minimal shared stack:

```bash
pip install torch transformers numpy scipy scikit-learn speechbrain
```

Grounding evaluations additionally use packages such as `datasets`, Pillow,
SoundFile, torchaudio, InsightFace, and the relevant Qwen multimodal model
dependencies.

## Data and encoders

Generated embeddings are stored under `runs/embeddings/` and are intentionally
gitignored.

| Signal | Data | Encoder | Extraction entry points |
|---|---|---|---|
| faces | LFW, AgeDB | ArcFace / AntelopeV2 | `src/extract_lfw_*.py`, `src/extract_agedb.py` |
| painter style | WikiArt | CLIP / DINOv2 | `src/nanochat_mm/extract_wikiart_xxl.py`, style scripts under `src/` |
| speakers | LibriSpeech, VoxCeleb | ECAPA-TDNN | `src/extract_voxceleb1.py`, `src/nanochat_mm/extract_wavlm_libri.py` |
| acoustic scenes | ESC-50 | AST | `src/extract_esc50_full.py` |
| vocal tone | RAVDESS | wav2vec2 emotion | `src/nanochat_mm/extract_more_embeddings.py` |

The exact cache filename expected by the main harness is listed in `MODE_PATHS`
inside `src/nanochat_mm/attmem_train_and_eval.py`.

Do not commit datasets, model weights, face/voice templates, or generated
embedding caches.

## Training-free paired recall

Run from the repository root. `n_steps=0` selects the final training-free read.
For a tied-embedding Qwen host, the paper uses a sharp attention temperature and
gain as fixed constants:

```bash
ATTMEM_INV_TEMP=300 \
ATTMEM_OUT_GAIN=64 \
ATTMEM_PAIRED_NS="10,100,300,1000" \
ATTMEM_PAIRED_SEEDS=20 \
python3 src/nanochat_mm/attmem_train_and_eval.py v-xc-id-xxxl 0 42
```

Other benchmark modes:

```text
v-xc-id-xxxl  face identity
v-sty-clip     painter style
a-xr-id        speaker identity
a-scn          acoustic scene
a-para         tone of voice
```

For the paired hard-distractor face cell:

```bash
ATTMEM_INV_TEMP=300 \
ATTMEM_OUT_GAIN=64 \
ATTMEM_PAIRED_NS=20 \
ATTMEM_PAIRED_ADV_K=19 \
ATTMEM_PAIRED_SEEDS=20 \
python3 src/nanochat_mm/attmem_train_and_eval.py v-xc-id-xxxl 0 42
```

Untied-embedding and hybrid hosts use the same read with a larger fixed gain
(`ATTMEM_OUT_GAIN=256` in the reported grid). Select the host with
`ATTMEM_MODEL_ID`.

## Grounding evaluations

These scripts download their datasets/models through the providers' standard
APIs and can require substantial GPU memory:

```bash
# Face: VLM bounding box -> RetinaFace -> ArcFace
ATTMEM_VLM=Qwen/Qwen2.5-VL-7B-Instruct \
  python3 src/nanochat_mm/eval_agentic_production.py 40 2 0

# Painting: VLM bounding box -> CLIP
ATTMEM_VLM=Qwen/Qwen2.5-VL-7B-Instruct \
  python3 src/nanochat_mm/eval_agentic_paintings.py 40 2 0

# Audio: Qwen2.5-Omni time span -> ECAPA
python3 src/nanochat_mm/eval_agentic_audio.py 30 3 0
```

The face and painting scripts compare the grounded path with whole-scene and
correct-region-oracle controls. The audio script reports the corresponding
whole-window, grounded-span, and oracle-span results.

## Composition, open set, and agent evaluation

```bash
python3 src/nanochat_mm/composition_eval.py 10 12
python3 src/latentmem/openset_verification.py
python3 src/nanochat_mm/agent_benchmark.py 5 "10,25,50,80"
```

These rely on the same local encoder caches. Review the constants at the top of
each script before a large run; several chronology-preserving experimental
scripts expose their configuration directly in source rather than through a
uniform CLI.

## Paper and website

```bash
# Paper
cd paper
./build.sh

# Website
cd ../site
npm ci
npm run lint
npm run build
```

The canonical publication PDF is
<https://arxiv.org/pdf/2608.28609>; local paper builds are for source
verification only.
