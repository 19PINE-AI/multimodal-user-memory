#!/bin/bash
# Latent user-memory pilot: wait for GPU, then train + eval across budgets k.
# Decisive question: does k learned latent vectors (mem) beat k tokens of text
# (text) at equal budget, and approach full context (full)?
set -u
cd "$(dirname "$0")"

NEED_GB="${NEED_GB:-30}"
STEPS="${STEPS:-3000}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
echo "[run_pilot] model=${MODEL_ID}  steps=${STEPS}"
echo "[run_pilot] waiting for >= ${NEED_GB} GB free GPU…"
python3 gpu_wait.py "$NEED_GB"

# On-hardware smoke: load the real model and run 4 steps + an eval. If the GPU
# path is broken this fails fast, before the multi-hour sweep.
echo "===== SMOKE (validate GPU path) ====="
if ! python3 train.py --model_id "$MODEL_ID" --n_steps 4 --k 4 --batch 4 \
        --eval_every 2 --eval_n 32 --seed 7 --no_wait; then
  echo "[run_pilot] SMOKE FAILED — aborting before the sweep. See trace above."
  exit 1
fi
echo "[run_pilot] smoke OK."

# Core sweep: vary the latent budget k. Same data, same steps.
for K in 4 8 16 32; do
  echo "===== k=$K ====="
  python3 train.py --model_id "$MODEL_ID" --n_steps "$STEPS" --k "$K" --batch 8 \
      --n_settings 16 --n_relevant 3 --recall_frac 0.5 --seed 42 --no_wait
done

echo "[run_pilot] done. results in ../../results/latentmem_*.json"
