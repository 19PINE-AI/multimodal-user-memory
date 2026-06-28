#!/bin/bash
# Semantic/gist test: does latent M capture an aggregate preference (lossy-friendly)
# far better than it captured exact facts (mem~0.55)? Contrast vs oracle + text.
set -u
cd "$(dirname "$0")"
M="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
R="/home/ubuntu/multimodal-user-memory/results"
COMMON="--model_id $M --task gist --n_steps 3000 --k 16 --batch 8 --eval_every 250 --eval_n 512 --seed 42 --need_gb 22"
run () { name="$1"; shift
  echo "================ $name ================"
  python3 train.py $COMMON "$@" > "$R/sweep_${name}.log" 2>&1 || echo "  ($name failed/killed)"
  echo -n "  $name final: "; grep "eval @" "$R/sweep_${name}.log" | tail -1 || echo "(no eval)"
}
run gist_frozen --lora_rank 0
run gist_lora   --lora_rank 16
echo "================ GIST SWEEP SUMMARY ================"
echo "contrast: exact-fact recall mem topped out ~0.55 (oracle 0.96)"
for n in gist_frozen gist_lora; do echo -n "$n : "; grep "eval @" "$R/sweep_${n}.log" | tail -1 || echo "(none)"; done
echo "GIST SWEEP DONE"
