#!/bin/bash
# Decisive sufficiency test: dense multi-probe supervision (8 recall probes/doc).
# Does M finally encode retrievable per-fact state? mem must climb 0.50 -> ~0.96.
set -u
cd "$(dirname "$0")"
M="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
R="/home/ubuntu/multimodal-user-memory/results"
COMMON="--model_id $M --n_steps 3000 --k 16 --batch 4 --probes_per_doc 8 --recall_frac 1.0 --eval_every 250 --eval_n 512 --seed 42 --need_gb 30"
run () { name="$1"; shift
  echo "================ $name ================"
  python3 train.py $COMMON "$@" > "$R/sweep_${name}.log" 2>&1 || echo "  ($name failed/killed)"
  echo -n "  $name final: "; grep "eval @" "$R/sweep_${name}.log" | tail -1 || echo "(no eval)"
}
run mp8_lora   --lora_rank 16
run mp8_frozen --lora_rank 0
echo "================ MULTIPROBE SWEEP SUMMARY ================"
echo "refs: single-probe recall mem=0.498(chance), +LoRA mem=0.529 | recall oracle=0.963"
for n in mp8_lora mp8_frozen; do echo -n "$n : "; grep "eval @" "$R/sweep_${n}.log" | tail -1 || echo "(none)"; done
echo "MP SWEEP DONE"
