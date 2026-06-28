#!/bin/bash
# Decisive test: does LoRA on the read path let M store retrievable facts?
# recall_lora is the key cell (mem-recall must jump from chance 0.50 -> ~oracle).
set -u
cd "$(dirname "$0")"
M="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
R="/home/ubuntu/multimodal-user-memory/results"
COMMON="--model_id $M --n_steps 3000 --k 16 --batch 6 --eval_every 250 --eval_n 512 --seed 42 --need_gb 22 --lora_rank 16"
run () { name="$1"; shift
  echo "================ $name ================"
  python3 train.py $COMMON "$@" > "$R/sweep_${name}.log" 2>&1 || echo "  ($name failed/killed)"
  echo -n "  $name final: "; grep "eval @" "$R/sweep_${name}.log" | tail -1 || echo "(no eval)"
}
run recall_lora --recall_frac 1.0
run gated_lora  --recall_frac 0.5
echo "================ LORA SWEEP SUMMARY ================"
echo "no-LoRA refs: recall mem=0.498(chance) | gated mem=0.586 | recall oracle=0.963 | gated oracle=0.803"
for n in recall_lora gated_lora; do echo -n "$n : "; grep "eval @" "$R/sweep_${n}.log" | tail -1 || echo "(none)"; done
echo "LORA SWEEP DONE"
