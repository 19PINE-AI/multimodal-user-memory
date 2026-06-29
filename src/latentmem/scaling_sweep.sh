#!/bin/bash
# Does latent fact-capacity scale with model size? Codec exact-match at
# lengths 8/16/24 for Qwen2.5 1.5B/3B/7B, fair 3000 steps each.
set -u
cd "$(dirname "$0")"
R="/home/ubuntu/multimodal-user-memory/results"
declare -A M=( [1.5B]="Qwen/Qwen2.5-1.5B-Instruct" [3B]="Qwen/Qwen2.5-3B-Instruct" [7B]="Qwen/Qwen2.5-7B-Instruct" )
for sz in 1.5B 3B 7B; do
  for nch in 8 16 24; do
    echo "===== $sz / ${nch}-char ====="
    python3 multimem.py --mode codec --model_id "${M[$sz]}" --fact_chars $nch --steps 3000 \
      --batch 32 --lora_rank 16 --out "$R/codec_scale_${sz}_c${nch}.json" \
      > "$R/codec_scale_${sz}_c${nch}.log" 2>&1 || echo "  ($sz c$nch FAILED)"
    echo -n "  -> $sz ${nch}ch best_exact: "
    python3 -c "import json;print(json.load(open('$R/codec_scale_${sz}_c${nch}.json'))['rows'][0]['best_exact_match'])" 2>/dev/null || echo "(none)"
  done
done
echo "=========== CAPACITY x MODEL SIZE (exact-match) ==========="
printf "%6s | %7s %7s %7s\n" "model" "8-char" "16-char" "24-char"
for sz in 1.5B 3B 7B; do
  printf "%6s |" "$sz"
  for nch in 8 16 24; do
    v=$(python3 -c "import json;print('%.3f'%json.load(open('$R/codec_scale_${sz}_c${nch}.json'))['rows'][0]['best_exact_match'])" 2>/dev/null || echo "  -  ")
    printf " %7s" "$v"
  done; printf "\n"
done
echo "SCALING_SWEEP_DONE"
