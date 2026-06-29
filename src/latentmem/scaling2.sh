#!/bin/bash
# Larger models + longer training + clean length curve (decode length 32, no
# truncation). Answers: does latent capacity scale with model size / training?
set -u
cd "$(dirname "$0")"
R="/home/ubuntu/multimodal-user-memory/results"
declare -A M=( [1.5B]="Qwen/Qwen2.5-1.5B-Instruct" [3B]="Qwen/Qwen2.5-3B-Instruct" [7B]="Qwen/Qwen2.5-7B-Instruct" )
codec () { # model_size fact_chars steps tag
  python3 multimem.py --mode codec --model_id "${M[$1]}" --fact_chars $2 --n_decode 32 \
    --k 16 --steps $3 --batch 32 --lora_rank 16 --out "$R/codec_$4.json" > "$R/codec_$4.log" 2>&1 \
    || echo "  ($4 FAILED)"
  echo -n "  $4: "; python3 -c "import json;print(round(json.load(open('$R/codec_$4.json'))['rows'][0]['best_exact_match'],3))" 2>/dev/null || echo "(none)"
}
echo "### clean length curve (1.5B, k=16, decode 32, 2500 steps)"
for nch in 8 16 24 32; do echo "== 1.5B ${nch}ch =="; codec 1.5B $nch 2500 "len_1.5B_c${nch}"; done
echo "### model size at 24-char (k=16, decode 32, 2500 steps)"
for sz in 3B 7B; do echo "== $sz 24ch =="; codec $sz 24 2500 "len_${sz}_c24"; done
echo "### longer training (1.5B, 24-char, 6000 steps) -- undertraining vs capacity"
codec 1.5B 24 6000 "long_1.5B_c24"
echo "=========== SCALING2 SUMMARY ==========="
echo "length curve 1.5B:"; for n in 8 16 24 32; do echo -n " ${n}ch="; python3 -c "import json;print(round(json.load(open('$R/codec_len_1.5B_c${n}.json'))['rows'][0]['best_exact_match'],3))" 2>/dev/null||echo -n -; done; echo
echo "24-char by size: 1.5B=$(python3 -c "import json;print(round(json.load(open('$R/codec_len_1.5B_c24.json'))['rows'][0]['best_exact_match'],3))" 2>/dev/null) 3B=$(python3 -c "import json;print(round(json.load(open('$R/codec_len_3B_c24.json'))['rows'][0]['best_exact_match'],3))" 2>/dev/null) 7B=$(python3 -c "import json;print(round(json.load(open('$R/codec_len_7B_c24.json'))['rows'][0]['best_exact_match'],3))" 2>/dev/null)"
echo "1.5B 24ch 6000-step=$(python3 -c "import json;print(round(json.load(open('$R/codec_long_1.5B_c24.json'))['rows'][0]['best_exact_match'],3))" 2>/dev/null)"
echo "SCALING2_DONE"
