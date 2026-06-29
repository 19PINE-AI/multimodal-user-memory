#!/bin/bash
# Full 3-architecture exact-fact bench within (8-char) and beyond (16-char) latent capacity.
set -u
cd "$(dirname "$0")"
R="/home/ubuntu/multimodal-user-memory/results"
for nch in 8 16; do
  echo "================ ${nch}-char facts ================"
  python3 multimem.py --mode bench --fact_chars $nch --steps 1500 --batch 32 \
      --lora_rank 16 --seeds 0 1 2 --ns 10 50 100 300 \
      --out "$R/multimem_bench_c${nch}.json" > "$R/bench_c${nch}.log" 2>&1 || echo "  (c${nch} failed)"
  grep -A8 "MULTIMODAL MEMORY BENCHMARK" "$R/bench_c${nch}.log" 2>/dev/null
done
echo "BENCH SWEEP DONE"
