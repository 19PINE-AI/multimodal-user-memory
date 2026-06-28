#!/bin/bash
# Latent fact-capacity curve: can a latent hold an exact code of N chars?
# One process per length (clean memory). exact-match should fall as entropy rises.
set -u
cd "$(dirname "$0")"
R="/home/ubuntu/multimodal-user-memory/results"
for nch in 2 4 8 16; do
  echo "================ ${nch}-char facts ================"
  python3 multimem.py --mode codec --fact_chars $nch --steps 1500 --batch 32 \
      --lora_rank 16 --out "$R/multimem_codec_c${nch}.json" > "$R/codec_c${nch}.log" 2>&1 \
      || echo "  (c${nch} failed)"
  echo -n "  c${nch}: "; grep "exact_match" "$R/codec_c${nch}.log" | tail -1
  tail -2 "$R/codec_c${nch}.log" | grep "exact=" | tail -1
done
echo "CODEC SWEEP DONE"
