#!/bin/bash
# Does increasing the number of latent tokens (k) raise capacity?
# Fixed hard fact (24-char code, ~19.5 tokens), decode length 32 (no truncation),
# 1.5B, 2500 steps. exact-match should rise with k; ~perfect once k >= content.
set -u
cd "$(dirname "$0")"
R="/home/ubuntu/multimodal-user-memory/results"
for k in 8 16 32 64; do
  echo "===== k=${k} latent tokens ====="
  python3 multimem.py --mode codec --fact_chars 24 --n_decode 32 --k $k --steps 2500 \
    --batch 32 --lora_rank 16 --out "$R/codec_k${k}_c24.json" > "$R/codec_k${k}_c24.log" 2>&1 \
    || echo "  (k$k FAILED)"
  echo -n "  -> k=${k}: "; python3 -c "import json;print(json.load(open('$R/codec_k${k}_c24.json'))['rows'][0]['best_exact_match'])" 2>/dev/null || echo "(none)"
done
echo "=== LATENT TOKENS (k) vs CAPACITY @ 24-char fact (~19.5 content tokens) ==="
for k in 8 16 32 64; do echo -n "k=${k}: "; python3 -c "import json;print(round(json.load(open('$R/codec_k${k}_c24.json'))['rows'][0]['best_exact_match'],3))" 2>/dev/null || echo "-"; done
echo "K_SWEEP_DONE"
