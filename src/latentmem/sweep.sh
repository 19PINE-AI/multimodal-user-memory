#!/bin/bash
# Recipe sweep: isolate each lever at 3k steps (~9 min each) against the
# bridge-informed AdamW baseline (mem=0.586). Each run self-gates on GPU and
# writes its own log + json. Run sequentially; a killed run does not stop the rest.
set -u
cd "$(dirname "$0")"
M="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
R="/home/ubuntu/multimodal-user-memory/results"
COMMON="--model_id $M --n_steps 3000 --k 16 --batch 8 --eval_every 250 --eval_n 512 --seed 42 --need_gb 20"

run () {
  name="$1"; shift
  echo "================ $name ================"
  python3 train.py $COMMON "$@" > "$R/sweep_${name}.log" 2>&1 || echo "  ($name failed/killed)"
  echo -n "  $name final: "; grep "eval @" "$R/sweep_${name}.log" | tail -1 || echo "(no eval)"
}

run muon          --optimizer muon
run recon         --recon_weight 0.5
run muon_recon    --optimizer muon --recon_weight 0.5
run recall_sanity --recall_frac 1.0

echo "================ SWEEP SUMMARY ================"
echo "reference  AdamW+bridge 3k : mem=0.586  (results/latentmem_k16_v2_run.log)"
for n in muon recon muon_recon recall_sanity; do
  echo -n "$n : "; grep "eval @" "$R/sweep_${n}.log" | tail -1 || echo "(none)"
done
echo "SWEEP DONE"
