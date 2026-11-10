#!/bin/bash
# T3 queue with accuracy-based stop for M>=2 (loss rule kept for M=1). LoRA r32.
# Key block seeds 42,43 (c6 M=1/2/3 + role M=2/3), then grid c2/4/8 x M1/2/3 seed 42.
set -u
cd ~/clean-tests/t3; . ../.venv/bin/activate; mkdir -p logs
while pgrep -f "python t3_binding" >/dev/null; do sleep 30; done
ana() { tag=$1; [ -f ../results/T3/$tag.json ] || python t3_binding.py analyze --ckpt ckpt/$tag.pt --full > logs/analyze_$tag.log 2>&1 || echo "ANALYZE FAIL $tag"; echo "done $tag $(date +%T)"; }
run() { tag=$1; cap=$2; shift 2
  if python needs_train.py $tag $cap; then
    if [ -f ckpt/$tag.pt ]; then   # keep the v7-rule run (analyzed) as a lower bound
      [ -f ../results/T3/$tag.json ] || ana $tag
      mv ckpt/$tag.pt ckpt/${tag}_v7stop.pt; mv ../results/T3/$tag.json ../results/T3/${tag}_v7stop.json 2>/dev/null
      mv logs/train_$tag.log logs/train_${tag}_v7stop.log 2>/dev/null; mv logs/analyze_$tag.log logs/analyze_${tag}_v7stop.log 2>/dev/null
      echo "rerun $tag under acc rule (old run kept as ${tag}_v7stop)"
    fi
    python t3_binding.py train --steps $cap --conv --suffix _conv "$@" > logs/train_$tag.log 2>&1 || echo "TRAIN FAIL $tag"
  else echo "skip-train $tag (finished)"; fi; }
cap() { [ $1 -ge 2 ] && echo 40000 || echo 15000; }
for s in 42 43; do
  for m in 1 2 3; do t=c6_m${m}_s${s}_conv; run $t $(cap $m) --chars 6 --mmax $m --seed $s; ana $t; done
  for m in 2 3; do t=c6_m${m}_s${s}_role_conv; run $t $(cap $m) --chars 6 --mmax $m --seed $s --role; ana $t; done
  echo "SEED_BLOCK_DONE $s"
done
echo "KEY_BLOCK_DONE"
for c in 2 4 8; do for m in 1 2 3; do
  t=c${c}_m${m}_s42_conv; run $t $(cap $m) --chars $c --mmax $m --seed 42; ana $t
done; done
echo T3_DRIVER_DONE
