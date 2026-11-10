#!/bin/bash
# Every training + analysis command behind the binding follow-up (paper Appendix "Binding exact
# facts in a latent memory"). Run from the repo root. GPU required (~14 GB per run); runs are
# sequential and each 40k-step run takes on the order of 1-2 h. Outputs: runs/binding_followup/*.pt
# (gitignored) and results/binding_followup/runs/*.json. Then: python3 .../summarize.py
# Seeds: the study ran the 3k arm on 42/43/44 and everything else on 42/43. The seed-42 rank-16
# reference runs (*.r16) came from an earlier harness revision; seed 43 used the commands below.
set -eu
H=src/latentmem/binding_followup/t3_binding.py
CK=${BINDING_CKPT:-runs/binding_followup}
SEEDS_LONG=${SEEDS_LONG:-42 43}
ana() { python "$H" analyze --ckpt "$CK/$1.pt" --full; }

# 1) original protocol: 3,000 steps, cosine to 0, LoRA r32, three seeds
for S in 42 43 44; do for M in 1 2 3; do
  python "$H" train --chars 6 --mmax $M --seed $S --steps 3000 --suffix _paper; ana c6_m${M}_s${S}_paper
done; done

for S in $SEEDS_LONG; do
  # 2) shared latent trained past the plateau (v8 stop rule for M>=2, loss rule for M=1)
  python "$H" train --chars 6 --mmax 1 --seed $S --conv --steps 15000 --suffix _conv; ana c6_m1_s${S}_conv
  for M in 2 3; do
    python "$H" train --chars 6 --mmax $M --seed $S --conv --steps 40000 --suffix _conv; ana c6_m${M}_s${S}_conv
  done
  # 3) hard per-line write mask (needs line boundaries at write time)
  for M in 2 3; do
    python "$H" train --chars 6 --mmax $M --seed $S --conv --steps 40000 --suffix _conv --role; ana c6_m${M}_s${S}_role_conv
  done
  # 4) LoRA rank 16 (current-code equivalent of the earlier-revision reference runs *.r16)
  for M in 1 2 3; do
    python "$H" train --chars 6 --mmax $M --seed $S --conv --steps 15000 --lora_rank 16 --suffix _conv_r16; ana c6_m${M}_s${S}_conv_r16
  done
done
python3 src/latentmem/binding_followup/summarize.py
