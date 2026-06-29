#!/bin/bash
# Does more latent budget k rescue multi-code retrieval, or is associative
# retrieval of exact content fundamentally brittle? Sweep k at M=2 and M=4.
set -u
cd "$(dirname "$0")"; R="/home/ubuntu/multimodal-user-memory/results"
cell() {  # M k
  for try in 1 2 3 4; do
    python3 gpu_wait.py 28
    python3 code_memory.py --k $2 --code_chars 6 --m_eval $1 --m_max $1 --steps 3000 \
      --lora_rank 32 --batch 8 --out "$R/codemem_Mk_M$1_k$2.json" > "$R/codemem_Mk_M$1_k$2.log" 2>&1
    if grep -qi "out of memory\|outofmemory" "$R/codemem_Mk_M$1_k$2.log"; then
      echo "  M=$1 k=$2 OOM (try $try) -> wait"; sleep 150; continue; fi
    break
  done
  echo -n "  M=$1 k=$2 -> "; python3 -c "import json;print(round(json.load(open('$R/codemem_Mk_M$1_k$2.json'))['rows'][0]['exact'],3))" 2>/dev/null || echo "(none)"
}
for M in 2 4; do for k in 16 32 64 128; do echo "=== M=$M k=$k ==="; cell $M $k; done; done
echo "=== MULTI-CODE k-SWEEP: exact-match by (M codes) x (k tokens) ==="
printf "%6s | %6s %6s %6s %6s\n" "M\\k" 16 32 64 128
for M in 2 4; do
  printf "%6s |" "$M"
  for k in 16 32 64 128; do
    v=$(python3 -c "import json;print('%.3f'%json.load(open('$R/codemem_Mk_M${M}_k${k}.json'))['rows'][0]['exact'])" 2>/dev/null || echo "  -  ")
    printf " %6s" "$v"; done; printf "\n"
done
echo "CODEMEM_KSWEEP_DONE"
