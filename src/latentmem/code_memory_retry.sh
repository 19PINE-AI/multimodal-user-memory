#!/bin/bash
# Persistent gated retry for the code_memory fix. Waits for a GPU window, verifies
# M=1 (3000 steps, rank 32). OOM (contention) -> retry; clean success -> per-M sweep;
# clean failure -> structural bug, stop.
set -u
cd "$(dirname "$0")"; R="/home/ubuntu/multimodal-user-memory/results"
a=0
while true; do
  a=$((a+1))
  python3 gpu_wait.py 30 || { sleep 300; continue; }
  echo "[$(date +%H:%M) attempt $a] verify code_memory M=1 (3000 steps, rank 32)"
  python3 code_memory.py --k 16 --code_chars 6 --m_eval 1 --m_max 1 --steps 3000 \
    --lora_rank 32 --batch 8 --out /tmp/cm_retry.json > "$R/codemem_retry_a${a}.log" 2>&1
  rc=$?
  if grep -qi "out of memory\|outofmemory" "$R/codemem_retry_a${a}.log"; then
    echo "[attempt $a] OOM (contention) -> retry in 180s"; sleep 180; continue; fi
  if [ $rc -ne 0 ]; then echo "[attempt $a] crashed (non-OOM):"; tail -4 "$R/codemem_retry_a${a}.log"; break; fi
  ex=$(python3 -c "import json;print(json.load(open('/tmp/cm_retry.json'))['rows'][0]['exact'])" 2>/dev/null || echo 0)
  echo "[attempt $a] M=1 exact-match = $ex"
  if python3 -c "import sys;sys.exit(0 if float('$ex')>0.3 else 1)"; then
    echo "[attempt $a] FIX WORKS -> per-M sweep"
    for Mx in 1 2 4 8 16; do
      python3 code_memory.py --k 16 --code_chars 6 --m_eval $Mx --m_max $Mx --steps 3000 \
        --lora_rank 32 --batch 8 --out "$R/codemem_perm_M${Mx}.json" > "$R/codemem_perm_M${Mx}.log" 2>&1 || echo "(M$Mx fail)"
      echo -n "  M=$Mx -> "; python3 -c "import json;print(round(json.load(open('$R/codemem_perm_M${Mx}.json'))['rows'][0]['exact'],3))" 2>/dev/null || echo "(none)"
    done
    echo "MULTICODE_SWEEP_OK"; break
  else
    echo "[attempt $a] M=1 still fails after 3000 steps -> structural bug, stopping"; break
  fi
done
echo "GATED_RETRY_FINISHED"
