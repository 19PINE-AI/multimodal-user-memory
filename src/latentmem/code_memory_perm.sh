#!/bin/bash
# Wait for GPU, verify the code_memory fix (M=1 must learn), then per-M sweep:
# store M name->code pairs in k=16 shared tokens, retrieve one. Trains per-M.
set -u
cd "$(dirname "$0")"; R="/home/ubuntu/multimodal-user-memory/results"
python3 gpu_wait.py 24
echo "=== GPU ready; verifying fix (code_memory M=1, 600 steps) ==="
python3 code_memory.py --k 16 --code_chars 6 --m_eval 1 --m_max 1 --steps 600 --batch 8 \
  --out /tmp/cm_verify.json > /tmp/cm_verify.log 2>&1 || echo "(verify crashed)"
ok=$(python3 -c "import json;print(json.load(open('/tmp/cm_verify.json'))['rows'][0]['exact'])" 2>/dev/null || echo 0)
echo "  M=1 exact-match after fix: $ok"
if python3 -c "import sys;sys.exit(0 if float('$ok')>0.1 else 1)"; then
  echo "=== fix works; running per-M sweep (train per M) ==="
  for M in 1 2 4 8 16; do
    python3 code_memory.py --k 16 --code_chars 6 --m_eval $M --m_max $M --steps 2500 --batch 8 \
      --out "$R/codemem_perm_M${M}.json" > "$R/codemem_perm_M${M}.log" 2>&1 || echo "(M$M failed)"
    echo -n "  M=$M codes -> exact-match "; python3 -c "import json;print(round(json.load(open('$R/codemem_perm_M${M}.json'))['rows'][0]['exact'],3))" 2>/dev/null || echo "(none)"
  done
  echo "=== MULTI-CODE RETRIEVAL (k=16, 6-char codes), exact-match by M ==="
  for M in 1 2 4 8 16; do echo -n "M=$M: "; python3 -c "import json;print(round(json.load(open('$R/codemem_perm_M${M}.json'))['rows'][0]['exact'],3))" 2>/dev/null||echo -; done
else
  echo "=== FIX DID NOT WORK (M=1 still ~0); needs more debugging ==="; tail -6 /tmp/cm_verify.log
fi
echo CODEMEM_PERM_DONE
