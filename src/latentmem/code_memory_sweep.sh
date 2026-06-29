#!/bin/bash
set -u
cd "$(dirname "$0")"; R="/home/ubuntu/multimodal-user-memory/results"
for k in 16 32; do
  echo "===== k=${k} ====="
  python3 code_memory.py --k $k --code_chars 6 --m_eval 1 2 4 8 16 --m_max 16 \
    --steps 2500 --batch 8 --out "$R/codemem_k${k}.json" > "$R/codemem_k${k}.log" 2>&1 || echo "(k$k FAILED)"
  grep -A8 "MULTI-CODE MEMORY" "$R/codemem_k${k}.log" | grep -E "^\s+[0-9]"
done
echo "=== MULTI-CODE RETRIEVAL: exact-match by (k tokens) x (M codes), 6-char codes ==="
for k in 16 32; do echo -n "k=$k: "; python3 -c "import json;d=json.load(open('$R/codemem_k${k}.json'));print(' '.join(f\"M{r['M']}={r['exact']:.2f}\" for r in d['rows']))" 2>/dev/null||echo -; done
echo CODEMEM_DONE
