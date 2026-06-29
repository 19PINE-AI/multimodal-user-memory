#!/bin/bash
# How many latent tokens to remember M faces/voices? recall@1 vs M, per k.
set -u
cd "$(dirname "$0")"
R="/home/ubuntu/multimodal-user-memory/results"
FACE="$R/../runs/embeddings/arcface_lfw_xxxl.npz"
VOICE="$R/../runs/embeddings/ecapa_libri_large.npz"
for k in 4 8 16 32; do
  echo "===== FACE k=${k} ====="
  python3 set_memory.py --k $k --emb_file "$FACE" --m_eval 2 4 8 16 32 64 --m_max 64 \
    --steps 3000 --out "$R/setmem_face_k${k}.json" > "$R/setmem_face_k${k}.log" 2>&1 || echo "(face k$k FAILED)"
  grep -A8 "SET MEMORY" "$R/setmem_face_k${k}.log" | grep -E "^\s+[0-9]"
done
for k in 8 16; do
  echo "===== VOICE k=${k} ====="
  python3 set_memory.py --k $k --emb_file "$VOICE" --m_eval 2 4 8 16 --m_max 16 \
    --steps 3000 --out "$R/setmem_voice_k${k}.json" > "$R/setmem_voice_k${k}.log" 2>&1 || echo "(voice k$k FAILED)"
  grep -A6 "SET MEMORY" "$R/setmem_voice_k${k}.log" | grep -E "^\s+[0-9]"
done
echo "=========== SET-MEMORY CAPACITY (recall@1) ==========="
python3 - <<'PY'
import json, glob, os
R="/home/ubuntu/multimodal-user-memory/results"
for kind in ("face","voice"):
    print(f"\n[{kind}] recall@1 by (k latent tokens) x (M identities):")
    files=sorted(glob.glob(f"{R}/setmem_{kind}_k*.json"), key=lambda f:int(f.split('_k')[-1].split('.')[0]))
    if not files: continue
    Ms=sorted({r['M'] for f in files for r in json.load(open(f))['rows']})
    print("  k\\M " + "".join(f"{m:>7}" for m in Ms))
    for f in files:
        d=json.load(open(f)); rec={r['M']:r['recall'] for r in d['rows']}
        print(f"  {d['k']:>3} " + "".join(f"{rec.get(m,float('nan')):>7.2f}" for m in Ms))
PY
echo "SETMEM_SWEEP_DONE"
