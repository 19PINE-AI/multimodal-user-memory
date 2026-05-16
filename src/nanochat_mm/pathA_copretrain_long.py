"""Long-form Path A + STE co-pretraining (DeepSeek-Engram-style scale).

Sessions 11/12 honest verdict: the bolt-on Engram mechanism (code-match
retrieval) is already 0.5–1.0 across most cells, but the codebook miss
rate gates retr@1. Top-K addressing (#23) showed the codebook miss rate
CAN be reduced, but the same intervention saturates the gate at higher N.

This script runs the DeepSeek-Engram canonical recipe — codebook +
Engram + perc_emb co-trained jointly with the LM (LM frozen, others
trainable) — at extended pretrain length (5000-10000 steps vs the
current 400–600 step baseline). The expectation is that the longer
co-training settles the codebook into LM-useful positions, lifting
the mechanism term and (indirectly) helping codebook-miss too.

Re-uses the STE machinery in pathA_ste.py (QwenEngramBoltSTE + the
quantiser-in-the-loop training path). Difference from pathA_ste.main():
  - Configurable n_steps (default 5000 vs the existing 600)
  - One modality at a time (so we can budget per-modality)
  - Records intermediate eval at multiple checkpoints

Usage:
  python3 pathA_copretrain_long.py <mode> <K> [n_steps]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_AUDIO, MODALITY_VISION
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from pathA_ste import (
    QwenEngramBoltSTE, pretrain_with_ste, evaluate_ste, MODEL_ID, DEVICE,
)
from id_codebook_v2 import MODE_PATHS

torch.manual_seed(42); np.random.seed(42)

MODE_TO_MODALITY = {
    "a-xr-id": MODALITY_AUDIO,
    "a-scn":   MODALITY_AUDIO,
    "a-para":  MODALITY_AUDIO,
    "v-xc-id": MODALITY_VISION,
    "v-sty":   MODALITY_VISION,
    "v-sty-clip": MODALITY_VISION,
    "v-sty-xxl": MODALITY_VISION,
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "a-para"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    print("=" * 70)
    print(f"Long-form Path A + STE co-pretrain — mode={mode}  K={K}  "
          f"n_steps={n_steps}  seed={seed}")
    print("=" * 70)

    torch.manual_seed(seed); np.random.seed(seed)
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary, _ = MODE_PATHS[mode]
    d = np.load(EMB / primary)
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    print(f"  train {len(set(tr_pid))} IDs / {len(tr_emb)} samp  D={emb.shape[1]}")

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    modality_id = MODE_TO_MODALITY[mode]
    bolt = QwenEngramBoltSTE(
        qwen, tok,
        vis_emb_dim=emb.shape[1] if modality_id == MODALITY_VISION else 192,
        aud_emb_dim=emb.shape[1] if modality_id == MODALITY_AUDIO else 192,
        V_vis=K if modality_id == MODALITY_VISION else 32,
        V_aud=K if modality_id == MODALITY_AUDIO else 32,
        engram_attach_layer=24,
    ).to(DEVICE)
    # Init the relevant codebook from k-means on the train pool
    if modality_id == MODALITY_VISION:
        bolt.vis_q.init_from_kmeans(tr_emb)
        bolt.vis_q.to(dtype=torch.bfloat16)
    else:
        bolt.aud_q.init_from_kmeans(tr_emb)
        bolt.aud_q.to(dtype=torch.bfloat16)
    bolt.install_hook()
    n_train = sum(p.numel() for p in bolt.parameters() if p.requires_grad)
    print(f"  bolt built; trainable params: {n_train:,}")

    # Long-form co-pretrain
    print(f"\n[co-pretrain] {n_steps} steps (vs prior 400-600 baseline)")
    t0 = time.time()
    losses = pretrain_with_ste(
        bolt, tr_emb, tr_pid, modality_id, tok,
        n_steps=n_steps, lr=3e-4, batch=4, T=64,
        frac_perceptual=0.15, vq_weight=0.1,
    )
    elapsed = time.time() - t0
    print(f"  pretrain elapsed: {elapsed:.0f}s ({elapsed/n_steps*1000:.0f}ms/step)")

    print("\n[eval]")
    Ns = [5, 10, 20]; nq = 5
    results = {}
    for N in Ns:
        if N > len(set(ev_pid)): continue
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate_ste(bolt, ev_emb, ev_pid, modality_id, tok,
                          N_subset=N, n_queries_per_id=nq,
                          max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} on {100*r['fraction_code_match']:.0f}%  "
              f"collisions={r['N_collision_codes']}")
        results[N] = {"rag": rag, **r}

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/"
                f"pathA_copretrain_{mode}_K{K}_steps{n_steps}.json")
    with open(out, "w") as f:
        json.dump({"mode": mode, "K": K, "n_steps": n_steps, "seed": seed,
                    "elapsed_s": elapsed, "results": results},
                   f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Brief comparison to single-seed Path A baselines for this modality
    legacy = {
        "a-para": ("pathA_idcb_a-para_K32.json", "K=32 single-seed baseline"),
        "a-xr-id": ("pathA_idcb_a-xr-id_K32.json", "K=32 single-seed baseline"),
        "a-scn": ("pathA_idcb_a-scn_K32.json", "K=32 single-seed baseline"),
        "v-xc-id": ("pathA_idcb_v-xc-id_K64.json", "K=64 single-seed baseline"),
    }
    if mode in legacy:
        try:
            ref = json.load(open(
                f"/home/ubuntu/multimodal-user-memory/results/{legacy[mode][0]}"))
            print(f"\nComparison — {legacy[mode][1]} vs co-pretrain {n_steps} steps")
            print(f"{'N':>4} | {'baseline':>9} | {'copretrain':>10}")
            print("-" * 36)
            for N in Ns:
                base = ref.get("results", {}).get(str(N), {}).get("retrieval_at_1")
                co   = results.get(N, {}).get("retrieval_at_1")
                if base is None or co is None: continue
                print(f"{N:>4} | {base:>9.3f} | {co:>10.3f}   (Δ {co-base:+.3f})")
        except Exception as e:
            print(f"(comparison skipped: {e})")


if __name__ == "__main__":
    sys.exit(main())
