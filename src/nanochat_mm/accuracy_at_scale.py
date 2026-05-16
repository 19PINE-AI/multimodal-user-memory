"""Accuracy vs N — the missing axis from the latency benchmark.

Sess-14b showed Path A is 32× faster than RAG-with-LM-consumption at
N=10,000. The natural next question: at those scales, what's the
retrieval ACCURACY?

This script computes both Path A retr@1 AND RAG retr@1 at the same N
values, using the combined LFW-XXL + AgeDB face data (1401 IDs available).

Two RAG variants are compared:
  - RAG-cosine-only: pick top-1 via cosine NN, return its registered
    marker directly (no LM forward). This is the realistic deployment.
  - RAG-with-LM: cosine NN + inject top-1 marker into LM context (the
    "smart" version of the sess-14b latency comparison).

The latter has the same retrieval accuracy as RAG-cosine-only (since
the LM only sees the chosen marker). The latency story uses the
inject-all variant; the accuracy story uses the inject-top-1 variant.

For Path A we use the K=64 v2 codebook (adapter+largepool) — saturates
fast as N grows past K, which is the structural ceiling.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_VISION
from v2_retrieval import split_by_identity
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import (
    QwenEngramBolt, evaluate, build_fixed_context, get_touched_rows, MODEL_ID, DEVICE,
)
from id_codebook_v2 import load_pipeline_apply

torch.manual_seed(42); np.random.seed(42)


def rag_cosine_only(ev_emb, ev_pid, N_subset, n_queries_per_id=3):
    """Cosine-NN retrieval — RAG without any LM cost. Returns retr@1 accuracy.

    This is the realistic deployment of RAG: store N embeddings, at query
    cosine-NN to find top-1, return its marker. The LM never sees the
    full candidate list — it just gets told "you're looking at identity
    37" (i.e., the chosen marker).
    """
    by_id = defaultdict(list)
    for i, p in enumerate(ev_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None: ids_sorted = ids_sorted[:N_subset]
    rng = np.random.default_rng(99)
    reg_idx, reg_lab = [], []
    q_idx, q_lab = [], []
    for pid_v in ids_sorted:
        idxs = list(by_id[pid_v]); rng.shuffle(idxs)
        reg_idx.append(idxs[0]); reg_lab.append(pid_v)
        for qi in idxs[1:1 + n_queries_per_id]:
            q_idx.append(qi); q_lab.append(pid_v)
    if not q_idx: return 0.0
    R = ev_emb[reg_idx].astype(np.float32)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    Q = ev_emb[q_idx].astype(np.float32)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    sims = Q @ R.T
    pred = sims.argmax(axis=1)
    return sum(1 for k in range(len(q_lab)) if reg_lab[pred[k]] == q_lab[k]) / len(q_lab)


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    Ns = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3
                            else "20,100,300,500,700").split(",")]

    print("=" * 75)
    print(f"Accuracy at scale — K={K}  seed={seed}  Ns={Ns}")
    print("=" * 75)

    torch.manual_seed(seed); np.random.seed(seed)
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    d = np.load(EMB / "arcface_face_combined.npz")
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    n_eval_ids = len(set(ev_pid.tolist()))
    Ns = [N for N in Ns if N <= n_eval_ids]
    print(f"  data: {n_eval_ids} eval IDs, {len(ev_emb)} eval samples")

    # === Phase 1: RAG cosine-only (fast, no LM) ===
    print("\n[RAG cosine-only — the realistic deployment]")
    rag_results = {}
    for N in Ns:
        t0 = time.time()
        rag = rag_cosine_only(ev_emb, ev_pid, N_subset=N, n_queries_per_id=3)
        rag_results[N] = rag
        print(f"  N={N:>4}  RAG retr@1 = {rag:.3f}  ({time.time()-t0:.1f}s)")

    # === Phase 2: Path A end-to-end ===
    print("\n[Path A — bolt-on with v2 codebook]")
    cb_path = Path(f"/home/ubuntu/multimodal-user-memory/runs/codebooks/"
                    f"id_v2_codebook_v-xc-id-face_K{K}.pt")
    if not cb_path.exists():
        # Fall back to V-XC-ID-XL K=64 codebook trained on LFW-XXL
        cb_path = Path("/home/ubuntu/multimodal-user-memory/runs/codebooks/"
                       "id_v2_codebook_v-xc-id_K64.pt")
    print(f"  codebook: {cb_path.name}")
    apply_fn = load_pipeline_apply(cb_path)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()
    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                            engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    print("[pretrain] generic-NTP 400 steps")
    pretrain_generic(bolt, tr_emb, tr_pid, apply_fn, MODALITY_VISION, tok,
                     n_steps=400, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    path_a_results = {}
    print(f"\n[Path A eval — N values up to {n_eval_ids}]")
    print(f"{'N':>5} | {'RAG retr@1':>10} | {'Path A retr@1':>13} | "
          f"{'code-match':>11} | {'frac-code':>10} | {'elapsed':>8}")
    print("-" * 75)
    for N in Ns:
        t0 = time.time()
        r = evaluate(bolt, apply_fn, ev_emb, ev_pid, MODALITY_VISION, tok,
                       N_subset=N, n_queries_per_id=3, max_steps=60, lr=1.0, T=24)
        elapsed = time.time() - t0
        path_a_results[N] = {"retr@1": r["retrieval_at_1"],
                              "code_match": r["code_match_retr"],
                              "frac_code_match": r["fraction_code_match"]}
        rag = rag_results[N]
        print(f"{N:>5} | {rag:>10.3f} | {r['retrieval_at_1']:>13.3f} | "
              f"{r['code_match_retr']:>11.3f} | {r['fraction_code_match']:>10.3f} | "
              f"{elapsed:>7.0f}s")

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/accuracy_at_scale_K{K}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump({"K": K, "seed": seed, "Ns": Ns,
                    "rag_cosine_only": {str(N): v for N, v in rag_results.items()},
                    "path_a": {str(N): v for N, v in path_a_results.items()}},
                   f, indent=2)
    print(f"\n[done] {out}")

    print("\n" + "=" * 75)
    print(f"HEADLINE — accuracy at scale (K={K} codebook)")
    print("=" * 75)
    print(f"{'N':>5} | {'RAG':>6} | {'Path A':>7} | {'Path A / RAG':>13}")
    print("-" * 50)
    for N in Ns:
        rag = rag_results[N]; pa = path_a_results[N]["retr@1"]
        ratio = pa / rag if rag > 0 else float("nan")
        print(f"{N:>5} | {rag:>6.3f} | {pa:>7.3f} | {ratio:>13.3f}")


if __name__ == "__main__":
    sys.exit(main())
