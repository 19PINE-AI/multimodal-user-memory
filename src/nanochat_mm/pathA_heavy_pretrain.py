"""Heavy pretraining of Path A — testing whether parametric memory scales
when given proper pretraining compute (the way pretrained LMs and
DeepSeek-Engram learn millions of facts).

The hypothesis under test: Path A's small-data, 400-step generic-NTP
pretraining isn't a fair test of the parametric primitive. Pretrained
LMs memorize via gradient descent over many exposures; DeepSeek-Engram
co-trains the table+codebook+gate over web-scale text. If Path A's
codebook + Engram + perc_emb are pretrained with comparable compute
density (10k–50k steps, K≥N, dense exposures per identity), its
accuracy at scale may close the gap to RAG.

This script:
  1. Builds a large-K (e.g., 512) STE-trainable codebook on the combined
     LFW-XXL + AgeDB face data (1401 IDs, 5703 cross-condition samples).
  2. Co-pretrains Engram + codebook + perc_emb for many steps with the
     LM frozen (Qwen2.5-3B).
  3. Evaluates Path A retr@1 at N=20, 50, 100, 300, 700 with the heavily
     pretrained codebook + Engram (no further training at eval).
  4. Compares to the naive 400-step baseline and to RAG.

If retr@1 climbs with N (vs the K=64-naive curve that collapsed to 0.09),
the parametric primitive scales given proper pretraining. If not, the
architectural ceiling is real and the paper's claim narrows accordingly.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_VISION
from v2_retrieval import split_by_identity
from pathA_ste import (
    QwenEngramBoltSTE, pretrain_with_ste, evaluate_ste, MODEL_ID, DEVICE,
)
from accuracy_at_scale import rag_cosine_only

torch.manual_seed(42); np.random.seed(42)


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    Ns_str = sys.argv[4] if len(sys.argv) > 4 else "20,100,300"
    Ns = [int(x) for x in Ns_str.split(",")]

    print("=" * 75)
    print(f"Heavy pretrain Path A — K={K}  n_steps={n_steps}  seed={seed}  Ns={Ns}")
    print("=" * 75)

    torch.manual_seed(seed); np.random.seed(seed)
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    d = np.load(EMB / "arcface_face_combined.npz")
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    n_train_ids = len(set(tr_pid.tolist())); n_eval_ids = len(set(ev_pid.tolist()))
    Ns = [N for N in Ns if N <= n_eval_ids]
    print(f"  data: train {n_train_ids} IDs / {len(tr_emb)} samp, "
          f"eval {n_eval_ids} IDs / {len(ev_emb)} samp")

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    bolt = QwenEngramBoltSTE(
        qwen, tok,
        vis_emb_dim=emb.shape[1], aud_emb_dim=192,
        V_vis=K, V_aud=32,
        engram_attach_layer=24,
    ).to(DEVICE)
    print(f"  bolt built; init codebook from k-means K={K}")
    bolt.vis_q.init_from_kmeans(tr_emb)
    bolt.vis_q.to(dtype=torch.bfloat16)
    bolt.install_hook()
    n_train = sum(p.numel() for p in bolt.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train:,}")

    print(f"\n[heavy co-pretrain] {n_steps} steps (STE codebook + Engram + perc_emb)")
    t0 = time.time()
    losses = pretrain_with_ste(
        bolt, tr_emb, tr_pid, MODALITY_VISION, tok,
        n_steps=n_steps, lr=3e-4, batch=4, T=64,
        frac_perceptual=0.20, vq_weight=0.1,
    )
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.0f}s ({elapsed/n_steps*1000:.1f} ms/step)")

    # RAG baseline (cosine-NN only, no LM cost)
    print("\n[RAG cosine-only baseline at the same N values]")
    rag_at_N = {}
    for N in Ns:
        rag = rag_cosine_only(ev_emb, ev_pid, N_subset=N, n_queries_per_id=3)
        rag_at_N[N] = rag
        print(f"  N={N:>4}  RAG retr@1 = {rag:.3f}")

    print(f"\n[Path A eval — heavy-pretrained codebook+Engram, K={K}]")
    print(f"{'N':>5} | {'RAG':>6} | {'Path A':>8} | {'code-match':>11} | "
          f"{'frac-code':>10} | {'elapsed':>8}")
    print("-" * 70)
    results = {}
    for N in Ns:
        t0 = time.time()
        r = evaluate_ste(bolt, ev_emb, ev_pid, MODALITY_VISION, tok,
                          N_subset=N, n_queries_per_id=3,
                          max_steps=60, lr=1.0, T=24)
        elapsed = time.time() - t0
        rag = rag_at_N[N]
        ratio = r["retrieval_at_1"] / rag if rag > 0 else float("nan")
        print(f"{N:>5} | {rag:>6.3f} | {r['retrieval_at_1']:>8.3f} | "
              f"{r['code_match_retr']:>11.3f} | {r['fraction_code_match']:>10.3f} | "
              f"{elapsed:>7.0f}s")
        results[N] = {"rag": rag, **r, "ratio_to_rag": ratio,
                       "elapsed_s": elapsed}

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/"
                f"pathA_heavy_pretrain_K{K}_steps{n_steps}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump({"K": K, "n_steps": n_steps, "seed": seed,
                    "pretrain_elapsed_s": elapsed,
                    "results": {str(N): v for N, v in results.items()}},
                   f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Headline summary
    print("\n" + "=" * 70)
    print(f"HEADLINE — heavy pretrain (K={K}, {n_steps} steps) vs RAG")
    print("=" * 70)
    print(f"{'N':>5} | {'RAG':>6} | {'Path A':>8} | {'ratio':>6} | {'verdict'}")
    print("-" * 50)
    for N in Ns:
        r = results[N]; rag = r["rag"]; pa = r["retr@1"]
        ratio = r["ratio_to_rag"]
        verdict = ("BEATS" if pa > rag else
                   ("near" if ratio > 0.8 else
                    ("competitive" if ratio > 0.5 else "below")))
        print(f"{N:>5} | {rag:>6.3f} | {pa:>8.3f} | {ratio:>6.2f} | {verdict}")


if __name__ == "__main__":
    sys.exit(main())
