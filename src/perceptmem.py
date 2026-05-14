"""PerceptMem v0.1 — unified benchmark for cross-condition perceptual memory.

Defines a single evaluation surface across the five sub-modalities tested
in this project:
  V-XC-ID: cross-condition face identity (LFW)
  V-STY:   cross-period painter style (WikiArt, distinctive painters, PCA-Gram)
  A-XR-ID: cross-recording speaker identity (LibriSpeech test-clean+other)
  A-SCN:   acoustic-scene identity (ESC-50)
  A-PARA:  paralinguistic state (RAVDESS, blocked at N>=5)

Standard register/recall API for every task:
  register(modality, label, perceptual_input) -> stores identity in memory
  recall(modality, perceptual_input) -> predicted label

Scorecard per task: retrieval@1, code-match-retrieval, code-match-fraction,
RAG-ceiling, N=5/10/20.

This script computes the FULL scorecard for Path A best-recipe on each
task by running pathA_submodality.py-style pipelines from cached
embeddings, plus the embedding-RAG baseline as the per-task ceiling.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent / "nanochat_mm"))
from engram_module_mm import MODALITY_VISION, MODALITY_AUDIO
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import QwenEngramBolt, evaluate, MODEL_ID, DEVICE
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)

EMB_DIR = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")

# PerceptMem v0.1 task definitions
TASKS = {
    "V-XC-ID": {
        "name": "Cross-condition face identity",
        "embeddings": "arcface_lfw_large.npz",
        "modality": "vision",
        "K": 32,
        "Ns": [5, 10, 20, 40, 60],
    },
    "V-STY": {
        "name": "Cross-period painter style",
        "embeddings": "style_pca_gram.npz",
        "modality": "vision",
        "K": 16,
        "Ns": [5, 8],  # only 15 painters
    },
    "A-XR-ID": {
        "name": "Cross-recording speaker identity",
        "embeddings": "ecapa_libri_large.npz",
        "modality": "audio",
        "K": 64,  # K=64 was the audio winner
        "Ns": [5, 10, 20, 29],
    },
    "A-SCN": {
        "name": "Acoustic-scene identity",
        "embeddings": "ast_esc50.npz",
        "modality": "audio",
        "K": 32,
        "Ns": [5, 10, 20],
    },
    # A-PARA blocked at N>=5 by RAVDESS class count
}


def run_task(task_id, task_cfg, qwen, tok):
    print(f"\n{'=' * 70}\nTask {task_id}: {task_cfg['name']}\n{'=' * 70}")
    emb_path = EMB_DIR / task_cfg["embeddings"]
    if not emb_path.exists():
        print(f"  [skip] {emb_path} not found")
        return None
    d = np.load(emb_path)
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    print(f"  loaded {emb.shape}, {len(set(pid))} identities")

    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    K = task_cfg["K"]
    apply_fn = fit_naive_rq(tr_emb, n_levels=1, k_per=K)
    print(f"  train: {len(tr_emb)}/{len(set(tr_pid))} ids; eval: {len(ev_emb)}/{len(set(ev_pid))} ids; K={K}")

    mid = MODALITY_VISION if task_cfg["modality"] == "vision" else MODALITY_AUDIO
    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                           engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()

    losses = pretrain_generic(bolt, tr_emb, tr_pid, apply_fn, mid, tok,
                              n_steps=400, lr=3e-4, batch=4, T=64,
                              frac_perceptual=0.15)
    print(f"  pretrain final loss: {float(np.mean(losses[-30:])):.4f}")

    results = {}
    nq = 5
    n_max = len(set(ev_pid))
    for N in task_cfg["Ns"]:
        if N > n_max:
            print(f"  N={N} exceeds available eval ids ({n_max}); skipping")
            continue
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate(bolt, apply_fn, ev_emb, ev_pid, mid, tok,
                      N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}")
        results[N] = {"rag": rag, **r}

    bolt.remove_hook()
    return {"task_id": task_id, "name": task_cfg["name"], "modality": task_cfg["modality"],
             "K": K, "n_eval_ids": int(n_max), "results": results}


def main():
    print("=" * 70)
    print("PerceptMem v0.1 — cross-condition perceptual memory benchmark")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct (shared across all tasks) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    all_results = {}
    for task_id, cfg in TASKS.items():
        try:
            res = run_task(task_id, cfg, qwen, tok)
            if res is not None:
                all_results[task_id] = res
        except Exception as e:
            print(f"  ERROR on {task_id}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    out = Path("/home/ubuntu/multimodal-user-memory/results/perceptmem_v0_1.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Headline scorecard
    print("\n" + "=" * 100)
    print("PerceptMem v0.1 — Path A scorecard")
    print("=" * 100)
    print(f"{'Task':>10} | {'N':>3} | {'eval-ids':>8} | {'RAG':>5} | {'retr@1':>7} | {'code-match':>10} | {'match-frac':>10}")
    print("-" * 100)
    for task_id, res in all_results.items():
        for N, r in res["results"].items():
            print(f"{task_id:>10} | {N:>3} | {res['n_eval_ids']:>8} | "
                  f"{r['rag']:>5.3f} | {r['retrieval_at_1']:>7.3f} | "
                  f"{r['code_match_retr']:>10.3f} | {r['fraction_code_match']:>10.3f}")


if __name__ == "__main__":
    sys.exit(main())
