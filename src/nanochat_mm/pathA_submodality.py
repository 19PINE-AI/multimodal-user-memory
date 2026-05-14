"""Generic Path A runner for any sub-modality.

Given a perceptual embedding npz (emb, pid), runs the full pipeline:
  1. Train/eval split by identity (50/50)
  2. Fit naive k-means codebook on TRAIN embeddings
  3. Construct QwenEngramBolt
  4. Generic-NTP pretraining
  5. Held-out surgical insertion + retrieval at N=5/10/20

Reports the same metrics as pathA_generic_pretrain.py so we can extend
the cross-modality comparison table from face+speaker to face+speaker+
scene+paralinguistic+style.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_VISION, MODALITY_AUDIO
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import QwenEngramBolt, evaluate, MODEL_ID, DEVICE
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="Path to npz with (emb, pid)")
    ap.add_argument("--name", required=True, help="Sub-modality name")
    ap.add_argument("--modality", choices=["vision", "audio"], required=True,
                    help="Which modality slot to use (vision/audio); not the sub-modality itself")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--Ns", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--nq", type=int, default=5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    mid = MODALITY_VISION if args.modality == "vision" else MODALITY_AUDIO

    print(f"\n{'=' * 70}")
    print(f"Path A sub-modality: {args.name} (slot={args.modality}, K={args.K})")
    print('=' * 70)

    d = np.load(args.emb)
    emb = d["emb"]; pid = d["pid"]
    print(f"  loaded {emb.shape}, {len(set(pid))} identities")

    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    apply_fn = fit_naive_rq(tr_emb, n_levels=1, k_per=args.K)
    print(f"  train: {len(tr_emb)}/{len(set(tr_pid))} ids; eval: {len(ev_emb)}/{len(set(ev_pid))} ids")

    print(f"\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    bolt = QwenEngramBolt(qwen, tok, V_vis=args.K, V_aud=args.K,
                           engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()

    print(f"\n[pretrain] generic-NTP, {args.steps} steps ...")
    losses = pretrain_generic(bolt, tr_emb, tr_pid, apply_fn, mid, tok,
                              n_steps=args.steps, lr=args.lr, batch=4, T=64,
                              frac_perceptual=0.15)
    print(f"  pretrain final loss: {float(np.mean(losses[-30:])):.4f}")

    print("\n[eval] held-out surgical-insertion retrieval ...")
    results = {}
    n_max = len(set(ev_pid))
    for N in args.Ns:
        if N > n_max:
            print(f"  N={N} exceeds available eval ids ({n_max}); skipping")
            continue
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=args.nq)
        r = evaluate(bolt, apply_fn, ev_emb, ev_pid, mid, tok,
                      N_subset=N, n_queries_per_id=args.nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}  insert-loss={r['avg_insert_loss']:.3f}")
        results[N] = {"rag": rag, **r}

    out = Path("/home/ubuntu/multimodal-user-memory/results") / f"pathA_{args.name}.json"
    with open(out, "w") as f:
        json.dump({"name": args.name, "modality": args.modality, "K": args.K,
                    "n_train_ids": int(len(set(tr_pid))),
                    "n_eval_ids": int(len(set(ev_pid))),
                    "pretrain_loss_final": float(np.mean(losses[-30:])),
                    "results": results}, f, indent=2, default=str)
    print(f"\n[done] saved {out}")


if __name__ == "__main__":
    sys.exit(main())
