"""Path A continual pretraining at K=1024 with embedding-space augmentation.

Per the session 16 plan: extended training compute (100k steps) + larger
training pool (LFW min_faces=2 + AgeDB combined, ~2000 IDs) + augmentation
in embedding space (per-step Gaussian noise simulating cross-condition
variation) should drive the codebook centroids into substantially better
cross-condition positions.

Embedding-space augmentation specifics:
  For each perceptual sample drawn at training time, add Gaussian noise
  ~ N(0, sigma) with sigma scaled to the natural intra-identity std of
  L2-normalised ArcFace embeddings (~0.02 in 512-d). The codebook +
  Engram must produce the SAME code+marker output despite the noise →
  learned invariance.
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
from engram_module_mm import MODALITY_VISION, MODALITY_TEXT
from v2_retrieval import split_by_identity
from pathA_ste import (
    QwenEngramBoltSTE, build_pretrain_batch_continuous, evaluate_ste,
    MODEL_ID, DEVICE,
)
from accuracy_at_scale import rag_cosine_only

torch.manual_seed(42); np.random.seed(42)


def pretrain_with_ste_aug(bolt, train_emb, train_pid, modality_id, tok,
                            n_steps, lr=3e-4, batch=4, T=64,
                            frac_perceptual=0.20, vq_weight=0.1,
                            aug_sigma=0.02, print_every=2000):
    """STE co-pretraining with embedding-space Gaussian augmentation."""
    q_mod = bolt.vis_q if modality_id == MODALITY_VISION else bolt.aud_q
    proj = bolt.vis_proj if modality_id == MODALITY_VISION else bolt.aud_proj
    resid_emb = bolt.vis_residual_emb if modality_id == MODALITY_VISION else bolt.aud_residual_emb
    eng = bolt.engram.engrams[str(modality_id)]

    params = list(eng.parameters()) + list(q_mod.parameters()) + list(proj.parameters()) + [resid_emb.weight]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    rng = np.random.default_rng(0)
    t0 = time.time()
    log = []
    for step in range(n_steps):
        input_ids, modality_ids, raw_perc = build_pretrain_batch_continuous(
            rng, train_emb, train_pid, modality_id, tok.vocab_size,
            T=T, batch=batch, frac_perceptual=frac_perceptual,
        )
        # Augmentation: add Gaussian noise to perceptual embeddings
        if aug_sigma > 0:
            noise = torch.randn_like(raw_perc) * aug_sigma
            # Re-normalise after noise to stay on the unit sphere
            raw_perc = raw_perc + noise
            raw_perc = raw_perc / raw_perc.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        logits, vq_loss = bolt.pretrain_forward(input_ids, modality_ids, raw_perc, modality_id)
        target_mids = modality_ids[:, 1:]
        text_mask = (target_mids == MODALITY_TEXT)
        if not text_mask.any():
            continue
        pred = logits[:, :-1, :]
        target = input_ids[:, 1:]
        pred_text = pred[text_mask]; target_text = target[text_mask]
        ntp_loss = F.cross_entropy(pred_text, target_text)
        loss = ntp_loss + vq_weight * vq_loss.to(ntp_loss.dtype)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % print_every == 0:
            print(f"    step {step+1:6d}  ntp={ntp_loss.item():.4f}  "
                  f"vq={vq_loss.item():.4f}  ({time.time()-t0:.0f}s)")
            log.append((step+1, float(ntp_loss.item()), float(vq_loss.item())))
    return log


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    aug_sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    Ns_str = sys.argv[5] if len(sys.argv) > 5 else "20,100,300,700"
    Ns = [int(x) for x in Ns_str.split(",")]
    data_path = sys.argv[6] if len(sys.argv) > 6 else \
        "/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_face_xxxl.npz"

    print("=" * 75)
    print(f"Continual pretrain — K={K}  n_steps={n_steps}  aug_sigma={aug_sigma}  "
          f"seed={seed}")
    print(f"  data: {data_path}")
    print("=" * 75)

    torch.manual_seed(seed); np.random.seed(seed)
    d = np.load(data_path)
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    if pid.dtype.kind != "U":
        pid = np.array([str(p) for p in pid])
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    n_train_ids = len(set(tr_pid.tolist())); n_eval_ids = len(set(ev_pid.tolist()))
    Ns = [N for N in Ns if N <= n_eval_ids]
    print(f"  train {n_train_ids} IDs / {len(tr_emb)} samp; eval {n_eval_ids} IDs / {len(ev_emb)} samp")
    print(f"  Ns after clamp: {Ns}")

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

    print(f"\n[continual pretrain] {n_steps} steps with embedding aug (sigma={aug_sigma})")
    t0 = time.time()
    log = pretrain_with_ste_aug(
        bolt, tr_emb, tr_pid, MODALITY_VISION, tok,
        n_steps=n_steps, lr=3e-4, batch=4, T=64,
        frac_perceptual=0.20, vq_weight=0.1, aug_sigma=aug_sigma,
    )
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.0f}s ({elapsed/n_steps*1000:.1f} ms/step)")

    # RAG baseline
    print("\n[RAG cosine-only baseline]")
    rag_at_N = {}
    for N in Ns:
        rag = rag_cosine_only(ev_emb, ev_pid, N_subset=N, n_queries_per_id=3)
        rag_at_N[N] = rag
        print(f"  N={N:>4}  RAG retr@1 = {rag:.3f}")

    print(f"\n[Path A eval — continual-pretrained K={K}, aug={aug_sigma}]")
    print(f"{'N':>5} | {'RAG':>6} | {'Path A':>8} | {'code-match':>11} | "
          f"{'frac-code':>10} | {'elapsed':>8}")
    print("-" * 70)
    results = {}
    for N in Ns:
        t0 = time.time()
        r = evaluate_ste(bolt, ev_emb, ev_pid, MODALITY_VISION, tok,
                          N_subset=N, n_queries_per_id=3,
                          max_steps=60, lr=1.0, T=24)
        dt = time.time() - t0
        rag = rag_at_N[N]
        ratio = r["retrieval_at_1"] / rag if rag > 0 else float("nan")
        print(f"{N:>5} | {rag:>6.3f} | {r['retrieval_at_1']:>8.3f} | "
              f"{r['code_match_retr']:>11.3f} | {r['fraction_code_match']:>10.3f} | "
              f"{dt:>7.0f}s")
        results[N] = {"rag": rag, "retr_at_1": r["retrieval_at_1"],
                       "code_match": r["code_match_retr"],
                       "frac_code_match": r["fraction_code_match"],
                       "ratio_to_rag": ratio, "elapsed_s": dt}

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/"
                f"pathA_continual_K{K}_steps{n_steps}_aug{aug_sigma}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump({"K": K, "n_steps": n_steps, "aug_sigma": aug_sigma, "seed": seed,
                    "pretrain_elapsed_s": elapsed,
                    "n_train_ids": n_train_ids, "n_eval_ids": n_eval_ids,
                    "results": {str(N): v for N, v in results.items()}},
                   f, indent=2, default=str)
    print(f"\n[done] {out}")

    print("\n" + "=" * 70)
    print(f"HEADLINE — continual pretrain K={K}, {n_steps} steps, aug σ={aug_sigma}")
    print("=" * 70)
    print(f"{'N':>5} | {'RAG':>6} | {'Path A':>8} | {'ratio':>6} | {'verdict'}")
    print("-" * 50)
    for N in Ns:
        r = results[N]; rag = r["rag"]; pa = r["retr_at_1"]
        ratio = r["ratio_to_rag"]
        verdict = ("BEATS" if pa > rag else
                   ("near" if ratio > 0.8 else
                    ("competitive" if ratio > 0.5 else
                     ("partial" if ratio > 0.3 else "below"))))
        print(f"{N:>5} | {rag:>6.3f} | {pa:>8.3f} | {ratio:>6.2f} | {verdict}")


if __name__ == "__main__":
    sys.exit(main())
