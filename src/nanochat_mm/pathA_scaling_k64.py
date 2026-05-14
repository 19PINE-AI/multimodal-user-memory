"""Path A scaling curve with K=64 codebook — does richer codebook lift the
ceiling at higher N?

At K=32 vision plateaus at retr@1 ~0.10 from N=20 onwards because the
codebook can't address > 32 distinct identities. With 79 eval IDs in
the large vision set, ~half the codes must be shared. With K=64 we
should see fewer collisions and a higher achievable retrieval rate.

Trade-off: larger K → less intra-cluster cohesion → some cross-condition
pairs that shared a code at K=32 won't share at K=64. We expect a
crossover N where K=64 starts to dominate K=32.
"""
import json
import sys
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
from pathA_two_layer import QwenEngramBoltMultiLayer, evaluate_multi

torch.manual_seed(42); np.random.seed(42)
LARGE = "/home/ubuntu/multimodal-user-memory/runs/embeddings"


def main():
    print("=" * 70)
    print("Path A scaling curve — K=64 codebook")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    aud = np.load(f"{LARGE}/ecapa_libri_large.npz")
    vis = np.load(f"{LARGE}/arcface_lfw_large.npz")
    aud_tr_emb, aud_tr_pid, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr_emb, vis_tr_pid, vis_ev_emb, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])

    K = 64
    aud_apply = fit_naive_rq(aud_tr_emb, n_levels=1, k_per=K)
    vis_apply = fit_naive_rq(vis_tr_emb, n_levels=1, k_per=K)
    print(f"  K={K}; audio train ids: {len(set(aud_tr_pid))} | vision train ids: {len(set(vis_tr_pid))}")

    Ns = [5, 10, 20, 40, 60]; nq = 5

    # AUDIO 1-layer
    print("\n[AUDIO] 1-layer + generic-NTP, K=64")
    bolt_aud = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                                 engram_attach_layer=24).to(DEVICE)
    bolt_aud.install_hook()
    aud_losses = pretrain_generic(bolt_aud, aud_tr_emb, aud_tr_pid, aud_apply,
                                    MODALITY_AUDIO, tok, n_steps=600, lr=3e-4,
                                    batch=4, T=64, frac_perceptual=0.15)
    print(f"  pretrain final loss: {float(np.mean(aud_losses[-30:])):.4f}")
    audio_results = {}
    n_max = len(set(aud_ev_pid))
    for N in Ns:
        if N > n_max: continue
        rag = embedding_rag_ceiling(aud_ev_emb, aud_ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate(bolt_aud, aud_apply, aud_ev_emb, aud_ev_pid, MODALITY_AUDIO, tok,
                      N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}")
        audio_results[N] = {"rag": rag, **r}
    bolt_aud.remove_hook()
    del bolt_aud

    # VISION 2-layer
    print("\n[VISION] 2-layer + generic-NTP, K=64")
    bolt_vis = QwenEngramBoltMultiLayer(qwen, tok, V_vis=K, V_aud=K,
                                           engram_attach_layers=(16, 28)).to(DEVICE)
    bolt_vis.install_hook()
    vis_losses = pretrain_generic(bolt_vis, vis_tr_emb, vis_tr_pid, vis_apply,
                                    MODALITY_VISION, tok, n_steps=600, lr=3e-4,
                                    batch=4, T=64, frac_perceptual=0.15)
    print(f"  pretrain final loss: {float(np.mean(vis_losses[-30:])):.4f}")
    vision_results = {}
    n_max = len(set(vis_ev_pid))
    for N in Ns:
        if N > n_max: continue
        rag = embedding_rag_ceiling(vis_ev_emb, vis_ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate_multi(bolt_vis, vis_apply, vis_ev_emb, vis_ev_pid, MODALITY_VISION, tok,
                            N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}")
        vision_results[N] = {"rag": rag, **r}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_scaling_k64.json")
    with open(out, "w") as f:
        json.dump({"K": K, "audio": audio_results, "vision": vision_results}, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Compare K=32 vs K=64
    k32_path = Path("/home/ubuntu/multimodal-user-memory/results/pathA_scaling.json")
    if k32_path.exists():
        with open(k32_path) as f: k32 = json.load(f)
        print("\n" + "=" * 95)
        print("K=32 vs K=64 head-to-head (large data)")
        print("=" * 95)
        print(f"{'N':>4} | {'k32 audio cm':>12} | {'k64 audio cm':>12} | {'k32 vision cm':>13} | {'k64 vision cm':>13} | "
              f"{'k32 vision retr':>15} | {'k64 vision retr':>15}")
        print("-" * 95)
        for N in Ns:
            a32 = k32["audio"].get(str(N), {}); a64 = audio_results.get(N, {})
            v32 = k32["vision"].get(str(N), {}); v64 = vision_results.get(N, {})
            print(f"{N:>4} | {a32.get('code_match_retr', float('nan')):>12.3f} | "
                  f"{a64.get('code_match_retr', float('nan')):>12.3f} | "
                  f"{v32.get('code_match_retr', float('nan')):>13.3f} | "
                  f"{v64.get('code_match_retr', float('nan')):>13.3f} | "
                  f"{v32.get('retrieval_at_1', float('nan')):>15.3f} | "
                  f"{v64.get('retrieval_at_1', float('nan')):>15.3f}")


if __name__ == "__main__":
    sys.exit(main())
