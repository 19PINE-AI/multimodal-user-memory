"""Path A scaling curve — how does retrieval degrade as N grows?

Uses the larger embedding sets (158 vision IDs, 58 audio IDs) to test
Path A + generic-NTP pretrain at progressive N: 5, 10, 20, 40, 60.

Two questions:
  1. Does the audio code-match retrieval at 0.89 (small N) hold up?
  2. At what N does Path A fall below v1 chained's 0.60 audio baseline?

We use:
  - Audio: 1-layer attach + generic-NTP (the session-5 audio optimum)
  - Vision: 2-layer attach + generic-NTP (the session-5 vision optimum)
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
    print("Path A scaling curve — larger embedding sets, N up to ~60")
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
    K = 32
    aud_apply = fit_naive_rq(aud_tr_emb, n_levels=1, k_per=K)
    vis_apply = fit_naive_rq(vis_tr_emb, n_levels=1, k_per=K)
    print(f"  audio train: {len(aud_tr_emb)}/{len(set(aud_tr_pid))}ids; eval: {len(aud_ev_emb)}/{len(set(aud_ev_pid))}ids")
    print(f"  vision train: {len(vis_tr_emb)}/{len(set(vis_tr_pid))}ids; eval: {len(vis_ev_emb)}/{len(set(vis_ev_pid))}ids")

    Ns = [5, 10, 20, 40, 60]; nq = 5

    # ---- Audio: 1-layer + generic-NTP ----
    print("\n" + "=" * 70)
    print("[AUDIO] 1-layer attach + generic-NTP pretrain")
    print("=" * 70)
    bolt_aud = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                                 engram_attach_layer=24).to(DEVICE)
    bolt_aud.install_hook()
    print("\nPretraining (audio generic-NTP) ...")
    aud_losses = pretrain_generic(bolt_aud, aud_tr_emb, aud_tr_pid, aud_apply,
                                    MODALITY_AUDIO, tok, n_steps=600, lr=3e-4,
                                    batch=4, T=64, frac_perceptual=0.15)
    print(f"  final loss: {float(np.mean(aud_losses[-30:])):.4f}")

    audio_results = {}
    n_max_aud = len(set(aud_ev_pid))
    for N in Ns:
        if N > n_max_aud:
            print(f"  N={N} exceeds available eval ids ({n_max_aud}); skipping")
            continue
        rag = embedding_rag_ceiling(aud_ev_emb, aud_ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate(bolt_aud, aud_apply, aud_ev_emb, aud_ev_pid, MODALITY_AUDIO, tok,
                      N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}")
        audio_results[N] = {"rag": rag, **r}
    bolt_aud.remove_hook()
    del bolt_aud

    # ---- Vision: 2-layer + generic-NTP ----
    print("\n" + "=" * 70)
    print("[VISION] 2-layer attach + generic-NTP pretrain")
    print("=" * 70)
    bolt_vis = QwenEngramBoltMultiLayer(qwen, tok, V_vis=K, V_aud=K,
                                           engram_attach_layers=(16, 28)).to(DEVICE)
    bolt_vis.install_hook()
    print("\nPretraining (vision generic-NTP) ...")
    vis_losses = pretrain_generic(bolt_vis, vis_tr_emb, vis_tr_pid, vis_apply,
                                    MODALITY_VISION, tok, n_steps=600, lr=3e-4,
                                    batch=4, T=64, frac_perceptual=0.15)
    print(f"  final loss: {float(np.mean(vis_losses[-30:])):.4f}")

    vision_results = {}
    n_max_vis = len(set(vis_ev_pid))
    for N in Ns:
        if N > n_max_vis:
            print(f"  N={N} exceeds available eval ids ({n_max_vis}); skipping")
            continue
        rag = embedding_rag_ceiling(vis_ev_emb, vis_ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate_multi(bolt_vis, vis_apply, vis_ev_emb, vis_ev_pid, MODALITY_VISION, tok,
                            N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}")
        vision_results[N] = {"rag": rag, **r}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_scaling.json")
    with open(out, "w") as f:
        json.dump({"audio": audio_results, "vision": vision_results,
                    "n_train_audio": int(len(set(aud_tr_pid))),
                    "n_eval_audio":  int(len(set(aud_ev_pid))),
                    "n_train_vision": int(len(set(vis_tr_pid))),
                    "n_eval_vision":  int(len(set(vis_ev_pid)))}, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Headline
    print("\n" + "=" * 90)
    print("HEADLINE — Path A scaling curve (per-modality best recipe)")
    print("=" * 90)
    print(f"{'N':>4} | {'audio RAG':>10} | {'audio retr@1':>13} | {'audio code-match':>16} | "
          f"{'vision RAG':>10} | {'vision retr@1':>13} | {'vision code-match':>17}")
    print("-" * 90)
    for N in Ns:
        a = audio_results.get(N, {})
        v = vision_results.get(N, {})
        a_rag = a.get('rag', float('nan'))
        a_ret = a.get('retrieval_at_1', float('nan'))
        a_cm  = a.get('code_match_retr', float('nan'))
        v_rag = v.get('rag', float('nan'))
        v_ret = v.get('retrieval_at_1', float('nan'))
        v_cm  = v.get('code_match_retr', float('nan'))
        print(f"{N:>4} | {a_rag:>10.3f} | {a_ret:>13.3f} | {a_cm:>16.3f} | "
              f"{v_rag:>10.3f} | {v_ret:>13.3f} | {v_cm:>17.3f}")


if __name__ == "__main__":
    sys.exit(main())
