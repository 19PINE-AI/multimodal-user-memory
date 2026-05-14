"""STE codebook + K=64 + generic-NTP on the LARGE audio data.

Combines the three best levers from prior experiments:
  - K=64 (decisive audio win at all N)
  - STE-trained codebook (joint-trains codebook with Engram via NTP)
  - generic-NTP pretraining (the right Engram pretraining objective)
  on the larger 58-speaker audio data.

If the combination pushes audio overall retrieval past v1 chained's
0.60 RAG-cheated baseline at N=20, Path A becomes the dominant
parametric recipe for audio.

We also test the same combination on large vision data, but the prior
K=32 K=64 trade-off (small N favours K=32) means we don't expect a
big vision lift at small N.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_VISION, MODALITY_AUDIO
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import MODEL_ID, DEVICE
from pathA_ste import QwenEngramBoltSTE, pretrain_with_ste, evaluate_ste

torch.manual_seed(42); np.random.seed(42)
LARGE = "/home/ubuntu/multimodal-user-memory/runs/embeddings"


def main():
    print("=" * 70)
    print("Path A + STE + K=64 + generic-NTP on LARGE data")
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

    print(f"  K={K}; train aud/vis ids: {len(set(aud_tr_pid))}/{len(set(vis_tr_pid))}")
    print(f"  eval aud/vis ids: {len(set(aud_ev_pid))}/{len(set(vis_ev_pid))}")

    bolt = QwenEngramBoltSTE(qwen, tok,
                              vis_emb_dim=vis_tr_emb.shape[1],
                              aud_emb_dim=aud_tr_emb.shape[1],
                              V_vis=K, V_aud=K,
                              engram_attach_layer=24).to(DEVICE)
    bolt.vis_q.init_from_kmeans(vis_tr_emb)
    bolt.aud_q.init_from_kmeans(aud_tr_emb)
    bolt.vis_q.to(dtype=torch.bfloat16)
    bolt.aud_q.to(dtype=torch.bfloat16)
    bolt.install_hook()

    print("\n[pretrain STE+K=64] vision ...")
    vis_losses = pretrain_with_ste(bolt, vis_tr_emb, vis_tr_pid, MODALITY_VISION, tok,
                                     n_steps=800, lr=3e-4, batch=4, T=64,
                                     frac_perceptual=0.15)
    print("\n[pretrain STE+K=64] audio ...")
    aud_losses = pretrain_with_ste(bolt, aud_tr_emb, aud_tr_pid, MODALITY_AUDIO, tok,
                                     n_steps=800, lr=3e-4, batch=4, T=64,
                                     frac_perceptual=0.15)

    print("\n" + "=" * 70)
    print("Held-out retrieval after STE+K=64 pretraining (LARGE data)")
    print("=" * 70)
    Ns = [5, 10, 20, 40, 60]; nq = 5
    results = {"K": K}
    for mid, name, emb, pids in [
        (MODALITY_AUDIO,  "audio",  aud_ev_emb, aud_ev_pid),
        (MODALITY_VISION, "vision", vis_ev_emb, vis_ev_pid),
    ]:
        print(f"\n[{name}]")
        n_max = len(set(pids))
        out_eval = {}
        for N in Ns:
            if N > n_max:
                print(f"  N={N} exceeds max eval ids ({n_max}); skipping")
                continue
            rag = embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq)
            r = evaluate_ste(bolt, emb, pids, mid, tok,
                              N_subset=N, n_queries_per_id=nq,
                              max_steps=80, lr=1.0, T=24)
            print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
                  f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
                  f"collisions={r['N_collision_codes']}  insert-loss={r['avg_insert_loss']:.3f}")
            out_eval[N] = {"rag": rag, **r}
        results[name] = out_eval

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_ste_k64.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Compare to K=32 generic-NTP large and K=64 generic-NTP large
    k32 = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_scaling.json"))
    k64 = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_scaling_k64.json"))
    print("\n" + "=" * 100)
    print("HEADLINE — audio retrieval@1 (large data, parametric variants)")
    print("=" * 100)
    print(f"{'N':>4} | {'RAG':>6} | {'K=32 generic-NTP':>17} | {'K=64 generic-NTP':>17} | {'K=64 +STE':>11}")
    print("-" * 100)
    for N in Ns:
        a32 = k32["audio"].get(str(N), {})
        a64 = k64["audio"].get(str(N), {})
        ste = results["audio"].get(N, {})
        if not ste: continue
        print(f"{N:>4} | {ste.get('rag', 1.0):>6.3f} | "
              f"{a32.get('retrieval_at_1', float('nan')):>17.3f} | "
              f"{a64.get('retrieval_at_1', float('nan')):>17.3f} | "
              f"{ste.get('retrieval_at_1', float('nan')):>11.3f}")
    print()
    print(f"{'N':>4} | code-match | K=32 | K=64 | K=64+STE")
    print("-" * 40)
    for N in Ns:
        a32 = k32["audio"].get(str(N), {})
        a64 = k64["audio"].get(str(N), {})
        ste = results["audio"].get(N, {})
        if not ste: continue
        print(f"{N:>4} | code-match | "
              f"{a32.get('code_match_retr', float('nan')):>4.2f} | "
              f"{a64.get('code_match_retr', float('nan')):>4.2f} | "
              f"{ste.get('code_match_retr', float('nan')):>8.2f}")


if __name__ == "__main__":
    sys.exit(main())
