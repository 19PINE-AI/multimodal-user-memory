"""STE codebook on V-AGE (AgeDB cross-age).

V-AGE naive K=64 gave code-match 0.55-0.66 but match-fraction 0.29-0.44
— the codebook miss rate is the binding constraint. STE on audio K=64
lifted code-match to 1.00 by training the codebook jointly with the
LM. This applies the same fix to cross-age faces.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_VISION
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import MODEL_ID, DEVICE
from pathA_ste import QwenEngramBoltSTE, pretrain_with_ste, evaluate_ste

torch.manual_seed(42); np.random.seed(42)


def main():
    print("=" * 70)
    print("Path A + STE codebook + K=64 on V-AGE (AgeDB cross-age)")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_agedb.npz")
    vis_tr_emb, vis_tr_pid, vis_ev_emb, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    K = 64
    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri_large.npz")
    aud_tr_emb, _, _, _ = split_by_identity(aud['emb'], aud['pid'])

    print(f"  V-AGE train: {len(vis_tr_emb)}/{len(set(vis_tr_pid))}ids; eval: {len(vis_ev_emb)}/{len(set(vis_ev_pid))}ids")

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
    print(f"  trainable params: {sum(p.numel() for p in bolt.parameters() if p.requires_grad):,}")

    print("\n[pretrain STE+K=64 on V-AGE] 800 steps ...")
    losses = pretrain_with_ste(bolt, vis_tr_emb, vis_tr_pid, MODALITY_VISION, tok,
                                 n_steps=800, lr=3e-4, batch=4, T=64,
                                 frac_perceptual=0.15)
    if losses:
        last = losses[-30:]
        avg_ntp = sum(l[0] for l in last) / len(last)
        print(f"  pretrain final NTP loss: {avg_ntp:.4f}")

    print("\n[eval] V-AGE retrieval")
    Ns = [5, 20, 50, 100, 200]; nq = 5
    results = {}
    for N in Ns:
        if N > len(set(vis_ev_pid)): continue
        rag = embedding_rag_ceiling(vis_ev_emb, vis_ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate_ste(bolt, vis_ev_emb, vis_ev_pid, MODALITY_VISION, tok,
                          N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>3}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}")
        results[N] = {"rag": rag, **r}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_V-AGE-ste.json")
    with open(out, "w") as f:
        json.dump({"K": K, "results": results}, f, indent=2, default=str)
    print(f"\n[done] {out}")

    naive = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_V-AGE.json"))
    print("\n" + "=" * 80)
    print("V-AGE: K=64 naive generic-NTP vs K=64 +STE")
    print("=" * 80)
    print(f"{'N':>4} | {'RAG':>5} | {'naive retr@1':>13} | {'+STE retr@1':>12} | "
          f"{'naive cm':>9} | {'+STE cm':>8} | {'naive mf':>9} | {'+STE mf':>8}")
    print("-" * 100)
    for N in Ns:
        n_res = naive["results"].get(str(N), {})
        s_res = results.get(N, {})
        if not s_res: continue
        print(f"{N:>4} | {s_res['rag']:>5.3f} | "
              f"{n_res.get('retrieval_at_1', float('nan')):>13.3f} | "
              f"{s_res['retrieval_at_1']:>12.3f} | "
              f"{n_res.get('code_match_retr', float('nan')):>9.3f} | "
              f"{s_res['code_match_retr']:>8.3f} | "
              f"{n_res.get('fraction_code_match', float('nan')):>9.3f} | "
              f"{s_res['fraction_code_match']:>8.3f}")


if __name__ == "__main__":
    sys.exit(main())
