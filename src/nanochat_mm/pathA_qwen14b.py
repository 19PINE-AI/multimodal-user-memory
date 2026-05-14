"""Path A scale-up: Qwen2.5-14B (cached) vs the Qwen2.5-3B baseline.

Quick check: does scaling the frozen base LM 5x (3B → 14B) improve
Path A on the strongest modality (audio at K=64)? If yes, the paper
has a 'scales with LM size' result for free.

Memory budget: 14B in bf16 = ~28 GB; plus ~1 GB Engram + perc emb +
3-5 GB activations. Comfortably under the 102 GB Blackwell.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_AUDIO, MODALITY_VISION
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import QwenEngramBolt, evaluate, DEVICE
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)

MODEL_ID_14B = "Qwen/Qwen2.5-14B-Instruct"
LARGE = "/home/ubuntu/multimodal-user-memory/runs/embeddings"


def main():
    print("=" * 70)
    print(f"Path A scale-up — {MODEL_ID_14B}")
    print("=" * 70)

    print("\nLoading Qwen2.5-14B (bf16) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID_14B, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_14B, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()
    n_params = sum(p.numel() for p in qwen.parameters())
    print(f"  loaded; {n_params/1e9:.2f}B params, hidden={qwen.config.hidden_size}, layers={qwen.config.num_hidden_layers}")
    if torch.cuda.is_available():
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    aud = np.load(f"{LARGE}/ecapa_libri_large.npz")
    aud_tr_emb, aud_tr_pid, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    K = 64
    apply_fn = fit_naive_rq(aud_tr_emb, n_levels=1, k_per=K)

    # 14B has more layers; attach at a layer proportional to 3B's 24/36
    n_layers = qwen.config.num_hidden_layers
    attach = int(0.66 * n_layers)
    print(f"  attaching Engram at layer {attach} / {n_layers}")

    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                           engram_attach_layer=attach).to(DEVICE)
    bolt.install_hook()
    print(f"  trainable params: {sum(p.numel() for p in bolt.parameters() if p.requires_grad):,}")
    if torch.cuda.is_available():
        print(f"  GPU mem after Engram: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\n[pretrain] audio generic-NTP, 14B base, K=64 ...")
    losses = pretrain_generic(bolt, aud_tr_emb, aud_tr_pid, apply_fn, MODALITY_AUDIO, tok,
                              n_steps=400, lr=3e-4, batch=2, T=64, frac_perceptual=0.15)
    print(f"  pretrain final loss: {float(np.mean(losses[-30:])):.4f}")

    print("\n[eval]")
    Ns = [5, 10, 20]; nq = 5
    results = {}
    for N in Ns:
        if N > len(set(aud_ev_pid)): continue
        rag = embedding_rag_ceiling(aud_ev_emb, aud_ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate(bolt, apply_fn, aud_ev_emb, aud_ev_pid, MODALITY_AUDIO, tok,
                      N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}")
        results[N] = {"rag": rag, **r}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_qwen14b.json")
    with open(out, "w") as f:
        json.dump({"model": MODEL_ID_14B, "n_params": int(n_params),
                    "attach_layer": int(attach), "K": K,
                    "results": results}, f, indent=2, default=str)

    # Headline vs 3B
    p3 = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_scaling_k64.json"))
    print("\n" + "=" * 80)
    print("HEADLINE — audio K=64 generic-NTP: 3B vs 14B base LM (large data)")
    print("=" * 80)
    print(f"{'N':>4} | {'3B retr@1':>10} | {'14B retr@1':>11} | {'3B code-match':>13} | {'14B code-match':>14}")
    print("-" * 80)
    for N in Ns:
        a3 = p3["audio"].get(str(N), {})
        a14 = results.get(N, {})
        if not a14: continue
        print(f"{N:>4} | {a3.get('retrieval_at_1', float('nan')):>10.3f} | "
              f"{a14.get('retrieval_at_1', float('nan')):>11.3f} | "
              f"{a3.get('code_match_retr', float('nan')):>13.3f} | "
              f"{a14.get('code_match_retr', float('nan')):>14.3f}")


if __name__ == "__main__":
    sys.exit(main())
