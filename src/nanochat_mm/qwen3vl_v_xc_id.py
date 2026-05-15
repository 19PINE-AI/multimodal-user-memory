"""End-to-end V-XC-ID on Qwen3-VL-8B-Thinking.

The audio path on Qwen3-VL was validated in `pathA_qwen3vl.json` (code-match
1.00 at N=5). The visual stream was deferred (experiment #11). This script
closes that gap by running V-XC-ID (cross-condition face identity) on
Qwen3-VL with the same generic-NTP recipe, ArcFace LFW-XL features, K=64
codebook. It does NOT route faces through Qwen3-VL's visual encoder
(the audio probe established that the LM backbone is not the bottleneck —
the codebook and Engram are); rather, it uses ArcFace as the encoder and
treats the resulting code as MODALITY_VISION inputs to the bolt.

This validates: with the strongest published face encoder, does Qwen3-VL
match or exceed Qwen2.5-3B on V-XC-ID? It answers the paper claim
"recipe transfers across LM scales and architectures" for vision.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_VISION
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import evaluate as evaluate_qwen
from qwen3vl_engram_bolt import Qwen3VLEngramBolt, MODEL_ID, DEVICE

torch.manual_seed(42); np.random.seed(42)


def main():
    print("=" * 70)
    print(f"V-XC-ID on {MODEL_ID}")
    print("=" * 70)

    free_gb = (torch.cuda.mem_get_info(0)[0] / 1e9) if torch.cuda.is_available() else 0.0
    print(f"\nGPU free: {free_gb:.1f} GB")

    kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                   low_cpu_mem_usage=True)
    if free_gb < 18.0:
        # Constrain GPU footprint so accelerate keeps activations headroom.
        # The lm_head + a few decoder layers fit; the rest stays on CPU.
        budget_gb = max(8, int(free_gb - 3))
        print(f"  Low GPU; device_map='auto' max_memory={budget_gb}GiB on GPU.")
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: f"{budget_gb}GiB", "cpu": "120GiB"}
    else:
        kwargs["device_map"] = {"": DEVICE}

    print("\nLoading Qwen3-VL-8B-Thinking ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    qwen.eval()
    n_layers = qwen.config.text_config.num_hidden_layers
    print(f"  loaded; {sum(p.numel() for p in qwen.parameters())/1e9:.2f}B params; "
          f"{n_layers} text layers")

    # ArcFace LFW XL: ~423 identities; same encoder as Qwen2.5-3B V-XC-ID-XL.
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw_xl.npz")
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(vis['emb'], vis['pid'])
    print(f"  vision: {len(set(tr_pid))} train IDs, {len(set(ev_pid))} eval IDs")
    K = 64
    apply_fn = fit_naive_rq(tr_emb, n_levels=1, k_per=K)

    attach = int(0.66 * n_layers)
    print(f"  attach Engram at layer {attach} / {n_layers}, K={K}")

    bolt = Qwen3VLEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                              engram_attach_layer=attach)
    # Only move the *new* modules (engram + perc_emb) to GPU. Qwen itself has
    # offloaded (meta) params under device_map='auto'; recursive .to() would fail.
    bolt.engram.to(DEVICE)
    bolt.vis_perc_emb.to(DEVICE)
    bolt.aud_perc_emb.to(DEVICE)
    bolt.install_hook()
    print(f"  trainable params: {sum(p.numel() for p in bolt.parameters() if p.requires_grad):,}")

    # Pretrain budget reduced for offloaded run: batch=1, T=32 to keep
    # activations small under tight GPU memory.
    print("\n[pretrain] vision generic-NTP K=64, 200 steps, batch=1, T=32")
    losses = pretrain_generic(bolt, tr_emb, tr_pid, apply_fn, MODALITY_VISION, tok,
                              n_steps=200, lr=3e-4, batch=1, T=32, frac_perceptual=0.15)
    print(f"  pretrain final loss: {float(np.mean(losses[-30:])):.4f}")

    print("\n[eval]")
    Ns = [5, 10, 20]; nq = 5
    results = {}
    for N in Ns:
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate_qwen(bolt, apply_fn, ev_emb, ev_pid, MODALITY_VISION, tok,
                          N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}")
        results[N] = {"rag": rag, **r}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_qwen3vl_face.json")
    with open(out, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "modality": "V-XC-ID",
            "encoder": "ArcFace R50",
            "data": "LFW-XL (423 IDs)",
            "attach_layer": int(attach),
            "K": K,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Compare with Qwen2.5-3B V-XC-ID-XL
    try:
        q3 = json.load(open(
            "/home/ubuntu/multimodal-user-memory/results/pathA_V-XC-ID-xl.json"))
        print("\n" + "=" * 80)
        print("Comparison — V-XC-ID K=64 across LM backbones")
        print("=" * 80)
        print(f"{'N':>4} | {'Qwen2.5-3B':>11} | {'Qwen3-VL-8B':>11} || code-match: 3B / 3-VL")
        print("-" * 80)
        for N in Ns:
            v3 = q3.get(str(N), {})
            vl = results.get(N, {})
            print(f"{N:>4} | {v3.get('retrieval_at_1', float('nan')):>11.3f} | "
                  f"{vl.get('retrieval_at_1', float('nan')):>11.3f} || "
                  f"{v3.get('code_match_retr', float('nan')):.3f} / "
                  f"{vl.get('code_match_retr', float('nan')):.3f}")
    except FileNotFoundError as e:
        print(f"\n(comparison file not found: {e})")


if __name__ == "__main__":
    sys.exit(main())
