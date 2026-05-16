"""PerceptMem scale eval — Path A vs RAG at N up to 1000+.

The PerceptMem v0.2 scorecard tops at N=20 for most cells. At that scale,
strong encoders (ECAPA, ArcFace, AST) give cosine-NN RAG ceilings of ~1.00,
which Path A's parametric mechanism cannot beat. The original framing's
"1000+ ID per user" scale was never actually evaluated.

This script evaluates Path A and RAG at scale-spaced N (50, 100, 200,
500, 1000 where data permits). The hypothesis: at large N, RAG's
cosine-NN ceiling degrades faster than Path A's parametric mechanism,
because Path A's per-row write is independent of N while RAG's
1-vs-N argmax becomes harder as more identities crowd the embedding
space.

Currently feasible scales:
  V-XC-ID: combined LFW-XXL + AgeDB = 1401 IDs (700 eval after 50/50).
           Plus we can use full pool (train on identities 0..N, test on
           the SAME identities' cross-condition queries) for N up to 1400.
  A-PARA:  168 IDs (84 eval).
  V-STY:   128 painters (64 eval).
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
from engram_module_mm import MODALITY_VISION, MODALITY_AUDIO
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import QwenEngramBolt, evaluate, MODEL_ID, DEVICE
from id_codebook_v2 import load_pipeline_apply, MODE_PATHS

torch.manual_seed(42); np.random.seed(42)


# Larger eval pools added for scale tests
SCALE_PATHS = {
    "v-xc-id-face": ("arcface_face_combined.npz", "arcface_lfw_xxl.npz"),  # 1401 / 901
    **MODE_PATHS,
}

MODE_TO_MODALITY = {
    "a-xr-id": MODALITY_AUDIO, "a-scn": MODALITY_AUDIO, "a-para": MODALITY_AUDIO,
    "v-xc-id": MODALITY_VISION, "v-sty": MODALITY_VISION,
    "v-sty-clip": MODALITY_VISION, "v-sty-xxl": MODALITY_VISION,
    "v-xc-id-face": MODALITY_VISION,
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "v-xc-id-face"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    # Default N sweep — adjust if data is smaller
    Ns_str = sys.argv[4] if len(sys.argv) > 4 else "20,50,100,200,400,700"
    Ns = [int(x) for x in Ns_str.split(",")]

    print("=" * 70)
    print(f"PerceptMem scale eval — mode={mode}  K={K}  seed={seed}  Ns={Ns}")
    print("=" * 70)

    torch.manual_seed(seed); np.random.seed(seed)
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary, _ = SCALE_PATHS[mode]
    d = np.load(EMB / primary)
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    n_eval_ids = len(set(ev_pid.tolist()))
    print(f"  data: train {len(set(tr_pid.tolist()))} IDs / {len(tr_emb)} samp, "
          f"eval {n_eval_ids} IDs / {len(ev_emb)} samp")
    # Clamp N values to available eval IDs
    Ns = [N for N in Ns if N <= n_eval_ids]
    print(f"  Ns (after clamp): {Ns}")

    # Load the codebook — use the existing per-modality v2 codebook when available,
    # otherwise fit naive k-means on this run's train split.
    if mode == "v-xc-id-face":
        # Reuse V-XC-ID-XL codebook (adapter+largepool); it was trained on LFW-XXL
        # which is a subset of the combined data. Fine to reuse the centroids.
        cb_path = Path("/home/ubuntu/multimodal-user-memory/runs/codebooks/"
                       "id_v2_codebook_v-xc-id_K64.pt")
    else:
        cb_path = Path(f"/home/ubuntu/multimodal-user-memory/runs/codebooks/"
                       f"id_v2_codebook_{mode}_K{K}.pt")
    if cb_path.exists():
        print(f"  using codebook: {cb_path.name}")
        apply_fn = load_pipeline_apply(cb_path)
    else:
        from real_encoder_train import fit_naive_rq
        print(f"  no v2 codebook at K={K}; fitting naive k-means on train split")
        f = fit_naive_rq(tr_emb, n_levels=1, k_per=K, seed=seed)
        apply_fn = lambda e: (f(e)[:, 0] if f(e).ndim == 2 else f(e))

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    modality_id = MODE_TO_MODALITY[mode]
    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                            engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()

    print("\n[pretrain] generic-NTP 400 steps")
    pretrain_generic(bolt, tr_emb, tr_pid, apply_fn, modality_id, tok,
                     n_steps=400, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    # Evaluate at each N
    nq = 3  # fewer queries per ID to keep eval time tractable at large N
    print(f"\n[eval — {nq} queries per ID]")
    print(f"{'N':>5} | {'RAG':>6} | {'PathA retr@1':>13} | {'code-match':>11} | "
          f"{'frac-code':>10} | {'insert s':>8}")
    print("-" * 75)
    results = {}
    for N in Ns:
        if N > n_eval_ids:
            print(f"  N={N} > eval IDs ({n_eval_ids}); skip"); continue
        t0 = time.time()
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate(bolt, apply_fn, ev_emb, ev_pid, modality_id, tok,
                       N_subset=N, n_queries_per_id=nq, max_steps=60, lr=1.0, T=24)
        elapsed = time.time() - t0
        print(f"{N:>5} | {rag:>6.3f} | {r['retrieval_at_1']:>13.3f} | "
              f"{r['code_match_retr']:>11.3f} | {r['fraction_code_match']:>10.3f} | "
              f"{elapsed:>7.0f}s")
        results[N] = {"rag": rag, "retr@1": r["retrieval_at_1"],
                       "code_match": r["code_match_retr"],
                       "frac_code_match": r["fraction_code_match"],
                       "collisions": r["N_collision_codes"],
                       "elapsed_s": elapsed}

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/"
                f"pathA_scale_{mode}_K{K}_seed{seed}.json")
    with open(out, "w") as f: json.dump({"mode": mode, "K": K, "seed": seed,
                                           "Ns": Ns, "results": results},
                                          f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Headline summary
    print("\n" + "=" * 70)
    print("HEADLINE — RAG ceiling degradation vs Path A retention")
    print("=" * 70)
    print(f"{'N':>5} | {'RAG':>6} | {'Path A':>8} | {'Δ (Path A - RAG)':>17}")
    print("-" * 50)
    for N in Ns:
        r = results.get(N)
        if r is None: continue
        delta = r["retr@1"] - r["rag"]
        mk = " ↑" if delta > 0 else (" ≈" if abs(delta) < 0.02 else " ↓")
        print(f"{N:>5} | {r['rag']:>6.3f} | {r['retr@1']:>8.3f} | {delta:>+8.3f}{mk}")


if __name__ == "__main__":
    sys.exit(main())
