"""Path A with id-supervised codebook v2 — full Path A pipeline.

Loads the saved (adapter, centroids) bundle from `runs/codebooks/`, builds
an apply_fn compatible with the existing Path A pipeline, then runs:
  - generic-NTP pretraining (Engram + perc_emb) at the v2 codebook
  - surgical insertion at registration time
  - evaluation at N=5/10/20

This is the integration that turns "codebook has higher same-code rate"
into "Path A retr@1 lifts."

Usage:
  python3 pathA_idcb_run.py <mode> <K>
    mode in {a-xr-id, a-scn, v-xc-id, v-sty, a-para}
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
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import (
    QwenEngramBolt, evaluate, MODEL_ID, DEVICE,
)
from id_codebook_v2 import load_pipeline_apply, MODE_PATHS

torch.manual_seed(42); np.random.seed(42)


MODE_TO_MODALITY = {
    "a-xr-id": MODALITY_AUDIO,
    "a-scn":   MODALITY_AUDIO,
    "a-para":  MODALITY_AUDIO,
    "v-xc-id": MODALITY_VISION,
    "v-sty":   MODALITY_VISION,
    "v-sty-clip": MODALITY_VISION,
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "a-xr-id"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    print("=" * 70)
    print(f"Path A + id-codebook v2 — mode={mode}  K={K}")
    print("=" * 70)

    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary, _ = MODE_PATHS[mode]
    d = np.load(EMB / primary)
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    print(f"  data: train {len(set(tr_pid))} IDs / {len(tr_emb)} samples, "
          f"eval {len(set(ev_pid))} IDs / {len(ev_emb)} samples")

    # Load the v2 codebook
    cb_path = Path(f"/home/ubuntu/multimodal-user-memory/runs/codebooks/"
                   f"id_v2_codebook_{mode}_K{K}.pt")
    print(f"  loading id-codebook v2 from {cb_path}")
    apply_fn = load_pipeline_apply(cb_path)

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
    print(f"  bolt built (attach=24, K={K})")

    print(f"\n[pretrain] generic-NTP 400 steps  modality={modality_id}")
    losses = pretrain_generic(bolt, tr_emb, tr_pid, apply_fn, modality_id, tok,
                              n_steps=400, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)
    print(f"  pretrain final loss: {float(np.mean(losses[-30:])):.4f}")

    print("\n[eval]")
    Ns = [5, 10, 20]
    nq = 5
    results = {}
    for N in Ns:
        if N > len(set(ev_pid)): continue
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate(bolt, apply_fn, ev_emb, ev_pid, modality_id, tok,
                       N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f}  "
              f"on {100*r['fraction_code_match']:.0f}%  "
              f"collisions={r['N_collision_codes']}")
        results[N] = {"rag": rag, **r}

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/pathA_idcb_{mode}_K{K}.json")
    with open(out, "w") as f:
        json.dump({"mode": mode, "K": K, "results": results,
                    "codebook_path": str(cb_path)},
                   f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Compare to old Path A scores when available
    legacy_map = {
        "a-xr-id": "pathA_A-XR-VOX.json",
        "a-scn":   "pathA_scene.json",
        "v-xc-id": "pathA_V-XC-ID-xl.json",
        "v-sty":   "pathA_style_pca_gram.json",
        "v-sty-clip": "pathA_style_pca_gram.json",  # gram is the prior baseline
        "a-para":  "pathA_paralinguistic_se.json",
    }
    legacy = Path("/home/ubuntu/multimodal-user-memory/results/") / legacy_map[mode]
    if legacy.exists():
        try:
            lg = json.load(open(legacy))
            print("\n" + "=" * 70)
            print(f"Comparison — Path A naive vs Path A + id-codebook v2 ({mode})")
            print("=" * 70)
            print(f"{'N':>4} | {'RAG':>5} | {'PathA-naive':>12} | {'PathA-idcb':>10}")
            print("-" * 60)
            for N in Ns:
                naive = lg.get("results", lg).get(str(N), {}).get("retrieval_at_1", None) \
                        if "results" in lg else lg.get(str(N), {}).get("retrieval_at_1", None)
                ours = results.get(N, {}).get("retrieval_at_1", None)
                if naive is None or ours is None: continue
                print(f"{N:>4} |  {results[N]['rag']:.2f} | {naive:>12.3f} | {ours:>10.3f}   "
                      f"(Δ {ours - naive:+.3f})")
        except Exception as e:
            print(f"(legacy compare skipped: {e})")


if __name__ == "__main__":
    sys.exit(main())
