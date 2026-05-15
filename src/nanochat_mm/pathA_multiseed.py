"""A-PARA multi-seed verification.

Original framing's headline: at A-PARA N=10, Path A retr@1 (0.45) BEATS
the embedding-RAG cosine-NN ceiling (0.43). This rests on one seed in the
session-7/8 runs. To make the claim robust, run the same protocol across
5 seeds and report retr@1 mean + std.

Per-seed entropy sources:
  - Codebook k-means init (seed at fit_naive_rq)
  - Engram + perc_emb param init (torch.manual_seed)
  - Pretrain batch sampling (rng inside pretrain_generic)
  - Eval registration / query order (rng=99 inside evaluate)

We vary the *codebook* seed and the *bolt* seed together; the eval rng
is held at the default (99) so we measure recipe variance, not query-set
variance.

Runs the new id-codebook v2 setup (best K per modality from K-sweep) so
the multi-seed numbers track the strongest recipe.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_AUDIO, MODALITY_VISION
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from real_encoder_train import fit_naive_rq
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import QwenEngramBolt, evaluate, MODEL_ID, DEVICE
from id_codebook_v2 import MODE_PATHS

MODE_TO_MODALITY = {"a-para": MODALITY_AUDIO, "a-xr-id": MODALITY_AUDIO,
                     "a-scn": MODALITY_AUDIO,
                     "v-xc-id": MODALITY_VISION, "v-sty": MODALITY_VISION,
                     "v-sty-clip": MODALITY_VISION}


def run_one_seed(mode, K, seed, qwen, tok):
    torch.manual_seed(seed); np.random.seed(seed)
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary, _ = MODE_PATHS[mode]
    d = np.load(EMB / primary)
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    apply_fn = fit_naive_rq(tr_emb, n_levels=1, k_per=K, seed=seed)
    def apply_1d(emb_np):
        out = apply_fn(emb_np)
        return out[:, 0] if out.ndim == 2 else out

    modality_id = MODE_TO_MODALITY[mode]
    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                            engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    pretrain_generic(bolt, tr_emb, tr_pid, apply_1d, modality_id, tok,
                     n_steps=400, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)
    Ns = [5, 10, 20]; nq = 5
    out = {}
    for N in Ns:
        if N > len(set(ev_pid)): continue
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate(bolt, apply_1d, ev_emb, ev_pid, modality_id, tok,
                       N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        out[N] = {"rag": rag, "retr@1": r["retrieval_at_1"],
                   "code_match": r["code_match_retr"],
                   "frac_code_match": r["fraction_code_match"]}
    bolt.remove_hook()
    del bolt; torch.cuda.empty_cache()
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "a-para"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    seeds = list(range(42, 42 + n_seeds))

    print("=" * 70)
    print(f"Multi-seed verification — mode={mode}  K={K}  n_seeds={n_seeds}")
    print(f"  seeds: {seeds}")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    all_results = {}
    for s in seeds:
        print(f"\n--- seed = {s} ---")
        try:
            all_results[s] = run_one_seed(mode, K, s, qwen, tok)
            for N, r in all_results[s].items():
                print(f"  N={N:>3}  retr@1={r['retr@1']:.3f}  RAG={r['rag']:.3f}  "
                      f"code-match={r['code_match']:.3f}")
        except Exception as e:
            print(f"  seed {s} failed: {e}")
            all_results[s] = {"error": str(e)}

    # Aggregate across seeds
    print("\n" + "=" * 70)
    print(f"AGGREGATE — mean ± std across {n_seeds} seeds")
    print("=" * 70)
    print(f"{'N':>4} | {'mean retr@1':>13} | {'std':>5} | {'min':>5} | {'max':>5} | {'RAG':>5}")
    print("-" * 60)
    Ns = [5, 10, 20]
    aggregate = {}
    for N in Ns:
        vals = []; rags = []
        for s in seeds:
            if isinstance(all_results.get(s), dict) and N in all_results[s] and "retr@1" in all_results[s][N]:
                vals.append(all_results[s][N]["retr@1"])
                rags.append(all_results[s][N]["rag"])
        if vals:
            mean = float(np.mean(vals)); std = float(np.std(vals))
            rag_mean = float(np.mean(rags))
            print(f"{N:>4} | {mean:>13.3f} | {std:>5.3f} | {min(vals):>5.3f} | "
                  f"{max(vals):>5.3f} | {rag_mean:>5.3f}")
            aggregate[N] = {"mean": mean, "std": std, "min": min(vals),
                             "max": max(vals), "vals": vals, "rag_mean": rag_mean}

    out_path = Path("/home/ubuntu/multimodal-user-memory/results/") / f"pathA_multiseed_{mode}_K{K}.json"
    with open(out_path, "w") as f:
        json.dump({"mode": mode, "K": K, "seeds": seeds,
                    "per_seed": all_results,
                    "aggregate": {str(k): v for k, v in aggregate.items()}},
                   f, indent=2, default=str)
    print(f"\n[done] {out_path}")

    # BEATS-RAG verdict (for A-PARA at N=10)
    if mode == "a-para" and 10 in aggregate:
        a = aggregate[10]
        print("\nA-PARA at N=10 BEATS-RAG verdict:")
        print(f"  RAG (mean across seeds):    {a['rag_mean']:.3f}")
        print(f"  Path A (mean across seeds): {a['mean']:.3f}  ({a['std']:.3f} std)")
        n_beats = sum(1 for v in a["vals"] if v >= a["rag_mean"])
        print(f"  Path A >= RAG in {n_beats}/{len(a['vals'])} seeds")
        if a["mean"] > a["rag_mean"]:
            print("  CLAIM SUPPORTED: mean across seeds beats RAG.")
        elif n_beats >= len(a["vals"]) // 2:
            print(f"  CLAIM PARTIALLY SUPPORTED: beats in majority of seeds.")
        else:
            print(f"  CLAIM NOT SUPPORTED: original BEATS-RAG was a lucky seed.")


if __name__ == "__main__":
    sys.exit(main())
