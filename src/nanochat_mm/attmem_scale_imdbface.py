"""Scale-test: register 10K-37K IMDB-Face celebrity embeddings as identities
and test retrieval at N=1k, 5k, 10k, 37k. Each celeb has only 1 avg embedding,
so we simulate cross-condition with bounded additive Gaussian noise.

This is an ARCHITECTURAL scale test, not a real cross-condition test (we lack
multi-sample-per-id IMDB data). The cosine NN baseline tells us the encoder
ceiling at this scale; AttMem tells us whether the LM's value-side prior
adds discriminative signal at 10x our face_xxxl scale.

Pre-trained AttMem weights are not available from earlier training runs, so
we use zero-shot AttMem (W_o=I, out_gain=8, log_inv_temp=log(20), no training)
+ a quick targeted pretraining on the IMDB pool.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION, MODALITY_TEXT
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    noise_sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    np.random.seed(seed); torch.manual_seed(seed)

    print(f"Loading IMDB-Face 37K celebrity embeddings ...")
    url = "https://huggingface.co/datasets/silk-road/IMDB-Face-Recognition/resolve/main/celeb_average_feature.parquet"
    df = pd.read_parquet(url)
    emb = np.stack([np.asarray(v[0], dtype=np.float32) for v in df["average_feature"]])
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    print(f"  total celebs: {len(emb)}, dim={emb.shape[1]}")

    print("Loading Qwen2.5-3B + AttMem bolt ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()
    bolt = QwenAttMemBolt(qwen, tok, vision_key_dim=512, audio_key_dim=192,
                          attach_layer=33, attach_lm_head=True).to(DEVICE)
    bolt.install_hook()

    T = 24
    pad_id = tok.pad_token_id or 0
    pref = tok.encode("You see", add_special_tokens=False)
    text_ids = list(pref) + [pad_id] * (T - 1 - len(pref))
    text_ids = (text_ids[: T - 1]) + [pad_id]
    text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
    modality_ids_t = torch.tensor(
        [[MODALITY_TEXT] * (T - 1) + [int(MODALITY_VISION)]],
        dtype=torch.long, device=DEVICE,
    )

    results = {}
    for N in [1000, 5000, 10000, 37283]:
        if N > len(emb):
            N = len(emb)
        print(f"\n=== N = {N} (synthetic cross-condition, noise σ={noise_sigma}) ===")
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(emb), size=N, replace=False)
        keys = emb[idxs]
        # Synthesise "cross-condition" queries via Gaussian noise + renorm
        noise = rng.normal(0, noise_sigma, keys.shape).astype(np.float32)
        queries = keys + noise
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)

        # RAG baseline
        # (full N×N cosine on N=37K is memory-heavy; pick 300 random queries to eval)
        n_eval_queries = min(300, N)
        q_eval_idx = rng.choice(N, size=n_eval_queries, replace=False)
        q_eval = queries[q_eval_idx]
        # Targets: the row-index of each query in [0..N)
        targets_local = q_eval_idx
        t0 = time.time()
        sim = q_eval @ keys.T  # [300, N]
        pred_rag = sim.argmax(axis=1)
        retr_rag = (pred_rag == targets_local).mean()
        t_rag = time.time() - t0

        # AttMem (zero-shot, restricted to the 300 candidate targets per query)
        # Use marker token IDs 30001..30001+N-1; insert all keys
        marker_offset = 30001
        marker_ids = list(range(marker_offset, marker_offset + N))
        bolt.reset_banks()
        keys_t = torch.from_numpy(keys).to(DEVICE)
        # Insert in chunks to avoid OOM at large N
        chunk = 2000
        for c0 in range(0, N, chunk):
            c1 = min(c0 + chunk, N)
            bolt.insert_batch(MODALITY_VISION, keys_t[c0:c1], marker_ids[c0:c1])

        correct_attmem = 0
        t0 = time.time()
        for k, tgt in enumerate(targets_local):
            q_key = torch.from_numpy(q_eval[k]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = bolt(modality_ids_t, text_ids_t, {int(MODALITY_VISION): q_key})
                last = logits[0, -1, :]
                # Argmax over the full set of N registered markers
                # (this is the full retrieval task — restriction would be cheating)
                ml = torch.stack([last[m] for m in marker_ids])
                pred_local = int(ml.argmax().item())
            if pred_local == tgt:
                correct_attmem += 1
        retr_attmem = correct_attmem / n_eval_queries
        t_attmem = time.time() - t0

        print(f"  n_eval_queries: {n_eval_queries}")
        print(f"  RAG cosine retr@1: {retr_rag:.3f}   ({t_rag*1000:.1f} ms total)")
        print(f"  AttMem retr@1:     {retr_attmem:.3f}   ({t_attmem*1000:.1f} ms total)")
        print(f"  AttMem ms/query: {t_attmem * 1000 / n_eval_queries:.1f}")

        results[N] = {
            "n_eval_queries": n_eval_queries,
            "rag_retr1": float(retr_rag),
            "attmem_retr1": float(retr_attmem),
            "noise_sigma": noise_sigma,
            "rag_total_ms": t_rag * 1000,
            "attmem_total_ms": t_attmem * 1000,
        }

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/attmem_scale_imdbface_seed{seed}_noise{noise_sigma}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
