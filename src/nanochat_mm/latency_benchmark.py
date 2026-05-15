"""Wall-clock latency benchmark — Path A surgical insertion + per-query vs RAG.

Compares the two operating points:

1. **Path A**:
   - Registration: surgical row insertion ~ 80 SGD steps on hashed rows.
   - Query: single forward pass through frozen Qwen (no extra context).

2. **RAG cosine-NN (baseline)**:
   - Registration: store one embedding (~512-dim).
   - Query: cosine NN over N stored embeddings (O(N) or with FAISS).
     For a fair compare, the LM also runs to produce the marker token,
     which means we either concatenate top-K candidates to the prompt
     (increasing context cost) or use the NN's marker directly.

We measure both per-registration and per-query wall-clock at
N = 10, 100, 1000, 10000.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_AUDIO
from real_encoder_train import fit_naive_rq
from qwen_engram_bolt import QwenEngramBolt, build_fixed_context, get_touched_rows, MODEL_ID, DEVICE
from pathA_generic_pretrain import pretrain_generic
from v2_retrieval import split_by_identity


def time_surgical_insertion(bolt, code, modality_id, marker, tok, max_steps=80, lr=1.0, T=24):
    """Time one surgical insertion."""
    eng = bolt.engram.engrams[str(modality_id)]
    input_ids, modality_ids = build_fixed_context(code, modality_id, tok, marker, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
    touched = get_touched_rows(eng, code, input_ids)
    params = [eng.tables[ks].embedding.weight for ks in touched]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.0)
    target = torch.tensor([marker], dtype=torch.long, device=DEVICE)
    import torch.nn.functional as F
    t0 = time.perf_counter()
    for step in range(max_steps):
        logits = bolt(input_ids, modality_ids)
        loss = F.cross_entropy(logits[:, -1, :], target)
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            for ks, rows in touched.items():
                W = eng.tables[ks].embedding.weight
                if W.grad is None: continue
                mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.grad.dtype)
                mask[torch.tensor(sorted(rows), device=W.device, dtype=torch.long)] = 1.0
                W.grad.mul_(mask)
        opt.step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def time_path_a_query(bolt, code, modality_id, tok, T=24):
    """Time one query forward pass."""
    input_ids, modality_ids = build_fixed_context(code, modality_id, tok, marker_text_id=0, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = bolt(input_ids, modality_ids)
        # extract logits at last position; argmax over markers
        _ = logits[0, -1, :].argmax()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def time_rag(query_emb, reg_embs):
    """Time one RAG cosine NN over reg_embs."""
    q = torch.from_numpy(query_emb).to(DEVICE).unsqueeze(0)
    R = torch.from_numpy(reg_embs).to(DEVICE)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
    R = R / (R.norm(dim=-1, keepdim=True) + 1e-9)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sims = q @ R.t()
    _ = sims.argmax(-1).item()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    print("=" * 70)
    print("Latency benchmark — Path A vs RAG cosine-NN")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B + Engram ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri_large.npz")
    aud_tr_emb, aud_tr_pid, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    K = 64
    apply_fn = fit_naive_rq(aud_tr_emb, n_levels=1, k_per=K)

    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K, engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    # Quick pretrain just to have a sensible model
    print("[setup] brief pretraining (100 steps) ...")
    pretrain_generic(bolt, aud_tr_emb, aud_tr_pid, apply_fn, MODALITY_AUDIO, tok,
                      n_steps=100, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    # ---- Per-registration latency (surgical insertion) ----
    print("\n[bench] surgical insertion per identity (5 runs each):")
    ins_times = []
    for i in range(5):
        code = int(np.random.randint(0, K))
        marker = 30001 + i
        t = time_surgical_insertion(bolt, code, MODALITY_AUDIO, marker, tok)
        ins_times.append(t)
    print(f"  Path A insertion: mean = {1000*np.mean(ins_times):.1f} ms, median = {1000*np.median(ins_times):.1f} ms")

    # ---- Per-query latency ----
    print("\n[bench] per-query forward (10 runs each):")
    q_times = []
    for i in range(10):
        code = int(np.random.randint(0, K))
        t = time_path_a_query(bolt, code, MODALITY_AUDIO, tok)
        q_times.append(t)
    print(f"  Path A query: mean = {1000*np.mean(q_times):.1f} ms, median = {1000*np.median(q_times):.1f} ms")

    # ---- RAG latency at various N ----
    print("\n[bench] RAG cosine-NN at various N:")
    rag_times = {}
    for N in [10, 100, 1000, 10000, 100000]:
        # Create synthetic reg_embs
        reg = np.random.randn(N, aud_tr_emb.shape[1]).astype(np.float32)
        runs = []
        # Warm-up
        for _ in range(2):
            time_rag(reg[0], reg)
        for _ in range(10):
            q_emb = reg[np.random.randint(0, N)]
            t = time_rag(q_emb, reg)
            runs.append(t)
        rag_times[N] = float(np.mean(runs))
        print(f"  RAG N={N:>6}: mean = {1000*np.mean(runs):.3f} ms, median = {1000*np.median(runs):.3f} ms")

    out = Path("/home/ubuntu/multimodal-user-memory/results/latency.json")
    with open(out, "w") as f:
        json.dump({
            "path_a_insertion_ms": float(1000 * np.mean(ins_times)),
            "path_a_query_ms": float(1000 * np.mean(q_times)),
            "rag_query_ms": {str(N): float(1000 * t) for N, t in rag_times.items()},
        }, f, indent=2)
    print(f"\n[done] {out}")

    print("\n" + "=" * 70)
    print("HEADLINE: end-to-end per-query latency")
    print("=" * 70)
    pa_q = 1000 * np.mean(q_times)
    print(f"  Path A: {pa_q:.1f} ms (LM forward + Engram hook, NO context overhead for stored identities)")
    print(f"\n  RAG cosine-NN ONLY (no LM forward):")
    for N, t in rag_times.items():
        print(f"    N={N:>6}: {1000*t:.3f} ms")
    print(f"\n  Real RAG: cosine-NN time + LM forward with N candidate tokens in context (~N*40 extra tokens for marker names)")


if __name__ == "__main__":
    sys.exit(main())
