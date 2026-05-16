"""Extreme-scale latency benchmark — Path A vs RAG at N up to 50,000.

The sess-14 latency_at_scale.py topped at N=2000. To make the structural
advantage of parametric memory visible, push to N=5k, 10k, 25k, 50k.
At those scales:
  - RAG-with-LM-consumption: context grows to 100k+ tokens. Either OOM
    or Qwen2.5's context window (32k) clips. We measure both "fits"
    and "OOM" regimes.
  - Path A: query time is O(1) in N. The constant doesn't move.

Also report:
  - Memory cost per registered identity (RAG: raw 512-dim emb = 1 KB;
    Path A: O(K) table, amortised across all IDs).
  - End-to-end "session" cost: 1 insertion + 1000 queries.
  - Context-window utilisation as a fraction of Qwen's 32k limit.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_AUDIO, MODALITY_TEXT
from qwen_engram_bolt import QwenEngramBolt, build_fixed_context, MODEL_ID, DEVICE

torch.manual_seed(42); np.random.seed(42)


def time_path_a_query(bolt, tok, n_warmup=5, n_iter=30, T=24):
    """One Path A query = 1 forward pass at constant context (T=24)."""
    code = 0
    inp, mids = build_fixed_context(code, MODALITY_AUDIO, tok, marker_text_id=0, T=T)
    inp = inp.to(DEVICE); mids = mids.to(DEVICE)
    for _ in range(n_warmup):
        with torch.no_grad():
            bolt(inp, mids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            bolt(inp, mids)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def time_rag_query(qwen, tok, N, n_warmup=1, n_iter=3, extra_per_id=2,
                    context_limit=32000):
    """RAG query: cosine NN + 1 LM forward at context = base + N × extra_per_id tokens.

    Returns (query_ms, context_tokens, status) where status is 'ok' / 'truncated' / 'oom'.
    """
    D = 192
    embs = np.random.randn(N, D).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    q = np.random.randn(1, D).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9

    base = tok.encode("Memory:", add_special_tokens=False)
    query = tok.encode(" Query: who is this?", add_special_tokens=False)
    target_tokens = N * extra_per_id
    full_ids = base + [42] * target_tokens + query

    status = "ok"
    if len(full_ids) > context_limit:
        # Truncate
        avail = context_limit - len(base) - len(query)
        full_ids = base + [42] * max(0, avail) + query
        status = f"truncated_to_{context_limit}"

    inp = torch.tensor([full_ids], dtype=torch.long).to(DEVICE)
    attn = torch.ones_like(inp)
    try:
        # Warmup
        for _ in range(n_warmup):
            _ = embs @ q.T
            with torch.no_grad():
                qwen(input_ids=inp, attention_mask=attn, use_cache=False)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = embs @ q.T  # cosine NN
            with torch.no_grad():
                qwen(input_ids=inp, attention_mask=attn, use_cache=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / n_iter * 1000
        torch.cuda.empty_cache()
        return elapsed_ms, len(full_ids), status
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan"), len(full_ids), "oom"


def memory_cost_per_id_kb(D_emb=192, dtype_bytes=4):
    """RAG: store the raw encoder embedding per ID."""
    return D_emb * dtype_bytes / 1024  # KB


def main():
    print("=" * 75)
    print("Extreme-scale latency — Path A (parametric, O(1)) vs RAG (context-grow)")
    print("=" * 75)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()
    ctx_limit = getattr(qwen.config, "max_position_embeddings", 32768)
    print(f"  context window: {ctx_limit} tokens")

    bolt = QwenEngramBolt(qwen, tok, V_vis=64, V_aud=64,
                            engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    path_a_ms = time_path_a_query(bolt, tok)
    print(f"\nPath A query: {path_a_ms:.2f} ms (constant in N)")
    bolt.remove_hook()  # remove for RAG forward to avoid hook context mismatch

    # Path A storage: K rows × (n_heads × dim) + hashes (negligible).
    # Approximate: K=64, max_ngram=3, head=4, emb=128 → ~32 K params per layer.
    # Total bolt's trainable: ~5M params ≈ 10 MB. Amortised across all IDs (O(1) per ID).
    print(f"  Path A table size: ~10 MB (independent of N)")
    print(f"  RAG per-ID storage: {memory_cost_per_id_kb():.2f} KB raw embedding")

    Ns = [10, 100, 1000, 5000, 10000, 25000, 50000]

    print(f"\n{'N':>6} | {'ctx tokens':>11} | {'RAG ms':>10} | {'Path A ms':>10} | "
          f"{'speedup':>8} | {'status':>16}")
    print("-" * 85)
    results = {"path_a_query_ms": path_a_ms, "context_limit": ctx_limit,
                "rag_with_context": {}}
    for N in Ns:
        rag_ms, ctx_n, status = time_rag_query(qwen, tok, N, context_limit=ctx_limit)
        if np.isnan(rag_ms):
            speedup_str = "OOM"
        else:
            sp = rag_ms / path_a_ms
            speedup_str = f"{sp:>6.1f}x" if sp > 1.0 else f"{sp:>6.2f}x"
        print(f"{N:>6} | {ctx_n:>11d} | {rag_ms:>10.2f} | {path_a_ms:>10.2f} | "
              f"{speedup_str:>8} | {status:>16}")
        results["rag_with_context"][N] = {
            "query_ms": (float(rag_ms) if not np.isnan(rag_ms) else None),
            "context_tokens": ctx_n,
            "status": status,
        }

    out = Path("/home/ubuntu/multimodal-user-memory/results/latency_extreme_scale.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[done] {out}")

    # Headline table — "session cost" at N=1000 and N=10000
    print("\n" + "=" * 75)
    print("HEADLINE — Path A's structural advantage at very large N")
    print("=" * 75)
    print(f"  Path A: {path_a_ms:.1f} ms per query at ANY N (table is O(K), not O(N))")

    for N in [1000, 5000, 10000]:
        r = results["rag_with_context"].get(N)
        if not r: continue
        rag_ms = r["query_ms"]
        if rag_ms is None:
            print(f"  N={N:>6}: RAG = OOM (context {r['context_tokens']} tokens)")
            continue
        print(f"  N={N:>6}: RAG = {rag_ms:.1f} ms (context {r['context_tokens']} tokens, "
              f"{100*r['context_tokens']/ctx_limit:.0f}% of window)  → Path A is "
              f"{rag_ms/path_a_ms:.1f}× faster")

    # Session cost: 1 insertion + 1000 queries
    path_a_insert_ms = 1500  # from #5 latency benchmark, conservative
    print(f"\n  Session: 1 insert + 1000 queries (Path A insert ~1.5s, includes Engram SGD)")
    for N in [1000, 10000]:
        r = results["rag_with_context"].get(N)
        if not r or r["query_ms"] is None: continue
        p_a_total = path_a_insert_ms + 1000 * path_a_ms
        rag_total = 0 + 1000 * r["query_ms"]  # RAG insert is cheap
        print(f"    N={N:>5}: Path A total {p_a_total/1000:.1f}s, "
              f"RAG total {rag_total/1000:.1f}s → Path A is "
              f"{rag_total/p_a_total:.1f}× faster end-to-end")


if __name__ == "__main__":
    sys.exit(main())
