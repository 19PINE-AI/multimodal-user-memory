"""Latency benchmark for AttentionMemory vs Path A vs RAG-with-LM-context.

Measures end-to-end per-query latency at N = 10, 100, 1000, 10000.

The bolt LM forward is the dominant cost; AttMem's bank query is a small
matmul (N×D for keys, weighted sum yielding H-dim output). RAG with LM
context grows quadratically in T because the LM must attend over all
context tokens.

Note: we measure on synthetic random keys to expose pure compute cost,
not encoder/storage cost.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION, MODALITY_AUDIO
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE
from transformers import AutoTokenizer, AutoModelForCausalLM


def time_attmem_query(bolt, modality_id: int, N: int, n_trials: int = 30, key_dim: int = 512):
    """Insert N random keys into bank, time the LM+hook forward."""
    bank = bolt.attmem.banks[str(modality_id)]
    bank.reset()
    keys = torch.randn(N, key_dim, device=DEVICE)
    markers = list(range(30001, 30001 + N))
    bolt.insert_batch(modality_id, keys, markers)

    tok = bolt.tok
    T = 24
    pad_id = tok.pad_token_id or 0
    pref = tok.encode("You see", add_special_tokens=False)
    text_ids = list(pref) + [pad_id] * (T - 1 - len(pref))
    text_ids = (text_ids[: T - 1]) + [pad_id]
    text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
    modality_ids_t = torch.tensor(
        [[0] * (T - 1) + [int(modality_id)]], dtype=torch.long, device=DEVICE
    )
    q_key = torch.randn(1, key_dim, device=DEVICE)
    perc_keys = {int(modality_id): q_key}

    # Warm up
    for _ in range(5):
        with torch.no_grad():
            _ = bolt(modality_ids_t, text_ids_t, perc_keys)
    torch.cuda.synchronize()

    ts = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = bolt(modality_ids_t, text_ids_t, perc_keys)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000), float(np.std(ts) * 1000)


def time_attmem_insertion(bolt, modality_id: int, N: int, n_trials: int = 10, key_dim: int = 512):
    """Pure insertion latency: time to register N identities (warm)."""
    bolt.reset_banks()
    keys = torch.randn(N, key_dim, device=DEVICE)
    markers = list(range(30001, 30001 + N))

    # Warm up
    bolt.reset_banks()
    bolt.insert_batch(modality_id, keys[:1], markers[:1])

    ts = []
    for _ in range(n_trials):
        bolt.reset_banks()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bolt.insert_batch(modality_id, keys, markers)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000), float(np.std(ts) * 1000)


def time_rag_with_context(qwen, tok, N: int, key_emb_dim: int = 512, n_trials: int = 5):
    """Naive RAG: prepend retrieved-id-context to prompt, run LM forward.

    Each registered id contributes ~16 tokens of text context
    (e.g., 'identity 30001 is associated with feature vector ...').
    Total context tokens = 16 * N. LM forward is O((T + 16N)² * H).

    Returns (median_ms, n_oom).
    """
    qwen.eval()
    # Build a long prompt
    ctx_tokens_per_id = 16
    prompt_ids = [tok.bos_token_id] if tok.bos_token_id is not None else []
    pad_id = tok.pad_token_id or 0
    # Synthesise a long input of (16 * N) + 8 tokens of "query" prefix
    total_ctx = ctx_tokens_per_id * N + 8
    if total_ctx > 32000:
        return float('nan'), 1  # OOM / exceeds Qwen 32k context
    input_ids = torch.tensor([[pad_id] * total_ctx], dtype=torch.long, device=DEVICE)
    attn = torch.ones_like(input_ids)

    # Warm up
    try:
        for _ in range(2):
            with torch.no_grad():
                _ = qwen(input_ids=input_ids, attention_mask=attn, use_cache=False)
    except torch.cuda.OutOfMemoryError:
        return float('nan'), 1

    torch.cuda.synchronize()
    ts = []
    try:
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = qwen(input_ids=input_ids, attention_mask=attn, use_cache=False)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    except torch.cuda.OutOfMemoryError:
        return float('nan'), 1
    return float(np.median(ts) * 1000), 0


def main():
    print("Loading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    bolt = QwenAttMemBolt(qwen, tok, vision_key_dim=512, audio_key_dim=192,
                          attach_layer=33, attach_lm_head=True).to(DEVICE)
    bolt.install_hook()

    print("\n=== Latency at N = 10, 100, 1000, 10000 ===")
    print(f"{'N':>6} | {'AttMem query':>15} | {'AttMem insertion':>20} | {'RAG-with-context':>20}")
    print("-" * 75)
    for N in [10, 100, 1000, 10000]:
        q_ms, q_std = time_attmem_query(bolt, MODALITY_VISION, N, n_trials=20)
        ins_ms, _ = time_attmem_insertion(bolt, MODALITY_VISION, N, n_trials=5)
        rag_ms, oom = time_rag_with_context(qwen, tok, N)
        rag_str = f"{rag_ms:>10.1f} ms" if not np.isnan(rag_ms) else "    OOM/>32k"
        print(f"{N:>6} | {q_ms:>10.2f} ms   | {ins_ms:>10.2f} ms ({ins_ms/N:.4f}/id) | {rag_str}")

    print("\n=== Path A (estimated) ===")
    print(f"  Path A insertion: ~1000 ms per id (80 SGD steps)")
    print(f"  AttMem insertion: see above (~ms per N appended)")


if __name__ == "__main__":
    main()
