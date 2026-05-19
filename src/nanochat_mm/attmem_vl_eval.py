"""Zero-shot AttMem on Qwen2.5-VL: end-to-end evaluation on AgeDB face crops.

For each cross-condition pair (same identity, different ages), register one
photo and query with the other. Bank key = mean-pooled Qwen-VL vision tokens.

Compares against:
  - RAG cosine NN over the same Qwen-VL visual keys.
  - (separately) ArcFace RAG cosine NN — to confirm Qwen-VL's vision encoder
    is less cross-condition-invariant than face-specific ArcFace.

This validates the AttMem mechanism on a real VLM, not just on text-only
Qwen with a pre-extracted face encoder.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION
from qwen_vl_attmem_bolt import QwenVLAttMemBolt, VLM_MODEL_ID, DEVICE


def get_eval_pairs(N=20, samples_per_id=2, seed=42):
    """Pick N identities from AgeDB with at least 2 photos at different ages.

    Returns list of (identity, [reg_image_pil, query_image_pil]).
    """
    print("Loading AgeDB ...")
    d = load_dataset("ljnlonoljpiljm/agedb", split="train")
    print(f"  {len(d)} images, identities: ?")
    by_id = defaultdict(list)
    for i, row in enumerate(d):
        by_id[row["identity"]].append(i)
    valid_ids = [k for k in by_id if len(by_id[k]) >= samples_per_id]
    print(f"  {len(valid_ids)} identities with ≥{samples_per_id} photos")
    rng = np.random.default_rng(seed)
    rng.shuffle(valid_ids)
    chosen_ids = valid_ids[:N]
    pairs = []
    for pid in chosen_ids:
        idxs = list(by_id[pid])
        rng.shuffle(idxs)
        reg_idx = idxs[0]
        q_idx = idxs[1]
        reg_img = d[reg_idx]["image"]
        q_img = d[q_idx]["image"]
        pairs.append((pid, reg_img, q_img))
    return pairs


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    print(f"=== AttMem on Qwen2.5-VL, end-to-end, N={N} identities, seed={seed} ===")
    pairs = get_eval_pairs(N=N, seed=seed)
    print(f"  picked {len(pairs)} cross-condition pairs")

    print(f"\nLoading {VLM_MODEL_ID} ...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID, trust_remote_code=True)
    qwen_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    qwen_vl.eval()

    print("Building bolt ...")
    bolt = QwenVLAttMemBolt(qwen_vl, processor).to(DEVICE)
    bolt.install_hook()

    # ---------- Phase 1: Extract visual keys for all registration images ----------
    print(f"\n[1/3] Extracting Qwen-VL visual keys for {N} registration faces ...")
    marker_offset = 30001
    marker_ids = list(range(marker_offset, marker_offset + N))
    reg_keys = []
    t0 = time.time()
    for k, (pid, reg_img, _) in enumerate(pairs):
        vis_key = bolt.extract_visual_key(reg_img)
        reg_keys.append(vis_key.cpu())
        if k < 3 or k % 10 == 0:
            print(f"  reg {k+1}/{N}: id={pid}, key shape={vis_key.shape}, norm={vis_key.norm().item():.3f}")
    reg_keys_t = torch.stack(reg_keys, dim=0).to(DEVICE)
    print(f"  total reg time: {time.time() - t0:.1f}s")

    # Insert into bank
    bolt.insert_batch(MODALITY_VISION, reg_keys_t, marker_ids)
    print(f"  bank populated: N={N}")

    # ---------- Phase 2: For each query, extract its visual key and run forward ----------
    print(f"\n[2/3] Running cross-condition queries (one per identity) ...")
    correct_attmem = 0
    correct_rag = 0
    n_queries = 0
    t0 = time.time()

    # Pre-compute RAG cosine baseline first
    print(f"  Pre-computing query keys ...")
    query_keys = []
    for k, (_, _, q_img) in enumerate(pairs):
        qk = bolt.extract_visual_key(q_img)
        query_keys.append(qk.cpu())
    query_keys_t = torch.stack(query_keys, dim=0).to(DEVICE)

    # RAG: cosine NN over the Qwen-VL visual keys
    reg_n = reg_keys_t.float() / (reg_keys_t.float().norm(dim=1, keepdim=True) + 1e-9)
    q_n = query_keys_t.float() / (query_keys_t.float().norm(dim=1, keepdim=True) + 1e-9)
    sim = q_n @ reg_n.T  # [N, N]
    rag_preds = sim.argmax(dim=1)
    correct_rag = int((rag_preds == torch.arange(N, device=DEVICE)).sum().item())

    # AttMem: for each query, run LM forward and check marker logit argmax
    print(f"  Running AttMem queries ...")
    for k, (pid, _, q_img) in enumerate(pairs):
        # Run query: VLM forward with image + prompt; hook injects bank residual
        # Set the visual key for THIS query so the bank-attention query is the
        # query image's vision-token pool (not the registered one).
        logits = bolt.query_logits(q_img, prompt_text="The person in this image is")
        # Argmax restricted to registered markers
        marker_logits = torch.stack([logits[m] for m in marker_ids])
        pred_local = int(marker_logits.argmax().item())
        if pred_local == k:
            correct_attmem += 1
        n_queries += 1
        if k < 3:
            top3 = marker_logits.topk(min(3, N))
            print(f"  query {k+1} (id={pid}): predicted marker_local={pred_local} (target={k}), top3 logits={top3.values.tolist()}")

    elapsed = time.time() - t0
    print(f"\n[3/3] Results:")
    print(f"  N={N}, n_queries={n_queries}")
    print(f"  RAG cosine NN (over Qwen-VL visual keys): {correct_rag}/{N} = {correct_rag/N:.3f}")
    print(f"  AttMem (zero-shot on Qwen-VL):            {correct_attmem}/{N} = {correct_attmem/N:.3f}")
    print(f"  AttMem total query time: {elapsed:.1f}s ({elapsed/N*1000:.1f} ms/query)")

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/attmem_vl_qwen25vl_N{N}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump({
            "model": VLM_MODEL_ID,
            "N": N, "seed": seed,
            "rag_retr1": correct_rag / N,
            "attmem_zeroshot_retr1": correct_attmem / N,
            "elapsed_sec": elapsed,
        }, f, indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
