"""Mechanism analysis: probe what AttMem is actually doing.

For each held-out query at N=10:
 1. Compute the bank-attention weights (softmax over the 10 registered keys).
   Show that the right ID typically gets the highest weight — but how peaked?
 2. Compute the marker-logit boost from the residual injection.
   How much of the boost lands on the *correct* marker vs on neighbours?
 3. Compare to the encoder-cosine ranking (what RAG would do).
   How often does AttMem disagree with cosine NN — and when it does,
   does it produce a better answer or a worse one?

Output: a JSON of attention weight matrices + logit deltas + agreement stats,
       plus a visualisation as a PNG.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION, MODALITY_TEXT
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE
from v2_retrieval import split_by_identity


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    torch.manual_seed(seed); np.random.seed(seed)

    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    d = np.load(EMB / "arcface_face_xxxl.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    _, _, ev_emb, ev_pid = split_by_identity(emb, pid)

    by_id = defaultdict(list)
    for i, p in enumerate(ev_pid):
        by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())[:N]

    print(f"Loading Qwen2.5-3B + AttMem bolt ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()
    bolt = QwenAttMemBolt(qwen, tok, vision_key_dim=512, audio_key_dim=192,
                          attach_layer=33, attach_lm_head=True).to(DEVICE)
    bolt.install_hook()

    # Insert: one registration sample per id, marker tokens 30001..30001+N-1
    marker_offset = 30001
    marker_ids = list(range(marker_offset, marker_offset + N))
    rng = np.random.default_rng(99)
    reg_idx_per_id = []
    for pid_str in ids_sorted:
        idxs = list(by_id[pid_str]); rng.shuffle(idxs)
        reg_idx_per_id.append(idxs[0])
    reg_emb_np = ev_emb[reg_idx_per_id].astype(np.float32)
    reg_emb_n  = reg_emb_np / (np.linalg.norm(reg_emb_np, axis=1, keepdims=True) + 1e-9)
    reg_keys = torch.from_numpy(reg_emb_np).to(DEVICE)
    bolt.insert_batch(MODALITY_VISION, reg_keys, marker_ids)

    # For each ID, run one cross-condition query and capture attention + logit info
    T = 24
    pad_id = tok.pad_token_id or 0
    pref = tok.encode("You see", add_special_tokens=False)
    text_ids = list(pref) + [pad_id] * (T - 1 - len(pref))
    text_ids = (text_ids[: T - 1]) + [pad_id]
    text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
    modality_ids_t = torch.tensor(
        [[MODALITY_TEXT] * (T - 1) + [int(MODALITY_VISION)]], dtype=torch.long, device=DEVICE
    )

    attention_matrix = np.zeros((N, N))
    cosine_matrix = np.zeros((N, N))
    marker_logit_matrix = np.zeros((N, N))
    rag_argmax = np.zeros(N, dtype=int)
    attmem_argmax = np.zeros(N, dtype=int)

    bank = bolt.attmem.banks[str(MODALITY_VISION)]

    for k, pid_str in enumerate(ids_sorted):
        idxs = list(by_id[pid_str]); rng.shuffle(idxs)
        q_idxs = [i for i in idxs if i != reg_idx_per_id[k]][:1]
        if not q_idxs: continue
        q_idx = q_idxs[0]
        q_emb_np = ev_emb[q_idx].astype(np.float32)
        q_emb_n  = q_emb_np / (np.linalg.norm(q_emb_np) + 1e-9)
        # Cosine to each registered key
        cos = reg_emb_n @ q_emb_n
        cosine_matrix[k] = cos
        rag_argmax[k] = int(np.argmax(cos))

        q_key = torch.from_numpy(q_emb_np).unsqueeze(0).to(DEVICE)
        # Compute the actual bank attention weights for this query
        with torch.no_grad():
            keys = bank.keys.float()  # [N, D]
            q_n = F.normalize(torch.from_numpy(q_emb_n).to(DEVICE).float().unsqueeze(0), dim=-1)
            inv_temp = torch.exp(bank.log_inv_temp).clamp_max(500.0)
            logits = (q_n @ keys.T) * inv_temp
            weights = F.softmax(logits, dim=-1)
            attention_matrix[k] = weights[0].cpu().numpy()

        # Forward pass through LM to get marker logits
        with torch.no_grad():
            lm_logits = bolt(modality_ids_t, text_ids_t, {int(MODALITY_VISION): q_key})
            last = lm_logits[0, -1, :]
            mlogits = torch.stack([last[m] for m in marker_ids]).float().cpu().numpy()
            marker_logit_matrix[k] = mlogits
            attmem_argmax[k] = int(np.argmax(mlogits))

    # Stats
    cosine_correct = int((rag_argmax == np.arange(N)).sum())
    attmem_correct = int((attmem_argmax == np.arange(N)).sum())
    agreement = int((rag_argmax == attmem_argmax).sum())
    # Where AttMem and RAG disagree — who's right?
    disagree_idx = np.where(rag_argmax != attmem_argmax)[0]
    attmem_better = sum(1 for k in disagree_idx if attmem_argmax[k] == k and rag_argmax[k] != k)
    cosine_better = sum(1 for k in disagree_idx if rag_argmax[k] == k and attmem_argmax[k] != k)

    print(f"\n=== Stats (N={N}) ===")
    print(f"  RAG cosine retr@1:     {cosine_correct}/{N}")
    print(f"  AttMem retr@1:         {attmem_correct}/{N}")
    print(f"  Agreement (RAG=AttMem): {agreement}/{N}")
    print(f"  Disagreements: {len(disagree_idx)}")
    print(f"    AttMem right, RAG wrong: {attmem_better}")
    print(f"    RAG right, AttMem wrong: {cosine_better}")

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    # Panel 1: attention weights
    im = axes[0].imshow(attention_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("(a) Bank attention weights\nrow=query, col=registered key")
    axes[0].set_xlabel("registered key index")
    axes[0].set_ylabel("query index")
    # Mark diagonal (correct match)
    for i in range(N):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                               edgecolor="#00aa00", linewidth=1.2)
        axes[0].add_patch(rect)
    plt.colorbar(im, ax=axes[0], fraction=0.04)

    # Panel 2: cosine similarities
    im2 = axes[1].imshow(cosine_matrix, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("(b) Encoder-cosine similarity\nrow=query, col=registered key")
    axes[1].set_xlabel("registered key index")
    axes[1].set_ylabel("query index")
    for i in range(N):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                               edgecolor="#00aa00", linewidth=1.2)
        axes[1].add_patch(rect)
    plt.colorbar(im2, ax=axes[1], fraction=0.04)

    # Panel 3: marker logits (normalised per row)
    ml_norm = marker_logit_matrix - marker_logit_matrix.min(axis=1, keepdims=True)
    ml_norm = ml_norm / (ml_norm.max(axis=1, keepdims=True) + 1e-9)
    im3 = axes[2].imshow(ml_norm, cmap="Purples", vmin=0, vmax=1, aspect="auto")
    axes[2].set_title("(c) LM marker logits\n(row-normalised)")
    axes[2].set_xlabel("marker index")
    axes[2].set_ylabel("query index")
    for i in range(N):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                               edgecolor="#00aa00", linewidth=1.2)
        axes[2].add_patch(rect)
    plt.colorbar(im3, ax=axes[2], fraction=0.04)

    plt.tight_layout()
    outp = Path("/home/ubuntu/multimodal-user-memory/paper/figs/fig9_mechanism.pdf")
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  -> wrote {outp}")

    # Save the data as JSON for paper
    out_json = Path("/home/ubuntu/multimodal-user-memory/results/attmem_mechanism_analysis.json")
    with open(out_json, "w") as f:
        json.dump({
            "N": N,
            "seed": seed,
            "ids": ids_sorted,
            "cosine_correct": cosine_correct,
            "attmem_correct": attmem_correct,
            "agreement": agreement,
            "attmem_right_rag_wrong": attmem_better,
            "rag_right_attmem_wrong": cosine_better,
            "attention_matrix_diag_mean": float(np.mean(np.diag(attention_matrix))),
            "cosine_matrix_diag_mean": float(np.mean(np.diag(cosine_matrix))),
        }, f, indent=2, default=str)
    print(f"  -> wrote {out_json}")


if __name__ == "__main__":
    main()
