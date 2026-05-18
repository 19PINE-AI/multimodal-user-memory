"""MyVLM-style baseline: train a per-concept binary classifier head over the
ArcFace encoder embeddings, then at recall time pick the concept whose head
fires strongest.

This is the most-charitable reimplementation of MyVLM's CORE mechanism
adapted to our task. The original MyVLM has additional vision-language
training components, but on the V-XC-ID-XXXL cross-condition retrieval task
those don't add anything — the heads are linear classifiers over the same
encoder our embedding-RAG baseline uses.

Cost model:
  - Insertion of new concept: train a small classifier (~5K params) for
    ~100 SGD steps. ~1 s per concept on H100-class GPU.
  - Query: forward the encoder embedding through N classifiers, take argmax.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from v2_retrieval import split_by_identity


def train_per_concept_heads(reg_emb, n_concepts, sgd_steps=100, lr=1e-2,
                              n_neg_samples=64, device="cuda"):
    """Train N binary heads, each classifying its concept vs all-others.

    Returns: head weights W [N, D] (no bias — same as ArcFace's prototype layer).
    """
    D = reg_emb.shape[1]
    # Each head: a single linear projection. Initialise at the prototype
    # (mean) of its concept's reg samples — equivalent to cosine NN at init.
    W = reg_emb.copy().astype(np.float32)  # [N, D]
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

    # Train with random negatives drawn from the registered pool itself
    rng = np.random.default_rng(0)
    for step in range(sgd_steps):
        # For each concept, gradient on "make this head fire +1 on own emb,
        # fire -1 on n_neg_samples random other concepts' embs"
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
        for c in range(n_concepts):
            # Positive
            pos = reg_emb[c]
            pos_n = pos / (np.linalg.norm(pos) + 1e-9)
            # Random negatives
            neg_idx = rng.choice(n_concepts, size=min(n_neg_samples, n_concepts - 1), replace=False)
            neg_idx = neg_idx[neg_idx != c]
            if len(neg_idx) == 0: continue
            negs = reg_emb[neg_idx]
            negs_n = negs / (np.linalg.norm(negs, axis=1, keepdims=True) + 1e-9)
            # Hinge-style update: push W[c] toward pos, away from negs (if confused)
            pred_pos = Wn[c] @ pos_n
            pred_neg = negs_n @ Wn[c]  # [n_neg]
            margin = 0.4
            err_pos = max(0, margin - pred_pos)
            err_neg = np.maximum(0, pred_neg - (-margin))  # we want neg < -margin
            grad = -err_pos * pos_n + (err_neg[:, None] * negs_n).sum(axis=0)
            W[c] = W[c] - lr * grad
    return W


def evaluate(W, query_emb, query_pid_local):
    """For each query, classify via argmax over per-concept heads.
    Returns retr@1."""
    W_n = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    q_n = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9)
    # Score: cosine of query with each head
    scores = q_n @ W_n.T  # [n_queries, n_concepts]
    pred = scores.argmax(axis=1)
    correct = (pred == query_pid_local).sum()
    return correct / len(query_emb)


def evaluate_adversarial(W, reg_emb, query_emb, query_pid_local, K_distractors=19, n_queries=None):
    """Adversarial eval: for each query, restrict argmax to (target + top-K
    most-cosine-similar non-matching head indices)."""
    N = W.shape[0]
    if K_distractors + 1 > N:
        K_distractors = N - 1
    W_n = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    reg_n = reg_emb / (np.linalg.norm(reg_emb, axis=1, keepdims=True) + 1e-9)
    cos_mat = reg_n @ reg_n.T
    np.fill_diagonal(cos_mat, -1)
    top_distractors = np.argsort(-cos_mat, axis=1)[:, :K_distractors]

    q_n = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9)
    all_scores = q_n @ W_n.T  # [n_queries, N]
    correct = 0
    for i, target in enumerate(query_pid_local):
        bank_ids = np.concatenate([[target], top_distractors[target]])
        bank_scores = all_scores[i, bank_ids]
        pred_bank = int(np.argmax(bank_scores))
        if pred_bank == 0:  # index 0 in bank is the target
            correct += 1
    return correct / len(query_emb)


def main():
    print("=" * 70)
    print("MyVLM-style baseline (per-concept linear heads on ArcFace embeddings)")
    print("=" * 70)

    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    d = np.load(EMB / "arcface_face_xxxl.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    _, _, ev_emb, ev_pid = split_by_identity(emb, pid)
    n_eval_ids = len(set(ev_pid.tolist()))
    print(f"  eval pool: {n_eval_ids} unique identities")

    by_id = defaultdict(list)
    for i, p in enumerate(ev_pid):
        by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())

    Ns = [N for N in [5, 10, 20, 50, 100, 300, 700, 1000] if N <= n_eval_ids]
    print(f"\n{'N':>5} | {'RAG cosine':>10} | {'MyVLM-style':>11} | {'AttMem(*)':>9} | n_queries")
    print("-" * 65)
    results = {}
    for N in Ns:
        # Register: one sample per id
        rng = np.random.default_rng(99)
        reg_idx_per_id = []
        for pid_str in ids_sorted[:N]:
            idxs = list(by_id[pid_str]); rng.shuffle(idxs)
            reg_idx_per_id.append(idxs[0])
        reg_emb = ev_emb[reg_idx_per_id].astype(np.float32)

        # Train per-concept heads (insertion cost)
        t0 = time.time()
        W = train_per_concept_heads(reg_emb, N, sgd_steps=100, lr=1e-2,
                                       n_neg_samples=min(N, 64))
        t_train = time.time() - t0

        # Eval queries: 3 cross-condition per id
        query_emb = []; query_pid_local = []
        for k, pid_str in enumerate(ids_sorted[:N]):
            idxs = list(by_id[pid_str]); rng.shuffle(idxs)
            q_idxs = [i for i in idxs if i != reg_idx_per_id[k]][:3]
            for qi in q_idxs:
                query_emb.append(ev_emb[qi])
                query_pid_local.append(k)
        query_emb = np.stack(query_emb, axis=0).astype(np.float32)
        query_pid_local = np.array(query_pid_local)

        # MyVLM-style retr@1
        retr_my = evaluate(W, query_emb, query_pid_local)

        # RAG cosine for reference
        reg_n = reg_emb / (np.linalg.norm(reg_emb, axis=1, keepdims=True) + 1e-9)
        q_n = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9)
        sim = q_n @ reg_n.T
        retr_rag = float((sim.argmax(axis=1) == query_pid_local).mean())

        # From paper for AttMem comparison
        attmem_vals = {5: 0.933, 10: 0.992, 20: 0.808, 50: 0.733, 100: 0.742,
                        300: 0.637, 700: 0.629, 1000: 0.594}
        attmem_ref = attmem_vals.get(N, float("nan"))

        print(f"{N:>5} | {retr_rag:>10.3f} | {retr_my:>11.3f} | {attmem_ref:>9.3f} | {len(query_emb)}")
        results[N] = {
            "n_queries": int(len(query_emb)),
            "rag_retr1": float(retr_rag),
            "myvlm_retr1": float(retr_my),
            "myvlm_train_seconds": float(t_train),
            "attmem_retr1_from_paper": attmem_ref,
        }

    # Adversarial eval at K=19
    print(f"\n=== Adversarial eval (target + top-19 cosine-similar distractors) ===")
    print(f"  Running on full eval pool ({n_eval_ids} IDs) ...")
    reg_idx_per_id = []
    rng = np.random.default_rng(99)
    for pid_str in ids_sorted:
        idxs = list(by_id[pid_str]); rng.shuffle(idxs)
        reg_idx_per_id.append(idxs[0])
    reg_emb = ev_emb[reg_idx_per_id].astype(np.float32)
    W = train_per_concept_heads(reg_emb, n_eval_ids, sgd_steps=100, lr=1e-2)
    query_emb = []; query_pid_local = []
    for k, pid_str in enumerate(ids_sorted):
        idxs = list(by_id[pid_str]); rng.shuffle(idxs)
        q_idxs = [i for i in idxs if i != reg_idx_per_id[k]][:3]
        for qi in q_idxs:
            query_emb.append(ev_emb[qi])
            query_pid_local.append(k)
    query_emb = np.stack(query_emb, axis=0).astype(np.float32)
    query_pid_local = np.array(query_pid_local)
    retr_adv = evaluate_adversarial(W, reg_emb, query_emb, query_pid_local, K_distractors=19)
    results["adversarial_K19"] = {"myvlm_retr1": float(retr_adv)}
    print(f"  MyVLM-style retr@1 (K=19 adversarial): {retr_adv:.3f}")
    print(f"  (Compare: RAG cosine 0.841; AttMem-standard 0.808; AttMem-advtrain 0.986)")

    out = Path("/home/ubuntu/multimodal-user-memory/results/myvlm_style_baseline.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
