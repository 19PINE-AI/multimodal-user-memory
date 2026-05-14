"""End-to-end Perceptual Engram retrieval test.

This is the test that gates the entire method. Given:
  - A frozen perceptual encoder (ArcFace / ECAPA-TDNN)
  - A frozen quantiser (naive k-means, the §5 winner)
  - A hash-keyed parametric memory table

Can we register N identities one-shot and retrieve them correctly from
new (cross-condition) examples?

Protocol:
  1. Fit k-means codebook on TRAIN identities (not used for retrieval).
  2. Split HELDOUT identities further:
       - 1 example/identity → REGISTRATION set (the "first time the agent
         meets you")
       - remaining examples → QUERY set (the "agent sees you again
         under different conditions")
  3. Register: for each registration example, compute its code via the
     frozen codebook, insert into the Engram table with the identity
     label as payload.
  4. Query: for each query example, compute code, look up the table.
     Score retrieval@1 (and top-K with chained-slot disambiguation).
  5. Sweep N (number of registered identities) to measure scaling.

Collision-handling strategies tested:
  - 'first-write-wins': only the first identity registered at a hash slot
    stays; later identities silently displace nothing (we record the
    miss).
  - 'chained-with-disambiguation': store list of (code, label, ref_emb)
    per slot; retrieval re-ranks the list by cosine similarity of the
    query embedding to ref_emb. This is what a real per-user Engram
    would do.

The chained variant is the actually-fair comparison vs. embedding RAG —
it's still parametric per-user with O(1) hash lookup at the slot, but
within-slot disambiguation uses the embedding. Win condition: chained
variant matches embedding RAG accuracy at much higher scale (many more
slots than entities).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from learned_rqvae import (
    extract_or_load_audio_embeddings,
    extract_or_load_vision_embeddings,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def split_train_eval_by_identity(emb, pid, train_frac=0.5):
    rng = np.random.RandomState(SEED)
    unique = sorted(set(pid.tolist()))
    rng.shuffle(unique)
    n_train = int(len(unique) * train_frac)
    train_ids = set(unique[:n_train])
    train_mask = np.array([str(p) in train_ids for p in pid])
    return emb[train_mask], pid[train_mask], emb[~train_mask], pid[~train_mask]


def fit_naive_rq(train_emb, n_levels, k_per):
    """Fit residual k-means codebook on train_emb. Return apply_fn."""
    import faiss
    D = train_emb.shape[1]
    centroids = []
    residual = train_emb.copy()
    for L in range(n_levels):
        km = faiss.Kmeans(D, k_per, niter=20, verbose=False, seed=SEED + L)
        km.train(residual)
        _, c = km.index.search(residual, 1)
        c = c.squeeze(1)
        centroids.append(km.centroids.copy())
        residual = residual - km.centroids[c]

    def apply(emb_np):
        r = emb_np.copy()
        codes = np.zeros((len(emb_np), n_levels), dtype=np.int64)
        for L, c_arr in enumerate(centroids):
            d2 = (r ** 2).sum(1, keepdims=True) - 2 * r @ c_arr.T + (c_arr ** 2).sum(1)
            idx = d2.argmin(1)
            codes[:, L] = idx
            r = r - c_arr[idx]
        return codes
    return apply


class PerceptualEngram:
    """Hash-keyed parametric memory.

    Key = tuple(code) from the quantiser.
    Row = list of (label, reference_embedding) — supports chained disambiguation.
    """
    def __init__(self):
        self.table = defaultdict(list)

    def insert(self, code_tuple, label, ref_emb):
        self.table[code_tuple].append((label, ref_emb.astype(np.float32)))

    def lookup_first(self, code_tuple):
        """First-write-wins: just return the first label at the slot."""
        rows = self.table.get(code_tuple)
        if rows is None:
            return None
        return rows[0][0]

    def lookup_chained(self, code_tuple, query_emb):
        """Within-slot disambiguation via cosine similarity."""
        rows = self.table.get(code_tuple)
        if rows is None:
            return None, 0
        if len(rows) == 1:
            return rows[0][0], 1
        # Re-rank within slot by cosine sim
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        sims = []
        for (lbl, ref) in rows:
            r = ref / (np.linalg.norm(ref) + 1e-9)
            sims.append(float(q @ r))
        best = int(np.argmax(sims))
        return rows[best][0], len(rows)

    def slot_occupancy(self):
        return [len(v) for v in self.table.values()]

    def num_slots(self):
        return len(self.table)


def per_user_engram_retrieval(eval_emb, eval_pid, apply_codebook, N_subset=None, n_queries_per_id=None):
    """Run the registration + retrieval protocol on eval_emb.

    Args:
      eval_emb, eval_pid: held-out embeddings (eval split).
      apply_codebook: function emb -> codes (shape [N, L]).
      N_subset: register only first N identities (sorted). None = all.
      n_queries_per_id: cap queries per identity for fairness.

    Returns dict of metrics.
    """
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid):
        by_id[str(p)].append(i)

    # Sort identities for reproducibility
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None:
        ids_sorted = ids_sorted[:N_subset]

    # Build registration & query sets
    eng = PerceptualEngram()
    queries = []  # list of (query_emb, ground_truth_label)
    rng = np.random.RandomState(SEED)

    for pid in ids_sorted:
        idxs = by_id[pid]
        if len(idxs) < 2:
            continue
        rng.shuffle(idxs)
        reg_idx = idxs[0]
        query_idxs = idxs[1:]
        if n_queries_per_id is not None:
            query_idxs = query_idxs[:n_queries_per_id]
        # Register
        reg_emb = eval_emb[reg_idx]
        reg_code = tuple(apply_codebook(reg_emb[None])[0].tolist())
        eng.insert(reg_code, pid, reg_emb)
        # Queue queries
        for qi in query_idxs:
            queries.append((eval_emb[qi], pid))

    # Compute query codes (batch)
    if not queries:
        return None
    q_embs = np.stack([q[0] for q in queries])
    q_codes = apply_codebook(q_embs)
    q_labels = [q[1] for q in queries]

    # Score retrieval
    correct_first = 0
    correct_chained = 0
    missed = 0
    chained_slot_sizes = []
    for k in range(len(queries)):
        ct = tuple(q_codes[k].tolist())
        pred1 = eng.lookup_first(ct)
        if pred1 is None:
            missed += 1
        elif pred1 == q_labels[k]:
            correct_first += 1
        pred_c, slot_size = eng.lookup_chained(ct, q_embs[k])
        if pred_c is None:
            pass  # already counted as miss
        elif pred_c == q_labels[k]:
            correct_chained += 1
        if pred_c is not None:
            chained_slot_sizes.append(slot_size)

    return {
        "N_registered": len(ids_sorted),
        "N_queries": len(queries),
        "retrieval_first_write_wins": correct_first / len(queries),
        "retrieval_chained_disambig": correct_chained / len(queries),
        "missed_no_slot_hit": missed / len(queries),
        "num_slots_used": eng.num_slots(),
        "mean_slot_occupancy": float(np.mean(eng.slot_occupancy())) if eng.slot_occupancy() else 0.0,
        "max_slot_occupancy": int(np.max(eng.slot_occupancy())) if eng.slot_occupancy() else 0,
        "mean_query_slot_size": float(np.mean(chained_slot_sizes)) if chained_slot_sizes else 0.0,
    }


def embedding_rag_baseline(eval_emb, eval_pid, N_subset=None, n_queries_per_id=None):
    """Strong baseline: cosine-similarity NN over registered embeddings.
    This is what production face-/voice-recognition systems do; the Engram
    table is competitive only if it can match this at lower context cost
    and lower per-query latency (O(1) hash vs O(N) cosine).
    """
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid):
        by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None:
        ids_sorted = ids_sorted[:N_subset]
    reg_embs = []
    reg_labels = []
    queries = []
    rng = np.random.RandomState(SEED)
    for pid in ids_sorted:
        idxs = by_id[pid]
        if len(idxs) < 2:
            continue
        rng.shuffle(idxs)
        reg_embs.append(eval_emb[idxs[0]])
        reg_labels.append(pid)
        query_idxs = idxs[1:]
        if n_queries_per_id is not None:
            query_idxs = query_idxs[:n_queries_per_id]
        for qi in query_idxs:
            queries.append((eval_emb[qi], pid))
    if not queries:
        return None
    reg_M = np.stack(reg_embs).astype(np.float32)
    q_M = np.stack([q[0] for q in queries]).astype(np.float32)
    # cosine
    reg_M /= np.linalg.norm(reg_M, axis=1, keepdims=True) + 1e-9
    q_M /= np.linalg.norm(q_M, axis=1, keepdims=True) + 1e-9
    sims = q_M @ reg_M.T
    pred = sims.argmax(axis=1)
    correct = sum(1 for k in range(len(queries)) if reg_labels[pred[k]] == queries[k][1])
    return correct / len(queries)


def main():
    cache_dir = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    results = {}

    # Codebook configs to test, plus retrieval scales
    configs = [(1, 32), (1, 128), (2, 16), (2, 32), (2, 64), (3, 16)]
    # For LFW/LibriSpeech we have ~20 held-out IDs total — scan up to that
    scales = [5, 10, 20]
    # n_queries_per_id capped so scale comparisons are fair
    nq = 5

    for modality, extractor in [
        ("audio", lambda: extract_or_load_audio_embeddings(cache_dir / "ecapa_libri.npz")),
        ("vision", lambda: extract_or_load_vision_embeddings(cache_dir / "arcface_lfw.npz")),
    ]:
        print(f"\n{'=' * 75}\n{modality.upper()} — Perceptual Engram end-to-end retrieval\n{'=' * 75}")
        emb, pid = extractor()
        train_emb, train_pid, eval_emb, eval_pid = split_train_eval_by_identity(emb, pid)
        print(f"  Train: {len(train_emb)} embs / {len(set(train_pid))} ids (codebook only)")
        print(f"  Eval:  {len(eval_emb)} embs / {len(set(eval_pid))} ids (register + query)")

        # Baseline embedding RAG at each scale (no quantiser dependence)
        print(f"\n  [embedding-RAG cosine NN baseline]")
        rag_results = {}
        for N in scales:
            r = embedding_rag_baseline(eval_emb, eval_pid, N_subset=N, n_queries_per_id=nq)
            print(f"    N={N:>3d}  retrieval@1 = {r:.4f}")
            rag_results[N] = r

        modality_results = {"embedding_rag": rag_results, "engram": {}}

        for n_levels, k_per in configs:
            cfg = f"L{n_levels}_K{k_per}"
            eff_K = k_per ** n_levels
            print(f"\n  [Perceptual Engram | codebook {n_levels}×{k_per} = eff_K {eff_K}]")
            apply = fit_naive_rq(train_emb, n_levels, k_per)
            cfg_results = {}
            for N in scales:
                m = per_user_engram_retrieval(eval_emb, eval_pid, apply, N_subset=N, n_queries_per_id=nq)
                if m is None:
                    continue
                cfg_results[N] = m
                print(f"    N={N:>3d}  ret@1(first)={m['retrieval_first_write_wins']:.4f}  ret@1(chained)={m['retrieval_chained_disambig']:.4f}  "
                      f"miss={m['missed_no_slot_hit']:.4f}  slots_used={m['num_slots_used']}  mean_slot_occ={m['mean_slot_occupancy']:.2f}  max_occ={m['max_slot_occupancy']}")
            modality_results["engram"][cfg] = cfg_results

        results[modality] = modality_results

    out = Path("/home/ubuntu/multimodal-user-memory/results/engram_retrieval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] Wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
