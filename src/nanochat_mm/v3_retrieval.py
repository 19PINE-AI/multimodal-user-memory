"""v3.1 — row-targeted surgical insertion.

Fixes the v2 retrieval failure mode where SGD over the *entire* Engram module
trampled earlier-inserted identities. Identifies the specific rows that
hash(code) addresses and updates ONLY those rows, leaving other rows
untouched. Multiple identities can be co-registered without cross-talk so
long as their hashed row sets are disjoint (collisions on K small remain
a hard limit, but those were a hard limit in v1 too).

Key implementation: build a gradient mask over each MultiHeadEmbedding's
embedding.weight that's 1.0 only on the rows hash(code) touches. Apply
that mask to .grad after backward, then step. Standard trick.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import (
    MultimodalEngramSet, MultimodalEngramConfig,
    MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO,
)
from toy_gpt_train import ToyGPTWithEngram
from real_encoder_train import fit_naive_rq, flatten_codes
from v2_retrieval import (
    load_model, split_by_identity, build_query_context, embedding_rag_ceiling,
)

torch.manual_seed(42); np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "/home/ubuntu/multimodal-user-memory/runs/v2_toy_realencoder.pt"


def get_touched_rows(eng, code_token):
    """For a single code token, identify which rows of each layer's embedding
    table are addressed by hash(code).

    Returns: dict {layer_id_str: set of row indices in embedding.weight}.
    """
    # Build a single-position single-batch input
    input_ids_np = np.array([[code_token]], dtype=np.int64)
    hashes_per_layer = eng.hash_mapping.hash_all_layers(input_ids_np, user_salt=int(eng.user_salt))
    touched = {}
    for lid, h in hashes_per_layer.items():
        # h: [1, 1, total_heads]  → per-head local row indices
        local = h[0, 0]  # [total_heads]
        # Add the per-head offsets to get global row indices in MultiHeadEmbedding.embedding
        tbl = eng.tables[str(lid)]
        global_rows = local + tbl.offsets.cpu().numpy()  # [total_heads]
        touched[str(lid)] = set(int(r) for r in global_rows.tolist())
    return touched


def row_targeted_surgical_insert(model, code_token, modality_id, marker_token,
                                   n_steps=20, lr=0.3, T=16, V_text=512):
    """Optimise ONLY the embedding rows that hash(code) touches, leaving
    everything else (including other Engram rows) unchanged.
    """
    eng = model.engram.engrams[str(modality_id)]
    touched = get_touched_rows(eng, code_token)

    # Snapshot original state
    snap = {}
    for lid_str in touched:
        emb_weight = eng.tables[lid_str].embedding.weight
        snap[lid_str] = emb_weight.detach().clone()

    # The optimiser must only touch the embedding.weight tensors;
    # we'll mask the gradient after backward.
    params_to_opt = [eng.tables[lid_str].embedding.weight for lid_str in touched]
    opt = torch.optim.SGD(params_to_opt, lr=lr, momentum=0.9)

    rng = np.random.default_rng(int(code_token) * 13 + int(marker_token) * 17 + 1)
    for step in range(n_steps):
        input_ids, modality_ids = build_query_context(code_token, modality_id, T=T, V_text=V_text, rng=rng)
        input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
        h = model(input_ids, modality_ids)
        last = h[:, -1, :]
        logits = model.head_text(last)
        target = torch.tensor([marker_token], dtype=torch.long, device=DEVICE)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        # Apply row mask: zero out gradients on rows NOT in `touched`
        with torch.no_grad():
            for lid_str, rows in touched.items():
                W = eng.tables[lid_str].embedding.weight
                if W.grad is None:
                    continue
                # Build mask: 1 where row is touched, 0 elsewhere
                mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.grad.dtype)
                row_idx = torch.tensor(sorted(rows), device=W.device, dtype=torch.long)
                mask[row_idx] = 1.0
                W.grad.mul_(mask)
        opt.step()
    return snap


def evaluate_retrieval_v3(model, codebook_apply, eval_emb, eval_pid, modality_id,
                          N_subset=None, n_queries_per_id=None, n_steps=20, lr=0.3,
                          V_text=512, marker_offset=400, T=16):
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid):
        by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None:
        ids_sorted = ids_sorted[:N_subset]
    markers = {pid: marker_offset + i for i, pid in enumerate(ids_sorted)}
    eng = model.engram.engrams[str(modality_id)]

    # Snapshot whole embedding table at start for clean restore
    init_snap = {lid_str: tbl.embedding.weight.detach().clone()
                  for lid_str, tbl in eng.tables.items()}

    # --- Registration: all identities accumulated ---
    register_codes = {}
    code_to_pid = defaultdict(list)
    rng = np.random.default_rng(99)
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_emb = eval_emb[idxs[0]]
        reg_code = int(codebook_apply(reg_emb[None])[0].item() if hasattr(codebook_apply(reg_emb[None])[0], 'item') else codebook_apply(reg_emb[None])[0])
        # Surgical insert (accumulating)
        row_targeted_surgical_insert(model, reg_code, modality_id, markers[pid],
                                       n_steps=n_steps, lr=lr, T=T, V_text=V_text)
        register_codes[pid] = reg_code
        code_to_pid[reg_code].append(pid)

    # Track which codes are "collision" codes (shared by multiple registered identities)
    collision_codes = {c: pids for c, pids in code_to_pid.items() if len(pids) > 1}

    # --- Query phase ---
    correct = 0; total = 0
    misclassified_due_to_collision = 0
    per_id_total = defaultdict(int); per_id_correct = defaultdict(int)
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        q_idxs = idxs[1:]
        if n_queries_per_id is not None:
            q_idxs = q_idxs[:n_queries_per_id]
        for qi in q_idxs:
            q_emb = eval_emb[qi]
            q_code_arr = codebook_apply(q_emb[None])[0]
            q_code = int(q_code_arr.item() if hasattr(q_code_arr, 'item') else q_code_arr)
            input_ids, modality_ids = build_query_context(q_code, modality_id, T=T, V_text=V_text,
                                                            rng=np.random.default_rng(qi))
            input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
            with torch.no_grad():
                h = model(input_ids, modality_ids)
                last = h[:, -1, :]
                logits = model.head_text(last)
                marker_logits = logits[0, marker_offset: marker_offset + len(ids_sorted)]
                pred_local_idx = int(marker_logits.argmax().item())
                pred_pid = ids_sorted[pred_local_idx]
            total += 1; per_id_total[pid] += 1
            if pred_pid == pid:
                correct += 1; per_id_correct[pid] += 1
            else:
                if q_code in collision_codes and pid in collision_codes[q_code]:
                    misclassified_due_to_collision += 1

    # Restore original Engram state
    with torch.no_grad():
        for lid_str, w in init_snap.items():
            eng.tables[lid_str].embedding.weight.copy_(w)

    return {
        "N_registered": len(ids_sorted),
        "N_queries": total,
        "retrieval_at_1": correct / total if total > 0 else 0.0,
        "N_collision_codes": len(collision_codes),
        "fraction_queries_in_collision_codes": misclassified_due_to_collision / total if total > 0 else 0.0,
    }


def main():
    print("=" * 70)
    print("v3.1 — row-targeted surgical insertion (vs v2 naive, v1 chained, RAG)")
    print("=" * 70)

    print(f"\n[load] model from {CKPT_PATH}")
    model, cfg = load_model(CKPT_PATH)
    print(f"  V_text={cfg['V_text']}, V_vis={cfg['V_vis']}, V_aud={cfg['V_aud']}")

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")
    aud_tr_emb, _, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr_emb, _, vis_ev_emb, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    K = 32  # matches checkpoint
    audio_apply = fit_naive_rq(aud_tr_emb, n_levels=1, k_per=K)
    vision_apply = fit_naive_rq(vis_tr_emb, n_levels=1, k_per=K)

    Ns = [5, 10, 20]
    nq = 5
    results = {}

    for modality_id, name, emb, pids, apply_fn, marker_offset in [
        (MODALITY_VISION, "vision", vis_ev_emb, vis_ev_pid, vision_apply, 400),
        (MODALITY_AUDIO, "audio",   aud_ev_emb, aud_ev_pid, audio_apply, 400),
    ]:
        print(f"\n[{name}]")
        rag = {N: embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq) for N in Ns}
        for N in Ns:
            print(f"  embedding-RAG ceiling   N={N:>2}  retr@1={rag[N]:.4f}")
        v3 = {}
        for N in Ns:
            print(f"  v3 row-targeted        N={N:>2}  ...", end="", flush=True)
            r = evaluate_retrieval_v3(model, apply_fn, emb, pids, modality_id,
                                       N_subset=N, n_queries_per_id=nq,
                                       n_steps=20, lr=0.3, V_text=cfg['V_text'],
                                       marker_offset=marker_offset)
            print(f"  retr@1={r['retrieval_at_1']:.4f}  "
                  f"(N_collision_codes={r['N_collision_codes']}, "
                  f"frac_queries_in_collisions={r['fraction_queries_in_collision_codes']:.3f})")
            v3[N] = r
        results[name] = {"embedding_rag": rag, "v3_row_targeted": v3}

    out = Path("/home/ubuntu/multimodal-user-memory/results/v3_retrieval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] Wrote {out}")

    # Side-by-side full table
    v1_path = Path("/home/ubuntu/multimodal-user-memory/results/engram_retrieval.json")
    v2_path = Path("/home/ubuntu/multimodal-user-memory/results/v2_retrieval.json")
    if v1_path.exists() and v2_path.exists():
        with open(v1_path) as f: v1 = json.load(f)
        with open(v2_path) as f: v2 = json.load(f)
        print("\n" + "=" * 95)
        print("HEADLINE: full progression — RAG | v1 hash+chained | v2 naive surgical | v3 row-targeted")
        print("=" * 95)
        print(f"{'modality':>8} | {'N':>3} | {'RAG':>6} | {'v1 best':>8} | {'v2 naive':>9} | {'v3 row-tgt':>11} | v3 − v1")
        print("-" * 95)
        for name in ["vision", "audio"]:
            for N in Ns:
                rag_v = results[name]["embedding_rag"][N]
                v3_v = results[name]["v3_row_targeted"][N]["retrieval_at_1"]
                v1_best = 0.0
                for cfg_name, cfg_res in v1[name].get("engram", {}).items():
                    if str(N) in cfg_res:
                        v1_best = max(v1_best, cfg_res[str(N)].get("retrieval_chained_disambig", 0.0))
                v2_v = v2[name]["v2_engram_surgical"][str(N)]
                delta = v3_v - v1_best
                mark = " ✓ v3 beats v1" if delta > 0.02 else (" ≈ tie" if abs(delta) <= 0.02 else " ✗ v3 worse")
                print(f"{name:>8} | {N:>3} | {rag_v:>6.3f} | {v1_best:>8.3f} | {v2_v:>9.3f} | {v3_v:>11.3f} | {delta:>+7.3f}{mark}")


if __name__ == "__main__":
    sys.exit(main())
