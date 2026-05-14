"""v2 retrieval head-to-head vs v1.

Tests whether the v2-trained multimodal Engram + surgical row insertion
supports per-user identity retrieval that matches or beats v1's
0.48 vision / 0.60 audio retrieval@1 at N=20.

Protocol (mirrors v1 `engram_retrieval.py` so numbers are comparable):
  - Split held-out identities; assign each a reserved "marker" text token.
  - For each held-out identity:
      1. Take ONE registration embedding.
      2. Compute its code with the frozen quantiser used during training.
      3. Find the Engram row that hash(code) points to (per modality).
      4. **Surgical insertion (OPT-15-style)**: run 15 SGD steps on that
         single row to minimise cross-entropy of the model's next-token
         prediction toward the marker token, conditioned on a context
         that ends with the perceptual code.
  - For each query embedding of the same identity (cross-condition):
      compute code → put in context → run model → check whether argmax
      of next-token == registered marker.

Compare to:
  - Embedding-RAG cosine NN (the v1 ceiling at 0.96 vision / 1.0 audio)
  - v1 Engram retrieval@1 (the v1 floor at 0.48 vision / 0.60 audio)

If v2 retrieval@1 > v1, the paper has its headline. If v2 only equals v1,
we still have the "gate fires on recurrence" intermediate result; if
even that doesn't help retrieval, the path forward is to scale up the
model (larger d, more layers, more pretraining) — the toy may simply be
under-capacity.
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

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "/home/ubuntu/multimodal-user-memory/runs/v2_toy_realencoder.pt"
QUANTPATH = "/home/ubuntu/multimodal-user-memory/runs/v2_quantisers.npz"


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = ToyGPTWithEngram(
        d=cfg["d"], n_layer=cfg["n_layer"], max_T=cfg["max_T"],
        V_text=cfg["V_text"], V_vis=cfg["V_vis"], V_aud=cfg["V_aud"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg


def split_by_identity(emb, pid, train_frac=0.5):
    rng = np.random.RandomState(42)
    unique = sorted(set(pid.tolist()))
    rng.shuffle(unique)
    n_train = int(len(unique) * train_frac)
    train_ids = set(unique[:n_train])
    train_mask = np.array([str(p) in train_ids for p in pid])
    return emb[train_mask], pid[train_mask], emb[~train_mask], pid[~train_mask]


def build_query_context(code_token, modality_id, T=16, V_text=512, rng=None, sep_token=0):
    """Build a context ending with the perceptual code. The model will then
    predict the NEXT token; surgical insertion tunes the Engram row to make
    that next-token prediction be the identity's marker."""
    rng = rng or np.random.default_rng()
    # Random text-token prefix, then SEP, then the perceptual code
    text_prefix_len = T - 2
    input_ids = np.zeros(T, dtype=np.int64)
    modality_ids = np.zeros(T, dtype=np.int64)
    input_ids[: text_prefix_len] = rng.integers(1, V_text, size=text_prefix_len)
    modality_ids[: text_prefix_len] = MODALITY_TEXT
    # SEP at penultimate
    input_ids[T - 2] = sep_token
    modality_ids[T - 2] = MODALITY_TEXT
    # Perceptual code at last
    input_ids[T - 1] = code_token
    modality_ids[T - 1] = modality_id
    return torch.from_numpy(input_ids).unsqueeze(0), torch.from_numpy(modality_ids).unsqueeze(0)


def surgical_insert(model, code_token, modality_id, marker_token, n_steps=15, lr=0.1, T=16, V_text=512):
    """Optimise the Engram row for this code so the model predicts `marker_token` next.

    Concretely: locate the embedding-table row that hash(code) addresses at every
    Engram-attached layer; collect those parameters as a small leaf set; SGD on
    them while keeping the rest of the model frozen.

    This is the analogue of user-as-engram's OPT-15 surgical insertion.
    """
    # We treat ALL Engram parameters of the relevant modality as the surgical
    # surface (small subset of model params; the hash already restricts which
    # rows actually update because grad flows only through the touched rows).
    eng = model.engram.engrams[str(modality_id)]
    params_to_train = list(eng.parameters())
    opt = torch.optim.SGD(params_to_train, lr=lr, momentum=0.9)

    # Snapshot starting state so we can revert after the experiment
    snap = {name: p.detach().clone() for name, p in eng.named_parameters()}

    rng = np.random.default_rng(int(code_token) * 13 + int(marker_token) * 17)
    for step in range(n_steps):
        # Build a fresh random-context query each step (so we don't overfit context structure)
        input_ids, modality_ids = build_query_context(code_token, modality_id, T=T, V_text=V_text, rng=rng)
        input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
        # Forward
        model.eval()  # freeze attention scaling etc.
        h = model(input_ids, modality_ids)
        # Last-position logits for text head (marker tokens are text)
        last = h[:, -1, :]
        logits_text = model.head_text(last)
        target = torch.tensor([marker_token], dtype=torch.long, device=DEVICE)
        loss = F.cross_entropy(logits_text, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return snap, params_to_train


def restore(eng, snap):
    with torch.no_grad():
        for name, p in eng.named_parameters():
            p.copy_(snap[name])


def evaluate_retrieval(model, codebook_apply, eval_emb, eval_pid, modality_id, K_codebook,
                       N_subset=None, n_queries_per_id=None, n_steps=15, lr=0.1,
                       V_text=512, marker_offset=400):
    """Run the registration + retrieval protocol."""
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid):
        by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None:
        ids_sorted = ids_sorted[:N_subset]

    # Assign each identity a unique marker text token
    markers = {pid: marker_offset + i for i, pid in enumerate(ids_sorted)}
    assert max(markers.values()) < V_text, f"too many identities for V_text={V_text}"

    eng = model.engram.engrams[str(modality_id)]

    # Snapshot original state once; we'll restore after each identity
    # Actually we KEEP the surgical insertions accumulated, so all identities are
    # registered simultaneously (the realistic multi-user case).
    full_snap = {name: p.detach().clone() for name, p in eng.named_parameters()}

    # ---- Registration phase ----
    register_codes = {}  # pid -> (code token, registered marker)
    rng = np.random.default_rng(99)
    for pid in ids_sorted:
        idxs = by_id[pid]
        idxs_shuffled = list(idxs); rng.shuffle(idxs_shuffled)
        reg_emb = eval_emb[idxs_shuffled[0]]
        reg_code = int(codebook_apply(reg_emb[None])[0])  # flat single-int (since L=1)
        # Surgical insert (accumulating, no restore between identities)
        _, _ = surgical_insert(model, reg_code, modality_id, markers[pid],
                                n_steps=n_steps, lr=lr, T=16, V_text=V_text)
        register_codes[pid] = (reg_code, markers[pid])

    # ---- Query phase ----
    correct = 0; total = 0
    per_id_correct = defaultdict(int)
    per_id_total = defaultdict(int)
    for pid in ids_sorted:
        idxs = by_id[pid]
        idxs_shuffled = list(idxs); rng.shuffle(idxs_shuffled)
        # Skip the registration index (always the first after shuffle)
        query_idxs = idxs_shuffled[1:]
        if n_queries_per_id is not None:
            query_idxs = query_idxs[:n_queries_per_id]
        for qi in query_idxs:
            q_emb = eval_emb[qi]
            q_code = int(codebook_apply(q_emb[None])[0])
            input_ids, modality_ids = build_query_context(q_code, modality_id, T=16, V_text=V_text,
                                                            rng=np.random.default_rng(qi))
            input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
            with torch.no_grad():
                h = model(input_ids, modality_ids)
                last = h[:, -1, :]
                logits_text = model.head_text(last)
                # Among the registered markers, which one is the model's top pick?
                marker_logits = logits_text[0, marker_offset: marker_offset + len(ids_sorted)]
                pred_local_idx = int(marker_logits.argmax().item())
                pred_pid = ids_sorted[pred_local_idx]
            total += 1
            per_id_total[pid] += 1
            if pred_pid == pid:
                correct += 1
                per_id_correct[pid] += 1

    # Restore Engram for clean run on next configuration
    restore(eng, full_snap)

    per_id = {pid: per_id_correct[pid] / per_id_total[pid] for pid in ids_sorted if per_id_total[pid] > 0}
    return {
        "N_registered": len(ids_sorted),
        "N_queries": total,
        "retrieval_at_1": correct / total if total > 0 else 0.0,
        "per_identity_accuracy": per_id,
        "marker_assignments": markers,
        "registered_codes": {pid: int(rc) for pid, (rc, _) in register_codes.items()},
    }


def embedding_rag_ceiling(eval_emb, eval_pid, N_subset=None, n_queries_per_id=None):
    """Baseline: cosine NN over registered embeddings (matches v1 protocol)."""
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid):
        by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None:
        ids_sorted = ids_sorted[:N_subset]
    reg_embs, reg_labels = [], []
    queries = []
    rng = np.random.default_rng(99)
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_embs.append(eval_emb[idxs[0]])
        reg_labels.append(pid)
        q_idxs = idxs[1:]
        if n_queries_per_id is not None:
            q_idxs = q_idxs[:n_queries_per_id]
        for qi in q_idxs:
            queries.append((eval_emb[qi], pid))
    if not queries: return 0.0
    R = np.stack(reg_embs).astype(np.float32); R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
    Q = np.stack([q[0] for q in queries]).astype(np.float32); Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    sims = Q @ R.T
    pred = sims.argmax(axis=1)
    return sum(1 for k in range(len(queries)) if reg_labels[pred[k]] == queries[k][1]) / len(queries)


def main():
    print("=" * 70)
    print("v2 retrieval head-to-head vs v1")
    print("=" * 70)

    print(f"\n[load] model from {CKPT_PATH}")
    model, cfg = load_model(CKPT_PATH)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model loaded: {n_params:,} params; V_text={cfg['V_text']}, V_vis={cfg['V_vis']}, V_aud={cfg['V_aud']}")

    print("\n[load] cached embeddings + refit quantiser ...")
    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")

    aud_tr_emb, aud_tr_pid, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr_emb, vis_tr_pid, vis_ev_emb, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    K_aud = int(np.sqrt(cfg["V_aud"])) if cfg["V_aud"] in (256, 1024, 4096) else cfg["V_aud"]
    K_vis = int(np.sqrt(cfg["V_vis"])) if cfg["V_vis"] in (256, 1024, 4096) else cfg["V_vis"]
    N_LEVELS = 1
    K_aud = cfg["V_aud"]; K_vis = cfg["V_vis"]  # we used L=1 in the trainer
    print(f"  K_aud={K_aud} (n_levels={N_LEVELS}), K_vis={K_vis}")
    audio_apply = fit_naive_rq(aud_tr_emb, n_levels=N_LEVELS, k_per=K_aud)
    vision_apply = fit_naive_rq(vis_tr_emb, n_levels=N_LEVELS, k_per=K_vis)

    print(f"  audio eval: {len(aud_ev_emb)} embs / {len(set(aud_ev_pid))} ids")
    print(f"  vision eval: {len(vis_ev_emb)} embs / {len(set(vis_ev_pid))} ids")

    results = {}
    Ns = [5, 10, 20]
    nq = 5

    for modality_id, name, emb, pids, apply_fn, K, marker_offset in [
        (MODALITY_VISION, "vision", vis_ev_emb, vis_ev_pid, vision_apply, K_vis, 400),
        (MODALITY_AUDIO, "audio",  aud_ev_emb, aud_ev_pid, audio_apply, K_aud, 400),
    ]:
        print(f"\n[{name}]")
        rag = {}
        for N in Ns:
            r = embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq)
            print(f"  embedding-RAG ceiling   N={N:>2}  retr@1={r:.4f}")
            rag[N] = r

        v2 = {}
        for N in Ns:
            print(f"  v2 surgical insert    N={N:>2}  (n_steps=15 per ident, lr=0.1) ...", end="", flush=True)
            r = evaluate_retrieval(
                model, apply_fn, emb, pids, modality_id, K,
                N_subset=N, n_queries_per_id=nq, n_steps=15, lr=0.1,
                V_text=cfg["V_text"], marker_offset=marker_offset,
            )
            print(f"  retr@1={r['retrieval_at_1']:.4f}  ({r['N_queries']} queries)")
            v2[N] = r

        results[name] = {"embedding_rag": rag, "v2_engram_surgical": {N: v2[N]["retrieval_at_1"] for N in Ns}}

    out = Path("/home/ubuntu/multimodal-user-memory/results/v2_retrieval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] Wrote {out}")

    # Side-by-side vs v1 (from results/engram_retrieval.json)
    v1_path = Path("/home/ubuntu/multimodal-user-memory/results/engram_retrieval.json")
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        print("\n" + "=" * 80)
        print("HEADLINE: v2 vs v1 retrieval@1 (chained-disambig variant for v1)")
        print("=" * 80)
        print(f"{'modality':>8} | {'N':>3} | {'RAG (ceiling)':>14} | {'v1 best Engram':>14} | {'v2 Engram surgical':>18}")
        print("-" * 80)
        for name in ["vision", "audio"]:
            for N in Ns:
                rag = results[name]["embedding_rag"][N]
                v2_acc = results[name]["v2_engram_surgical"][N]
                # v1 best per N: take max across all configs of chained-disambig
                v1_best = 0.0
                for cfg_name, cfg_results in v1[name].get("engram", {}).items():
                    if str(N) in cfg_results:
                        v1_best = max(v1_best, cfg_results[str(N)].get("retrieval_chained_disambig", 0.0))
                delta = v2_acc - v1_best
                marker = " ✓ v2 beats v1" if delta > 0.02 else (" ✗ v2 worse" if delta < -0.02 else " ≈ tie")
                print(f"{name:>8} | {N:>3} | {rag:>14.4f} | {v1_best:>14.4f} | {v2_acc:>18.4f}{marker}")


if __name__ == "__main__":
    sys.exit(main())
