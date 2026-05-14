"""v3.4 — aggressive surgical insertion on midscale.

The midscale retrieval test surfaced that when codes match (~50% of queries),
audio surgical insertion hits 0.41 on 10 classes — well above chance,
but well short of the embedding-RAG ceiling of 1.0. Hypothesis: the
default n_steps=30, lr=0.5 with momentum carrying-over across identities
isn't pushing the rows hard enough.

This script tests:
  - More steps (n_steps=100)
  - Fresh optimiser per identity (no momentum carry-over)
  - Per-identity convergence: stop when CE on the target token < threshold
  - Higher lr (1.0)

It also breaks out retrieval@1 conditional on code-match status, so we
can see whether the "scale solves it eventually" path is plausible.
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
from engram_module_mm import MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO
from midscale_train import MidScaleGPTWithEngram
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, build_query_context, embedding_rag_ceiling
from v3_retrieval import get_touched_rows

torch.manual_seed(42); np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/home/ubuntu/multimodal-user-memory/runs/v3_midscale.pt"


def load_midscale():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = MidScaleGPTWithEngram(
        d=cfg["d"], n_layer=cfg["n_layer"], max_T=cfg["max_T"],
        V_text=cfg["V_text"], V_vis=cfg["V_vis"], V_aud=cfg["V_aud"],
        n_head=cfg["n_head"], engram_layers=tuple(cfg["engram_layers"]),
        n_embed_per_ngram=cfg["n_embed_per_ngram"],
        engram_vocab_per_ngram=cfg["engram_vocab_per_ngram"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg


def aggressive_surgical_insert(model, code_token, modality_id, marker_token,
                                 max_steps=100, lr=1.0, T=16, V_text=512,
                                 early_stop_loss=0.5):
    """Optimise touched rows with a fresh optimiser, no momentum, until loss
    drops below early_stop_loss or max_steps reached.

    Returns the number of steps actually taken and final loss.
    """
    eng = model.engram.engrams[str(modality_id)]
    touched = get_touched_rows(eng, code_token)

    params_to_opt = [eng.tables[ks].embedding.weight for ks in touched]
    opt = torch.optim.SGD(params_to_opt, lr=lr, momentum=0.0)

    rng = np.random.default_rng(int(code_token) * 13 + int(marker_token) * 17 + 1)
    target = torch.tensor([marker_token], dtype=torch.long, device=DEVICE)
    last_loss = float("inf")
    for step in range(max_steps):
        input_ids, modality_ids = build_query_context(code_token, modality_id, T=T,
                                                        V_text=V_text, rng=rng)
        input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
        h = model(input_ids, modality_ids)
        last = h[:, -1, :]
        logits = model.head_text(last)
        loss = F.cross_entropy(logits, target)
        last_loss = float(loss.item())
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for ks, rows in touched.items():
                W = eng.tables[ks].embedding.weight
                if W.grad is None: continue
                mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.grad.dtype)
                row_idx = torch.tensor(sorted(rows), device=W.device, dtype=torch.long)
                mask[row_idx] = 1.0
                W.grad.mul_(mask)
        opt.step()
        if last_loss < early_stop_loss:
            return step + 1, last_loss
    return max_steps, last_loss


def evaluate(model, codebook_apply, eval_emb, eval_pid, modality_id,
             N_subset=None, n_queries_per_id=None,
             max_steps=100, lr=1.0, V_text=512, marker_offset=400, T=16):
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid):
        by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None:
        ids_sorted = ids_sorted[:N_subset]
    markers = {pid: marker_offset + i for i, pid in enumerate(ids_sorted)}
    eng = model.engram.engrams[str(modality_id)]
    init_snap = {ks: tbl.embedding.weight.detach().clone()
                  for ks, tbl in eng.tables.items()}

    rng = np.random.default_rng(99)
    register_codes = {}; code_to_pid = defaultdict(list)
    insert_stats = []
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_emb = eval_emb[idxs[0]]
        reg_code_arr = codebook_apply(reg_emb[None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        steps, final_loss = aggressive_surgical_insert(
            model, reg_code, modality_id, markers[pid],
            max_steps=max_steps, lr=lr, T=T, V_text=V_text,
        )
        insert_stats.append((steps, final_loss))
        register_codes[pid] = reg_code
        code_to_pid[reg_code].append(pid)
    collision_codes = {c: pids for c, pids in code_to_pid.items() if len(pids) > 1}

    correct = 0; total = 0
    code_match_correct = 0; code_match_total = 0
    code_mismatch_correct = 0; code_mismatch_total = 0
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
                                                            rng=np.random.default_rng(qi + 1))
            input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
            with torch.no_grad():
                h = model(input_ids, modality_ids)
                last = h[:, -1, :]
                logits = model.head_text(last)
                marker_logits = logits[0, marker_offset: marker_offset + len(ids_sorted)]
                pred_local_idx = int(marker_logits.argmax().item())
                pred_pid = ids_sorted[pred_local_idx]
            total += 1
            ok = (pred_pid == pid)
            if ok: correct += 1
            if q_code == register_codes[pid]:
                code_match_total += 1
                if ok: code_match_correct += 1
            else:
                code_mismatch_total += 1
                if ok: code_mismatch_correct += 1

    with torch.no_grad():
        for ks, w in init_snap.items():
            eng.tables[ks].embedding.weight.copy_(w)

    avg_steps = float(np.mean([s for s, _ in insert_stats]))
    avg_loss = float(np.mean([l for _, l in insert_stats]))
    return {
        "N_registered": len(ids_sorted),
        "N_queries": total,
        "retrieval_at_1": correct / total if total > 0 else 0.0,
        "code_match_retr": code_match_correct / code_match_total if code_match_total > 0 else float("nan"),
        "code_mismatch_retr": code_mismatch_correct / code_mismatch_total if code_mismatch_total > 0 else float("nan"),
        "fraction_code_match": code_match_total / total if total > 0 else 0.0,
        "N_collision_codes": len(collision_codes),
        "avg_insert_steps": avg_steps,
        "avg_insert_loss": avg_loss,
    }


def main():
    print("=" * 70)
    print("v3.4 — aggressive surgical insertion on midscale (max_steps=100, lr=1.0)")
    print("=" * 70)
    model, cfg = load_midscale()
    print(f"  model loaded; V_text={cfg['V_text']}, V_vis={cfg['V_vis']}, V_aud={cfg['V_aud']}")

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")
    aud_tr, _, aud_ev, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr, _, vis_ev, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    K = cfg["K_aud"]
    audio_apply = fit_naive_rq(aud_tr, n_levels=1, k_per=K)
    vision_apply = fit_naive_rq(vis_tr, n_levels=1, k_per=K)

    Ns = [5, 10, 20]; nq = 5
    results = {}
    for mid, name, emb, pids, apply_fn, marker_offset in [
        (MODALITY_VISION, "vision", vis_ev, vis_ev_pid, vision_apply, 400),
        (MODALITY_AUDIO,  "audio",  aud_ev, aud_ev_pid, audio_apply,  400),
    ]:
        print(f"\n[{name}]")
        rag = {N: embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq) for N in Ns}
        v3a = {}
        for N in Ns:
            print(f"  RAG ceiling N={N:>2}: {rag[N]:.4f}")
            print(f"  v3.4 aggressive N={N:>2} ...", end="", flush=True)
            r = evaluate(model, apply_fn, emb, pids, mid,
                          N_subset=N, n_queries_per_id=nq,
                          max_steps=100, lr=1.0,
                          V_text=cfg['V_text'], marker_offset=marker_offset)
            print(f"  retr@1={r['retrieval_at_1']:.4f}  "
                  f"(insert: avg {r['avg_insert_steps']:.0f} steps, final loss {r['avg_insert_loss']:.3f})  "
                  f"code-match-retr={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
                  f"mismatch={r['code_mismatch_retr']:.3f}")
            v3a[N] = r
        results[name] = {"rag": rag, "v3_4_aggressive": v3a}

    out = Path("/home/ubuntu/multimodal-user-memory/results/v3_aggressive_insert.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Cross-table comparison
    print("\n" + "=" * 100)
    print("HEADLINE — RAG | v1 chained | v3 midscale standard | v3.4 aggressive")
    print("=" * 100)
    v1_path = Path("/home/ubuntu/multimodal-user-memory/results/engram_retrieval.json")
    v3m_path = Path("/home/ubuntu/multimodal-user-memory/results/v3_retrieval_midscale.json")
    if v1_path.exists() and v3m_path.exists():
        with open(v1_path) as f: v1 = json.load(f)
        with open(v3m_path) as f: v3m = json.load(f)
        print(f"{'modality':>8} | {'N':>3} | {'RAG':>6} | {'v1 best':>8} | {'v3 mid std':>10} | {'v3.4 aggr':>10} | aggr − std")
        print("-" * 100)
        for name in ["vision", "audio"]:
            for N in Ns:
                rag_v = results[name]["rag"][N]
                v3_aggr = results[name]["v3_4_aggressive"][N]["retrieval_at_1"]
                v3_std = v3m[name]["v3_midscale"][str(N)]["retrieval_at_1"]
                v1_best = 0.0
                for cfg_name, cfg_res in v1[name].get("engram", {}).items():
                    if str(N) in cfg_res:
                        v1_best = max(v1_best, cfg_res[str(N)].get("retrieval_chained_disambig", 0.0))
                delta = v3_aggr - v3_std
                mark = " ↑↑" if delta > 0.1 else (" ↑" if delta > 0.02 else (" ≈" if abs(delta) <= 0.02 else " ↓"))
                print(f"{name:>8} | {N:>3} | {rag_v:>6.3f} | {v1_best:>8.3f} | {v3_std:>10.3f} | {v3_aggr:>10.3f} | {delta:>+9.3f}{mark}")


if __name__ == "__main__":
    sys.exit(main())
