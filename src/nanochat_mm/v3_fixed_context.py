"""v3.5 — fixed-context surgical insertion.

Bug found in v3.4 follow-up: the Engram hashes suffix N-grams (n in {2,3}),
so the rows touched at the perceptual position depend on the *preceding*
tokens. With random text prefixes, registration and query hit different
rows, and my row-target mask zeros exactly the rows that actually carry
gradient. The losses didn't budge in v3.4 because of this.

Fix tested here: use a FIXED deterministic context (all sep_token before
the perceptual code). Now the N-gram for the perceptual position is always
(SEP, SEP, code), so the hash is stable across registration and query.

This is a hack — the principled fix is to add a unigram (n=1) head to the
Engram so the perceptual modality has a context-free address. Recorded as
v3.5b for the next session.
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
from v2_retrieval import split_by_identity, embedding_rag_ceiling

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


def build_fixed_context(code_token, modality_id, T=16, sep_token=0):
    """Fixed prefix: all sep_tokens, then perceptual code at last position.
    Hash at position T-1 uses (sep, sep, code) which is deterministic."""
    input_ids = np.full(T, sep_token, dtype=np.int64)
    modality_ids = np.full(T, MODALITY_TEXT, dtype=np.int64)
    input_ids[T - 1] = code_token
    modality_ids[T - 1] = modality_id
    return torch.from_numpy(input_ids).unsqueeze(0), torch.from_numpy(modality_ids).unsqueeze(0)


def get_touched_rows_with_context(eng, code_token, input_ids):
    """Compute touched rows using the actual full context (not a single-token input)."""
    inp = input_ids.cpu().numpy() if isinstance(input_ids, torch.Tensor) else input_ids
    if inp.ndim == 1: inp = inp[None]
    hashes_per_layer = eng.hash_mapping.hash_all_layers(inp, user_salt=int(eng.user_salt))
    touched = {}
    last_pos = inp.shape[1] - 1
    for lid, h in hashes_per_layer.items():
        # h: [B, T, total_heads]; we want the rows hashed at the last position of the only batch entry
        local = h[0, last_pos]
        tbl = eng.tables[str(lid)]
        global_rows = local + tbl.offsets.cpu().numpy()
        touched[str(lid)] = set(int(r) for r in global_rows.tolist())
    return touched


def surgical_insert_fixed(model, code_token, modality_id, marker_token,
                          max_steps=100, lr=1.0, T=16, V_text=512,
                          early_stop_loss=0.5):
    eng = model.engram.engrams[str(modality_id)]
    # Build the exact context that will be used during training, and use IT
    # to compute touched rows.
    input_ids, modality_ids = build_fixed_context(code_token, modality_id, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
    touched = get_touched_rows_with_context(eng, code_token, input_ids)

    params_to_opt = [eng.tables[ks].embedding.weight for ks in touched]
    opt = torch.optim.SGD(params_to_opt, lr=lr, momentum=0.0)

    target = torch.tensor([marker_token], dtype=torch.long, device=DEVICE)
    last_loss = float("inf"); step_taken = 0
    for step in range(max_steps):
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
        step_taken = step + 1
        if last_loss < early_stop_loss:
            break
    return step_taken, last_loss


def evaluate(model, codebook_apply, eval_emb, eval_pid, modality_id,
             N_subset=None, n_queries_per_id=None,
             max_steps=100, lr=1.0, V_text=512, marker_offset=400, T=16):
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None: ids_sorted = ids_sorted[:N_subset]
    markers = {pid: marker_offset + i for i, pid in enumerate(ids_sorted)}
    eng = model.engram.engrams[str(modality_id)]
    init_snap = {ks: tbl.embedding.weight.detach().clone() for ks, tbl in eng.tables.items()}

    rng = np.random.default_rng(99)
    register_codes = {}; code_to_pid = defaultdict(list); insert_stats = []
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_emb = eval_emb[idxs[0]]
        reg_code_arr = codebook_apply(reg_emb[None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        steps, fl = surgical_insert_fixed(
            model, reg_code, modality_id, markers[pid],
            max_steps=max_steps, lr=lr, T=T, V_text=V_text,
        )
        insert_stats.append((steps, fl))
        register_codes[pid] = reg_code
        code_to_pid[reg_code].append(pid)
    collision_codes = {c: pids for c, pids in code_to_pid.items() if len(pids) > 1}

    correct = 0; total = 0
    code_match_correct = 0; code_match_total = 0
    code_mismatch_correct = 0; code_mismatch_total = 0
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        q_idxs = idxs[1:]
        if n_queries_per_id is not None: q_idxs = q_idxs[:n_queries_per_id]
        for qi in q_idxs:
            q_emb = eval_emb[qi]
            q_code_arr = codebook_apply(q_emb[None])[0]
            q_code = int(q_code_arr.item() if hasattr(q_code_arr, 'item') else q_code_arr)
            # USE THE SAME FIXED CONTEXT at query time as at registration time
            input_ids, modality_ids = build_fixed_context(q_code, modality_id, T=T)
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
    print("v3.5 — FIXED-CONTEXT surgical insertion (stable hash address)")
    print("=" * 70)
    model, cfg = load_midscale()
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
        v3_5 = {}
        for N in Ns:
            print(f"  RAG ceiling N={N:>2}: {rag[N]:.4f}")
            print(f"  v3.5 fixed-context N={N:>2} ...", end="", flush=True)
            r = evaluate(model, apply_fn, emb, pids, mid, N_subset=N, n_queries_per_id=nq,
                         max_steps=100, lr=1.0, V_text=cfg['V_text'], marker_offset=marker_offset)
            print(f"  retr@1={r['retrieval_at_1']:.4f}  "
                  f"(insert: avg {r['avg_insert_steps']:.0f} steps, final loss {r['avg_insert_loss']:.3f})  "
                  f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
                  f"mismatch={r['code_mismatch_retr']:.3f}")
            v3_5[N] = r
        results[name] = {"rag": rag, "v3_5_fixed_context": v3_5}

    out = Path("/home/ubuntu/multimodal-user-memory/results/v3_fixed_context.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Final progression
    print("\n" + "=" * 100)
    print("HEADLINE — RAG | v1 chained | v3 midscale random ctx | v3.5 fixed ctx")
    print("=" * 100)
    v1_path = Path("/home/ubuntu/multimodal-user-memory/results/engram_retrieval.json")
    v3m_path = Path("/home/ubuntu/multimodal-user-memory/results/v3_retrieval_midscale.json")
    if v1_path.exists() and v3m_path.exists():
        with open(v1_path) as f: v1 = json.load(f)
        with open(v3m_path) as f: v3m = json.load(f)
        print(f"{'modality':>8} | {'N':>3} | {'RAG':>6} | {'v1 best':>8} | {'v3 rand ctx':>11} | {'v3.5 fixed':>10} | fixed − rand")
        print("-" * 100)
        for name in ["vision", "audio"]:
            for N in Ns:
                rag_v = results[name]["rag"][N]
                v3_5v = results[name]["v3_5_fixed_context"][N]["retrieval_at_1"]
                v3_r = v3m[name]["v3_midscale"][str(N)]["retrieval_at_1"]
                v1_best = 0.0
                for cfg_name, cfg_res in v1[name].get("engram", {}).items():
                    if str(N) in cfg_res:
                        v1_best = max(v1_best, cfg_res[str(N)].get("retrieval_chained_disambig", 0.0))
                delta = v3_5v - v3_r
                mark = " ↑↑↑" if delta > 0.2 else (" ↑↑" if delta > 0.1 else (" ↑" if delta > 0.02 else (" ≈" if abs(delta) <= 0.02 else " ↓")))
                print(f"{name:>8} | {N:>3} | {rag_v:>6.3f} | {v1_best:>8.3f} | {v3_r:>11.3f} | {v3_5v:>10.3f} | {delta:>+11.3f}{mark}")


if __name__ == "__main__":
    sys.exit(main())
