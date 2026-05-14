"""v3.3 retrieval — does mid-scale move the needle?

Re-runs the v3.1 row-targeted surgical insertion + retrieval protocol on the
mid-scale (15.5M) checkpoint trained in midscale_train.py. The hypothesis:
at meaningful scale (28% of params in Engram, deeper LM, more training),
surgical insertion can drive the LM output enough to retrieve correctly.

Direct comparison vs:
  - v1 hash+chained (0.48 vision / 0.60 audio at N=20)
  - v3 row-targeted at toy 3M (0.12 vision / 0.09 audio at N=20)
  - embedding-RAG ceiling
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
from midscale_train import MidScaleGPTWithEngram
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, build_query_context, embedding_rag_ceiling
from v3_retrieval import row_targeted_surgical_insert, get_touched_rows

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


def evaluate(model, codebook_apply, eval_emb, eval_pid, modality_id,
             N_subset=None, n_queries_per_id=None, n_steps=30, lr=0.5,
             V_text=512, marker_offset=400, T=16):
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
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_emb = eval_emb[idxs[0]]
        reg_code_arr = codebook_apply(reg_emb[None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        row_targeted_surgical_insert(model, reg_code, modality_id, markers[pid],
                                       n_steps=n_steps, lr=lr, T=T, V_text=V_text)
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

    # Restore
    with torch.no_grad():
        for ks, w in init_snap.items():
            eng.tables[ks].embedding.weight.copy_(w)

    return {
        "N_registered": len(ids_sorted),
        "N_queries": total,
        "retrieval_at_1": correct / total if total > 0 else 0.0,
        "N_collision_codes": len(collision_codes),
        "code_match_retrieval": code_match_correct / code_match_total if code_match_total > 0 else float("nan"),
        "code_mismatch_retrieval": code_mismatch_correct / code_mismatch_total if code_mismatch_total > 0 else float("nan"),
        "fraction_code_match": code_match_total / total if total > 0 else 0.0,
    }


def main():
    print("=" * 70)
    print("v3.3 midscale retrieval (15.5M params, Engram=4.3M)")
    print("=" * 70)
    model, cfg = load_midscale()
    print(f"  V_text={cfg['V_text']}, V_vis={cfg['V_vis']}, V_aud={cfg['V_aud']}")

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")
    aud_tr, _, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr, _, vis_ev_emb, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    K = cfg["K_aud"]  # 32
    audio_apply = fit_naive_rq(aud_tr, n_levels=1, k_per=K)
    vision_apply = fit_naive_rq(vis_tr, n_levels=1, k_per=K)

    Ns = [5, 10, 20]; nq = 5
    out_results = {}
    for mid, name, emb, pids, apply_fn, marker_offset in [
        (MODALITY_VISION, "vision", vis_ev_emb, vis_ev_pid, vision_apply, 400),
        (MODALITY_AUDIO,  "audio",  aud_ev_emb, aud_ev_pid, audio_apply,  400),
    ]:
        print(f"\n[{name}]")
        rag = {N: embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq) for N in Ns}
        for N in Ns:
            print(f"  embedding-RAG ceiling   N={N:>2}  retr@1={rag[N]:.4f}")
        v3 = {}
        for N in Ns:
            print(f"  v3.3 midscale (n_steps=30,lr=0.5)   N={N:>2}  ...", end="", flush=True)
            r = evaluate(model, apply_fn, emb, pids, mid, N_subset=N, n_queries_per_id=nq,
                         n_steps=30, lr=0.5, V_text=cfg['V_text'], marker_offset=marker_offset)
            print(f"  retr@1={r['retrieval_at_1']:.4f}  "
                  f"(collisions={r['N_collision_codes']}, "
                  f"code-match-retr={r['code_match_retrieval']:.3f} on {100*r['fraction_code_match']:.0f}% of queries, "
                  f"code-mismatch-retr={r['code_mismatch_retrieval']:.3f})")
            v3[N] = r
        out_results[name] = {"rag": rag, "v3_midscale": v3}

    out = Path("/home/ubuntu/multimodal-user-memory/results/v3_retrieval_midscale.json")
    with open(out, "w") as f: json.dump(out_results, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Full progression headline
    v1_path = Path("/home/ubuntu/multimodal-user-memory/results/engram_retrieval.json")
    v3toy_path = Path("/home/ubuntu/multimodal-user-memory/results/v3_retrieval.json")
    if v1_path.exists() and v3toy_path.exists():
        with open(v1_path) as f: v1 = json.load(f)
        with open(v3toy_path) as f: v3t = json.load(f)
        print("\n" + "=" * 95)
        print("HEADLINE — RAG | v1 chained | v3 toy (3M) | v3 midscale (15.5M)")
        print("=" * 95)
        print(f"{'modality':>8} | {'N':>3} | {'RAG':>6} | {'v1 best':>8} | {'v3 toy':>7} | {'v3 mid':>7} | mid − toy | mid − v1")
        print("-" * 95)
        for name in ["vision", "audio"]:
            for N in Ns:
                rag_v = out_results[name]["rag"][N]
                v3_mid = out_results[name]["v3_midscale"][N]["retrieval_at_1"]
                v3_toy = v3t[name]["v3_row_targeted"][str(N)]["retrieval_at_1"]
                v1_best = 0.0
                for cfg_name, cfg_res in v1[name].get("engram", {}).items():
                    if str(N) in cfg_res:
                        v1_best = max(v1_best, cfg_res[str(N)].get("retrieval_chained_disambig", 0.0))
                dmt = v3_mid - v3_toy; dmv1 = v3_mid - v1_best
                mark_scale = " ↑↑" if dmt > 0.1 else (" ↑" if dmt > 0.02 else (" ≈" if abs(dmt) <= 0.02 else " ↓"))
                mark_v1 = "✓ beats v1" if dmv1 > 0.02 else ("= ties v1" if abs(dmv1) <= 0.02 else "✗ under v1")
                print(f"{name:>8} | {N:>3} | {rag_v:>6.3f} | {v1_best:>8.3f} | {v3_toy:>7.3f} | {v3_mid:>7.3f} | {dmt:>+8.3f}{mark_scale} | {dmv1:>+7.3f} {mark_v1}")


if __name__ == "__main__":
    sys.exit(main())
