"""Cross-modal Path A — register with voice, query with face (or vice versa).

Real agent scenario: agent meets Alice via her voice (e.g., a phone call).
Later, sees her face. Can the per-user memory bridge across modalities?

Test setup: synthetic pairing. For each (LibriSpeech speaker, LFW identity)
pair, we synthetically declare them to be 'the same identity.' At
registration time, we surgical-insert the marker AT BOTH the face code's
hash address AND the voice code's hash address. At query time, we use
ONE modality and check whether the right marker is retrieved.

If this works, the mechanism supports cross-modal user memory.
Limitation: this is a mechanism test on synthetic pairings — real
celebrity face+voice data is needed for a 'cross-modal recognition'
claim, but the mechanism question is settled here.

The deeper question (whether the LM's cross-modal generalisation helps
because the same marker is reachable from face OR voice path) is what
distinguishes this from independent per-modality memories.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_VISION, MODALITY_AUDIO
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import QwenEngramBolt, get_touched_rows, MODEL_ID, DEVICE, surgical_insert, build_fixed_context
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)


def evaluate_cross_modal(bolt, vis_apply, aud_apply,
                          vis_emb, vis_pid, aud_emb, aud_pid, tok,
                          N=20, n_queries=5, marker_offset=30001):
    """Pair vis_pid[k] with aud_pid[k] synthetically. Register marker_k at BOTH
    face and voice codes. Then query one modality at a time, check retrieval."""
    rng = np.random.default_rng(42)

    # Sample N pairings
    vis_by = defaultdict(list)
    for i, p in enumerate(vis_pid): vis_by[str(p)].append(i)
    aud_by = defaultdict(list)
    for i, p in enumerate(aud_pid): aud_by[str(p)].append(i)
    vis_keys = list(vis_by.keys())
    aud_keys = list(aud_by.keys())
    rng.shuffle(vis_keys); rng.shuffle(aud_keys)

    if N > min(len(vis_keys), len(aud_keys)):
        N = min(len(vis_keys), len(aud_keys))
    pairs = list(zip(vis_keys[:N], aud_keys[:N]))
    markers = list(range(marker_offset, marker_offset + N))

    # Snapshot
    snap = {}
    for mid in [MODALITY_VISION, MODALITY_AUDIO]:
        eng = bolt.engram.engrams[str(mid)]
        snap[mid] = {ks: tbl.embedding.weight.detach().clone() for ks, tbl in eng.tables.items()}
    vis_snap_emb = bolt.vis_perc_emb.weight.detach().clone()
    aud_snap_emb = bolt.aud_perc_emb.weight.detach().clone()

    # ---- Registration: install marker at BOTH modality codes for each pair ----
    for k, (vis_id, aud_id) in enumerate(pairs):
        vis_idx = vis_by[vis_id][0]
        aud_idx = aud_by[aud_id][0]
        vis_code = int(vis_apply(vis_emb[vis_idx][None])[0])
        aud_code = int(aud_apply(aud_emb[aud_idx][None])[0])
        marker = markers[k]
        # Insert at face code
        surgical_insert(bolt, vis_code, MODALITY_VISION, marker, tok,
                         max_steps=60, lr=1.0, T=24)
        # Insert at voice code
        surgical_insert(bolt, aud_code, MODALITY_AUDIO, marker, tok,
                         max_steps=60, lr=1.0, T=24)

    # ---- Query: from each modality, attempt to retrieve the right marker ----
    correct_vis2vis = 0; n_vis = 0
    correct_aud2aud = 0; n_aud = 0
    for k, (vis_id, aud_id) in enumerate(pairs):
        true_marker = markers[k]
        # Vision queries (skip the registration idx)
        vis_idxs = vis_by[vis_id][1: 1 + n_queries]
        for q_idx in vis_idxs:
            q_code = int(vis_apply(vis_emb[q_idx][None])[0])
            inp, mids = build_fixed_context(q_code, MODALITY_VISION, tok, marker_text_id=0, T=24)
            inp = inp.to(DEVICE); mids = mids.to(DEVICE)
            with torch.no_grad():
                logits = bolt(inp, mids)
                marker_logits = torch.stack([logits[0, -1, m] for m in markers])
                pred_idx = int(marker_logits.argmax())
                n_vis += 1
                if pred_idx == k: correct_vis2vis += 1
        # Audio queries
        aud_idxs = aud_by[aud_id][1: 1 + n_queries]
        for q_idx in aud_idxs:
            q_code = int(aud_apply(aud_emb[q_idx][None])[0])
            inp, mids = build_fixed_context(q_code, MODALITY_AUDIO, tok, marker_text_id=0, T=24)
            inp = inp.to(DEVICE); mids = mids.to(DEVICE)
            with torch.no_grad():
                logits = bolt(inp, mids)
                marker_logits = torch.stack([logits[0, -1, m] for m in markers])
                pred_idx = int(marker_logits.argmax())
                n_aud += 1
                if pred_idx == k: correct_aud2aud += 1

    # Restore
    with torch.no_grad():
        for mid in [MODALITY_VISION, MODALITY_AUDIO]:
            eng = bolt.engram.engrams[str(mid)]
            for ks, w in snap[mid].items():
                eng.tables[ks].embedding.weight.copy_(w)
        bolt.vis_perc_emb.weight.copy_(vis_snap_emb)
        bolt.aud_perc_emb.weight.copy_(aud_snap_emb)

    return {
        "N": N,
        "vis_to_vis_retr1": correct_vis2vis / n_vis if n_vis else 0.0,
        "aud_to_aud_retr1": correct_aud2aud / n_aud if n_aud else 0.0,
        "n_vis_queries": n_vis,
        "n_aud_queries": n_aud,
    }


def main():
    print("=" * 70)
    print("Cross-modal Path A — synthetic pairing test")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw_xl.npz")
    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri_large.npz")
    vis_tr, vis_tr_pid, vis_ev, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    aud_tr, aud_tr_pid, aud_ev, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    K = 32  # smaller K for cross-modal; per-N optimum
    vis_apply = fit_naive_rq(vis_tr, n_levels=1, k_per=K)
    aud_apply = fit_naive_rq(aud_tr, n_levels=1, k_per=K)

    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K, engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()

    # Pretrain on BOTH modalities sequentially (small)
    print("\n[pretrain] vision generic-NTP ...")
    pretrain_generic(bolt, vis_tr, vis_tr_pid, vis_apply, MODALITY_VISION, tok,
                      n_steps=300, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)
    print("\n[pretrain] audio generic-NTP ...")
    pretrain_generic(bolt, aud_tr, aud_tr_pid, aud_apply, MODALITY_AUDIO, tok,
                      n_steps=300, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    print("\n[eval] cross-modal retrieval:")
    results = {}
    for N in [5, 10, 20]:
        r = evaluate_cross_modal(bolt, vis_apply, aud_apply,
                                   vis_ev, vis_ev_pid, aud_ev, aud_ev_pid, tok,
                                   N=N, n_queries=5)
        print(f"  N={N:>2}: vision-query retrieval = {r['vis_to_vis_retr1']:.3f}, "
              f"audio-query retrieval = {r['aud_to_aud_retr1']:.3f}  "
              f"(both should be > chance 1/{N})")
        results[N] = r

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_cross_modal.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print(" Both modalities register the SAME marker per identity.")
    print(" Vision-query retrieval tests: did inserting at face code propagate to query?")
    print(" Audio-query retrieval tests: did inserting at voice code propagate to query?")
    print(" If both are >> chance: the per-user identity slot supports cross-modal access.")


if __name__ == "__main__":
    sys.exit(main())
