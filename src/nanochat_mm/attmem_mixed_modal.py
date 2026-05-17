"""Mixed-modal test: register vision + audio identities in the same model,
verify per-modality retrieval is independent.

Tests that AttMem's per-modality bank architecture supports concurrent
multi-modal memory without cross-modal interference.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE
from v2_retrieval import split_by_identity


def evaluate_mixed(bolt, vis_emb, vis_pid, aud_emb, aud_pid, tok,
                    N_vis=100, N_aud=20, n_queries_per_id=3,
                    marker_offset_vis=30001, marker_offset_aud=31001, T=24):
    """Register N_vis vision IDs and N_aud audio IDs in the same bolt.
    Then query each modality separately; cross-modal queries should
    never match vision markers from audio queries or vice versa."""
    bolt.reset_banks()

    # Vision bank
    by_id_v = defaultdict(list)
    for i, p in enumerate(vis_pid):
        by_id_v[str(p)].append(i)
    ids_v = sorted(by_id_v.keys())[:N_vis]
    markers_v = list(range(marker_offset_vis, marker_offset_vis + len(ids_v)))
    rng = np.random.default_rng(99)
    reg_idx_v = [rng.choice(by_id_v[p]) for p in ids_v]
    reg_keys_v = torch.from_numpy(vis_emb[reg_idx_v].astype(np.float32)).to(DEVICE)
    bolt.insert_batch(MODALITY_VISION, reg_keys_v, markers_v)

    # Audio bank
    by_id_a = defaultdict(list)
    for i, p in enumerate(aud_pid):
        by_id_a[str(p)].append(i)
    ids_a = sorted(by_id_a.keys())[:N_aud]
    markers_a = list(range(marker_offset_aud, marker_offset_aud + len(ids_a)))
    reg_idx_a = [rng.choice(by_id_a[p]) for p in ids_a]
    reg_keys_a = torch.from_numpy(aud_emb[reg_idx_a].astype(np.float32)).to(DEVICE)
    bolt.insert_batch(MODALITY_AUDIO, reg_keys_a, markers_a)

    print(f"  Bank sizes: vision={bolt.attmem.banks[str(MODALITY_VISION)].size}, "
          f"audio={bolt.attmem.banks[str(MODALITY_AUDIO)].size}")

    pad_id = tok.pad_token_id or 0
    pref = tok.encode("You see", add_special_tokens=False)
    text_ids = list(pref) + [pad_id] * (T - 1 - len(pref))
    text_ids = (text_ids[: T - 1]) + [pad_id]

    # Vision queries
    correct_v = 0; cross_modal_match_v = 0; total_v = 0
    for k, pid in enumerate(ids_v):
        idxs = list(by_id_v[pid])
        q_idxs = [i for i in idxs if i != reg_idx_v[k]][:n_queries_per_id]
        for qi in q_idxs:
            q_key = torch.from_numpy(vis_emb[qi].astype(np.float32)).unsqueeze(0).to(DEVICE)
            modality_ids_t = torch.tensor(
                [[MODALITY_TEXT] * (T - 1) + [MODALITY_VISION]], dtype=torch.long, device=DEVICE
            )
            text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                logits = bolt(modality_ids_t, text_ids_t, {MODALITY_VISION: q_key})
                last = logits[0, -1, :]
            # Argmax over all registered markers (vis + aud)
            all_markers = markers_v + markers_a
            ml = torch.stack([last[m] for m in all_markers])
            pred_marker = all_markers[int(ml.argmax().item())]
            total_v += 1
            if pred_marker == markers_v[k]:
                correct_v += 1
            elif pred_marker in markers_a:
                cross_modal_match_v += 1

    # Audio queries
    correct_a = 0; cross_modal_match_a = 0; total_a = 0
    for k, pid in enumerate(ids_a):
        idxs = list(by_id_a[pid])
        q_idxs = [i for i in idxs if i != reg_idx_a[k]][:n_queries_per_id]
        for qi in q_idxs:
            q_key = torch.from_numpy(aud_emb[qi].astype(np.float32)).unsqueeze(0).to(DEVICE)
            modality_ids_t = torch.tensor(
                [[MODALITY_TEXT] * (T - 1) + [MODALITY_AUDIO]], dtype=torch.long, device=DEVICE
            )
            text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                logits = bolt(modality_ids_t, text_ids_t, {MODALITY_AUDIO: q_key})
                last = logits[0, -1, :]
            all_markers = markers_v + markers_a
            ml = torch.stack([last[m] for m in all_markers])
            pred_marker = all_markers[int(ml.argmax().item())]
            total_a += 1
            if pred_marker == markers_a[k]:
                correct_a += 1
            elif pred_marker in markers_v:
                cross_modal_match_a += 1

    return {
        "vision": {"retr@1": correct_v / total_v if total_v else 0,
                   "cross_modal_leak": cross_modal_match_v / total_v if total_v else 0,
                   "n_queries": total_v},
        "audio": {"retr@1": correct_a / total_a if total_a else 0,
                   "cross_modal_leak": cross_modal_match_a / total_a if total_a else 0,
                   "n_queries": total_a},
    }


def main():
    print("Loading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    print("Loading vision (arcface_face_xxxl) and audio (ecapa_libri_large) ...")
    vd = np.load(EMB / "arcface_face_xxxl.npz")
    ad = np.load(EMB / "ecapa_libri_large.npz")
    vis_emb = vd["emb"].astype(np.float32)
    aud_emb = ad["emb"].astype(np.float32)
    vis_pid = vd["pid"] if vd["pid"].dtype.kind == "U" else np.array([str(p) for p in vd["pid"]])
    aud_pid = ad["pid"] if ad["pid"].dtype.kind == "U" else np.array([str(p) for p in ad["pid"]])
    _, _, ev_v, evp_v = split_by_identity(vis_emb, vis_pid)
    _, _, ev_a, evp_a = split_by_identity(aud_emb, aud_pid)
    print(f"  vision eval: {len(set(evp_v.tolist()))} IDs / {len(ev_v)} samples")
    print(f"  audio eval: {len(set(evp_a.tolist()))} IDs / {len(ev_a)} samples")

    print("\nInitializing bolt (no pretraining — zero-shot mixed-modal test) ...")
    bolt = QwenAttMemBolt(qwen, tok,
                          vision_key_dim=vis_emb.shape[1],
                          audio_key_dim=aud_emb.shape[1],
                          attach_layer=33, attach_lm_head=True).to(DEVICE)
    bolt.install_hook()

    print("\n=== Mixed-modal eval (random init, zero-shot architectural test) ===")
    r = evaluate_mixed(bolt, ev_v, evp_v, ev_a, evp_a, tok, N_vis=20, N_aud=15)
    print(f"\n  Vision: retr@1 = {r['vision']['retr@1']:.3f}  cross-modal leak = {r['vision']['cross_modal_leak']:.3f}  (n={r['vision']['n_queries']})")
    print(f"  Audio:  retr@1 = {r['audio']['retr@1']:.3f}  cross-modal leak = {r['audio']['cross_modal_leak']:.3f}  (n={r['audio']['n_queries']})")

    out = Path("/home/ubuntu/multimodal-user-memory/results/attmem_mixed_modal.json")
    with open(out, "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
