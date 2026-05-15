"""Catastrophic forgetting probe.

Sequential insertion test: register identity 1, measure retrieval@1 on
identity 1. Register identity 2, RE-measure retrieval on identity 1.
... up to identity N. Plot the retrieval-drift curve.

If retrieval@1 on identity 1 stays high across all N insertions, the
Engram architecture has good multi-user capacity. If it drops sharply,
catastrophic forgetting is happening (later writes corrupting earlier rows).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import MODALITY_AUDIO
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity
from qwen_engram_bolt import (
    QwenEngramBolt, surgical_insert, build_fixed_context, MODEL_ID, DEVICE,
)
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)


def query_one(bolt, code, modality_id, tok, marker_offset, N_registered, true_marker_idx):
    inp, mids = build_fixed_context(code, modality_id, tok, marker_text_id=0, T=24)
    inp = inp.to(DEVICE); mids = mids.to(DEVICE)
    markers = list(range(marker_offset, marker_offset + N_registered))
    with torch.no_grad():
        logits = bolt(inp, mids)
        marker_logits = torch.stack([logits[0, -1, m] for m in markers])
        pred_idx = int(marker_logits.argmax().item())
    return pred_idx == true_marker_idx


def main():
    print("=" * 70)
    print("Catastrophic forgetting probe — sequential insertion test")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri_large.npz")
    _, _, ev_emb, ev_pid = split_by_identity(aud['emb'], aud['pid'])
    K = 64
    apply_fn = fit_naive_rq(aud['emb'][np.in1d(aud['pid'], list(set(ev_pid))[:14])], n_levels=1, k_per=K)

    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K, engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    print("[pretrain] generic-NTP 200 steps ...")
    pretrain_generic(bolt, aud['emb'], aud['pid'], apply_fn, MODALITY_AUDIO, tok,
                      n_steps=200, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    by_id = defaultdict(list)
    for i, p in enumerate(ev_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())[:20]  # use up to 20 identities
    marker_offset = 30001

    print(f"\nSequential insertion of {len(ids_sorted)} identities, "
          f"re-measuring retrieval on identity 1 (and 5, 10) after each insertion ...")

    # Probe identities: track 3 (1st, 5th, 10th inserted) over the sequence
    probe_indices = [0, 4, 9]  # 1st, 5th, 10th inserted
    retrieval_curve = defaultdict(list)
    rng = np.random.default_rng(42)

    for k, pid in enumerate(ids_sorted):
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_code = int(apply_fn(ev_emb[idxs[0]][None])[0].item() if hasattr(apply_fn(ev_emb[idxs[0]][None])[0], 'item') else apply_fn(ev_emb[idxs[0]][None])[0])
        marker = marker_offset + k
        surgical_insert(bolt, reg_code, MODALITY_AUDIO, marker, tok, max_steps=60, lr=1.0, T=24)

        # After this insertion, re-query the probe identities (those already inserted)
        for probe_idx in probe_indices:
            if probe_idx > k: continue
            probe_pid = ids_sorted[probe_idx]
            probe_idxs = list(by_id[probe_pid])
            # Use 3 queries per probe (cross-condition: NOT the registration sample)
            correct = 0; total = 0
            for q_local in range(1, min(4, len(probe_idxs))):
                q_emb = ev_emb[probe_idxs[q_local]]
                q_code_arr = apply_fn(q_emb[None])[0]
                q_code = int(q_code_arr.item() if hasattr(q_code_arr, 'item') else q_code_arr)
                if query_one(bolt, q_code, MODALITY_AUDIO, tok, marker_offset, k + 1, probe_idx):
                    correct += 1
                total += 1
            retrieval_curve[probe_idx].append(correct / max(total, 1))
        if (k + 1) % 5 == 0:
            print(f"  after {k+1} insertions: probe-1 retr {retrieval_curve[0][-1]:.2f}, "
                  f"probe-5 retr {retrieval_curve[4][-1] if retrieval_curve[4] else float('nan'):.2f}, "
                  f"probe-10 retr {retrieval_curve[9][-1] if retrieval_curve[9] else float('nan'):.2f}")

    out = Path("/home/ubuntu/multimodal-user-memory/results/forgetting_probe.json")
    with open(out, "w") as f:
        json.dump({
            "probe_indices": probe_indices,
            "retrieval_curves": {str(k): v for k, v in retrieval_curve.items()},
            "n_insertions_tested": len(ids_sorted),
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("HEADLINE: retention curves")
    print("=" * 70)
    print(f"{'Insertions':>11} | {'probe 1 (1st)':>14} | {'probe 5 (5th)':>14} | {'probe 10 (10th)':>16}")
    print("-" * 70)
    for i in range(len(ids_sorted)):
        p1 = retrieval_curve[0][i] if i < len(retrieval_curve[0]) else None
        p5 = retrieval_curve[4][i-4] if i >= 4 and (i-4) < len(retrieval_curve[4]) else None
        p10 = retrieval_curve[9][i-9] if i >= 9 and (i-9) < len(retrieval_curve[9]) else None
        line = f"{i+1:>11} | "
        line += f"{p1:>14.3f} | " if p1 is not None else f"{'--':>14} | "
        line += f"{p5:>14.3f} | " if p5 is not None else f"{'--':>14} | "
        line += f"{p10:>16.3f}" if p10 is not None else f"{'--':>16}"
        print(line)

    # Initial retrieval (when only that identity was registered):
    init_p1 = retrieval_curve[0][0] if retrieval_curve[0] else None
    final_p1 = retrieval_curve[0][-1] if retrieval_curve[0] else None
    print(f"\nProbe 1: initial retrieval (N=1) = {init_p1:.3f}, after {len(ids_sorted)} insertions = {final_p1:.3f}")
    if init_p1 is not None and final_p1 is not None:
        print(f"  drift: {final_p1 - init_p1:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
