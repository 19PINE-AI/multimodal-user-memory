"""Catastrophic forgetting probe v2 — with row-freeze mitigation.

Experiment #8 showed Probe-1 retrieval collapsing from 1.0 to 0 after
only 4 additional sequential insertions. Inspection of the gradient
mask in `surgical_insert` shows the issue: when inserting identity k,
the optimiser writes to all rows touched by the n-gram hash for k's
code, but those rows are ALSO touched (via the same multi-head hash
table with bounded slot count) by previously-inserted codes. The
later writes overwrite the earlier ones' marker bias.

Mitigation: track rows previously written by surgical_insert across
sequential calls. Subsequent insertions mask out those rows from the
gradient step. This is the "surgical insertion" property in its
strict form — each row is written once and frozen thereafter.

Additionally: freeze the corresponding perc_emb row(s) — a code that
has been registered cannot have its perceptual embedding changed by
later registrations of the SAME code (a separate identity that
happened to quantise to the same code is a real collision and not
something a freeze can solve; we measure both with and without the
perc-emb freeze).
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
from engram_module_mm import MODALITY_AUDIO
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity
from qwen_engram_bolt import (
    QwenEngramBolt, build_fixed_context, get_touched_rows, MODEL_ID, DEVICE,
)
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)


def surgical_insert_freeze(bolt, code, modality_id, marker, tok,
                            frozen_rows, frozen_perc_codes,
                            max_steps=60, lr=1.0, T=24):
    """surgical_insert with row freeze.

    frozen_rows: dict[layer_str -> set(int)] of Engram rows that must NOT
                 be updated (used by earlier insertions).
    frozen_perc_codes: set[int] of perc_emb rows that must NOT be updated.

    The function returns the touched rows for THIS insertion so the caller
    can add them to the frozen set for subsequent calls.
    """
    eng = bolt.engram.engrams[str(modality_id)]
    input_ids, modality_ids = build_fixed_context(code, modality_id, tok, marker, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
    touched = get_touched_rows(eng, code, input_ids)

    if modality_id == 1:  # MODALITY_VISION
        perc_emb_param = bolt.vis_perc_emb.weight
    else:
        perc_emb_param = bolt.aud_perc_emb.weight

    # Effective writable rows = touched minus frozen.
    writable = {ks: set(rows) - frozen_rows.get(ks, set()) for ks, rows in touched.items()}
    params_to_opt = [eng.tables[ks].embedding.weight for ks in writable] + [perc_emb_param]
    opt = torch.optim.SGD(params_to_opt, lr=lr, momentum=0.0)

    target = torch.tensor([marker], dtype=torch.long, device=DEVICE)
    last_loss = float("inf"); steps_taken = 0
    perc_frozen_now = int(code) in frozen_perc_codes
    for step in range(max_steps):
        logits = bolt(input_ids, modality_ids)
        last = logits[:, -1, :]
        loss = F.cross_entropy(last, target)
        last_loss = float(loss.item())
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            # Engram: only writable (touched - frozen) rows
            for ks, rows in writable.items():
                W = eng.tables[ks].embedding.weight
                if W.grad is None: continue
                if not rows:
                    W.grad.zero_(); continue
                mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.grad.dtype)
                row_idx = torch.tensor(sorted(rows), device=W.device, dtype=torch.long)
                mask[row_idx] = 1.0
                W.grad.mul_(mask)
            # Perc-emb: only this code's row, and only if not frozen
            if perc_emb_param.grad is not None:
                if perc_frozen_now:
                    perc_emb_param.grad.zero_()
                else:
                    pmask = torch.zeros(perc_emb_param.shape[0], 1,
                                         device=perc_emb_param.device,
                                         dtype=perc_emb_param.grad.dtype)
                    pmask[int(code)] = 1.0
                    perc_emb_param.grad.mul_(pmask)
        opt.step()
        steps_taken = step + 1
        if last_loss < 0.5:
            break
    return steps_taken, last_loss, touched


def query_one(bolt, code, modality_id, tok, marker_offset, N_registered, true_marker_idx, T=24):
    inp, mids = build_fixed_context(code, modality_id, tok, marker_text_id=0, T=T)
    inp = inp.to(DEVICE); mids = mids.to(DEVICE)
    markers = list(range(marker_offset, marker_offset + N_registered))
    with torch.no_grad():
        logits = bolt(inp, mids)
        marker_logits = torch.stack([logits[0, -1, m] for m in markers])
        pred_idx = int(marker_logits.argmax().item())
    return pred_idx == true_marker_idx


def run_sequence(bolt, tok, apply_fn, ev_emb, by_id, ids_sorted, marker_offset,
                  freeze: bool, probe_indices, label, rng):
    frozen_rows = defaultdict(set)
    frozen_perc_codes = set()
    retrieval_curve = defaultdict(list)
    inserted_codes = []
    print(f"\n--- run: {label} ({'FREEZE' if freeze else 'NO-FREEZE'}) ---")
    for k, pid in enumerate(ids_sorted):
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_emb = ev_emb[idxs[0]]
        reg_code_arr = apply_fn(reg_emb[None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        marker = marker_offset + k
        if freeze:
            steps, fl, touched = surgical_insert_freeze(
                bolt, reg_code, MODALITY_AUDIO, marker, tok,
                frozen_rows, frozen_perc_codes, max_steps=60, lr=1.0, T=24,
            )
            # After this insertion, lock its rows
            for ks, rows in touched.items():
                frozen_rows[ks].update(rows)
            frozen_perc_codes.add(reg_code)
        else:
            steps, fl, touched = surgical_insert_freeze(
                bolt, reg_code, MODALITY_AUDIO, marker, tok,
                defaultdict(set), set(), max_steps=60, lr=1.0, T=24,
            )
        inserted_codes.append(reg_code)
        for probe_idx in probe_indices:
            if probe_idx > k: continue
            probe_pid = ids_sorted[probe_idx]
            probe_idxs = list(by_id[probe_pid])
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
            print(f"  after {k+1}: p1={retrieval_curve[0][-1]:.2f}, "
                  f"p5={retrieval_curve[4][-1] if retrieval_curve[4] else float('nan'):.2f}, "
                  f"p10={retrieval_curve[9][-1] if retrieval_curve[9] else float('nan'):.2f}")
    return {str(k): v for k, v in retrieval_curve.items()}, inserted_codes


def snapshot_state(bolt, modality_id):
    eng = bolt.engram.engrams[str(modality_id)]
    eng_snap = {ks: tbl.embedding.weight.detach().clone() for ks, tbl in eng.tables.items()}
    perc_snap = bolt.aud_perc_emb.weight.detach().clone() if modality_id == 2 else bolt.vis_perc_emb.weight.detach().clone()
    return eng_snap, perc_snap


def restore_state(bolt, modality_id, eng_snap, perc_snap):
    eng = bolt.engram.engrams[str(modality_id)]
    with torch.no_grad():
        for ks, w in eng_snap.items():
            eng.tables[ks].embedding.weight.copy_(w)
        if modality_id == 2:
            bolt.aud_perc_emb.weight.copy_(perc_snap)
        else:
            bolt.vis_perc_emb.weight.copy_(perc_snap)


def main():
    print("=" * 70)
    print("Forgetting probe v2 — row freeze mitigation")
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
    apply_fn = fit_naive_rq(
        aud['emb'][np.in1d(aud['pid'], list(set(ev_pid))[:14])], n_levels=1, k_per=K)

    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K, engram_attach_layer=24).to(DEVICE)
    bolt.install_hook()
    print("[pretrain] generic-NTP 200 steps ...")
    pretrain_generic(bolt, aud['emb'], aud['pid'], apply_fn, MODALITY_AUDIO, tok,
                     n_steps=200, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)

    by_id = defaultdict(list)
    for i, p in enumerate(ev_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())[:20]
    probe_indices = [0, 4, 9]
    marker_offset = 30001

    # Snapshot post-pretrain state to compare with/without freeze fairly
    eng_snap, perc_snap = snapshot_state(bolt, 2)
    rng_a = np.random.default_rng(42)
    curve_nofreeze, codes_nf = run_sequence(
        bolt, tok, apply_fn, ev_emb, by_id, ids_sorted, marker_offset,
        freeze=False, probe_indices=probe_indices, label="baseline (no freeze)",
        rng=rng_a)

    restore_state(bolt, 2, eng_snap, perc_snap)
    rng_b = np.random.default_rng(42)
    curve_freeze, codes_f = run_sequence(
        bolt, tok, apply_fn, ev_emb, by_id, ids_sorted, marker_offset,
        freeze=True, probe_indices=probe_indices, label="freeze",
        rng=rng_b)

    out = Path("/home/ubuntu/multimodal-user-memory/results/forgetting_probe_v2.json")
    with open(out, "w") as f:
        json.dump({
            "probe_indices": probe_indices,
            "n_insertions_tested": len(ids_sorted),
            "baseline_no_freeze": curve_nofreeze,
            "freeze": curve_freeze,
            "codes_baseline": codes_nf,
            "codes_freeze": codes_f,
        }, f, indent=2)
    print(f"\n[done] {out}")

    print("\n" + "=" * 70)
    print("Probe-1 retention across sequential insertions")
    print("=" * 70)
    print(f"{'k':>3} | {'no-freeze':>10} | {'freeze':>8}")
    print("-" * 30)
    for i in range(len(ids_sorted)):
        a = curve_nofreeze.get("0", [])
        b = curve_freeze.get("0", [])
        ai = f"{a[i]:.2f}" if i < len(a) else "--"
        bi = f"{b[i]:.2f}" if i < len(b) else "--"
        print(f"{i+1:>3} | {ai:>10} | {bi:>8}")

    # Summarize: mean Probe-1 retrieval across the last 10 insertions
    a0 = curve_nofreeze.get("0", [])
    b0 = curve_freeze.get("0", [])
    if len(a0) >= 10 and len(b0) >= 10:
        a_tail = float(np.mean(a0[-10:]))
        b_tail = float(np.mean(b0[-10:]))
        print(f"\nMean Probe-1 retrieval over last 10 insertions:")
        print(f"  no-freeze: {a_tail:.3f}")
        print(f"  freeze:    {b_tail:.3f}")
        print(f"  delta:     {b_tail - a_tail:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
