"""Path A + Engram pretraining.

After session 4 we have: bolt-on Engram on Qwen2.5-3B, no pretraining,
achieves 0.46-0.91 code-match retrieval. Goal here: pretrain the Engram
(+ perceptual-emb tables) on training-identity sessions, then test
held-out surgical insertion + retrieval.

The pretraining recipe is the simplest possible "teach the gate to use
perceptual codes for output":
  - Each training identity gets a unique training-time marker token.
  - Training example: [fixed-text prefix] [training perceptual code]
    → target = training marker.
  - NTP CE loss on the marker position. Engram + perceptual-emb update;
    Qwen frozen.

We expect this to teach the Engram a GENERAL capability of "fire on a
perceptual code, output a marker". At test time, held-out surgical
insertion installs new (held-out code, new marker) pairs into a small
disjoint subset of Engram rows. Hypothesis: code-match retrieval on
held-out should improve over the from-scratch surgical insertion result.

If it does → pretraining + Path A is the headline recipe.
If it doesn't (or hurts) → pretraining may interfere; STE codebook is the
remaining lever.
"""
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import (
    MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO,
)
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import (
    QwenEngramBolt, build_fixed_context, get_touched_rows, surgical_insert, evaluate,
    MODEL_ID, DEVICE,
)

torch.manual_seed(42); np.random.seed(42)


def pretrain_engram(bolt, train_emb, train_pid, codebook_apply, modality_id, tok,
                    train_marker_offset, n_steps=1500, lr=3e-4, batch=8, T=24,
                    print_every=100):
    """Train Engram + perceptual-emb (Qwen frozen) on (training code → training marker)."""
    eng = bolt.engram.engrams[str(modality_id)]
    if modality_id == MODALITY_VISION:
        perc_emb = bolt.vis_perc_emb
    else:
        perc_emb = bolt.aud_perc_emb

    # Group training samples by identity & assign each id a marker
    by_id = defaultdict(list)
    for i, p in enumerate(train_pid):
        by_id[str(p)].append(i)
    train_ids = sorted(by_id.keys())
    train_markers = {pid: train_marker_offset + i for i, pid in enumerate(train_ids)}
    print(f"  pretraining: {len(train_ids)} training identities; "
          f"markers {train_marker_offset}..{train_marker_offset + len(train_ids) - 1}")

    # Precompute codes for each training sample
    codes_all = codebook_apply(train_emb)
    # If shape is (N, n_levels=1), flatten to (N,) — handle both shapes
    if codes_all.ndim == 2 and codes_all.shape[1] == 1:
        codes_all = codes_all[:, 0]

    # Trainable params
    params_to_opt = list(eng.parameters()) + [perc_emb.weight]
    opt = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=0.01)

    # Build a fixed prefix once
    sample_input, _ = build_fixed_context(0, modality_id, tok, marker_text_id=0, T=T)
    fixed_prefix = sample_input[0, :-1].tolist()
    fixed_mids   = [MODALITY_TEXT] * len(fixed_prefix)

    rng = np.random.default_rng(0)
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        # Sample batch: random training identity each → its code + its marker
        ids_batch = rng.choice(train_ids, size=batch, replace=True)
        input_ids_b = []
        modality_ids_b = []
        target_b = []
        for pid in ids_batch:
            # Random sample of this identity
            samp_idx = int(rng.choice(by_id[pid]))
            code = int(codes_all[samp_idx])
            # Build sequence: fixed prefix + code at last position
            ids = fixed_prefix + [code]
            mids = fixed_mids + [int(modality_id)]
            input_ids_b.append(ids)
            modality_ids_b.append(mids)
            target_b.append(train_markers[pid])

        input_ids = torch.tensor(input_ids_b, dtype=torch.long, device=DEVICE)
        modality_ids = torch.tensor(modality_ids_b, dtype=torch.long, device=DEVICE)
        targets = torch.tensor(target_b, dtype=torch.long, device=DEVICE)

        logits = bolt(input_ids, modality_ids)  # [B, T, V]
        last = logits[:, -1, :]
        loss = F.cross_entropy(last, targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_opt, 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % print_every == 0:
            avg = float(np.mean(losses[-50:]))
            dt = time.time() - t0
            print(f"    step {step+1:4d}  loss={avg:.4f}  "
                  f"(elapsed {dt:.0f}s, ~{1000*dt/(step+1):.1f} ms/step)")

    # Quick sanity: training accuracy
    bolt.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for pid in train_ids:
            for samp_idx in by_id[pid]:
                code = int(codes_all[samp_idx])
                ids = fixed_prefix + [code]
                mids = fixed_mids + [int(modality_id)]
                input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                modality_ids = torch.tensor([mids], dtype=torch.long, device=DEVICE)
                logits = bolt(input_ids, modality_ids)
                # Restrict to training-marker range to measure capability
                marker_logits = logits[0, -1, train_marker_offset: train_marker_offset + len(train_ids)]
                pred_local = int(marker_logits.argmax().item())
                pred_pid = train_ids[pred_local]
                total += 1
                if pred_pid == pid: correct += 1
    print(f"  train accuracy on {len(train_ids)} ids × samples: {correct}/{total} = {correct/total:.4f}")
    return losses, correct / total


def main():
    print("=" * 70)
    print("Path A + Engram pretraining")
    print("=" * 70)

    # ---- Load Qwen + embeddings + codebook ----
    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()
    print(f"  {sum(p.numel() for p in qwen.parameters())/1e9:.2f}B params")

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")
    aud_tr_emb, aud_tr_pid, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr_emb, vis_tr_pid, vis_ev_emb, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    K = 32
    aud_apply = fit_naive_rq(aud_tr_emb, n_levels=1, k_per=K)
    vis_apply = fit_naive_rq(vis_tr_emb, n_levels=1, k_per=K)
    print(f"  audio train: {len(aud_tr_emb)} embs / {len(set(aud_tr_pid))} ids; eval: {len(aud_ev_emb)} / {len(set(aud_ev_pid))}")
    print(f"  vision train: {len(vis_tr_emb)} embs / {len(set(vis_tr_pid))} ids; eval: {len(vis_ev_emb)} / {len(set(vis_ev_pid))}")

    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                           engram_attach_layer=24,
                           engram_n_embed_per_ngram=128,
                           engram_vocab_per_ngram=503,
                           engram_n_head=4).to(DEVICE)
    bolt.install_hook()

    # ---- Pretrain ----
    # Use a separate marker block for training so eval-time markers don't collide
    TRAIN_MARKER_OFFSET = 20001  # arbitrary spot in Qwen vocab
    EVAL_MARKER_OFFSET = 30001   # disjoint from training markers (matches qwen_engram_bolt.py)

    print("\n[pretrain] vision Engram ...")
    vis_losses, vis_acc = pretrain_engram(
        bolt, vis_tr_emb, vis_tr_pid, vis_apply, MODALITY_VISION, tok,
        train_marker_offset=TRAIN_MARKER_OFFSET, n_steps=1000, lr=3e-4, batch=8, T=24,
    )

    print("\n[pretrain] audio Engram ...")
    aud_losses, aud_acc = pretrain_engram(
        bolt, aud_tr_emb, aud_tr_pid, aud_apply, MODALITY_AUDIO, tok,
        train_marker_offset=TRAIN_MARKER_OFFSET, n_steps=1000, lr=3e-4, batch=8, T=24,
    )

    # ---- Now evaluate on held-out with surgical insertion ----
    print("\n" + "=" * 70)
    print("Held-out surgical insertion retrieval (after pretraining)")
    print("=" * 70)
    Ns = [5, 10, 20]; nq = 5
    results = {"pretrain_losses": {"vision": vis_losses[-100:], "audio": aud_losses[-100:]},
                "pretrain_train_acc": {"vision": vis_acc, "audio": aud_acc}}

    for mid, name, emb, pids, apply_fn in [
        (MODALITY_VISION, "vision", vis_ev_emb, vis_ev_pid, vis_apply),
        (MODALITY_AUDIO,  "audio",  aud_ev_emb, aud_ev_pid, aud_apply),
    ]:
        print(f"\n[{name}]")
        rag = {N: embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq) for N in Ns}
        v_held = {}
        for N in Ns:
            print(f"  RAG ceiling N={N}: {rag[N]:.4f}")
            print(f"  Path A + pretrain — held-out surgical insertion N={N} ...", end="", flush=True)
            r = evaluate(bolt, apply_fn, emb, pids, mid, tok,
                          N_subset=N, n_queries_per_id=nq,
                          max_steps=80, lr=1.0, T=24)
            print(f"  retr@1={r['retrieval_at_1']:.4f}  "
                  f"(insert avg {r['avg_insert_steps']:.0f} steps, final loss {r['avg_insert_loss']:.3f})  "
                  f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
                  f"collisions={r['N_collision_codes']}")
            v_held[N] = r
        results[name] = {"rag": rag, "pathA_with_pretrain": v_held}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_pretrain.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[done] Wrote {out}")

    # Comparison vs no-pretrain
    pa_path = Path("/home/ubuntu/multimodal-user-memory/results/pathA_qwen_bolt.json")
    if pa_path.exists():
        with open(pa_path) as f: pa = json.load(f)
        print("\n" + "=" * 95)
        print("HEADLINE — RAG | Path A no-pretrain | Path A + pretrain | Δ code-match")
        print("=" * 95)
        print(f"{'modality':>8} | {'N':>3} | {'RAG':>6} | {'no-pretrain':>11} | {'+ pretrain':>10} |   delta  | code-match no-p | code-match + p")
        print("-" * 95)
        for name in ["vision", "audio"]:
            for N in Ns:
                rag_v = results[name]["rag"][N]
                with_p = results[name]["pathA_with_pretrain"][N]["retrieval_at_1"]
                with_p_cm = results[name]["pathA_with_pretrain"][N]["code_match_retr"]
                no_p = pa[name]["pathA_no_pretrain"][str(N)]["retrieval_at_1"]
                no_p_cm = pa[name]["pathA_no_pretrain"][str(N)]["code_match_retr"]
                delta = with_p - no_p
                mark = " ✓" if delta > 0.02 else (" ≈" if abs(delta) <= 0.02 else " ✗")
                print(f"{name:>8} | {N:>3} | {rag_v:>6.3f} | {no_p:>11.3f} | {with_p:>10.3f} | {delta:>+8.3f}{mark} | "
                      f"{no_p_cm:>14.3f} | {with_p_cm:>13.3f}")


if __name__ == "__main__":
    sys.exit(main())
