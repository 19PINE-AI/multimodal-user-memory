"""Path A + GENERIC-NTP pretraining.

Replaces the marker-supervised pretraining (which locked the gate onto training
markers and hurt held-out retrieval) with general next-token-prediction on
cross-sequence recurrence streams. The gate learns to use perceptual codes
for predicting natural text continuations, without ever seeing a "marker
token" during pretraining. Held-out surgical insertion at evaluation time
then writes (perceptual_code, marker) pairs into the gate's available
capacity without competing against a learned marker bias.

Training data structure: each session has several training-identity perceptual
codes interspersed in random Qwen text. The NTP target is the actual next
token (no specific marker). The Engram has to use the perceptual code to
predict the random text continuation correctly — which it can only do by
learning to attend to the recurrent identity in the session.
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
from engram_module_mm import MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import QwenEngramBolt, evaluate, MODEL_ID, DEVICE

torch.manual_seed(42); np.random.seed(42)


def build_pretrain_batch(rng, tok, train_emb, train_pid, codebook_apply,
                          modality_id, V_text=151643, T=64, batch=4,
                          frac_perceptual=0.15):
    """Build a NTP training batch with perceptual codes embedded in random
    Qwen-vocab text. Codes within a sequence are drawn from a few sampled
    training identities so recurrence exists.
    """
    by_id = defaultdict(list)
    for i, p in enumerate(train_pid):
        by_id[str(p)].append(i)
    train_ids = sorted(by_id.keys())

    # Precompute codes for the whole train set
    codes_all = codebook_apply(train_emb)
    if codes_all.ndim == 2 and codes_all.shape[1] == 1:
        codes_all = codes_all[:, 0]

    B = batch
    input_ids = np.zeros((B, T), dtype=np.int64)
    modality_ids = np.zeros((B, T), dtype=np.int64)
    for b in range(B):
        # Pick this session's identity (one focus identity for recurrence)
        focus_pid = str(rng.choice(train_ids))
        for t in range(T):
            if rng.random() < frac_perceptual:
                modality_ids[b, t] = modality_id
                # 70% focus identity, 30% random other identity (for distractors)
                if rng.random() < 0.7:
                    samp_idx = int(rng.choice(by_id[focus_pid]))
                else:
                    other_pid = str(rng.choice(train_ids))
                    samp_idx = int(rng.choice(by_id[other_pid]))
                input_ids[b, t] = int(codes_all[samp_idx])
            else:
                modality_ids[b, t] = MODALITY_TEXT
                # Random token from a moderate vocab range to keep loss meaningful
                input_ids[b, t] = int(rng.integers(1, 10000))
    return (torch.from_numpy(input_ids).to(DEVICE),
            torch.from_numpy(modality_ids).to(DEVICE))


def pretrain_generic(bolt, train_emb, train_pid, codebook_apply, modality_id, tok,
                      n_steps=600, lr=3e-4, batch=4, T=64,
                      frac_perceptual=0.15, print_every=100):
    """Generic NTP loss. Engram learns to use perceptual codes to improve
    text-token prediction, no marker bias."""
    eng = bolt.engram.engrams[str(modality_id)]
    if modality_id == MODALITY_VISION:
        perc_emb = bolt.vis_perc_emb
    else:
        perc_emb = bolt.aud_perc_emb
    params = list(eng.parameters()) + [perc_emb.weight]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    rng = np.random.default_rng(0)
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        input_ids, modality_ids = build_pretrain_batch(
            rng, tok, train_emb, train_pid, codebook_apply,
            modality_id, V_text=tok.vocab_size, T=T, batch=batch,
            frac_perceptual=frac_perceptual,
        )
        logits = bolt(input_ids, modality_ids)  # [B, T, V]
        # NTP on text positions only — predict next token where next is text
        target_mids = modality_ids[:, 1:]
        text_mask = (target_mids == MODALITY_TEXT)
        if not text_mask.any():
            continue
        pred = logits[:, :-1, :]
        target = input_ids[:, 1:]
        # Cross-entropy only on text-target positions
        pred_text = pred[text_mask]
        target_text = target[text_mask]
        loss = F.cross_entropy(pred_text, target_text)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % print_every == 0:
            avg = float(np.mean(losses[-50:]))
            dt = time.time() - t0
            print(f"    step {step+1:4d}  loss={avg:.4f}  (elapsed {dt:.0f}s)")
    return losses


def main():
    print("=" * 70)
    print("Path A + GENERIC-NTP pretraining (no marker supervision)")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")
    aud_tr_emb, aud_tr_pid, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr_emb, vis_tr_pid, vis_ev_emb, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    K = 32
    aud_apply = fit_naive_rq(aud_tr_emb, n_levels=1, k_per=K)
    vis_apply = fit_naive_rq(vis_tr_emb, n_levels=1, k_per=K)

    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                           engram_attach_layer=24,
                           engram_n_embed_per_ngram=128,
                           engram_vocab_per_ngram=503,
                           engram_n_head=4).to(DEVICE)
    bolt.install_hook()

    # Pretrain
    print("\n[pretrain] vision Engram (generic NTP) ...")
    vis_losses = pretrain_generic(bolt, vis_tr_emb, vis_tr_pid, vis_apply, MODALITY_VISION, tok,
                                    n_steps=600, lr=3e-4, batch=4, T=64,
                                    frac_perceptual=0.15)
    print(f"  vision pretrain final loss: {float(np.mean(vis_losses[-30:])):.4f}")

    print("\n[pretrain] audio Engram (generic NTP) ...")
    aud_losses = pretrain_generic(bolt, aud_tr_emb, aud_tr_pid, aud_apply, MODALITY_AUDIO, tok,
                                    n_steps=600, lr=3e-4, batch=4, T=64,
                                    frac_perceptual=0.15)
    print(f"  audio pretrain final loss: {float(np.mean(aud_losses[-30:])):.4f}")

    # Eval
    print("\n" + "=" * 70)
    print("Held-out surgical insertion retrieval after generic-NTP pretraining")
    print("=" * 70)
    Ns = [5, 10, 20]; nq = 5
    results = {"pretrain_losses": {"vision": vis_losses[-50:], "audio": aud_losses[-50:]}}
    for mid, name, emb, pids, apply_fn in [
        (MODALITY_VISION, "vision", vis_ev_emb, vis_ev_pid, vis_apply),
        (MODALITY_AUDIO,  "audio",  aud_ev_emb, aud_ev_pid, aud_apply),
    ]:
        print(f"\n[{name}]")
        rag = {N: embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq) for N in Ns}
        out_eval = {}
        for N in Ns:
            print(f"  RAG ceiling N={N}: {rag[N]:.4f}")
            print(f"  Path A + generic pretrain — held-out surgical N={N} ...", end="", flush=True)
            r = evaluate(bolt, apply_fn, emb, pids, mid, tok,
                         N_subset=N, n_queries_per_id=nq,
                         max_steps=80, lr=1.0, T=24)
            print(f"  retr@1={r['retrieval_at_1']:.4f}  "
                  f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
                  f"collisions={r['N_collision_codes']}  insert-loss={r['avg_insert_loss']:.3f}")
            out_eval[N] = r
        results[name] = {"rag": rag, "pathA_generic_pretrain": out_eval}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_generic_pretrain.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[done] Wrote {out}")

    # Side-by-side: no-pretrain vs marker-pretrain vs generic-pretrain
    pa_path = Path("/home/ubuntu/multimodal-user-memory/results/pathA_qwen_bolt.json")
    pm_path = Path("/home/ubuntu/multimodal-user-memory/results/pathA_pretrain.json")
    if pa_path.exists() and pm_path.exists():
        with open(pa_path) as f: pa = json.load(f)
        with open(pm_path) as f: pm = json.load(f)
        print("\n" + "=" * 100)
        print("HEADLINE — code-match retrieval (mechanism strength)")
        print("=" * 100)
        print(f"{'modality':>8} | {'N':>3} | {'no-pretrain':>11} | {'marker-pretrain':>15} | {'generic-pretrain':>16}")
        print("-" * 100)
        for name in ["vision", "audio"]:
            for N in Ns:
                no_p = pa[name]["pathA_no_pretrain"][str(N)]["code_match_retr"]
                m_p  = pm[name]["pathA_with_pretrain"][str(N)]["code_match_retr"]
                g_p  = results[name]["pathA_generic_pretrain"][N]["code_match_retr"]
                print(f"{name:>8} | {N:>3} | {no_p:>11.3f} | {m_p:>15.3f} | {g_p:>16.3f}")
        print()
        print(f"{'modality':>8} | {'N':>3} | {'no-pretrain':>11} | {'marker-pretrain':>15} | {'generic-pretrain':>16}  (overall retrieval)")
        print("-" * 100)
        for name in ["vision", "audio"]:
            for N in Ns:
                no_p = pa[name]["pathA_no_pretrain"][str(N)]["retrieval_at_1"]
                m_p  = pm[name]["pathA_with_pretrain"][str(N)]["retrieval_at_1"]
                g_p  = results[name]["pathA_generic_pretrain"][N]["retrieval_at_1"]
                print(f"{name:>8} | {N:>3} | {no_p:>11.3f} | {m_p:>15.3f} | {g_p:>16.3f}")


if __name__ == "__main__":
    sys.exit(main())
