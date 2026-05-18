"""AttentionMemory: pretrain + eval at multiple N.

Pretraining: build random (key, value) banks from the training pool;
forward a query designed to elicit the marker; CE loss on the LM's
predicted next-token at the perceptual position. Trainable params:
W_q, W_o, log_tau (~200K), plus the per-modality input projection
(~1M).

Eval: for each evaluation pool, reset bank, insert each registered
identity (encoder embedding → key; marker token id → value embedding),
then for each query sample compute LM forward and check the predicted
next-token marker against ground truth.

Usage:
  python3 attmem_train_and_eval.py <mode> <n_steps> [seed]
  mode in {a-para, v-xc-id-face, ...}
"""
import json
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
from attention_memory import MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE
from v2_retrieval import split_by_identity, embedding_rag_ceiling

torch.manual_seed(42); np.random.seed(42)


MODE_PATHS = {
    "a-xr-id": ("ecapa_libri_large.npz", MODALITY_AUDIO),
    "a-scn":   ("ast_esc50_full.npz",    MODALITY_AUDIO),
    "a-para":  ("wav2vec_para_spk_emo.npz", MODALITY_AUDIO),
    "v-xc-id": ("arcface_lfw_xl.npz",    MODALITY_VISION),
    "v-xc-id-face": ("arcface_face_combined.npz", MODALITY_VISION),
    "v-xc-id-xxxl": ("arcface_face_xxxl.npz", MODALITY_VISION),
    "v-sty":   ("style_pca_gram.npz",    MODALITY_VISION),
    "v-sty-clip": ("clip_mid_wikiart.npz", MODALITY_VISION),
}


def build_query_context(tok, marker_token_id: int, T: int = 24):
    """Build the [text-prefix, perceptual-position] input. The perceptual
    key is supplied separately; here we just produce the modality_ids and
    text_input_ids."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    prompt = "You see"
    pref_ids = tok.encode(prompt, add_special_tokens=False)
    pref = list(pref_ids) + [pad_id] * (T - 1 - len(pref_ids))
    pref = pref[: T - 1]
    text_input_ids = pref + [pad_id]  # last position is perceptual (id ignored)
    return text_input_ids


def pretrain(bolt, train_emb, train_pid, modality_id, tok,
              n_steps=5000, lr=3e-4, batch_banks=8, bank_size=64,
              T=24, marker_offset=30001, print_every=200,
              bank_size_max=None):
    """Pretrain W_q, W_o, log_tau on synthetic recall.

    Each step:
      - Sample `bank_size` train IDs at random; one sample per ID as the
        registration key, and a (random) marker token id assigned per ID.
      - Insert all into a fresh bank.
      - Sample a query: pick one of those IDs and a DIFFERENT sample of
        the same ID (cross-condition).
      - Forward through the LM; CE loss on the predicted next-token at
        the perceptual position vs the marker for the queried ID.

    Trainable: bolt.vis_proj or aud_proj, bolt.attmem.banks[mod].W_q,
    W_o, log_tau.
    """
    bank = bolt.attmem.banks[str(modality_id)]
    proj = bolt.vis_proj if modality_id == MODALITY_VISION else bolt.aud_proj
    params = list(bank.parameters()) + list(proj.parameters())
    n_train = sum(p.numel() for p in params if p.requires_grad)
    print(f"  pretrain trainable params: {n_train:,}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    by_id = defaultdict(list)
    for i, p in enumerate(train_pid):
        by_id[str(p)].append(i)
    ids = [p for p in by_id if len(by_id[p]) >= 2]
    rng = np.random.default_rng(0)
    t0 = time.time()
    losses = []
    for step in range(n_steps):
        # Sample a fresh bank — if bank_size_max set, sample uniformly between
        # bank_size and bank_size_max to expose the model to varying scales
        # (key fix for distribution shift between train and large-N eval).
        if bank_size_max is not None and bank_size_max > bank_size:
            bs_step = int(rng.integers(bank_size, bank_size_max + 1))
            bs_step = min(bs_step, len(ids))
        else:
            bs_step = bank_size
        bank.reset()
        chosen = rng.choice(len(ids), size=bs_step, replace=(bs_step > len(ids)))
        marker_ids = [marker_offset + k for k in range(bs_step)]
        # Per-id pick one registration sample
        reg_idxs = [int(rng.choice(by_id[ids[ix]])) for ix in chosen]
        reg_keys = torch.from_numpy(train_emb[reg_idxs].astype(np.float32)).to(DEVICE)
        bolt.insert_batch(modality_id, reg_keys, marker_ids)

        # Build a query batch (B=1 for simplicity; we can scale to multi-query later)
        q_pid_local = int(rng.integers(0, bs_step))
        q_id = ids[chosen[q_pid_local]]
        # Pick a different sample from the same ID as cross-condition query
        q_candidates = [i for i in by_id[q_id] if i != reg_idxs[q_pid_local]]
        if not q_candidates:
            q_candidates = by_id[q_id]
        q_idx = int(rng.choice(q_candidates))
        q_key = torch.from_numpy(train_emb[q_idx].astype(np.float32)).unsqueeze(0).to(DEVICE)

        text_ids = build_query_context(tok, marker_ids[q_pid_local], T=T)
        text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
        modality_ids_t = torch.tensor(
            [[MODALITY_TEXT] * (T - 1) + [int(modality_id)]],
            dtype=torch.long, device=DEVICE,
        )
        perc_keys_by_mod = {int(modality_id): q_key}

        logits = bolt(modality_ids_t, text_ids_t, perc_keys_by_mod)
        # Predict the NEXT token after the perceptual position (i.e. the marker)
        # The LM outputs logits[:, -1, :] for predicting position T (after T-1).
        # Target: the marker token for q_pid_local.
        target = torch.tensor([marker_ids[q_pid_local]], dtype=torch.long, device=DEVICE)
        loss = F.cross_entropy(logits[:, -1, :], target)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % print_every == 0:
            recent = float(np.mean(losses[-50:]))
            print(f"    step {step+1:5d}  loss={recent:.4f}  inv_temp={float(torch.exp(bank.log_inv_temp).item()):.1f}  gain={float(bank.out_gain.item()):.2f}  ({time.time()-t0:.0f}s)")
    return losses


def evaluate(bolt, eval_emb, eval_pid, modality_id, tok,
              N_subset=None, n_queries_per_id=3, marker_offset=30001, T=24):
    """Eval: insert N_subset registered identities, then for each id run
    n_queries cross-condition queries and check argmax marker."""
    bank = bolt.attmem.banks[str(modality_id)]
    bank.reset()

    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid):
        by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None:
        ids_sorted = ids_sorted[:N_subset]
    marker_ids = list(range(marker_offset, marker_offset + len(ids_sorted)))

    # Insert: one registration sample per id
    rng = np.random.default_rng(99)
    reg_idx_per_id = []
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_idx_per_id.append(idxs[0])
    reg_keys = torch.from_numpy(eval_emb[reg_idx_per_id].astype(np.float32)).to(DEVICE)
    bolt.insert_batch(modality_id, reg_keys, marker_ids)

    # Query: for each id, n_queries_per_id cross-condition queries
    correct = 0; total = 0
    for k, pid in enumerate(ids_sorted):
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        q_idxs = [i for i in idxs if i != reg_idx_per_id[k]][:n_queries_per_id]
        for qi in q_idxs:
            q_key = torch.from_numpy(eval_emb[qi].astype(np.float32)).unsqueeze(0).to(DEVICE)
            text_ids = build_query_context(tok, marker_offset, T=T)
            text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
            modality_ids_t = torch.tensor(
                [[MODALITY_TEXT] * (T - 1) + [int(modality_id)]],
                dtype=torch.long, device=DEVICE,
            )
            with torch.no_grad():
                logits = bolt(modality_ids_t, text_ids_t, {int(modality_id): q_key})
                last = logits[0, -1, :]
                # Argmax restricted to the registered markers
                marker_logits = torch.stack([last[m] for m in marker_ids])
                pred_local = int(marker_logits.argmax().item())
            total += 1
            if pred_local == k:
                correct += 1
    return {
        "N_registered": len(ids_sorted),
        "N_queries": total,
        "retrieval_at_1": correct / total if total else 0.0,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "a-para"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    bank_size_max = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0 = fixed bank_size=64

    print("=" * 70)
    print(f"AttentionMemory train+eval — mode={mode}  n_steps={n_steps}  seed={seed}")
    print("=" * 70)

    torch.manual_seed(seed); np.random.seed(seed)
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary, modality_id = MODE_PATHS[mode]
    d = np.load(EMB / primary)
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    n_train_ids = len(set(tr_pid.tolist())); n_eval_ids = len(set(ev_pid.tolist()))
    print(f"  data: train {n_train_ids} IDs / {len(tr_emb)} samp, "
          f"eval {n_eval_ids} IDs / {len(ev_emb)} samp; modality={modality_id}")

    print(f"\nLoading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    key_dim = emb.shape[1]
    bolt = QwenAttMemBolt(
        qwen, tok,
        vision_key_dim=(key_dim if modality_id == MODALITY_VISION else 512),
        audio_key_dim=(key_dim if modality_id == MODALITY_AUDIO else 192),
        attach_layer=33,  # Near the end of Qwen2.5-3B's 36 layers — closer to lm_head
    ).to(DEVICE)
    bolt.install_hook()

    if n_steps > 0:
        bs_max = bank_size_max if bank_size_max > 0 else None
        print(f"\n[pretrain] {n_steps} steps  bank_size=64"
              + (f"..{bs_max} (uniform)" if bs_max else " (fixed)"))
        losses = pretrain(bolt, tr_emb, tr_pid, modality_id, tok,
                           n_steps=n_steps, lr=3e-4, batch_banks=1, bank_size=64,
                           bank_size_max=bs_max,
                           T=24, marker_offset=30001, print_every=max(1, n_steps // 25))
        print(f"  final loss (last 50): {float(np.mean(losses[-50:])):.4f}")
    else:
        print(f"\n[ZERO-SHOT] skipping pretraining — eval with W_o=0.5*I, tau=1.0 only")
        losses = [0.0]

    # Eval at multiple N
    print(f"\n[eval — RAG cosine-only baseline + AttentionMemory]")
    Ns = [N for N in [5, 10, 20, 50, 100, 300, 700, 1000] if N <= n_eval_ids]
    print(f"{'N':>5} | {'RAG':>6} | {'AttMem':>7} | {'ratio':>6} | {'verdict'}")
    print("-" * 55)
    results = {}
    for N in Ns:
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=3)
        r = evaluate(bolt, ev_emb, ev_pid, modality_id, tok,
                       N_subset=N, n_queries_per_id=3, T=24)
        attmem = r["retrieval_at_1"]
        ratio = attmem / rag if rag > 0 else float("nan")
        verdict = ("BEATS" if attmem > rag else
                   ("near"  if ratio > 0.85 else
                    ("comp"  if ratio > 0.5 else "below")))
        print(f"{N:>5} | {rag:>6.3f} | {attmem:>7.3f} | {ratio:>6.2f} | {verdict}")
        results[N] = {"rag": rag, "attmem": attmem, "ratio": ratio,
                       "N_queries": r["N_queries"]}

    suffix = f"_bsmax{bank_size_max}" if bank_size_max > 0 else ""
    # If non-default model, add a model tag to filename so 7B/14B results don't collide with 3B.
    model_tag = ""
    if "3B" not in MODEL_ID:
        # e.g., "Qwen/Qwen2.5-7B-Instruct" -> "_qwen7b"
        if "7B" in MODEL_ID: model_tag = "_qwen7b"
        elif "14B" in MODEL_ID: model_tag = "_qwen14b"
        else: model_tag = "_" + MODEL_ID.split("/")[-1].lower().replace("-", "")
    out = Path(f"/home/ubuntu/multimodal-user-memory/results/attmem_{mode}_steps{n_steps}_seed{seed}{suffix}{model_tag}.json")
    with open(out, "w") as f:
        json.dump({"mode": mode, "n_steps": n_steps, "seed": seed,
                    "model_id": MODEL_ID,
                    "n_train_ids": n_train_ids, "n_eval_ids": n_eval_ids,
                    "final_loss": float(np.mean(losses[-50:])),
                    "results": {str(N): v for N, v in results.items()}},
                   f, indent=2, default=str)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    sys.exit(main())
