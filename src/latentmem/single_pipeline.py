"""Unified in-LM benchmark: text-only vs latent-only vs hybrid user memory.

A single pipeline that measures all three memory architectures end-to-end through
the *same* frozen LM + AttentionMemory mechanism, on the *same* registered face
population, so the comparison is apples-to-apples (no cross-experiment composition).

Task. Each user = a real ArcFace face (cross-condition, non-captionable) + a
private fact. Register one photo + the fact; query a DIFFERENT photo and recall
the fact. Recall needs both legs: match the face (perceptual), then recall its
fact (captionable).

Architectures (all over the identical registered population + trained memory):
  text_only    caption-coded identity -> fact dict   (no LM; weak PERCEPTUAL leg)
  latent_only  AttMem with the FACT as the marker-value; recall the fact marker
  hybrid       AttMem with the IDENTITY as the marker-value; recall identity,
               then expand to the fact via a text dict

Fact-cardinality sweep C in {2, 10, 50, unique=N}: tests whether a single latent
marker can stand in for the text store. For categorical facts a marker suffices,
so latent_only ~= hybrid; the text channel earns its keep only for exact /
multi-token fact CONTENT a single marker cannot hold (see EXACT note below and
HYBRID_FINDINGS.md).

Design notes
  * The perceptual memory (W_q, W_o, log_tau, projection) is trained ONCE on a
    disjoint identity split; statistics are over eval randomisation (fact
    assignment, photo and identity selection), which is what we want CIs on.
  * Recall is batched for throughput. The eval identities never overlap the
    memory's training identities.

Usage
  python3 single_pipeline.py --train_steps 4000 --seeds 0 1 2 \
      --ns 10 50 100 300 --out ../../results/single_pipeline.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

ROOT = Path("/home/ubuntu/multimodal-user-memory")
sys.path.insert(0, str(ROOT / "src" / "nanochat_mm"))

from attention_memory import MODALITY_TEXT, MODALITY_VISION          # noqa: E402
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE         # noqa: E402
from attmem_train_and_eval import pretrain, build_query_context      # noqa: E402
from v2_retrieval import split_by_identity                           # noqa: E402

log = logging.getLogger("single_pipeline")
MARKER_OFFSET = 30001
EMB_FILE = ROOT / "runs" / "embeddings" / "arcface_lfw_xxxl.npz"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_faces():
    d = np.load(EMB_FILE)
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    return (tr_emb, tr_pid), (ev_emb, ev_pid)


def group_by_id(pid):
    by = {}
    for i, p in enumerate(pid):
        by.setdefault(str(p), []).append(i)
    return by


def lsh_codes(X, R):
    """Coarse caption proxy: locality-sensitive hash (Path-A code regime)."""
    bits = (X @ R) > 0
    return (bits * (1 << np.arange(R.shape[1]))).sum(axis=1)


# ---------------------------------------------------------------------------
# Memory training (perceptual matcher) — trained ONCE, reused by all conditions
# ---------------------------------------------------------------------------
def build_and_train(train, steps: int, seed: int):
    tr_emb, tr_pid = train
    torch.manual_seed(seed); np.random.seed(seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log.info("loading %s ...", MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}, low_cpu_mem_usage=True).eval()
    bolt = QwenAttMemBolt(qwen, tok, vision_key_dim=tr_emb.shape[1],
                          audio_key_dim=192, attach_layer=33).to(DEVICE)
    bolt.install_hook()
    if steps > 0:
        log.info("pretraining perceptual memory: %d steps", steps)
        losses = pretrain(bolt, tr_emb, tr_pid, MODALITY_VISION, tok, n_steps=steps,
                          lr=3e-4, batch_banks=1, bank_size=64, bank_size_max=1024,
                          T=24, marker_offset=MARKER_OFFSET, print_every=max(1, steps // 10))
        log.info("  final loss (last 50): %.4f", float(np.mean(losses[-50:])))
    else:
        log.info("ZERO-SHOT memory (no pretraining)")
    return bolt, tok


# ---------------------------------------------------------------------------
# Batched marker recall through the LM
# ---------------------------------------------------------------------------
@torch.no_grad()
def recall_markers(bolt, tok, q_keys: torch.Tensor, marker_set: list[int],
                   T: int = 24, batch: int = 128) -> np.ndarray:
    """For each query embedding, return argmax index into `marker_set`."""
    base_text = build_query_context(tok, MARKER_OFFSET, T=T)
    mset = torch.tensor(marker_set, device=DEVICE)
    preds = []
    for s in range(0, q_keys.shape[0], batch):
        qb = q_keys[s:s + batch]
        b = qb.shape[0]
        text_ids = torch.tensor([base_text] * b, dtype=torch.long, device=DEVICE)
        mod_ids = torch.tensor([[MODALITY_TEXT] * (T - 1) + [MODALITY_VISION]] * b,
                               dtype=torch.long, device=DEVICE)
        logits = bolt(mod_ids, text_ids, {MODALITY_VISION: qb})
        last = logits[:, -1, :]                       # [b, V]
        preds.append(last[:, mset].argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def register(bolt, keys: torch.Tensor, marker_ids: list[int]):
    bank = bolt.attmem.banks[str(MODALITY_VISION)]
    bank.reset()
    bolt.insert_batch(MODALITY_VISION, keys, marker_ids)


# ---------------------------------------------------------------------------
# One eval cell: register N identities, assign facts (cardinality C), run all 3
# ---------------------------------------------------------------------------
def eval_cell(bolt, tok, ev, by_id, ids_all, seed: int, N: int, C: int):
    rng = np.random.default_rng(seed)
    ev_emb, _ = ev
    sel = list(rng.choice(ids_all, size=N, replace=False))
    reg_idx, q_idx, q_owner = [], [], []
    for k, pid in enumerate(sel):
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_idx.append(idxs[0])
        for qi in idxs[1:3]:                          # up to 2 cross-condition queries
            q_idx.append(qi); q_owner.append(k)
    reg_keys = torch.from_numpy(ev_emb[reg_idx].astype(np.float32)).to(DEVICE)
    q_keys = torch.from_numpy(ev_emb[q_idx].astype(np.float32)).to(DEVICE)
    q_owner = np.array(q_owner)

    Ceff = N if C == 0 else min(C, N)                 # C==0 -> unique fact per id
    facts = rng.integers(0, Ceff, size=N)             # arbitrary fact per identity
    true_fact = facts[q_owner]

    # ---- hybrid: identity markers -> recall identity -> fact dict
    register(bolt, reg_keys, list(range(MARKER_OFFSET, MARKER_OFFSET + N)))
    pred_id = recall_markers(bolt, tok, q_keys, list(range(MARKER_OFFSET, MARKER_OFFSET + N)))
    hybrid = float((facts[pred_id] == true_fact).mean())
    id_recall = float((pred_id == q_owner).mean())

    # ---- latent_only: fact markers as the stored value -> recall fact directly
    fact_markers = [MARKER_OFFSET + int(f) for f in facts]
    register(bolt, reg_keys, fact_markers)
    pred_fact = recall_markers(bolt, tok, q_keys,
                               list(range(MARKER_OFFSET, MARKER_OFFSET + Ceff)))
    latent = float((pred_fact == true_fact).mean())

    # ---- text_only: caption code (LSH) -> majority fact (no LM)
    R = rng.standard_normal((ev_emb.shape[1], 8)).astype(np.float32)
    kcode = lsh_codes(ev_emb[reg_idx], R); qcode = lsh_codes(ev_emb[q_idx], R)
    code_fact = {}
    for c, f in zip(kcode, facts):
        code_fact.setdefault(int(c), []).append(int(f))
    code_major = {c: int(np.bincount(fs).argmax()) for c, fs in code_fact.items()}
    text_pred = np.array([code_major.get(int(c), rng.integers(0, Ceff)) for c in qcode])
    text = float((text_pred == true_fact).mean())

    return {"text_only": text, "latent_only": latent, "hybrid": hybrid,
            "id_recall": id_recall, "n_queries": len(q_idx), "C_eff": Ceff}


def paired_t(a, b):
    d = np.array(a) - np.array(b)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float(d.mean()), float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    return float(d.mean()), float(1 - NormalDist().cdf(t))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 50, 100, 300])
    ap.add_argument("--cs", type=int, nargs="+", default=[2, 10, 50, 0],
                    help="fact cardinalities; 0 = unique fact per identity")
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "single_pipeline.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    train, ev = load_faces()
    by_id = group_by_id(ev[1])
    ids_all = [p for p, idx in by_id.items() if len(idx) >= 2]
    log.info("eval pool: %d identities (>=2 photos)", len(ids_all))

    t0 = time.time()
    bolt, tok = build_and_train(train, args.train_steps, args.train_seed)

    rows = []
    for C in args.cs:
        for N in args.ns:
            if N > len(ids_all):
                continue
            acc = {k: [] for k in ("text_only", "latent_only", "hybrid", "id_recall")}
            for seed in args.seeds:
                r = eval_cell(bolt, tok, ev, by_id, ids_all, seed, N, C)
                for k in acc:
                    acc[k].append(r[k])
            row = {"C": C, "C_eff_max": (N if C == 0 else min(C, N)), "N": N,
                   "n_seeds": len(args.seeds)}
            for k in acc:
                row[f"{k}_mean"] = float(np.mean(acc[k]))
                row[f"{k}_std"] = float(np.std(acc[k], ddof=1)) if len(acc[k]) > 1 else 0.0
            row["d_hyb_text"], row["p_hyb_text"] = paired_t(acc["hybrid"], acc["text_only"])
            row["d_hyb_lat"], row["p_hyb_lat"] = paired_t(acc["hybrid"], acc["latent_only"])
            rows.append(row)
            log.info("C=%s N=%d | text=%.3f latent=%.3f hybrid=%.3f (id=%.3f)",
                     C, N, row["text_only_mean"], row["latent_only_mean"],
                     row["hybrid_mean"], row["id_recall_mean"])

    Path(args.out).write_text(json.dumps(
        {"model": MODEL_ID, "train_steps": args.train_steps, "seeds": args.seeds,
         "elapsed_s": time.time() - t0, "rows": rows}, indent=2))

    print("\n=== UNIFIED IN-LM MEMORY BENCHMARK (recall@1 of the fact) ===")
    print(f"{'C':>5} {'N':>5} | {'text_only':>12} {'latent_only':>12} {'hybrid':>12} "
          f"{'id_rec':>7} | {'H-text':>8} {'H-lat':>8}")
    for r in rows:
        cl = "uniq" if r["C"] == 0 else str(r["C"])
        f = lambda k: f"{r[k+'_mean']:.3f}±{r[k+'_std']:.3f}"
        print(f"{cl:>5} {r['N']:>5} | {f('text_only'):>12} {f('latent_only'):>12} "
              f"{f('hybrid'):>12} {r['id_recall_mean']:>7.3f} | "
              f"{r['d_hyb_text']:>+8.3f} {r['d_hyb_lat']:>+8.3f}")
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
