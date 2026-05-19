"""Pretrain AttMem on Qwen2.5-VL using Qwen-VL's own visual encoder keys.

Step 1: extract visual keys for all AgeDB training images (cache to npz).
Step 2: pretrain the AttMem bolt parameters using the recipe from the
        text-only Qwen run, but with raw images going through Qwen-VL's
        image processor + visual encoder.
Step 3: eval cross-condition retrieval at multiple N.

Cached keys live in /home/ubuntu/multimodal-user-memory/runs/embeddings/qwenvl_agedb_keys.npz.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION, MODALITY_TEXT
from qwen_vl_attmem_bolt import QwenVLAttMemBolt, VLM_MODEL_ID, DEVICE


KEYS_PATH = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/qwenvl_agedb_keys.npz")


def extract_or_load_keys(bolt, n_samples_per_id=None):
    """Extract Qwen-VL visual keys for the AgeDB dataset, with caching."""
    if KEYS_PATH.exists():
        print(f"  Loading cached keys from {KEYS_PATH}")
        d = np.load(KEYS_PATH, allow_pickle=True)
        return d["keys"], d["pid"], d["age"]

    print(f"  Cache miss; extracting Qwen-VL keys for AgeDB ...")
    print(f"  Loading AgeDB ...")
    d = load_dataset("ljnlonoljpiljm/agedb", split="train")
    by_id = defaultdict(list)
    for i, row in enumerate(d):
        by_id[row["identity"]].append(i)

    # If n_samples_per_id given, cap per-id samples (e.g. 4 for speed)
    if n_samples_per_id:
        for pid in by_id:
            by_id[pid] = by_id[pid][:n_samples_per_id]
    n_total = sum(len(v) for v in by_id.values())
    print(f"  {len(by_id)} identities, {n_total} total samples (capped at {n_samples_per_id}/id)")

    keys = []
    pids = []
    ages = []
    t0 = time.time()
    n_done = 0
    for pid, idxs in by_id.items():
        for i in idxs:
            row = d[i]
            try:
                vk = bolt.extract_visual_key(row["image"]).float().cpu().numpy()
            except Exception as e:
                print(f"    SKIP {i} {pid}: {e}")
                continue
            keys.append(vk)
            pids.append(pid)
            ages.append(row["age"])
            n_done += 1
            if n_done % 100 == 0:
                print(f"    extracted {n_done}/{n_total}, {time.time()-t0:.0f}s elapsed")

    keys = np.stack(keys, axis=0)
    pids = np.array(pids)
    ages = np.array(ages)
    KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(KEYS_PATH, keys=keys, pid=pids, age=ages)
    print(f"  saved {len(keys)} keys to {KEYS_PATH} ({time.time()-t0:.0f}s)")
    return keys, pids, ages


def pretrain(bolt, train_keys, train_pid, n_steps=3000, lr=3e-4, bank_size=32,
              T=24, marker_offset=30001, adv_prob=0.0, adv_K=8, print_every=100):
    """Same recipe as attmem_train_and_eval.pretrain but Qwen-VL keys."""
    bank = bolt.attmem.banks[str(MODALITY_VISION)]
    params = list(bank.parameters())
    print(f"  pretrain trainable params: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    by_id = defaultdict(list)
    for i, p in enumerate(train_pid):
        by_id[str(p)].append(i)
    ids = [p for p in by_id if len(by_id[p]) >= 2]
    rng = np.random.default_rng(0)

    # Adversarial setup
    cos_mat = None; top_distractors = None
    if adv_prob > 0:
        canon = []
        for pid in ids:
            canon.append(train_keys[by_id[pid]].mean(axis=0))
        canon = np.stack(canon, axis=0).astype(np.float32)
        canon_n = canon / (np.linalg.norm(canon, axis=1, keepdims=True) + 1e-9)
        cos_mat = canon_n @ canon_n.T
        np.fill_diagonal(cos_mat, -1)
        top_distractors = np.argsort(-cos_mat, axis=1)[:, :adv_K]

    pad_id = bolt.processor.tokenizer.pad_token_id or 0
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        use_adv = (adv_prob > 0 and rng.random() < adv_prob)
        if use_adv:
            target_local = int(rng.integers(0, len(ids)))
            chosen = np.array([target_local] + top_distractors[target_local].tolist())
            q_local = 0
        else:
            chosen = rng.choice(len(ids), size=bank_size, replace=False)
            q_local = int(rng.integers(0, bank_size))
        bs_step = len(chosen)
        marker_ids = [marker_offset + k for k in range(bs_step)]
        reg_idxs = [int(rng.choice(by_id[ids[ix]])) for ix in chosen]
        reg_keys_np = train_keys[reg_idxs]
        reg_keys_t = torch.from_numpy(reg_keys_np.astype(np.float32)).to(DEVICE).to(torch.bfloat16)

        # Insert
        bank.reset()
        bolt.insert_batch(MODALITY_VISION, reg_keys_t, marker_ids)

        # Query: pick a different sample from same id
        q_id = ids[chosen[q_local]]
        q_candidates = [i for i in by_id[q_id] if i != reg_idxs[q_local]]
        if not q_candidates: q_candidates = by_id[q_id]
        q_idx = int(rng.choice(q_candidates))
        q_key_np = train_keys[q_idx]
        q_key = torch.from_numpy(q_key_np.astype(np.float32)).unsqueeze(0).to(DEVICE).to(torch.bfloat16)

        # Build a synthetic input: T text tokens, last one tagged as VISION
        # We bypass the VLM image encoder and just use the perceptual_keys path
        # so the bank-attention query uses our extracted Qwen-VL visual key.
        input_ids = torch.tensor([[pad_id] * T], device=DEVICE)
        modality_ids = torch.zeros(1, T, dtype=torch.long, device=DEVICE)
        modality_ids[:, -1] = MODALITY_VISION

        # Hook will use _last_modality_ids and _last_perc_keys_by_mod
        bolt._last_modality_ids = modality_ids
        bolt._last_perc_keys_by_mod = {MODALITY_VISION: q_key}

        # Run the full qwen_vl forward so the hook on lm_head fires
        out = bolt.qwen_vl(input_ids=input_ids, use_cache=False)
        logits = out.logits if hasattr(out, 'logits') else out[0]
        target = torch.tensor([marker_ids[q_local]], dtype=torch.long, device=DEVICE)
        loss = F.cross_entropy(logits[:, -1, :], target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % print_every == 0:
            recent = float(np.mean(losses[-50:]))
            print(f"    step {step+1:5d}  loss={recent:.3f}  ({time.time()-t0:.0f}s)")
    return losses


def evaluate(bolt, eval_keys, eval_pid, N, n_queries_per_id=1, marker_offset=30001):
    """For each of N identities, register one key, query with another, check argmax."""
    bank = bolt.attmem.banks[str(MODALITY_VISION)]
    bank.reset()

    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid):
        by_id[str(p)].append(i)
    ids_sorted = sorted(k for k in by_id if len(by_id[k]) >= 2)[:N]
    marker_ids = list(range(marker_offset, marker_offset + len(ids_sorted)))
    rng = np.random.default_rng(99)
    reg_idx_per_id = []
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_idx_per_id.append(idxs[0])
    reg_keys = torch.from_numpy(eval_keys[reg_idx_per_id].astype(np.float32)).to(DEVICE).to(torch.bfloat16)
    bolt.insert_batch(MODALITY_VISION, reg_keys, marker_ids)

    correct_attmem = 0; correct_rag = 0; total = 0
    T = 24
    pad_id = bolt.processor.tokenizer.pad_token_id or 0
    input_ids = torch.tensor([[pad_id] * T], device=DEVICE)
    modality_ids = torch.zeros(1, T, dtype=torch.long, device=DEVICE)
    modality_ids[:, -1] = MODALITY_VISION

    reg_n = eval_keys[reg_idx_per_id] / (np.linalg.norm(eval_keys[reg_idx_per_id], axis=1, keepdims=True) + 1e-9)

    for k, pid in enumerate(ids_sorted):
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        q_idxs = [i for i in idxs if i != reg_idx_per_id[k]][:n_queries_per_id]
        for qi in q_idxs:
            q_key_np = eval_keys[qi]
            q_n = q_key_np / (np.linalg.norm(q_key_np) + 1e-9)
            # RAG
            sim = reg_n @ q_n
            if int(np.argmax(sim)) == k: correct_rag += 1
            # AttMem
            q_key = torch.from_numpy(q_key_np.astype(np.float32)).unsqueeze(0).to(DEVICE).to(torch.bfloat16)
            bolt._last_modality_ids = modality_ids
            bolt._last_perc_keys_by_mod = {MODALITY_VISION: q_key}
            with torch.no_grad():
                out = bolt.qwen_vl(input_ids=input_ids, use_cache=False)
                logits = out.logits if hasattr(out, 'logits') else out[0]
                last = logits[0, -1, :]
                ml = torch.stack([last[m] for m in marker_ids])
                pred = int(ml.argmax().item())
            if pred == k: correct_attmem += 1
            total += 1
    return {"N": N, "n_queries": total,
            "rag_retr1": correct_rag / total if total else 0,
            "attmem_retr1": correct_attmem / total if total else 0}


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    adv_prob = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    print(f"\n=== AttMem-VL pretrain on Qwen2.5-VL (steps={n_steps}, seed={seed}, adv_prob={adv_prob}) ===")
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"Loading {VLM_MODEL_ID} ...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID, trust_remote_code=True)
    qwen_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        trust_remote_code=True, low_cpu_mem_usage=True,
    ); qwen_vl.eval()
    bolt = QwenVLAttMemBolt(qwen_vl, processor).to(DEVICE)
    bolt.install_hook()

    # Extract or load keys
    print(f"\n[1/3] Visual keys for AgeDB ...")
    keys, pid, age = extract_or_load_keys(bolt, n_samples_per_id=4)
    keys = keys.astype(np.float32)
    print(f"  loaded {len(keys)} keys, dim={keys.shape[1]}")

    # Train/eval split by identity
    unique_ids = sorted(set(pid.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    n_train = int(len(unique_ids) * 0.5)
    train_ids = set(unique_ids[:n_train])
    eval_ids = set(unique_ids[n_train:])
    tr_mask = np.array([p in train_ids for p in pid])
    tr_keys = keys[tr_mask]; tr_pid = pid[tr_mask]
    ev_keys = keys[~tr_mask]; ev_pid = pid[~tr_mask]
    print(f"  train: {len(set(tr_pid.tolist()))} IDs / {len(tr_keys)} samples")
    print(f"  eval:  {len(set(ev_pid.tolist()))} IDs / {len(ev_keys)} samples")

    # Pretrain
    print(f"\n[2/3] Pretraining for {n_steps} steps ...")
    losses = pretrain(bolt, tr_keys, tr_pid, n_steps=n_steps,
                       bank_size=32, adv_prob=adv_prob)

    # Eval at multiple N
    print(f"\n[3/3] Eval at multiple N ...")
    Ns = [N for N in [5, 10, 20, 50, 100] if N <= len(set(ev_pid.tolist()))]
    results = {}
    for N in Ns:
        r = evaluate(bolt, ev_keys, ev_pid, N=N)
        print(f"  N={N}: RAG={r['rag_retr1']:.3f}, AttMem={r['attmem_retr1']:.3f}, Δ={r['attmem_retr1']-r['rag_retr1']:+.3f}")
        results[N] = r

    suffix = f"_advp{int(adv_prob*100):02d}" if adv_prob > 0 else ""
    out = Path(f"/home/ubuntu/multimodal-user-memory/results/attmem_vl_train_steps{n_steps}_seed{seed}{suffix}.json")
    with open(out, "w") as f:
        json.dump({"model": VLM_MODEL_ID, "n_steps": n_steps, "seed": seed,
                    "adv_prob": adv_prob,
                    "final_loss": float(np.mean(losses[-50:])),
                    "results": {str(N): r for N, r in results.items()}},
                   f, indent=2, default=str)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
