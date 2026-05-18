"""Real-name qualitative demo: register 10 AgeDB celebrities by their real first name
(as single-token markers in Qwen's vocabulary), then query with held-out images.
Show the LM's next-token continuation = the registered first name.

This is the "concrete flavor" figure for the paper: it shows the bolt-on mechanism
actually produces human-readable continuations, not arbitrary 30001+ markers.

Usage: python3 src/nanochat_mm/attmem_named_demo.py [n_steps] [seed]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION, MODALITY_TEXT
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE
from v2_retrieval import split_by_identity


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = zero-shot
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    torch.manual_seed(seed); np.random.seed(seed)

    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    d = np.load(EMB / "arcface_face_xxxl.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])

    # Find AgeDB identities (prefix A) and split first/last name
    def first_name(p):
        if not p.startswith("A"): return None
        raw = p[1:]
        if not raw or not raw[0].isupper(): return None
        first = ""
        for ch in raw[0:]:
            if ch.isupper() and first: break
            first += ch
        return first if 3 <= len(first) <= 12 else None

    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)

    print("Loading tokenizer + Qwen ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    # Pick 10 IDs whose first name is single-token AND distinct
    chosen = []
    seen_first = set()
    rng = np.random.default_rng(seed)
    all_ids = list(by_id.keys()); rng.shuffle(all_ids)
    for p in all_ids:
        if len(chosen) >= 10:
            break
        f = first_name(p)
        if f is None or f in seen_first or len(by_id[p]) < 2:
            continue
        ids_t = tok.encode(f, add_special_tokens=False)
        if len(ids_t) != 1:
            continue
        seen_first.add(f)
        chosen.append((p, f, ids_t[0]))

    print(f"\n=== Registering {len(chosen)} face IDs with real first-name markers ===")
    for p, f, tid in chosen:
        full = p[1:]
        print(f"  {full:30s} -> register as '{f}' (token id {tid})")

    # Build the bolt
    bolt = QwenAttMemBolt(qwen, tok, vision_key_dim=512, audio_key_dim=192,
                          attach_layer=33, attach_lm_head=True).to(DEVICE)
    bolt.install_hook()
    marker_tokens = [c[2] for c in chosen]
    first_names = [c[1] for c in chosen]

    # If n_steps > 0, do a small pretrain over a separate AgeDB pool to teach
    # the bolt how to route encoder embeddings to value-side embeddings.
    if n_steps > 0:
        from attmem_train_and_eval import pretrain
        tr_emb, tr_pid, _, _ = split_by_identity(emb, pid)
        print(f"\n[pretrain] {n_steps} steps over training pool (markers refreshed each step)")
        pretrain(bolt, tr_emb, tr_pid, MODALITY_VISION, tok,
                  n_steps=n_steps, lr=3e-4, bank_size=64, bank_size_max=512,
                  T=24, marker_offset=30001, print_every=max(1, n_steps // 10))
    bolt.reset_banks()

    # Insert demo IDs with their real-name marker tokens
    reg_idxs = []
    for p, f, tid in chosen:
        idx_list = by_id[p]
        rng.shuffle(idx_list)
        reg_idxs.append(idx_list[0])
    reg_keys = torch.from_numpy(emb[reg_idxs].astype(np.float32)).to(DEVICE)
    bolt.insert_batch(MODALITY_VISION, reg_keys, marker_tokens)

    # Query: for each chosen ID, pick a held-out cross-condition sample and run LM forward
    T = 24
    pad_id = tok.pad_token_id or 0
    pref = tok.encode("You see", add_special_tokens=False)
    text_ids = list(pref) + [pad_id] * (T - 1 - len(pref))
    text_ids = (text_ids[: T - 1]) + [pad_id]
    text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
    modality_ids_t = torch.tensor(
        [[MODALITY_TEXT] * (T - 1) + [int(MODALITY_VISION)]], dtype=torch.long, device=DEVICE
    )

    print(f"\n=== Query: 'You see [face_emb] ->' (LM's top-3 continuations) ===")
    correct = 0
    for k, (p, expected_name, expected_tid) in enumerate(chosen):
        idx_list = by_id[p]
        rng.shuffle(idx_list)
        q_candidates = [i for i in idx_list if i != reg_idxs[k]]
        if not q_candidates: continue
        q_idx = q_candidates[0]
        q_key = torch.from_numpy(emb[q_idx].astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = bolt(modality_ids_t, text_ids_t, {int(MODALITY_VISION): q_key})
            last = logits[0, -1, :]
            # Top-3 over all of vocab — show what the LM actually predicts
            top3 = last.topk(3)
            top3_names = []
            for tid, lg in zip(top3.indices.tolist(), top3.values.tolist()):
                tok_str = tok.decode([tid]).strip()
                # Mark if it's one of our registered markers
                marker = "[REG]" if tid in marker_tokens else "    "
                top3_names.append(f"'{tok_str}' ({lg:+.1f}){marker}")
            # Also restrict to just registered markers for retr@1
            marker_logits = torch.stack([last[m] for m in marker_tokens])
            pred_local = int(marker_logits.argmax().item())

        full = p[1:]
        is_correct = (pred_local == k)
        correct += int(is_correct)
        check = "✓" if is_correct else "✗"
        print(f"  {full:25s} expected='{expected_name}'")
        print(f"      LM top-3:  {' | '.join(top3_names)}")
        print(f"      restricted argmax over markers: '{first_names[pred_local]}'  {check}")
    print(f"\n  retr@1 = {correct}/{len(chosen)} = {correct/len(chosen):.2f}")


if __name__ == "__main__":
    main()
