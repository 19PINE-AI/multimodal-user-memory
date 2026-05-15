"""Style encoder: end-to-end contrastive style head on top of frozen DINOv2.

PCA-Gram was the best fixed-encoder result (top-1 0.42). A trained head
should push past that. Recipe:
  - Frozen DINOv2-small backbone (features)
  - Small projection head (MLP, ~1M params)
  - Train on WikiArt painters as classes with supervised-contrastive
    loss (SupCon, Khosla et al. 2020): same-painter pairs pull, different
    pull apart, normalised by class.
  - Hold out 25% of painters for eval (the same train/eval split logic
    we use elsewhere). The head learns style structure from training
    painters; we measure transfer to UNSEEN painters via top-1 NN
    recall + collision-rate quantisation.

Win condition: head-projected features give top-1 painter recall > 0.50
on HELD-OUT painters, with K=32 ratio > 10. That would unblock V-STY
as a clean cell rather than a partial/limitation cell.
"""
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def supcon_loss(features, labels, temperature=0.07):
    """SupConLoss: same-class samples pull together, different push apart.
    features: [B, D] normalised.
    labels: [B] class ids.
    """
    B = features.size(0)
    sim = features @ features.t() / temperature  # [B, B]
    # Mask out self-pairs
    eye = torch.eye(B, device=features.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e4)
    # Positive mask: same label
    labels = labels.contiguous().view(-1, 1)
    pos_mask = (labels == labels.t()).float()
    pos_mask = pos_mask.masked_fill(eye, 0)
    # Log-softmax over all non-self entries
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # For each sample, average log_prob over positives
    pos_count = pos_mask.sum(1).clamp(min=1)
    loss = -(pos_mask * log_prob).sum(1) / pos_count
    # Drop samples with no positives
    return loss[pos_count > 0].mean()


def main():
    print("=" * 70)
    print("Style: contrastive head on frozen DINOv2")
    print("=" * 70)

    from transformers import AutoImageProcessor, AutoModel
    print("[load] DINOv2-small backbone (frozen) ...")
    backbone_id = "facebook/dinov2-small"
    proc = AutoImageProcessor.from_pretrained(backbone_id)
    backbone = AutoModel.from_pretrained(backbone_id, torch_dtype=torch.float32).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    D = backbone.config.hidden_size  # 384
    print(f"  D={D}")

    print("[data] WikiArt — pick 100 painters with >=12 paintings, split 75/25 ...")
    from datasets import load_dataset
    ds = load_dataset("huggan/wikiart", split="train")
    artist_col = ds["artist"]
    by_artist = defaultdict(list)
    for i, a in enumerate(artist_col):
        by_artist[int(a)].append(i)
    eligible = [(a, idxs) for a, idxs in by_artist.items() if len(idxs) >= 12]
    print(f"  {len(eligible)} painters with >= 12 paintings")
    chosen = random.sample(eligible, k=min(100, len(eligible)))
    random.shuffle(chosen)
    n_train = int(len(chosen) * 0.75)
    train_set = chosen[:n_train]
    eval_set = chosen[n_train:]
    print(f"  train: {len(train_set)} painters, eval (DISJOINT): {len(eval_set)} painters")

    # Pre-encode all paintings via DINOv2 (one-time cost)
    def encode_imgs(idxs, batch=32):
        out = []
        for k in range(0, len(idxs), batch):
            batch_idxs = idxs[k:k+batch]
            imgs = []
            for i in batch_idxs:
                im = ds[i]["image"]
                if not isinstance(im, Image.Image): continue
                if im.mode != "RGB": im = im.convert("RGB")
                imgs.append(im)
            inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                o = backbone(**inputs)
            f = o.pooler_output if o.pooler_output is not None else o.last_hidden_state.mean(dim=1)
            out.append(f.cpu())
        return torch.cat(out, dim=0)

    # Build train data: pick ~10 paintings per train painter
    train_feats = []
    train_labels = []
    label_map = {a: i for i, (a, _) in enumerate(train_set)}
    print("[encode] DINOv2 features for training painters ...")
    for a, idxs in train_set:
        sel = random.sample(idxs, k=min(12, len(idxs)))
        f = encode_imgs(sel)
        train_feats.append(f)
        train_labels.extend([label_map[a]] * f.size(0))
    train_feats = torch.cat(train_feats, dim=0).to(DEVICE)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=DEVICE)
    print(f"  train feats: {train_feats.shape}, {train_labels.unique().numel()} classes")

    # Eval data: ~8 paintings per held-out painter
    print("[encode] DINOv2 features for HELD-OUT painters ...")
    eval_feats = []
    eval_labels = []
    eval_label_map = {a: i for i, (a, _) in enumerate(eval_set)}
    for a, idxs in eval_set:
        sel = random.sample(idxs, k=min(8, len(idxs)))
        f = encode_imgs(sel)
        eval_feats.append(f)
        eval_labels.extend([a] * f.size(0))
    eval_feats = torch.cat(eval_feats, dim=0).to(DEVICE)
    print(f"  eval feats: {eval_feats.shape}, {len(set(eval_labels))} classes")

    # Define head: MLP from D -> D' with normalisation
    proj_dim = 256
    head = nn.Sequential(
        nn.Linear(D, 1024), nn.GELU(),
        nn.Linear(1024, proj_dim),
    ).to(DEVICE)

    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)

    print("\n[train] SupCon for 500 steps, batch=64 ...")
    BATCH = 64; STEPS = 500
    t0 = time.time()
    for step in range(STEPS):
        # Random batch of training samples
        idx = torch.randint(0, train_feats.size(0), (BATCH,), device=DEVICE)
        x = train_feats[idx]
        y = train_labels[idx]
        z = head(x)
        z = F.normalize(z, dim=-1)
        loss = supcon_loss(z, y, temperature=0.07)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  (elapsed {time.time()-t0:.0f}s)")

    # Project eval feats
    head.eval()
    with torch.no_grad():
        emb = head(eval_feats); emb = F.normalize(emb, dim=-1)
    emb = emb.cpu().numpy().astype(np.float32)
    pid = np.array([str(l) for l in eval_labels])
    print(f"\n[eval] projected eval feats: {emb.shape}")

    # Sanity
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn_idx = sims.argmax(axis=1)
    nn_same = (pid[nn_idx] == pid).mean()
    print(f"\n[sanity] contrastive-head top-1 painter recall (held-out): {nn_same:.4f}  (baseline DINOv2=0.24, Gram+PCA=0.42)")

    by_p = defaultdict(list)
    for i, p in enumerate(pid): by_p[p].append(i)
    intra, inter = [], []
    for p, idxs in by_p.items():
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                intra.append(sims[idxs[i], idxs[j]])
    rng = np.random.RandomState(SEED)
    for _ in range(5000):
        i, j = rng.randint(0, len(emb), size=2)
        if pid[i] != pid[j]: inter.append(sims[i, j])
    print(f"[sanity] intra-painter cosine: mean={np.mean(intra):.4f}")
    print(f"[sanity] inter-painter cosine: mean={np.mean(inter):.4f}")

    import faiss
    print(f"\n[sanity] quantisation:")
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
    results = {}
    for K in [8, 16, 32, 64]:
        if K > len(emb): continue
        km = faiss.Kmeans(proj_dim, K, niter=20, verbose=False, seed=SEED)
        km.train(emb)
        _, codes = km.index.search(emb, 1); codes = codes.squeeze(1)
        intra_agree = 0; intra_pairs = 0
        for p, idxs in by_p.items():
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    intra_pairs += 1
                    if codes[idxs[i]] == codes[idxs[j]]: intra_agree += 1
        rng2 = np.random.RandomState(SEED + K)
        inter_coll = 0; inter_pairs = 0
        for _ in range(10000):
            i, j = rng2.randint(0, len(emb), size=2)
            if pid[i] != pid[j]:
                inter_pairs += 1
                if codes[i] == codes[j]: inter_coll += 1
        intra_rate = intra_agree / max(intra_pairs, 1)
        inter_rate = inter_coll / max(inter_pairs, 1)
        ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
        print(f"  {K:>4d} | {intra_rate:>12.4f} | {inter_rate:>12.4f} | {ratio:>8.2f}")
        results[K] = (intra_rate, inter_rate, ratio)

    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/style_contrastive.npz")
    np.savez(out_emb, emb=emb, pid=pid)
    print(f"\n[done] saved {out_emb}")

    # Save head
    torch.save(head.state_dict(), "/home/ubuntu/multimodal-user-memory/runs/style_contrastive_head.pt")

    import json
    with open("/home/ubuntu/multimodal-user-memory/results/sanity_style_contrastive.json", "w") as f:
        json.dump({
            "encoder": f"{backbone_id} + contrastive head 1024->256",
            "n_train_painters": len(train_set), "n_eval_painters": len(eval_set),
            "top1_recall_heldout": float(nn_same),
            "intra_mean": float(np.mean(intra)),
            "inter_mean": float(np.mean(inter)),
            "quantisation": {str(K): {"intra_agree": float(a), "inter_coll": float(b),
                                       "ratio": float(r) if r != float("inf") else None}
                             for K, (a, b, r) in results.items()},
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
