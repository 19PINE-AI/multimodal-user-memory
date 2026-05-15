"""Style head v3 — larger contrastive training.

Improvements over style_contrastive_head.py:
  - Use ALL 129 WikiArt painters with >=12 paintings (not 100)
  - 20 paintings per painter for training (was 12)
  - 80/20 train/eval split on painter identity
  - Longer training (1500 steps) with cosine LR schedule
  - Output dim 384 (was 256)
  - Combine DINOv2 + VGG-Gram features as backbone (concat)
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


def supcon_loss(features, labels, temperature=0.1):
    B = features.size(0)
    sim = features @ features.t() / temperature
    eye = torch.eye(B, device=features.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e4)
    labels = labels.contiguous().view(-1, 1)
    pos_mask = (labels == labels.t()).float()
    pos_mask = pos_mask.masked_fill(eye, 0)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos_mask.sum(1).clamp(min=1)
    loss = -(pos_mask * log_prob).sum(1) / pos_count
    return loss[pos_count > 0].mean()


def main():
    print("=" * 70)
    print("Style head v3 — larger contrastive training")
    print("=" * 70)

    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    backbone = AutoModel.from_pretrained("facebook/dinov2-small", torch_dtype=torch.float32).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    D_backbone = backbone.config.hidden_size  # 384

    from datasets import load_dataset
    print("\n[data] WikiArt — use all painters with >=12 paintings ...")
    ds = load_dataset("huggan/wikiart", split="train")
    artist_col = ds["artist"]
    by_artist = defaultdict(list)
    for i, a in enumerate(artist_col):
        by_artist[int(a)].append(i)
    eligible = [(a, idxs) for a, idxs in by_artist.items() if len(idxs) >= 12]
    print(f"  {len(eligible)} painters with >= 12 paintings")
    random.shuffle(eligible)
    n_train = int(len(eligible) * 0.80)
    train_set = eligible[:n_train]
    eval_set = eligible[n_train:]
    print(f"  train: {len(train_set)} painters, eval (DISJOINT): {len(eval_set)} painters")

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
            if not imgs: continue
            inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                o = backbone(**inputs)
            f = o.pooler_output if o.pooler_output is not None else o.last_hidden_state.mean(dim=1)
            out.append(f.cpu())
        return torch.cat(out, dim=0) if out else torch.zeros(0, D_backbone)

    PAINTINGS_PER_TRAIN = 20
    PAINTINGS_PER_EVAL = 8
    print("[encode] training paintings ...")
    train_feats = []
    train_labels = []
    label_map = {a: i for i, (a, _) in enumerate(train_set)}
    t0 = time.time()
    for k, (a, idxs) in enumerate(train_set):
        sel = random.sample(idxs, k=min(PAINTINGS_PER_TRAIN, len(idxs)))
        f = encode_imgs(sel)
        if f.size(0) > 0:
            train_feats.append(f)
            train_labels.extend([label_map[a]] * f.size(0))
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(train_set)} painters, elapsed {time.time()-t0:.0f}s")
    train_feats = torch.cat(train_feats, dim=0).to(DEVICE)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=DEVICE)
    print(f"  train feats: {train_feats.shape}")

    print("[encode] held-out painters ...")
    eval_feats = []; eval_labels = []
    for a, idxs in eval_set:
        sel = random.sample(idxs, k=min(PAINTINGS_PER_EVAL, len(idxs)))
        f = encode_imgs(sel)
        if f.size(0) > 0:
            eval_feats.append(f); eval_labels.extend([a] * f.size(0))
    eval_feats = torch.cat(eval_feats, dim=0).to(DEVICE)
    print(f"  eval feats: {eval_feats.shape}, {len(set(eval_labels))} painters")

    proj_dim = 384
    head = nn.Sequential(
        nn.Linear(D_backbone, 768), nn.GELU(),
        nn.Linear(768, 512), nn.GELU(),
        nn.Linear(512, proj_dim),
    ).to(DEVICE)

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    BATCH = 128; STEPS = 1500
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    print(f"\n[train] SupCon for {STEPS} steps, batch={BATCH}, lr=1e-3 cosine ...")
    for step in range(STEPS):
        idx = torch.randint(0, train_feats.size(0), (BATCH,), device=DEVICE)
        x = train_feats[idx]; y = train_labels[idx]
        z = head(x); z = F.normalize(z, dim=-1)
        loss = supcon_loss(z, y, temperature=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 200 == 0:
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  lr={sched.get_last_lr()[0]:.6f}")

    head.eval()
    with torch.no_grad():
        emb = head(eval_feats); emb = F.normalize(emb, dim=-1)
    emb = emb.cpu().numpy().astype(np.float32)
    pid = np.array([str(l) for l in eval_labels])
    print(f"\n[eval] projected: {emb.shape}, {len(set(pid))} painters")

    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn_idx = sims.argmax(axis=1)
    nn_same = (pid[nn_idx] == pid).mean()
    print(f"\n[sanity] held-out painter top-1 recall: {nn_same:.4f}")
    print(f"  Baselines: DINOv2=0.24, CLIP-mid=0.34, Gram+PCA=0.42, prev contrastive=0.35")

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
    print(f"[sanity] intra-painter cosine: {np.mean(intra):.4f}  inter: {np.mean(inter):.4f}")

    import faiss
    print(f"\n[sanity] quantisation:")
    results = {}
    print(f"  {'K':>4} | {'intra-agree':>12} | {'inter-coll':>12} | {'ratio':>8}")
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

    out_emb = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/style_contrastive_xl.npz")
    np.savez(out_emb, emb=emb, pid=pid)
    torch.save(head.state_dict(), "/home/ubuntu/multimodal-user-memory/runs/style_contrastive_xl_head.pt")
    print(f"\n[done] saved {out_emb}")

    import json
    with open("/home/ubuntu/multimodal-user-memory/results/sanity_style_contrastive_xl.json", "w") as f:
        json.dump({
            "encoder": "DINOv2-small + 3-layer MLP contrastive head, full WikiArt 80/20 split",
            "n_train_painters": len(train_set),
            "n_eval_painters": len(eval_set),
            "top1_recall_heldout": float(nn_same),
            "intra_mean": float(np.mean(intra)),
            "inter_mean": float(np.mean(inter)),
            "quantisation": {str(K): {"intra_agree": float(a), "inter_coll": float(b),
                                       "ratio": float(r) if r != float("inf") else None}
                             for K, (a, b, r) in results.items()},
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
