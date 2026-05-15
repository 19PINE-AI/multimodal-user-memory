"""V-STY: contrastive style head trained on clip_mid_wikiart 50-painter pool.

Session 8's contrastive XL head trained on 75 painters didn't beat Gram+PCA
(top-1 0.35 vs 0.42). But session 8's eval was top-1 painter recall, not
PerceptMem same-code rate. We retry with three differences:

  1. Use the CLIP-mid encoder features (clip_mid_wikiart, 50 painters,
     400 samples) — more painters than the style_pca_gram pool (15
     painters). Identity-disjoint train/eval split: 25/25 painters.
  2. Train a small projection head + SupCon, output dim 128.
  3. Evaluate via same-code rate using id_codebook_v2's k-means in the
     projection output space. This is the V-STY-equivalent of V-XC-ID's
     winning recipe.

If this lifts V-STY's eval same-code rate, plug into Path A and rerun.
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

sys.path.insert(0, str(Path(__file__).parent))
from v2_retrieval import split_by_identity
from id_codebook_v2 import (
    InvarianceAdapter, supcon_loss, kmeans_centroids, apply_pipeline,
    evaluate_same_code_rate, evaluate_cross_id_collision, save_pipeline,
)

torch.manual_seed(42); np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StyleHead(nn.Module):
    """A larger projection head than the v2 adapter — D_in -> D_hidden -> D_out.
    For style: D_in 2304 (CLIP-mid concat), D_hidden 512, D_out 128. Output
    L2-normalised so cosine similarity = dot product."""
    def __init__(self, d_in, d_hidden=512, d_out=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


def train_style_head(tr_emb, tr_pid, d_hidden=512, d_out=128, dropout=0.2,
                     n_steps=4000, batch=128, lr=1e-3, temperature=0.1,
                     print_every=400, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    by_id = defaultdict(list)
    for i, p in enumerate(tr_pid):
        by_id[str(p)].append(i)
    ids = [p for p in by_id if len(by_id[p]) >= 2]
    print(f"  style head train: {len(ids)} painters with >=2 works")
    head = StyleHead(d_in=tr_emb.shape[1], d_hidden=d_hidden, d_out=d_out,
                      dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    emb_t = torch.from_numpy(tr_emb.astype(np.float32)).to(DEVICE)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(n_steps):
        K = batch // 2
        chosen_ids = rng.choice(len(ids), size=K, replace=(K > len(ids)))
        idx_list, lab_list = [], []
        for k, ix in enumerate(chosen_ids):
            pid_v = ids[ix]
            samps = by_id[pid_v]
            pair = rng.choice(len(samps), size=2, replace=(len(samps) < 2))
            idx_list.append(samps[pair[0]]); lab_list.append(k)
            idx_list.append(samps[pair[1]]); lab_list.append(k)
        x = emb_t[idx_list]
        y = torch.tensor(lab_list, dtype=torch.long, device=DEVICE)
        x = F.normalize(x, dim=-1)
        z = head(x)
        loss = supcon_loss(z, y, temperature=temperature)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % print_every == 0:
            print(f"    step {step+1:4d}  supcon={loss.item():.4f}  ({time.time()-t0:.0f}s)")
    head.eval()
    return head


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 16

    print("=" * 70)
    print(f"V-STY contrastive style head + id-codebook v2 — K={K}")
    print("=" * 70)

    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    # clip_mid_wikiart: 50 painters, 8 works each
    d = np.load(EMB / "clip_mid_wikiart.npz")
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    print(f"  data: {len(set(tr_pid))} train painters / {len(tr_emb)} works, "
          f"{len(set(ev_pid))} eval painters / {len(ev_emb)} works  D={emb.shape[1]}")

    # Train the style head with SupCon on train painters
    print("\n[train style head]")
    head = train_style_head(tr_emb, tr_pid, n_steps=4000, batch=128, lr=1e-3,
                              temperature=0.1, print_every=500, seed=42)

    # Project train embeddings; k-means in projection space; evaluate same-code
    print("\n[k-means + same-code rate]")
    with torch.no_grad():
        x = torch.from_numpy(tr_emb.astype(np.float32)).to(DEVICE)
        x = F.normalize(x, dim=-1)
        z_tr = head(x).cpu().numpy()
    centroids_np = kmeans_centroids(z_tr, K, seed=42)
    centroids_t = torch.from_numpy(centroids_np.astype(np.float32)).to(DEVICE)

    @torch.no_grad()
    def apply(emb_np):
        x = torch.from_numpy(emb_np.astype(np.float32)).to(DEVICE)
        x = F.normalize(x, dim=-1)
        z = head(x)
        d2 = (z.pow(2).sum(-1, keepdim=True)
              - 2 * z @ centroids_t.t()
              + centroids_t.pow(2).sum(-1))
        return d2.argmin(-1).cpu().numpy()

    tr_codes = apply(tr_emb); ev_codes = apply(ev_emb)
    tr_same, tr_used, _ = evaluate_same_code_rate(tr_codes, tr_pid, K)
    ev_same, ev_used, n_ev = evaluate_same_code_rate(ev_codes, ev_pid, K)
    ev_inter = evaluate_cross_id_collision(ev_codes, ev_pid)
    print(f"  train same-code = {tr_same:.3f} ({tr_used}/{K} codes used)")
    print(f"  eval  same-code = {ev_same:.3f} ({ev_used}/{K} codes used, "
          f"inter_coll={ev_inter:.3f})  n_pairs={n_ev}")

    # Compare to prior codebooks
    print("\n[baselines on the same eval split]")
    from real_encoder_train import fit_naive_rq
    # Naive k-means on raw clip_mid features
    naive_fn = fit_naive_rq(tr_emb, n_levels=1, k_per=K, seed=42)
    def naive_apply(e):
        c = naive_fn(e); return c[:, 0] if c.ndim == 2 else c
    naive_codes = naive_apply(ev_emb)
    n_same, _, _ = evaluate_same_code_rate(naive_codes, ev_pid, K)
    n_inter = evaluate_cross_id_collision(naive_codes, ev_pid)
    print(f"  naive k-means on raw clip_mid:  eval same-code = {n_same:.3f}  inter_coll = {n_inter:.3f}")

    # Save the style-head + centroids bundle
    out_dir = Path("/home/ubuntu/multimodal-user-memory/runs/codebooks")
    state = {
        "head_state": head.state_dict(),
        "head_d_in": tr_emb.shape[1], "head_d_hidden": 512, "head_d_out": 128,
        "centroids": centroids_np,
    }
    out_path = out_dir / f"id_v2_codebook_v-sty-head_K{K}.pt"
    torch.save(state, out_path)

    summary = {
        "K": K, "n_train_painters": len(set(tr_pid)), "n_eval_painters": len(set(ev_pid)),
        "train_same_code": tr_same,
        "eval_same_code": ev_same,
        "eval_inter_collision": ev_inter,
        "naive_kmeans_eval_same_code": n_same,
        "naive_kmeans_eval_inter": n_inter,
        "lift_over_naive": ev_same - n_same,
    }
    out_json = Path("/home/ubuntu/multimodal-user-memory/results/") / f"id_codebook_v2_v-sty-head_K{K}.json"
    with open(out_json, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_path}")
    print(f"[saved] {out_json}")

    print("\n" + "=" * 70)
    print("HEADLINE — V-STY eval same-code rate")
    print("=" * 70)
    print(f"  naive k-means (clip_mid):       {n_same:.3f}  inter_coll={n_inter:.3f}")
    print(f"  contrastive style head + km:    {ev_same:.3f}  inter_coll={ev_inter:.3f}  "
          f"Δ = {ev_same - n_same:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
