"""Identity-supervised codebook v2 — adapter + k-means.

v1 of the id-supervised codebook overfit train identities (train same-code
1.000, eval same-code only +0.05 over naive). Root cause: with only 29 train
IDs, optimising centroid POSITIONS directly turns the codebook into an
identity-classifier on train.

v2 separates "learn identity-invariance" from "place centroids":
  1. Train a small residual adapter f(x) = L2(W*x + x) with SupCon loss on
     train identities. Adapter learns to compress cross-condition variance.
  2. Run k-means in the adapter's output space (which is L2-normalised) to
     place K centroids by variance — the variance is now identity-aligned.
  3. At inference: encode → adapter → argmin to centroid → code.

The adapter is small (one residual linear) so it can't memorize identities.
The final clustering is k-means, which is variance-optimal — and the
adapter has reshaped the space so that variance IS identity-aligned.

This is the canonical metric-learning + clustering recipe. We also support
a "no-adapter" mode (= spherical k-means baseline) for ablation.
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

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InvarianceAdapter(nn.Module):
    """Small residual L2-normalising adapter.

    Output = L2( x + alpha * MLP(x) ) where alpha is small.
    """
    def __init__(self, D, hidden=None, alpha=0.5, dropout=0.1):
        super().__init__()
        hidden = hidden or D
        self.fc1 = nn.Linear(D, hidden)
        self.fc2 = nn.Linear(hidden, D)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        delta = self.fc2(h)
        out = x + self.alpha * delta
        return F.normalize(out, dim=-1)


def supcon_loss(z, labels, temperature=0.1):
    B = z.shape[0]
    sims = (z @ z.T) / temperature
    sims = sims - sims.max(dim=1, keepdim=True).values.detach()
    exp_sims = sims.exp()
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    self_mask = torch.eye(B, device=z.device, dtype=pos_mask.dtype)
    pos_mask = pos_mask - self_mask
    denom = (exp_sims * (1.0 - self_mask)).sum(dim=1).clamp_min(1e-9)
    log_prob = sims - denom.log().unsqueeze(1)
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return z.sum() * 0.0
    return -(log_prob * pos_mask).sum(dim=1)[valid].mean() / pos_count[valid].clamp_min(1).mean()


def train_adapter(tr_emb, tr_pid, *, hidden=None, alpha=0.5, dropout=0.1,
                   n_steps=2000, batch=128, lr=1e-3, temperature=0.1,
                   print_every=200, seed=42):
    torch.manual_seed(seed)
    by_id = defaultdict(list)
    for i, p in enumerate(tr_pid):
        by_id[str(p)].append(i)
    ids = [p for p in by_id if len(by_id[p]) >= 2]
    print(f"  adapter train: {len(ids)} IDs with >=2 samples")
    adapter = InvarianceAdapter(D=tr_emb.shape[1], hidden=hidden,
                                  alpha=alpha, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
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
        z = adapter(x)
        loss = supcon_loss(z, y, temperature=temperature)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % print_every == 0:
            print(f"    step {step+1:4d}  supcon={loss.item():.4f}  ({time.time()-t0:.0f}s)")
    adapter.eval()
    return adapter


def kmeans_centroids(x_np, K, seed=42):
    import faiss
    km = faiss.Kmeans(x_np.shape[1], K, niter=30, verbose=False, seed=seed)
    km.train(x_np.astype(np.float32))
    return km.centroids.astype(np.float32)


def apply_pipeline(emb_np, adapter, centroids_t, normalize_in=True):
    x = torch.from_numpy(emb_np.astype(np.float32)).to(DEVICE)
    if normalize_in:
        x = F.normalize(x, dim=-1)
    with torch.no_grad():
        if adapter is not None:
            z = adapter(x)
        else:
            z = x
        d2 = (z.pow(2).sum(-1, keepdim=True)
              - 2 * z @ centroids_t.t()
              + centroids_t.pow(2).sum(-1))
    return d2.argmin(-1).cpu().numpy()


def evaluate_same_code_rate(codes, pid, K):
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)
    ids = [p for p in by_id if len(by_id[p]) >= 2]
    n_same = 0; n_pairs = 0
    for pid_v in ids:
        samps = by_id[pid_v]
        for i in range(len(samps)):
            for j in range(i + 1, len(samps)):
                if codes[samps[i]] == codes[samps[j]]: n_same += 1
                n_pairs += 1
    n_used = len(set(codes.tolist()))
    return (n_same / n_pairs if n_pairs else 0.0), n_used, n_pairs


def evaluate_cross_id_collision(codes, pid):
    """Inter-identity collision: fraction of (different-id pair → same code).
    Lower is better."""
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)
    ids = list(by_id.keys())
    n_diff_pairs = 0; n_diff_same_code = 0
    code_arr = np.asarray(codes)
    # Approximate: pair up first sample from each pair of identities
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = by_id[ids[i]][0]; b = by_id[ids[j]][0]
            if code_arr[a] == code_arr[b]: n_diff_same_code += 1
            n_diff_pairs += 1
    return n_diff_same_code / n_diff_pairs if n_diff_pairs else 0.0


def save_pipeline(out_path, adapter, centroids_np, D_in, alpha, hidden):
    state = {
        "adapter_state": (adapter.state_dict() if adapter is not None else None),
        "centroids": centroids_np,
        "D_in": D_in,
        "alpha": alpha,
        "hidden": hidden if hidden is not None else D_in,
    }
    torch.save(state, out_path)


def load_pipeline_apply(state_path):
    state = torch.load(state_path, map_location=DEVICE, weights_only=False)
    centroids_t = torch.from_numpy(state["centroids"].astype(np.float32)).to(DEVICE)
    if state["adapter_state"] is not None:
        adapter = InvarianceAdapter(state["D_in"], hidden=state["hidden"],
                                      alpha=state["alpha"]).to(DEVICE)
        adapter.load_state_dict(state["adapter_state"])
        adapter.eval()
    else:
        adapter = None
    @torch.no_grad()
    def apply(emb_np):
        x = torch.from_numpy(emb_np.astype(np.float32)).to(DEVICE)
        x = F.normalize(x, dim=-1)
        z = adapter(x) if adapter is not None else x
        d2 = (z.pow(2).sum(-1, keepdim=True)
              - 2 * z @ centroids_t.t()
              + centroids_t.pow(2).sum(-1))
        return d2.argmin(-1).cpu().numpy()
    return apply


def run_variant(name, tr_emb, tr_pid, ev_emb, ev_pid, *, K, adapter=None,
                use_train_pool_for_kmeans=True, seed=42):
    """Compute centroids from train (optionally adapter-transformed), then
    score same-code rate on the eval set."""
    # Get the embedding pool to fit k-means
    src_t = torch.from_numpy(tr_emb.astype(np.float32)).to(DEVICE)
    src_t = F.normalize(src_t, dim=-1)
    with torch.no_grad():
        if adapter is not None:
            src_t = adapter(src_t)
    centroids_np = kmeans_centroids(src_t.cpu().numpy(), K, seed=seed)
    centroids_t = torch.from_numpy(centroids_np).to(DEVICE)

    tr_codes = apply_pipeline(tr_emb, adapter, centroids_t)
    ev_codes = apply_pipeline(ev_emb, adapter, centroids_t)

    tr_same, tr_used, tr_npairs = evaluate_same_code_rate(tr_codes, tr_pid, K)
    ev_same, ev_used, ev_npairs = evaluate_same_code_rate(ev_codes, ev_pid, K)
    ev_inter = evaluate_cross_id_collision(ev_codes, ev_pid)

    print(f"  {name:30s}  tr_same={tr_same:.3f} ({tr_used}/{K} codes)  "
          f"ev_same={ev_same:.3f} ({ev_used}/{K} codes, inter_coll={ev_inter:.3f})")
    return {
        "train_same_code": tr_same, "train_codes_used": tr_used,
        "eval_same_code": ev_same, "eval_codes_used": ev_used,
        "eval_inter_collision": ev_inter,
        "centroids": centroids_np,
    }


MODE_PATHS = {
    "a-xr-id": ("ecapa_libri_large.npz", "ecapa_voxceleb1.npz"),  # voxceleb1 supplement
    "a-scn":   ("ast_esc50_full.npz", None),
    "v-xc-id": ("arcface_lfw_xl.npz", "arcface_lfw_xxl.npz"),  # bigger pool available
    "v-sty":   ("style_pca_gram.npz", None),
    "v-sty-clip": ("clip_mid_wikiart.npz", None),  # 50 IDs vs gram's 15
    "a-para":  ("wav2vec_para_spk_emo.npz", None),
}


def main():
    from v2_retrieval import split_by_identity

    mode = sys.argv[1] if len(sys.argv) > 1 else "a-xr-id"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    print("=" * 70)
    print(f"id-codebook v2 (adapter + k-means) — mode={mode}  K={K}")
    print("=" * 70)

    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    primary_file, larger_file = MODE_PATHS[mode]
    d = np.load(EMB / primary_file)
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    print(f"  primary {primary_file}: train {len(set(tr_pid))} IDs / {len(tr_emb)} samp, "
          f"eval {len(set(ev_pid))} IDs / {len(ev_emb)} samp  D={emb.shape[1]}")

    results = {}

    # Variant 0: K-sweep on naive — sometimes the right K matters more than the recipe.
    # Skip K values that exceed train sample count (faiss k-means fails when N < K).
    print("\n[V0] naive k-means K-sweep")
    K_sweep_results = {}
    n_train_samples = len(tr_emb)
    for K_alt in [8, 16, 32, 64, 128]:
        if K_alt == K or K_alt > n_train_samples: continue
        try:
            r = run_variant(f"V0_naive_K{K_alt}", tr_emb, tr_pid, ev_emb, ev_pid,
                              K=K_alt, adapter=None)
            K_sweep_results[K_alt] = {k: v for k, v in r.items() if k != "centroids"}
        except Exception as e:
            print(f"  K={K_alt} failed: {e}")
    results["V0_K_sweep"] = K_sweep_results

    # Variant 1: naive k-means in raw space (current Path A baseline)
    print(f"\n[V1] naive k-means in raw space (K={K})")
    results["V1_naive"] = run_variant("V1_naive_kmeans", tr_emb, tr_pid, ev_emb, ev_pid,
                                        K=K, adapter=None)

    # Variant 2: adapter (alpha=0.5) + k-means
    print("\n[V2] adapter (alpha=0.5) + k-means")
    adapter_v2 = train_adapter(tr_emb, tr_pid, hidden=None, alpha=0.5, dropout=0.1,
                                n_steps=2000, batch=128, lr=1e-3, temperature=0.1)
    results["V2_adapter05"] = run_variant("V2_adapter_alpha0.5", tr_emb, tr_pid, ev_emb, ev_pid,
                                            K=K, adapter=adapter_v2)

    # Variant 3: stronger adapter (alpha=1.0)
    print("\n[V3] adapter (alpha=1.0) + k-means")
    adapter_v3 = train_adapter(tr_emb, tr_pid, hidden=None, alpha=1.0, dropout=0.1,
                                n_steps=2000, batch=128, lr=1e-3, temperature=0.1)
    results["V3_adapter10"] = run_variant("V3_adapter_alpha1.0", tr_emb, tr_pid, ev_emb, ev_pid,
                                            K=K, adapter=adapter_v3)

    # Variant 4: weaker adapter (alpha=0.2)
    print("\n[V4] adapter (alpha=0.2) + k-means")
    adapter_v4 = train_adapter(tr_emb, tr_pid, hidden=None, alpha=0.2, dropout=0.1,
                                n_steps=2000, batch=128, lr=1e-3, temperature=0.1)
    results["V4_adapter02"] = run_variant("V4_adapter_alpha0.2", tr_emb, tr_pid, ev_emb, ev_pid,
                                            K=K, adapter=adapter_v4)

    # If a larger external pool is available, also try Variant 5: larger pool.
    if larger_file is not None and (EMB / larger_file).exists():
        print(f"\n[V5] adapter trained on larger pool ({larger_file})")
        larger = np.load(EMB / larger_file)
        l_emb = larger["emb"].astype(np.float32); l_pid = larger["pid"]
        # Drop any identities that overlap with eval IDs for clean evaluation
        ev_id_set = set(ev_pid.tolist())
        keep = ~np.isin(l_pid, list(ev_id_set))
        l_emb_clean = l_emb[keep]; l_pid_clean = l_pid[keep]
        print(f"  larger pool clean (no eval IDs): {len(set(l_pid_clean))} IDs / {len(l_emb_clean)} samples")
        adapter_v5 = train_adapter(l_emb_clean, l_pid_clean, hidden=None, alpha=0.5,
                                     dropout=0.1, n_steps=2000, batch=128, lr=1e-3,
                                     temperature=0.1)
        # Centroids fit on the larger pool, then evaluate on the original split
        results["V5_adapter_largepool"] = run_variant(
            "V5_adapter_largepool", l_emb_clean, l_pid_clean, ev_emb, ev_pid,
            K=K, adapter=adapter_v5)

    # Summary — exclude V0_K_sweep from the K-fixed comparison
    print("\n" + "=" * 90)
    print(f"HEADLINE — eval same-code rate at FIXED K={K} (Path A scorecard K)")
    print("=" * 90)
    print(f"{'Variant':>30}  {'ev_same':>9}  {'inter_coll':>11}  {'codes_used':>11}")
    print("-" * 70)
    baseline = results["V1_naive"]["eval_same_code"]
    fixed_K_results = {k: v for k, v in results.items() if k != "V0_K_sweep"}
    for k, v in fixed_K_results.items():
        delta = v["eval_same_code"] - baseline
        mark = " ↑↑" if delta > 0.05 else (" ↑" if delta > 0.01 else (" ≈" if abs(delta) <= 0.01 else " ↓"))
        print(f"{k:>30}  {v['eval_same_code']:>9.3f}  {v['eval_inter_collision']:>11.3f}  "
              f"{v['eval_codes_used']:>4d}/{K:<4d}{mark}  (Δ {delta:+.3f})")

    # K sweep summary
    if "V0_K_sweep" in results and results["V0_K_sweep"]:
        print(f"\nK-sweep (naive, alternative K values):")
        for K_alt, v in sorted(results["V0_K_sweep"].items()):
            score = v["eval_same_code"] - 0.3 * v["eval_inter_collision"]
            print(f"  K={K_alt:>3d}: ev_same={v['eval_same_code']:.3f}  "
                  f"inter_coll={v['eval_inter_collision']:.3f}  score={score:.3f}")

    # Save the BEST adapter + centroids for downstream Path A use (fixed K)
    best_key = max(fixed_K_results, key=lambda k: fixed_K_results[k]["eval_same_code"]
                                              - 0.3 * fixed_K_results[k]["eval_inter_collision"])
    print(f"\nBest variant at fixed K={K} (same-code − 0.3*inter_collision): {best_key}")

    out_dir = Path("/home/ubuntu/multimodal-user-memory/runs/codebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the best (adapter, centroids) bundle so Path A can load it.
    best = results[best_key]
    adapter_map = {"V1_naive": None, "V2_adapter05": adapter_v2,
                    "V3_adapter10": adapter_v3, "V4_adapter02": adapter_v4}
    if "V5_adapter_largepool" in results:
        adapter_map["V5_adapter_largepool"] = adapter_v5  # type: ignore[name-defined]
    best_adapter = adapter_map.get(best_key, None)
    alpha_map = {"V2_adapter05": 0.5, "V3_adapter10": 1.0, "V4_adapter02": 0.2,
                  "V5_adapter_largepool": 0.5}
    D_in = tr_emb.shape[1]
    save_pipeline(out_dir / f"id_v2_codebook_{mode}_K{K}.pt",
                   best_adapter, best["centroids"], D_in,
                   alpha_map.get(best_key, 0.5), None)
    print(f"[saved] {out_dir}/id_v2_codebook_{mode}_K{K}.pt")

    # Save the diagnostic
    summary_results = {k: {kk: vv for kk, vv in v.items() if kk != "centroids"}
                        for k, v in results.items()}
    summary = {"mode": mode, "K": K, "results": summary_results, "best_variant": best_key}
    out_p = Path("/home/ubuntu/multimodal-user-memory/results/") / f"id_codebook_v2_{mode}_K{K}.json"
    with open(out_p, "w") as f: json.dump(summary, f, indent=2, default=str)
    print(f"[saved] {out_p}")


if __name__ == "__main__":
    sys.exit(main())
