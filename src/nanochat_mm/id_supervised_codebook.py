"""Identity-supervised codebook (the v1-plan §5/§11 "learned identity-preserving quantiser").

Replaces the naive k-means / vanilla STE codebook with a codebook trained
so that cross-condition same-identity embedding pairs map to the SAME code.

This directly attacks the codebook miss rate — the binding constraint on
Path A's end-to-end retr@1 across V-XC-ID, A-XR-ID, A-SCN, V-STY.

Training:
  - Initialise K codebook centroids from k-means.
  - Each step, draw a batch of (anchor, positive_same_id) pairs.
  - Compute soft assignments p(code | x) = softmax(-||x - c_k||² / tau).
  - Identity supervision: maximize the expected agreement between anchor
    and positive soft distributions — equivalently, minimize
    -log( <p_anchor, p_positive> ).
  - Reconstruction term: minimize ||x - sum_k p[k] c_k||², keeps centroids
    grounded in the embedding manifold.
  - Diversity term: penalize mean p[k] deviating from uniform 1/K,
    preventing codebook collapse to one bin.

Output: an `apply_fn(emb_np) → codes` interface compatible with `fit_naive_rq`,
plus the trained centroid matrix saved to disk.
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


class IDSupCodebook(nn.Module):
    """Single-level codebook with identity-supervised training.

    init_from_kmeans uses the v1 baseline as a warm start.
    train_id_supervised runs the SupCon-style loss until convergence.
    quantise() is the inference-time argmin used by Path A.
    """
    def __init__(self, D, K, tau=0.5, recon_weight=0.1, diversity_weight=0.05):
        super().__init__()
        self.D = D; self.K = K
        self.tau = tau
        self.recon_weight = recon_weight
        self.diversity_weight = diversity_weight
        self.centroids = nn.Parameter(torch.empty(K, D))
        nn.init.normal_(self.centroids, std=1.0 / math.sqrt(D))

    @torch.no_grad()
    def init_from_kmeans(self, train_emb, seed=42):
        import faiss
        km = faiss.Kmeans(self.D, self.K, niter=30, verbose=False, seed=seed)
        km.train(train_emb.astype(np.float32))
        self.centroids.data.copy_(torch.from_numpy(km.centroids))

    def soft_assign(self, x):
        # x: [B, D] → p: [B, K]
        d2 = (x.pow(2).sum(-1, keepdim=True)
              - 2 * x @ self.centroids.t()
              + self.centroids.pow(2).sum(-1))
        return F.softmax(-d2 / self.tau, dim=-1)

    def quantise(self, x):
        # Hard argmin
        d2 = (x.pow(2).sum(-1, keepdim=True)
              - 2 * x @ self.centroids.t()
              + self.centroids.pow(2).sum(-1))
        return d2.argmin(-1)


def train_id_supervised_codebook(emb, pid, *, K=32, tau=0.5, n_steps=3000,
                                   batch_pairs=64, lr=1e-2, seed=42,
                                   recon_weight=0.1, diversity_weight=0.05,
                                   print_every=200):
    """Train the codebook on identity-supervised pairs.

    Returns: trained codebook (CPU), centroid matrix (np), training history.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)
    ids = [p for p in by_id if len(by_id[p]) >= 2]
    print(f"  train id-sup codebook: {len(ids)} usable identities (>=2 samples), "
          f"K={K}, tau={tau}, n_steps={n_steps}")

    if len(ids) < 2:
        raise ValueError(f"Not enough identities with >=2 samples ({len(ids)})")

    cb = IDSupCodebook(D=emb.shape[1], K=K, tau=tau,
                        recon_weight=recon_weight, diversity_weight=diversity_weight).to(DEVICE)
    cb.init_from_kmeans(emb, seed=seed)

    emb_t = torch.from_numpy(emb.astype(np.float32)).to(DEVICE)
    opt = torch.optim.Adam(cb.parameters(), lr=lr)

    rng = np.random.default_rng(seed)
    hist = []; t0 = time.time()
    for step in range(n_steps):
        anchor_idx = []; positive_idx = []
        chosen = rng.choice(len(ids), size=batch_pairs,
                              replace=(batch_pairs > len(ids)))
        for ci in chosen:
            pid_v = ids[ci]
            samps = by_id[pid_v]
            i_a = int(rng.choice(len(samps)))
            i_p = int(rng.choice(len(samps)))
            # Force distinct samples when possible
            tries = 0
            while i_p == i_a and len(samps) > 1 and tries < 5:
                i_p = int(rng.choice(len(samps))); tries += 1
            anchor_idx.append(samps[i_a]); positive_idx.append(samps[i_p])

        a = emb_t[anchor_idx]; p = emb_t[positive_idx]
        # Soft assignments
        pa = cb.soft_assign(a); pp = cb.soft_assign(p)  # [B, K]
        # Identity supervision: -log <pa, pp> averaged over the batch
        agree = (pa * pp).sum(-1).clamp_min(1e-9)
        id_loss = -agree.log().mean()
        # Reconstruction: ||x - sum_k p[k] c_k||² for anchor (and positive)
        recon_a = (a - pa @ cb.centroids).pow(2).sum(-1).mean()
        recon_p = (p - pp @ cb.centroids).pow(2).sum(-1).mean()
        recon_loss = 0.5 * (recon_a + recon_p)
        # Diversity: KL from uniform to mean assignment
        mean_p = 0.5 * (pa.mean(0) + pp.mean(0))
        uniform = torch.full_like(mean_p, 1.0 / cb.K)
        div_loss = (mean_p * (mean_p.clamp_min(1e-9).log() - uniform.log())).sum()

        loss = id_loss + cb.recon_weight * recon_loss + cb.diversity_weight * div_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % print_every == 0:
            with torch.no_grad():
                hard_pa = cb.quantise(a); hard_pp = cb.quantise(p)
                hard_match = float((hard_pa == hard_pp).float().mean().item())
            print(f"    step {step+1:4d}  id_loss={id_loss.item():.4f}  "
                  f"recon={recon_loss.item():.4f}  div={div_loss.item():.4f}  "
                  f"hard-same-code(train)={hard_match:.3f}  ({time.time()-t0:.0f}s)")
            hist.append((step + 1, id_loss.item(), recon_loss.item(), div_loss.item(),
                          hard_match))
    return cb, cb.centroids.detach().cpu().numpy(), hist


@torch.no_grad()
def evaluate_same_code_rate(emb, pid, codebook):
    """Same-code rate: among held-out same-identity cross-condition pairs,
    what fraction quantise to the same code? Compared to chance (1/K)."""
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)
    ids = [p for p in by_id if len(by_id[p]) >= 2]
    if not ids:
        return float("nan"), float("nan"), float("nan"), 0
    emb_t = torch.from_numpy(emb.astype(np.float32)).to(DEVICE)
    codes = codebook.quantise(emb_t).cpu().numpy()
    n_pairs = 0; n_same = 0
    # For each identity, count all C(n, 2) same-id pairs
    for pid_v in ids:
        samps = by_id[pid_v]
        for i in range(len(samps)):
            for j in range(i + 1, len(samps)):
                if codes[samps[i]] == codes[samps[j]]:
                    n_same += 1
                n_pairs += 1
    same_rate = n_same / n_pairs if n_pairs else 0.0
    # Diversity: how many codes are used at all?
    n_used = len(set(codes.tolist()))
    return same_rate, 1.0 / codebook.K, n_used, n_pairs


def fit_id_supervised_apply(emb, pid, *, K=32, tau=0.5, n_steps=3000,
                              batch_pairs=64, lr=1e-2, seed=42):
    """Wrapper that returns a callable `apply_fn(emb_np) → codes` compatible
    with `fit_naive_rq` so it can drop into the existing Path A pipeline.
    Also returns the codebook for diagnostics."""
    cb, centroids_np, hist = train_id_supervised_codebook(
        emb, pid, K=K, tau=tau, n_steps=n_steps, batch_pairs=batch_pairs,
        lr=lr, seed=seed,
    )

    centroids_t = torch.from_numpy(centroids_np.astype(np.float32)).to(DEVICE)

    @torch.no_grad()
    def apply(emb_np):
        x = torch.from_numpy(emb_np.astype(np.float32)).to(DEVICE)
        d2 = (x.pow(2).sum(-1, keepdim=True)
              - 2 * x @ centroids_t.t()
              + centroids_t.pow(2).sum(-1))
        return d2.argmin(-1).cpu().numpy()
    return apply, cb, hist


def diagnose_codebook(name, codebook_apply, tr_emb, tr_pid, ev_emb, ev_pid, K):
    """Print same-code rate on TRAIN and EVAL splits. Compare to chance."""
    import torch as _torch
    # Build a temporary CB wrapper that uses the apply_fn
    class _Wrap:
        def __init__(self, fn, K_): self.fn = fn; self.K = K_
        def quantise(self, x):
            return _torch.from_numpy(self.fn(x.cpu().numpy())).to(x.device)
    cb_wrap = _Wrap(codebook_apply, K)
    tr_rate, chance, n_used_tr, n_tr = evaluate_same_code_rate(tr_emb, tr_pid, cb_wrap)
    ev_rate, _, n_used_ev, n_ev = evaluate_same_code_rate(ev_emb, ev_pid, cb_wrap)
    print(f"  {name}: train same-code = {tr_rate:.3f} (n_pairs={n_tr}, {n_used_tr}/{K} codes used);  "
          f"eval same-code = {ev_rate:.3f} (n_pairs={n_ev}, {n_used_ev}/{K} codes used);  "
          f"chance = {chance:.3f}")
    return {"train_same_code": tr_rate, "eval_same_code": ev_rate,
            "n_used_train": n_used_tr, "n_used_eval": n_used_ev,
            "chance": chance}


if __name__ == "__main__":
    from real_encoder_train import fit_naive_rq
    from v2_retrieval import split_by_identity

    EMB_ROOT = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")

    # Pick the modality from CLI: a-xr-id | a-scn | v-xc-id | v-sty | a-para
    mode = sys.argv[1] if len(sys.argv) > 1 else "a-xr-id"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    n_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3000

    print("=" * 70)
    print(f"Identity-supervised codebook — mode={mode}  K={K}  tau={tau}  steps={n_steps}")
    print("=" * 70)

    PATHS = {
        "a-xr-id": EMB_ROOT / "ecapa_libri_large.npz",
        "a-scn":   EMB_ROOT / "ast_esc50_full.npz",
        "v-xc-id": EMB_ROOT / "arcface_lfw_xl.npz",
        "v-sty":   EMB_ROOT / "style_pca_gram.npz",
        "a-para":  EMB_ROOT / "wav2vec_para_spk_emo.npz",
    }
    d = np.load(PATHS[mode])
    emb = d["emb"].astype(np.float32); pid = d["pid"]
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    print(f"  train: {len(set(tr_pid))} IDs / {len(tr_emb)} samples")
    print(f"  eval:  {len(set(ev_pid))} IDs / {len(ev_emb)} samples")
    print(f"  D = {emb.shape[1]}")

    # Baseline: naive k-means codebook
    print("\n[baseline] naive k-means codebook")
    naive_fn = fit_naive_rq(tr_emb, n_levels=1, k_per=K, seed=42)
    def naive_fn_1d(emb_np):
        out = naive_fn(emb_np)
        return out[:, 0] if out.ndim == 2 else out
    naive_diag = diagnose_codebook("naive k-means", naive_fn_1d,
                                    tr_emb, tr_pid, ev_emb, ev_pid, K)

    # ID-supervised codebook
    print("\n[id-sup] training identity-supervised codebook")
    id_apply, id_cb, hist = fit_id_supervised_apply(
        tr_emb, tr_pid, K=K, tau=tau, n_steps=n_steps,
        batch_pairs=64, lr=1e-2, seed=42,
    )
    id_diag = diagnose_codebook("id-supervised", id_apply,
                                  tr_emb, tr_pid, ev_emb, ev_pid, K)

    # Save the codebook and a summary
    out_dir = Path("/home/ubuntu/multimodal-user-memory/runs/codebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"id_sup_codebook_{mode}_K{K}_tau{tau}.npy",
            id_cb.centroids.detach().cpu().numpy())

    summary = {
        "mode": mode, "K": K, "tau": tau, "n_steps": n_steps,
        "naive": naive_diag, "id_sup": id_diag,
        "lift_eval": id_diag["eval_same_code"] - naive_diag["eval_same_code"],
    }
    out_path = Path("/home/ubuntu/multimodal-user-memory/results/") / f"id_sup_codebook_{mode}_K{K}.json"
    with open(out_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n[done] codebook → {out_dir}/id_sup_codebook_{mode}_K{K}_tau{tau}.npy")
    print(f"[done] diagnostic → {out_path}")
    print(f"\nHEADLINE: eval same-code rate (mechanism input quality)")
    print(f"  naive k-means:  {naive_diag['eval_same_code']:.3f}")
    print(f"  id-supervised:  {id_diag['eval_same_code']:.3f}")
    print(f"  LIFT:           {summary['lift_eval']:+.3f}")
