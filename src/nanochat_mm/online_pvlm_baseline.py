"""Online-PVLM-equivalent baseline on V-XC-ID.

baseline_positioning.md argues Online-PVLM's mechanism reduces to cosine
nearest-neighbour over a learned light projection of the frozen perceptual
encoder. We implement that literally for V-XC-ID:

  1. Frozen perceptual encoder: ArcFace R50 (same as Path A baselines).
  2. Instance normalisation (here = L2 norm of the 512-d ArcFace vector;
     no per-token or per-spatial-position norm since ArcFace returns a
     single pooled vector).
  3. Light learned MLP projection: linear → ReLU → linear, output dim
     equal to input dim. Trained on the LFW-XL identity-disjoint train
     split with a supervised-contrastive loss (same-identity pairs pulled
     together; cross-identity pushed apart).
  4. At inference: project registration + query through the trained MLP,
     L2 normalise, cosine top-1 over the registration bank.

This isolates "what would a strong train-free embedding-bank baseline
score on V-XC-ID?" against Path A's parametric retrieval. If the
projection lifts retr@1 over the raw-RAG cosine ceiling, Online-PVLM
genuinely exceeds the upper bound we cite. If it ties or underperforms,
the equivalence claim in baseline_positioning.md holds.
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
from v2_retrieval import split_by_identity, embedding_rag_ceiling

torch.manual_seed(42); np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LightProjector(nn.Module):
    def __init__(self, d_in=512, d_hidden=512, d_out=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.proj(x)


def supervised_contrastive_loss(z, labels, temperature=0.1):
    """SupCon loss. z: [B, D] L2-normalised; labels: [B]. Pairs with the same
    label are positives; everyone else is negative.
    """
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
    loss = -(log_prob * pos_mask).sum(dim=1)[valid] / pos_count[valid]
    return loss.mean()


def train_projector(tr_emb, tr_pid, *, n_steps=2000, batch=128, lr=1e-3,
                     temperature=0.1, print_every=200):
    """Train the light MLP projection on the train split."""
    by_id = defaultdict(list)
    for i, p in enumerate(tr_pid):
        by_id[str(p)].append(i)
    ids = [p for p in by_id if len(by_id[p]) >= 2]
    print(f"  train projector: {len(ids)} usable train identities (>=2 samples each)")

    model = LightProjector(d_in=tr_emb.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr_emb_t = torch.from_numpy(tr_emb.astype(np.float32)).to(DEVICE)
    pid_map = {p: i for i, p in enumerate(ids)}

    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(n_steps):
        # Build batch: K=batch//2 identities, 2 samples each
        K = batch // 2
        chosen_ids = rng.choice(len(ids), size=K, replace=(K > len(ids)))
        batch_idx = []
        batch_lab = []
        for k, ix in enumerate(chosen_ids):
            pid_v = ids[ix]
            samps = by_id[pid_v]
            pair = rng.choice(len(samps), size=2, replace=(len(samps) < 2))
            batch_idx.append(samps[pair[0]]); batch_lab.append(k)
            batch_idx.append(samps[pair[1]]); batch_lab.append(k)
        x = tr_emb_t[batch_idx]
        y = torch.tensor(batch_lab, dtype=torch.long, device=DEVICE)
        # Instance-norm (L2 on input)
        x = F.normalize(x, dim=-1)
        z = model(x)
        z = F.normalize(z, dim=-1)
        loss = supervised_contrastive_loss(z, y, temperature=temperature)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % print_every == 0:
            print(f"    step {step+1:4d}  loss={loss.item():.4f}  (elapsed {time.time()-t0:.0f}s)")
    return model


def projected_rag(eval_emb, eval_pid, projector, N_subset, n_queries_per_id):
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None:
        ids_sorted = ids_sorted[:N_subset]
    rng = np.random.default_rng(99)
    reg_idx, reg_lab = [], []
    q_idx, q_lab = [], []
    for pid_v in ids_sorted:
        idxs = list(by_id[pid_v]); rng.shuffle(idxs)
        reg_idx.append(idxs[0]); reg_lab.append(pid_v)
        for qi in idxs[1:1 + n_queries_per_id]:
            q_idx.append(qi); q_lab.append(pid_v)
    if not q_idx: return 0.0
    with torch.no_grad():
        Rx = torch.from_numpy(eval_emb[reg_idx].astype(np.float32)).to(DEVICE)
        Qx = torch.from_numpy(eval_emb[q_idx].astype(np.float32)).to(DEVICE)
        Rx = F.normalize(Rx, dim=-1)
        Qx = F.normalize(Qx, dim=-1)
        Rz = F.normalize(projector(Rx), dim=-1)
        Qz = F.normalize(projector(Qx), dim=-1)
        sims = Qz @ Rz.T
        pred = sims.argmax(dim=1).cpu().numpy()
    correct = sum(1 for k in range(len(q_lab)) if reg_lab[pred[k]] == q_lab[k])
    return correct / len(q_lab)


def main():
    print("=" * 70)
    print("Online-PVLM-equivalent baseline on V-XC-ID")
    print("=" * 70)

    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw_xl.npz")
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(vis['emb'], vis['pid'])
    print(f"  train: {len(set(tr_pid))} IDs, eval: {len(set(ev_pid))} IDs")

    print("\n[train light projector on train split, SupCon]")
    model = train_projector(tr_emb, tr_pid, n_steps=2000, batch=128, lr=1e-3,
                              temperature=0.1, print_every=200)
    model.eval()

    print("\n[eval]")
    Ns = [5, 10, 20, 50]; nq = 5
    results = {}
    for N in Ns:
        if N > len(set(ev_pid)): continue
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=nq)
        pvlm = projected_rag(ev_emb, ev_pid, model, N_subset=N, n_queries_per_id=nq)
        print(f"  N={N:>3}  raw cosine-NN={rag:.3f}  Online-PVLM-equiv (proj+cos)={pvlm:.3f}  "
              f"delta={pvlm - rag:+.3f}")
        results[N] = {"raw_cosine_nn": rag, "pvlm_projected": pvlm, "delta": pvlm - rag}

    # Comparison to Path A V-XC-ID-XL
    try:
        path_a = json.load(open(
            "/home/ubuntu/multimodal-user-memory/results/pathA_V-XC-ID-xl.json"))
        print("\n" + "=" * 80)
        print("Comparison — V-XC-ID-XL: cosine-NN vs Online-PVLM-equiv vs Path A")
        print("=" * 80)
        print(f"{'N':>4} | {'cos-NN':>7} | {'PVLM-equiv':>11} | {'Path A':>7}")
        print("-" * 50)
        for N in Ns:
            r = results.get(N)
            if r is None: continue
            pa = path_a.get(str(N), {}).get("retrieval_at_1", None)
            pa_str = f"{pa:.3f}" if pa is not None else "—"
            print(f"{N:>4} | {r['raw_cosine_nn']:>7.3f} | "
                  f"{r['pvlm_projected']:>11.3f} | {pa_str:>7}")
    except FileNotFoundError as e:
        print(f"(comparison file missing: {e})")

    out = Path("/home/ubuntu/multimodal-user-memory/results/online_pvlm_baseline.json")
    with open(out, "w") as f:
        json.dump({
            "modality": "V-XC-ID",
            "encoder": "ArcFace R50",
            "data": "LFW-XL (423 IDs)",
            "mechanism": "instance-norm + SupCon-trained MLP + cosine-NN",
            "results": {str(k): v for k, v in results.items()},
        }, f, indent=2)
    print(f"\n[done] {out}")
    print("\nINTERPRETATION:")
    print("  If pvlm_projected ≈ raw_cosine_nn → projection adds nothing on top;")
    print("  the upper-bound equivalence in baseline_positioning.md holds.")


if __name__ == "__main__":
    sys.exit(main())
