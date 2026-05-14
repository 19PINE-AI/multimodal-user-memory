"""Learned residual VQ vs naive residual k-means.

Hypothesis: a small learned RQ-VAE with reconstruction + identity-preservation
loss preserves intra-identity code agreement better than naive residual
k-means at the same effective codebook size — particularly in the regime
where naive RQ collapses (vision, depth >= 2).

We measure intra-identity tuple-agreement and inter-identity tuple-collision
on the same embeddings used in the naive sanity checks, and compare directly.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------- Embedding extraction (cached) ----------

def extract_or_load_audio_embeddings(cache_path: Path):
    """ECAPA-TDNN embeddings on LibriSpeech, same recipe as sanity_ecapa_collisions.py."""
    if cache_path.exists():
        d = np.load(cache_path)
        return d["emb"], d["pid"]
    import soundfile as sf
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    import random as rnd
    rnd.seed(SEED)

    libri = Path.home() / "data" / "LibriSpeech" / "test-clean"
    speakers = [d for d in sorted(libri.iterdir()) if d.is_dir() and len([c for c in d.iterdir() if c.is_dir()]) >= 2]
    speakers = rnd.sample(speakers, k=min(40, len(speakers)))
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/home/ubuntu/multimodal-user-memory/runs/pretrained-ecapa",
        run_opts={"device": DEVICE},
    )
    embs, pids = [], []
    for spk in speakers:
        chapters = sorted([c for c in spk.iterdir() if c.is_dir()])
        flacs = []
        for c in chapters:
            flacs.extend(sorted(c.glob("*.flac")))
        sel = rnd.sample(flacs, k=min(10, len(flacs)))
        for u in sel:
            data, sr = sf.read(str(u), always_2d=False, dtype="float32")
            if data.ndim > 1: data = data.mean(axis=1)
            wav = torch.from_numpy(data).unsqueeze(0)
            if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav[:, :16000 * 6]
            e = enc.encode_batch(wav.to(DEVICE)).squeeze().cpu().numpy()
            e = e / (np.linalg.norm(e) + 1e-9)
            embs.append(e); pids.append(spk.name)
    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, emb=emb, pid=pid)
    return emb, pid


def extract_or_load_vision_embeddings(cache_path: Path):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["emb"], d["pid"]
    import onnxruntime as ort
    import cv2
    from sklearn.datasets import fetch_lfw_people
    import random as rnd
    rnd.seed(SEED)

    lfw = fetch_lfw_people(min_faces_per_person=10, color=True, resize=1.0)
    by_person = defaultdict(list)
    for i, t in enumerate(lfw.target):
        by_person[int(t)].append(i)
    eligible = [(p, idxs) for p, idxs in by_person.items() if len(idxs) >= 10]
    chosen = rnd.sample(eligible, k=min(40, len(eligible)))

    sess = ort.InferenceSession(
        str(Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"),
        providers=['CPUExecutionProvider'],  # CUDA EP needs extra binding; CPU is fine for this volume
    )
    inp = sess.get_inputs()[0].name
    embs, pids = [], []
    for pid, idxs in chosen:
        sel = rnd.sample(idxs, k=min(10, len(idxs)))
        for i in sel:
            img = lfw.images[i]
            img = (img * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            arr = ((img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
            e = sess.run(None, {inp: arr})[0][0]
            e = e / (np.linalg.norm(e) + 1e-9)
            embs.append(e); pids.append(str(pid))
    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, emb=emb, pid=pid)
    return emb, pid


# ---------- Residual VQ module ----------

class ResidualVQ(nn.Module):
    def __init__(self, dim, n_levels, k_per_level, commitment=0.25):
        super().__init__()
        self.n_levels = n_levels
        self.k = k_per_level
        self.codebooks = nn.ModuleList([
            nn.Embedding(k_per_level, dim) for _ in range(n_levels)
        ])
        for cb in self.codebooks:
            nn.init.normal_(cb.weight, std=1.0 / (dim ** 0.5))
        self.commitment = commitment

    def forward(self, x):
        """Return (quantised, codes [B, n_levels], commitment_loss)."""
        B, D = x.shape
        codes = torch.zeros(B, self.n_levels, dtype=torch.long, device=x.device)
        q_total = torch.zeros_like(x)
        residual = x
        commit_loss = 0.0
        for L, cb in enumerate(self.codebooks):
            # Find nearest codebook entry for residual
            cb_weight = cb.weight  # (K, D)
            dist = (residual.pow(2).sum(-1, keepdim=True)
                    - 2 * residual @ cb_weight.t()
                    + cb_weight.pow(2).sum(-1))  # (B, K)
            idx = dist.argmin(-1)  # (B,)
            codes[:, L] = idx
            q = cb(idx)  # (B, D)
            # Commit + codebook losses (VQ-VAE)
            commit_loss = commit_loss + F.mse_loss(residual, q.detach())
            codebook_loss = F.mse_loss(q, residual.detach())
            commit_loss = commit_loss + codebook_loss  # accumulate
            # Straight-through estimator
            q_st = residual + (q - residual).detach()
            q_total = q_total + q_st
            residual = residual - q.detach()
        return q_total, codes, commit_loss


# ---------- Train + eval ----------

def train_rqvae(emb_np, pid_np, n_levels, k_per_level, lambda_cls=1.0, epochs=400, lr=3e-3):
    D = emb_np.shape[1]
    # Encode pids as ints
    unique_pids, pid_ints = np.unique(pid_np, return_inverse=True)
    n_classes = len(unique_pids)

    x = torch.from_numpy(emb_np).to(DEVICE)
    y = torch.from_numpy(pid_ints).long().to(DEVICE)

    rvq = ResidualVQ(D, n_levels, k_per_level).to(DEVICE)
    cls_head = nn.Linear(D, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(list(rvq.parameters()) + list(cls_head.parameters()), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        rvq.train()
        q, codes, cl = rvq(x)
        recon_loss = F.mse_loss(q, x)
        logits = cls_head(q)
        cls_loss = F.cross_entropy(logits, y)
        loss = recon_loss + 0.25 * cl + lambda_cls * cls_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"    epoch {epoch+1:3d}  recon={recon_loss.item():.4f}  cls={cls_loss.item():.4f}  cls_acc={acc:.3f}")

    # Eval: get codes deterministically
    rvq.eval()
    with torch.no_grad():
        _, codes, _ = rvq(x)
    return codes.cpu().numpy(), pid_np


def collision_stats(codes_per_emb, pid_arr, seed=SEED):
    """codes_per_emb: array of shape (N,) (flat code) or (N, L) (tuple code)."""
    pid_to_indices = defaultdict(list)
    for i, p in enumerate(pid_arr):
        pid_to_indices[str(p)].append(i)
    intra_agree = 0; intra_pairs = 0
    for p, idxs in pid_to_indices.items():
        if len(idxs) >= 2:
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    intra_pairs += 1
                    a = codes_per_emb[idxs[i]]; b = codes_per_emb[idxs[j]]
                    if tuple(np.atleast_1d(a)) == tuple(np.atleast_1d(b)):
                        intra_agree += 1
    intra_rate = intra_agree / intra_pairs if intra_pairs > 0 else 0.0
    rng = np.random.RandomState(seed + 17)
    inter_coll = 0; inter_pairs = 0
    N = len(codes_per_emb)
    for _ in range(20000):
        i, j = rng.randint(0, N, size=2)
        if str(pid_arr[i]) != str(pid_arr[j]):
            inter_pairs += 1
            a = codes_per_emb[i]; b = codes_per_emb[j]
            if tuple(np.atleast_1d(a)) == tuple(np.atleast_1d(b)):
                inter_coll += 1
    inter_rate = inter_coll / inter_pairs if inter_pairs > 0 else 0.0
    ratio = intra_rate / inter_rate if inter_rate > 0 else float("inf")
    return intra_rate, inter_rate, ratio


def naive_rq_kmeans(emb_np, n_levels, k_per_level):
    import faiss
    D = emb_np.shape[1]
    residual = emb_np.copy()
    codes = np.zeros((len(emb_np), n_levels), dtype=np.int64)
    for L in range(n_levels):
        km = faiss.Kmeans(D, k_per_level, niter=20, verbose=False, seed=SEED + L)
        km.train(residual)
        _, c = km.index.search(residual, 1)
        c = c.squeeze(1)
        codes[:, L] = c
        residual = residual - km.centroids[c]
    return codes


def main():
    cache_dir = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for modality, extractor in [
        ("audio", lambda: extract_or_load_audio_embeddings(cache_dir / "ecapa_libri.npz")),
        ("vision", lambda: extract_or_load_vision_embeddings(cache_dir / "arcface_lfw.npz")),
    ]:
        print(f"\n{'='*60}\n{modality.upper()}\n{'='*60}")
        print("[extract] loading embeddings ...")
        emb, pid = extractor()
        print(f"[extract] emb shape {emb.shape}, n_identities = {len(set(pid))}")

        modality_results = {}
        # Compare naive RQ vs learned RQ-VAE at matched configs
        configs = [(2, 16), (2, 64), (3, 16), (4, 16)]
        for n_levels, k_per in configs:
            cfg = f"L{n_levels}_K{k_per}"
            eff_K = k_per ** n_levels
            print(f"\n  config: {n_levels} levels x {k_per} codes = effective {eff_K}")

            # Naive
            naive_codes = naive_rq_kmeans(emb, n_levels, k_per)
            n_intra, n_inter, n_ratio = collision_stats(naive_codes, pid)
            print(f"  [naive RQ-kmeans]      intra={n_intra:.4f}  inter={n_inter:.4f}  ratio={n_ratio:.2f}")

            # Learned
            print(f"  [learned RQ-VAE] training ...")
            learned_codes, _ = train_rqvae(emb, pid, n_levels, k_per)
            l_intra, l_inter, l_ratio = collision_stats(learned_codes, pid)
            print(f"  [learned RQ-VAE]       intra={l_intra:.4f}  inter={l_inter:.4f}  ratio={l_ratio:.2f}")

            modality_results[cfg] = {
                "n_levels": n_levels, "k_per_level": k_per, "eff_K": eff_K,
                "naive": {"intra": float(n_intra), "inter": float(n_inter), "ratio": float(n_ratio) if n_ratio != float("inf") else None},
                "learned": {"intra": float(l_intra), "inter": float(l_inter), "ratio": float(l_ratio) if l_ratio != float("inf") else None},
                "delta_intra": float(l_intra - n_intra),
            }
        results[modality] = modality_results

    out = Path("/home/ubuntu/multimodal-user-memory/results/learned_rqvae.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] Wrote {out}")

    # Headline table
    print("\n" + "=" * 70)
    print("HEADLINE: learned RQ-VAE vs naive residual k-means")
    print("=" * 70)
    print(f"{'modality':>10} | {'config':>10} | {'eff_K':>8} | {'naive_intra':>11} | {'learned_intra':>13} | {'Δ':>7}")
    print("-" * 70)
    for mod, mr in results.items():
        for cfg, d in mr.items():
            print(f"{mod:>10} | {cfg:>10} | {d['eff_K']:>8} | {d['naive']['intra']:>11.4f} | {d['learned']['intra']:>13.4f} | {d['delta_intra']:>+7.4f}")


if __name__ == "__main__":
    sys.exit(main())
