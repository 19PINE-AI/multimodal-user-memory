"""Open-set recognition and verification across modalities --- the capabilities a user
memory needs beyond closed-set recall@1: (a) reject a stranger it has never enrolled,
and (b) decide whether two perceptions are the same identity. Pure cosine over the
cached encoder embeddings (== the memory's read), so this reports what the grounded
memory inherits, with standard operating-point metrics (AUROC, EER).

Verification: genuine pairs (same id) vs impostor pairs (different id); report EER, AUROC.
Open-set ID: enroll N identities (1 sample each); known probes = held-out samples of the
enrolled, unknown probes = samples of identities never enrolled. Score = max cosine to the
gallery. Report AUROC(known vs unknown) and open-set accuracy at the balanced threshold
(accept-known top-1-correct-and-above-thresh; reject-unknown below-thresh), over draws.

Usage: python3 openset_verification.py [N] [draws]
"""
import sys, json
import numpy as np
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EMB = REPO_ROOT / "runs" / "embeddings"
# (display, file) per modality -- strong, diverse encoders
POOLS = [
    ("Face (ArcFace, LFW)",      "arcface_lfw_xxxl"),
    ("Face (AgeDB, cross-age)",  "arcface_agedb"),
    ("Speaker (ECAPA, VoxCeleb)","ecapa_voxceleb1"),
    ("Acoustic scene (AST)",     "ast_esc50_full"),
    ("Painting style (CLIP)",    "clip_mid_wikiart_xxl"),
]


def l2(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def load(name):
    d = np.load(EMB / f"{name}.npz", allow_pickle=True)
    k = "emb" if "emb" in d else "keys"
    pid = np.array([str(p) for p in d["pid"]])
    return l2(d[k].astype(np.float32)), pid


def by_id(pid):
    bi = defaultdict(list)
    for i, p in enumerate(pid): bi[p].append(i)
    return bi


def eer_auroc(gen, imp):
    """Equal-error rate and AUROC from genuine/impostor score arrays."""
    scores = np.concatenate([gen, imp])
    labels = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    order = np.argsort(-scores); labels = labels[order]
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    P, Nn = labels.sum(), (1 - labels).sum()
    tpr = tp / P; fpr = fp / Nn
    auroc = float(np.trapz(tpr, fpr))
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fpr[i] + fnr[i]) / 2)
    return eer, auroc


def verify_draw(emb, bi, ids, n_pairs, rng):
    gen, imp = [], []
    for _ in range(n_pairs):
        p = ids[rng.integers(len(ids))]; ix = bi[p]
        a, b = rng.choice(ix, 2, replace=False); gen.append(float(emb[a] @ emb[b]))
        p1, p2 = ids[rng.integers(len(ids))], ids[rng.integers(len(ids))]
        while p2 == p1: p2 = ids[rng.integers(len(ids))]
        imp.append(float(emb[bi[p1][rng.integers(len(bi[p1]))]] @ emb[bi[p2][rng.integers(len(bi[p2]))]]))
    return eer_auroc(np.array(gen), np.array(imp))


def openset_draw(emb, bi, ids, N, rng):
    rng.shuffle(ids)
    enroll = ids[:N]; unknown = ids[N:2 * N] if len(ids) >= 2 * N else ids[N:]
    gal, lab = [], []
    for p in enroll:
        ix = list(bi[p]); rng.shuffle(ix); gal.append(emb[ix[0]]); lab.append(p)
    G = np.stack(gal)
    # known probes: a held-out sample of each enrolled id
    kn_s, kn_correct = [], []
    for p in enroll:
        ix = list(bi[p])
        q = emb[ix[1]] if len(ix) > 1 else emb[ix[0]]
        sims = q @ G.T; j = int(sims.argmax())
        kn_s.append(float(sims.max())); kn_correct.append(lab[j] == p)
    # unknown probes: a sample of each never-enrolled id
    un_s = []
    for p in unknown:
        ix = list(bi[p]); q = emb[ix[rng.integers(len(ix))]]
        un_s.append(float((q @ G.T).max()))
    kn_s, un_s = np.array(kn_s), np.array(un_s); kn_correct = np.array(kn_correct)
    _, auroc = eer_auroc(kn_s, un_s)
    # balanced threshold (EER point between known/unknown max-sim distributions)
    alls = np.sort(np.concatenate([kn_s, un_s]))[::-1]
    best_t, best = alls[0], -1
    for t in alls:
        acc = 0.5 * np.mean((kn_s >= t) & kn_correct) + 0.5 * np.mean(un_s < t)
        if acc > best: best, best_t = acc, t
    return auroc, float(best), float(np.mean((kn_s >= best_t) & kn_correct)), float(np.mean(un_s < best_t))


def ci(v): return 1.96 * float(np.std(v, ddof=1)) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(f"=== Open-set recognition & verification across modalities (N={N} enrolled, {draws} draws) ===\n")
    print(f"{'modality':30} {'verif AUROC':>12} {'verif EER':>10}   {'openset AUROC':>13} {'accept-known':>12} {'reject-unk':>10}")
    rows = []
    for name, fname in POOLS:
        try:
            emb, pid = load(fname)
        except FileNotFoundError:
            print(f"{name:30} (missing)"); continue
        bi = by_id(pid); ids = [p for p in bi if len(bi[p]) >= 2]
        Nn = min(N, len(ids) // 2)
        vres = np.array([verify_draw(emb, bi, list(ids), 200, np.random.default_rng(1000 + s)) for s in range(draws)])
        ores = np.array([openset_draw(emb, bi, list(ids), Nn, np.random.default_rng(2000 + s)) for s in range(draws)])
        v_eer, v_auc = vres[:, 0].mean(), vres[:, 1].mean()
        o_auc, o_acc, o_kn, o_un = ores[:, 0].mean(), ores[:, 1].mean(), ores[:, 2].mean(), ores[:, 3].mean()
        rows.append({"modality": name, "N": Nn, "n_id": len(ids),
                     "verif_auroc": float(v_auc), "verif_auroc_ci": ci(vres[:, 1]),
                     "verif_eer": float(v_eer), "verif_eer_ci": ci(vres[:, 0]),
                     "openset_auroc": float(o_auc), "openset_auroc_ci": ci(ores[:, 0]),
                     "openset_acc": float(o_acc), "accept_known": float(o_kn), "reject_unknown": float(o_un)})
        print(f"{name:30} {v_auc:>12.3f} {v_eer:>10.3f}   {o_auc:>13.3f} {o_kn:>12.3f} {o_un:>10.3f}")
    (REPO_ROOT / "results" / "openset_verification.json").write_text(
        json.dumps({"N": N, "draws": draws, "rows": rows}, indent=2))
    print("\nwrote results/openset_verification.json")


if __name__ == "__main__":
    main()
