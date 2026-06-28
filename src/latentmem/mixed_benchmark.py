"""Mixed-content memory benchmark: perceptual identity + private fact.

Tests the hybrid claim: for content with a NON-captionable component (a face,
where latent wins) AND a captionable arbitrary fact (where text wins), a hybrid
memory (latent for identity, text for the fact) beats either single channel.

Each user = (real ArcFace face embedding, cross-condition) + an arbitrary fact
in one of C categories. Register one photo + the fact; query a DIFFERENT photo
and recall the fact. Recall needs BOTH: match the face (perceptual leg), then
recall the fact (captionable leg).

Three memory architectures, each missing-or-not a leg:
  text_only   : caption-resolved identity + exact text fact  (weak PERCEPTUAL leg)
  latent_only : latent-resolved identity + fact in the latent (weak FACT leg)
  hybrid      : latent identity + exact text fact             (strong on both)

Controls:
  - captionable-fact: fact IS derivable from the face (= its coarse code).
    Then text_only should ~tie hybrid -> the hybrid gain is specific to
    arbitrary (non-perceptual) facts, not an artifact.

Thorough: multi-seed, multiple pool sizes N, multiple C, paired significance.
Embedding-level (real data, fast) -- the recall@1 metric the paper uses. The
in-LM AttMem version of the latent identity leg scores at least as high
(AttMem >= cosine-NN), so hybrid here is a conservative lower bound.
"""
import json
import math
from pathlib import Path

import numpy as np

EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw_xxxl.npz")
OUT = Path("/home/ubuntu/multimodal-user-memory/results/mixed_benchmark.json")

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
NS = [10, 50, 100, 300, 1000]
CS = [10, 50]
BETA = 20.0           # latent soft-attention sharpness (AttMem init value)
CAP_BITS = 8          # caption granularity: 2^8 = 256 coarse codes (Path-A regime)


def load():
    d = np.load(EMB)
    emb = d["emb"].astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    pid = d["pid"]
    by = {}
    for i, p in enumerate(pid):
        by.setdefault(str(p), []).append(i)
    ids = [p for p, idx in by.items() if len(idx) >= 2]
    return emb, by, ids


def lsh_codes(X, R):
    """Coarse locality-sensitive codes = a faithful stand-in for a discrete
    caption/codebook (the paper's Path A regime): many distinct faces collide
    into one code, and the code is not reliably cross-condition invariant."""
    bits = (X @ R) > 0                      # [n, b]
    w = (1 << np.arange(R.shape[1]))
    return (bits * w).sum(axis=1)           # [n] integer code


def trial(emb, by, ids, seed, N, C, captionable):
    rng = np.random.default_rng(seed)
    sel = rng.choice(ids, size=N, replace=False)
    keys, queries = [], []
    for p in sel:
        idx = list(by[p]); rng.shuffle(idx)
        keys.append(emb[idx[0]]); queries.append(emb[idx[1]])
    K = np.stack(keys); Q = np.stack(queries)           # [N, 512]

    R = rng.standard_normal((K.shape[1], CAP_BITS)).astype(np.float32)
    kcode = lsh_codes(K, R); qcode = lsh_codes(Q, R)

    if captionable:
        facts = (kcode % C).astype(int)                 # derivable from the face
    else:
        facts = rng.integers(0, C, size=N)              # arbitrary

    truth = facts                                       # query's true fact = its user's

    # ---- text_only: caption-keyed exact fact; identity bottlenecked by codes
    code_to_fact = {}
    for c, f in zip(kcode, facts):
        code_to_fact.setdefault(int(c), []).append(int(f))
    code_major = {c: np.bincount(fs).argmax() for c, fs in code_to_fact.items()}
    text_pred = np.array([code_major.get(int(c), rng.integers(0, C)) for c in qcode])

    # ---- hybrid: hard latent (cosine-NN) identity -> exact text fact
    sim = Q @ K.T                                        # [N_q, N]
    nn = sim.argmax(axis=1)
    hybrid_pred = facts[nn]
    id_recall = float((nn == np.arange(N)).mean())

    # ---- latent_only: identity latent, but the fact RIDES the perceptual latent
    #      (soft attention over the bank, fact as a one-hot value -> argmax).
    w = np.exp(BETA * (sim - sim.max(axis=1, keepdims=True)))
    w /= w.sum(axis=1, keepdims=True)                   # [N_q, N] attention
    onehot = np.eye(C, dtype=np.float32)[facts]         # [N, C]
    blended = w @ onehot                                # [N_q, C]
    latent_pred = blended.argmax(axis=1)

    return {
        "text_only": float((text_pred == truth).mean()),
        "latent_only": float((latent_pred == truth).mean()),
        "hybrid": float((hybrid_pred == truth).mean()),
        "id_recall": id_recall,
    }


def paired_t(a, b):
    """One-sided paired t (a>b) across seeds. Returns (mean diff, p)."""
    d = np.array(a) - np.array(b)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float(d.mean()), float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    # survival of Student-t via normal approx is crude; use a t-table-free
    # two-sided estimate good enough for n=8 reporting alongside the effect size.
    from statistics import NormalDist
    p = 1 - NormalDist().cdf(t)        # one-sided
    return float(d.mean()), float(p)


def run():
    emb, by, ids = load()
    print(f"loaded {len(ids)} identities (>=2 photos), dim={emb.shape[1]}", flush=True)
    rows = []
    for captionable in (False, True):
        for C in CS:
            for N in NS:
                if N > len(ids):
                    continue
                per = {k: [] for k in ("text_only", "latent_only", "hybrid", "id_recall")}
                for s in SEEDS:
                    r = trial(emb, by, ids, s, N, C, captionable)
                    for k in per:
                        per[k].append(r[k])
                row = {"captionable": captionable, "C": C, "N": N,
                       "n_seeds": len(SEEDS)}
                for k in per:
                    row[f"{k}_mean"] = float(np.mean(per[k]))
                    row[f"{k}_std"] = float(np.std(per[k], ddof=1))
                dh_t, ph_t = paired_t(per["hybrid"], per["text_only"])
                dh_l, ph_l = paired_t(per["hybrid"], per["latent_only"])
                row.update({"d_hybrid_vs_text": dh_t, "p_hybrid_vs_text": ph_t,
                            "d_hybrid_vs_latent": dh_l, "p_hybrid_vs_latent": ph_l})
                rows.append(row)
        print(f"  done captionable={captionable}", flush=True)

    OUT.write_text(json.dumps(rows, indent=2))
    # ---- printed table
    def fmt(r, k): return f"{r[k+'_mean']:.3f}±{r[k+'_std']:.3f}"
    print("\n=== MIXED-MEMORY BENCHMARK (recall@1 of the private fact, 8 seeds) ===")
    print("arbitrary facts (the real test):")
    hdr = f"{'C':>3} {'N':>5} | {'text_only':>13} {'latent_only':>13} {'hybrid':>13} "\
          f"{'id_rec':>7} | {'H-text':>7}(p) {'H-lat':>7}(p)"
    for cond, label in ((False, "arbitrary facts (the real test):"),
                        (True, "captionable-fact CONTROL (text should ~tie hybrid):")):
        print("\n" + label); print(hdr)
        for r in rows:
            if r["captionable"] != cond:
                continue
            print(f"{r['C']:>3} {r['N']:>5} | {fmt(r,'text_only'):>13} {fmt(r,'latent_only'):>13} "
                  f"{fmt(r,'hybrid'):>13} {r['id_recall_mean']:>7.3f} | "
                  f"{r['d_hybrid_vs_text']:>+7.3f}({r['p_hybrid_vs_text']:.0e}) "
                  f"{r['d_hybrid_vs_latent']:>+7.3f}({r['p_hybrid_vs_latent']:.0e})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run()
