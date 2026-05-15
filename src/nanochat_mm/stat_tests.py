"""Statistical significance tests for the BEATS-RAG claims.

Reads the multi-seed result files and computes paired t-test and Wilcoxon
signed-rank tests of Path A retr@1 vs RAG retr@1. The "beats in K/N seeds"
narrative is informal; a p-value hardens the claim for reviewers.

We do paired tests because Path A and RAG share the registration/query
split (same seed → same data; only the memory mechanism differs). Pairing
removes data-side variance.
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import ttest_rel, wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def read_multiseed(path):
    """Returns dict[N -> {'path_a': [...], 'rag': [...]}] aligned by seed order."""
    d = json.load(open(path))
    per_seed = d.get("per_seed", {})
    seeds = d.get("seeds", []) or sorted(per_seed.keys(), key=lambda x: int(x))
    # seeds may be ints or str; per_seed keys may be str
    out = {}
    for N in (5, 10, 20):
        pa, rag = [], []
        for s in seeds:
            ps = per_seed.get(str(s)) or per_seed.get(s)
            if not isinstance(ps, dict): continue
            cell = ps.get(N) or ps.get(str(N))
            if not isinstance(cell, dict): continue
            if "retr@1" in cell and "rag" in cell:
                pa.append(cell["retr@1"]); rag.append(cell["rag"])
        if pa and rag:
            out[N] = {"path_a": pa, "rag": rag}
    return out


def beats_rag_test(path_a_vals, rag_vals, alpha=0.05):
    """Paired one-sided test: H1 is Path A > RAG."""
    path_a = np.asarray(path_a_vals, dtype=float)
    rag = np.asarray(rag_vals, dtype=float)
    n_beats = int((path_a >= rag).sum())
    diff = path_a - rag
    out = {
        "n_seeds": len(path_a),
        "path_a_mean": float(path_a.mean()),
        "path_a_std": float(path_a.std(ddof=1) if len(path_a) > 1 else 0.0),
        "rag_mean": float(rag.mean()),
        "rag_std": float(rag.std(ddof=1) if len(rag) > 1 else 0.0),
        "mean_diff": float(diff.mean()),
        "n_beats": n_beats,
        "frac_beats": n_beats / len(path_a),
    }
    if HAVE_SCIPY and len(path_a) >= 2:
        try:
            t_stat, t_p = ttest_rel(path_a, rag, alternative="greater")
            out["t_stat"] = float(t_stat); out["t_p_value"] = float(t_p)
            out["t_significant"] = bool(t_p < alpha)
        except Exception as e:
            out["t_error"] = str(e)
        try:
            # Wilcoxon needs nonzero differences
            nz = (diff != 0).any()
            if nz:
                w_stat, w_p = wilcoxon(path_a, rag, alternative="greater",
                                          zero_method="wilcox")
                out["w_stat"] = float(w_stat); out["w_p_value"] = float(w_p)
                out["w_significant"] = bool(w_p < alpha)
            else:
                out["w_error"] = "no nonzero diffs"
        except Exception as e:
            out["w_error"] = str(e)
    else:
        out["scipy_unavailable"] = True
    return out


def main():
    results_dir = Path("/home/ubuntu/multimodal-user-memory/results")
    files = list(results_dir.glob("pathA_multiseed_*.json"))
    if not files:
        print("No multi-seed files found.")
        return

    print("=" * 80)
    print("Statistical significance tests — Path A vs RAG (paired)")
    print("=" * 80)
    print(f"scipy: {'available' if HAVE_SCIPY else 'MISSING — falling back to descriptive only'}")
    print()

    summary = {}
    for f in sorted(files):
        ms = read_multiseed(f)
        name = f.stem.replace("pathA_multiseed_", "")
        print(f"\n--- {name}  ({f.name}) ---")
        per_N = {}
        for N in sorted(ms.keys()):
            pa = ms[N]["path_a"]; rag = ms[N]["rag"]
            if pa is None or rag is None or not pa or not rag:
                print(f"  N={N}: missing per-seed values, skipping")
                continue
            r = beats_rag_test(pa, rag)
            per_N[N] = r
            t_p = r.get("t_p_value", float("nan"))
            w_p = r.get("w_p_value", float("nan"))
            t_sig = r.get("t_significant", False)
            w_sig = r.get("w_significant", False)
            verdict = ""
            if t_sig and w_sig: verdict = "**** p<0.05 BOTH"
            elif t_sig or w_sig: verdict = "** p<0.05 ONE"
            elif r["frac_beats"] >= 0.6: verdict = "majority"
            elif r["frac_beats"] >= 0.5: verdict = "ties"
            else: verdict = "below"
            print(f"  N={N:>3}  Path A = {r['path_a_mean']:.3f} ± {r['path_a_std']:.3f}, "
                  f"RAG = {r['rag_mean']:.3f} ± {r['rag_std']:.3f},  "
                  f"Δ = {r['mean_diff']:+.3f},  "
                  f"beats {r['n_beats']}/{r['n_seeds']},  "
                  f"t p={t_p:.3f}  w p={w_p:.3f}  {verdict}")
        summary[name] = per_N

    out_p = results_dir / "stat_tests.json"
    with open(out_p, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_p}")


if __name__ == "__main__":
    sys.exit(main())
