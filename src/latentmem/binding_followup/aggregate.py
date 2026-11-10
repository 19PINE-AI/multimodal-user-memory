"""Aggregate T3 per-run JSONs into seed-averaged tables (markdown + JSON)."""
import json, glob, collections, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[3] / "results/binding_followup/runs"
runs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(R / "c*_conv.json")))]
groups = collections.defaultdict(list)
for r in runs:
    m = r["meta"]
    groups[(m["chars"], m["mmax"], m["role"])].append(r)


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return "—"
    return f"{np.mean(xs):.2f}±{np.std(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def g(r, *path):
    for p in path:
        if r is None: return None
        r = r.get(str(p)) if isinstance(r, dict) else None
    return r


out, lines = {}, []
lines.append("## LM retrieval (seed mean±sd)\n")
lines.append("| chars | M_train | write | seeds | stop step | final loss | tf_exact M=1 | tf_exact M=2 | tf_exact M=3 | greedy queried | greedy other code | char acc queried | char acc other |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(groups):
    c, m, role = key; rs = groups[key]; top = str(m)
    row = dict(n=len(rs),
               stop=[r["meta"].get("final_step") for r in rs],
               loss=[r["meta"]["curve"][-1]["loss"] if r["meta"].get("curve") else None for r in rs],
               tf={M: [g(r, "lm", M, "tf_exact") for r in rs] for M in (1, 2, 3)},
               gq=[g(r, "lm", top, "greedy_exact") for r in rs],
               go=[g(r, "lm", top, "greedy_other_code_exact") for r in rs],
               cq=[g(r, "lm", top, "char_acc_queried") for r in rs],
               co=[g(r, "lm", top, "char_acc_other") for r in rs])
    out[f"c{c}_m{m}_{'role' if role else 'std'}"] = row
    lines.append(f"| {c} | {m} | {'role-keyed' if role else 'shared'} | {len(rs)} | {ms(row['stop'])} | {ms(row['loss'])} | "
                 f"{ms(row['tf'][1])} | {ms(row['tf'][2])} | {ms(row['tf'][3])} | {ms(row['gq'])} | {ms(row['go'])} | {ms(row['cq'])} | {ms(row['co'])} |")

lines.append("\n## Probes on the latent (test-set char accuracy / full-code exact; chance char acc = 0.031)\n")
lines.append("| chars | M | write | slot probe linear (mean over slots) | slot probe MLP | name-cond linear | name-cond MLP | untrained head slot MLP | shuffled labels |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for key in sorted(groups):
    c, m, role = key; rs = groups[key]; P = [g(r, "probes", m) for r in rs]
    def slot(kind, field, src_suffix=""):
        v = []
        for p in P:
            if not p: continue
            xs = [p[f"slot{s}_{kind}{src_suffix}"][field] for s in range(m) if f"slot{s}_{kind}{src_suffix}" in p]
            if xs: v.append(float(np.mean(xs)))
        return v
    def one(name, field):
        return [p[name][field] for p in P if p and name in p]
    cell = lambda a, b: f"{ms(a)} / {ms(b)}"
    lines.append(f"| {c} | {m} | {'role' if role else 'shared'} | {cell(slot('linear','char_acc'), slot('linear','code_exact'))} | "
                 f"{cell(slot('mlp','char_acc'), slot('mlp','code_exact'))} | "
                 f"{cell(one('namecond_linear','char_acc'), one('namecond_linear','code_exact'))} | "
                 f"{cell(one('namecond_mlp','char_acc'), one('namecond_mlp','code_exact'))} | "
                 f"{ms(slot('mlp','char_acc','_untrained_head'))} | {ms(one('slot0_mlp_shuffled_labels','char_acc'))} |")
    out.setdefault(f"c{c}_m{m}_{'role' if role else 'std'}", {})["probes"] = P

lines.append("\n## Interference: latent of {A,B} vs latents of {A}, {B} (centered; M_train ≥ 2)\n")
lines.append("| chars | M | write | cos(AB, A) | cos(AB, B) | cos(AB, A+B) | cos(AB, unrelated) | cos(AB, BA) | R² AB≈αA+βB | α | β | R² unrelated control | ‖AB‖/‖A‖ |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(groups):
    c, m, role = key; rs = groups[key]
    I = [r["interference"] for r in rs if r.get("interference")]
    if not I: continue
    f = lambda k: ms([i[k] for i in I])
    lines.append(f"| {c} | {m} | {'role' if role else 'shared'} | {f('cos_AB_A')} | {f('cos_AB_B')} | {f('cos_AB_sum')} | {f('cos_AB_unrelated')} | "
                 f"{f('cos_AB_BA')} | {f('lin_R2')} | {f('lin_alpha')} | {f('lin_beta')} | {f('lin_R2_unrelated_control')} | {f('norm_ratio_AB_over_A')} |")

(R / "SUMMARY_TABLES.md").write_text("\n".join(lines) + "\n")
(R / "summary.json").write_text(json.dumps(out, indent=1, default=float))
print("\n".join(lines))
print(f"\n{len(runs)} runs aggregated", file=sys.stderr)
