"""Summarize the binding follow-up runs into small, citable tables (CPU only, no checkpoints).

Reads   results/binding_followup/runs/*.json   (written by `t3_binding.py analyze`)
Writes  results/binding_followup/summary.csv   one row per run
        results/binding_followup/curves.csv    in-training eval curve of every run
        results/binding_followup/PAPER_NUMBERS.md   every binding number quoted in the paper,
                                                    recomputed from the JSONs

Usage:  python3 src/latentmem/binding_followup/summarize.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results/binding_followup"
RUNS = R / "runs"


def load(tag):
    return json.loads((RUNS / f"{tag}.json").read_text())


def describe(tag, d):
    m = d["meta"]
    # Runs from the earlier harness revision (*.r16) have no lora_rank field; the harness
    # treats a missing field as 16 (see load() in t3_binding.py).
    rank = m.get("lora_rank", 16)
    if tag.endswith("_paper") or (tag.endswith(".r16") and "conv" not in tag):
        arm = "paper_protocol_3k"
    elif m["role"]:
        arm = "per_line_mask"
    else:
        arm = "shared_latent_trained_longer"
    return dict(tag=tag, arm=arm, lora_rank=rank, seed=m["seed"], m_train=m["mmax"],
                role_mask=int(bool(m["role"])), final_step=m.get("final_step", 3000),
                stop_rule=m.get("stop_rule") or ("fixed_3k_cosine" if arm == "paper_protocol_3k" else "loss"),
                stop_reason=m.get("stop_reason") or "")


def summary_rows():
    rows = []
    for p in sorted(RUNS.glob("*.json")):
        tag = p.stem
        d = json.loads(p.read_text())
        row = describe(tag, d)
        for M in ("1", "2", "3"):
            lm = d["lm"].get(M)
            row[f"tf_exact_M{M}"] = round(lm["tf_exact"], 4) if lm else ""
        top = d["lm"][str(max(int(k) for k in d["lm"]))]
        row["greedy_queried_topM"] = round(top["greedy_exact"], 4)
        row["greedy_other_code_topM"] = round(top["greedy_other_code_exact"], 4)
        row["greedy_any_stored_topM"] = round(top["greedy_any_stored_code"], 4)
        P = d["probes"].get(str(row["m_train"]), {})
        row["slot_probe_linear_char_acc"] = " ".join(
            f'{P[f"slot{i}_linear"]["char_acc"]:.3f}' for i in range(row["m_train"]) if f"slot{i}_linear" in P)
        I = d.get("interference") or {}
        row["cos_AB_BA"] = round(I["cos_AB_BA"], 3) if I else ""
        row["lin_R2"] = round(I["lin_R2"], 3) if I else ""
        rows.append(row)
    return rows


def curve_rows():
    out = []
    for p in sorted(RUNS.glob("*.json")):
        for c in json.loads(p.read_text())["meta"].get("curve") or []:
            out.append(dict(tag=p.stem, step=c["step"], loss=round(c["loss"], 4),
                            tf_M1=c.get("tf_M1", ""), tf_M2=c.get("tf_M2", ""), tf_M3=c.get("tf_M3", "")))
    return out


def get(tag, *path):
    x = load(tag)
    for k in path:
        x = x[k]
    return x


def first_step_at(tag, key, thr):
    for c in load(tag)["meta"]["curve"]:
        if c.get(key, 0) >= thr:
            return c["step"]
    return None


# (where in the paper, value as quoted, source file, how it is computed)
CHECKS = [
    ("3k-step reproduction, two codes, seeds 42/43/44 (0.00--0.04)", "0.000 / 0.037 / 0.000",
     "c6_m2_s{42,43,44}_paper.json", "lm.2.tf_exact",
     lambda: " / ".join(f'{get(f"c6_m2_s{s}_paper", "lm", "2", "tf_exact"):.3f}' for s in (42, 43, 44))),
    ("3k-step reproduction, one code, seeds 42/43/44", "0.648 / 0.395 / 0.951",
     "c6_m1_s{42,43,44}_paper.json", "lm.1.tf_exact",
     lambda: " / ".join(f'{get(f"c6_m1_s{s}_paper", "lm", "1", "tf_exact"):.3f}' for s in (42, 43, 44))),
    ("3k-step reproduction, three codes, seeds 42/43/44", "0.027 / 0.000 / 0.014",
     "c6_m3_s{42,43,44}_paper.json", "lm.3.tf_exact",
     lambda: " / ".join(f'{get(f"c6_m3_s{s}_paper", "lm", "3", "tf_exact"):.3f}' for s in (42, 43, 44))),
    ("trained to plateau, two codes, rank 32 (0.555)", "0.555", "c6_m2_s42_conv.json", "lm.2.tf_exact",
     lambda: f'{get("c6_m2_s42_conv", "lm", "2", "tf_exact"):.3f}'),
    ("  stop step of that run", "26000", "c6_m2_s42_conv.json", "meta.final_step",
     lambda: str(get("c6_m2_s42_conv", "meta", "final_step"))),
    ("  same model, loss stop rule at 18.5k (lower bound)", "0.541", "c6_m2_s42_conv_lossrule.json", "lm.2.tf_exact",
     lambda: f'{get("c6_m2_s42_conv_lossrule", "lm", "2", "tf_exact"):.3f}'),
    ("  first step with in-training two-code recall >= 0.5 (paper: 16,500; the study report's ~18,000 read 1k-step points)", "16500", "c6_m2_s42_conv.json",
     "meta.curve[].tf_M2", lambda: str(first_step_at("c6_m2_s42_conv", "tf_M2", 0.5))),
    ("  one-code recall of the two-code model", "0.990", "c6_m2_s42_conv.json", "lm.1.tf_exact",
     lambda: f'{get("c6_m2_s42_conv", "lm", "1", "tf_exact"):.3f}'),
    ("trained to plateau, two codes, rank 16 (0.715)", "0.715", "c6_m2_s42_conv.r16.json", "lm.2.tf_exact",
     lambda: f'{get("c6_m2_s42_conv.r16", "lm", "2", "tf_exact"):.3f}'),
    ("trained longer, one code, rank 32 / rank 16", "0.982 / 0.996", "c6_m1_s42_conv{,.r16}.json", "lm.1.tf_exact",
     lambda: f'{get("c6_m1_s42_conv", "lm", "1", "tf_exact"):.3f} / {get("c6_m1_s42_conv.r16", "lm", "1", "tf_exact"):.3f}'),
    ("swaps: answer is the other stored code, two codes, rank 32 / rank 16", "0.156 / 0.277",
     "c6_m2_s42_conv{,.r16}.json", "lm.2.greedy_other_code_exact",
     lambda: f'{get("c6_m2_s42_conv", "lm", "2", "greedy_other_code_exact"):.3f} / {get("c6_m2_s42_conv.r16", "lm", "2", "greedy_other_code_exact"):.3f}'),
    ("answer is one of the stored codes, rank 32 / rank 16", "0.711 / 0.988",
     "c6_m2_s42_conv{,.r16}.json", "lm.2.greedy_any_stored_code",
     lambda: f'{get("c6_m2_s42_conv", "lm", "2", "greedy_any_stored_code"):.3f} / {get("c6_m2_s42_conv.r16", "lm", "2", "greedy_any_stored_code"):.3f}'),
    ("slot probe (linear, char acc), two-code latent, r32 lines 1/2", "0.79 / 0.83",
     "c6_m2_s42_conv.json", "probes.2.slot{0,1}_linear.char_acc",
     lambda: " / ".join(f'{get("c6_m2_s42_conv", "probes", "2", f"slot{i}_linear", "char_acc"):.2f}' for i in (0, 1))),
    ("slot probe (linear, char acc), two-code latent, r16 lines 1/2", "0.90 / 0.90",
     "c6_m2_s42_conv.r16.json", "probes.2.slot{0,1}_linear.char_acc",
     lambda: " / ".join(f'{get("c6_m2_s42_conv.r16", "probes", "2", f"slot{i}_linear", "char_acc"):.2f}' for i in (0, 1))),
    ("slot probe (linear, char acc), one-code model", "0.92", "c6_m1_s42_conv.json", "probes.1.slot0_linear.char_acc",
     lambda: f'{get("c6_m1_s42_conv", "probes", "1", "slot0_linear", "char_acc"):.2f}'),
    ("slot probe, untrained write head (MLP), two codes r32", "0.24 / 0.23", "c6_m2_s42_conv.json",
     "probes.2.slot{0,1}_mlp_untrained_head.char_acc",
     lambda: " / ".join(f'{get("c6_m2_s42_conv", "probes", "2", f"slot{i}_mlp_untrained_head", "char_acc"):.2f}' for i in (0, 1))),
    ("cos(L_AB, L_BA): 3k / r32 / r16 / mask (two codes)", "0.93 / 0.81 / 0.32 / -0.03",
     "c6_m2_s42_{paper,conv,conv.r16,role_conv}.json", "interference.cos_AB_BA",
     lambda: " / ".join(f'{get(t, "interference", "cos_AB_BA"):.2f}' for t in
                        ("c6_m2_s42_paper", "c6_m2_s42_conv", "c6_m2_s42_conv.r16", "c6_m2_s42_role_conv"))),
    ("linear superposition fit R^2 (alpha, beta), 3k model", "0.88 (0.42, 0.43)", "c6_m2_s42_paper.json",
     "interference.lin_R2/lin_alpha/lin_beta",
     lambda: f'{get("c6_m2_s42_paper", "interference", "lin_R2"):.2f} ({get("c6_m2_s42_paper", "interference", "lin_alpha"):.2f}, {get("c6_m2_s42_paper", "interference", "lin_beta"):.2f})'),
    ("shared latent, three codes, rank 32, three stop rules (v8 / loss / v7)", "0.135 / 0.129 / 0.117",
     "c6_m3_s42_conv{,_lossrule,_v7stop}.json", "lm.3.tf_exact",
     lambda: " / ".join(f'{get(t, "lm", "3", "tf_exact"):.3f}' for t in
                        ("c6_m3_s42_conv", "c6_m3_s42_conv_lossrule", "c6_m3_s42_conv_v7stop"))),
    ("  their stop steps", "24500 / 25500 / 19000", "same", "meta.final_step",
     lambda: " / ".join(str(get(t, "meta", "final_step")) for t in
                        ("c6_m3_s42_conv", "c6_m3_s42_conv_lossrule", "c6_m3_s42_conv_v7stop"))),
    ("shared latent, three codes, rank 16", "0.311", "c6_m3_s42_conv.r16.json", "lm.3.tf_exact",
     lambda: f'{get("c6_m3_s42_conv.r16", "lm", "3", "tf_exact"):.3f}'),
    ("cos(L_AB, L_BA), three-code shared model r32", "0.88", "c6_m3_s42_conv.json", "interference.cos_AB_BA",
     lambda: f'{get("c6_m3_s42_conv", "interference", "cos_AB_BA"):.2f}'),
    ("per-line mask, two codes (40k cap, still rising)", "0.893", "c6_m2_s42_role_conv.json", "lm.2.tf_exact",
     lambda: f'{get("c6_m2_s42_role_conv", "lm", "2", "tf_exact"):.3f}'),
    ("  its other-code rate", "0.061", "c6_m2_s42_role_conv.json", "lm.2.greedy_other_code_exact",
     lambda: f'{get("c6_m2_s42_role_conv", "lm", "2", "greedy_other_code_exact"):.3f}'),
    ("  first step with nonzero in-training one-code recall (paper: zero for the first 7,000 steps)", "7500",
     "c6_m2_s42_role_conv.json", "meta.curve[].tf_M1", lambda: str(first_step_at("c6_m2_s42_role_conv", "tf_M1", 0.01))),
    ("per-line mask, three codes", "0.992", "c6_m3_s42_role_conv.json", "lm.3.tf_exact",
     lambda: f'{get("c6_m3_s42_role_conv", "lm", "3", "tf_exact"):.3f}'),
    ("  its other-code rate", "0.000", "c6_m3_s42_role_conv.json", "lm.3.greedy_other_code_exact",
     lambda: f'{get("c6_m3_s42_role_conv", "lm", "3", "greedy_other_code_exact"):.3f}'),
    ("  first step with in-training three-code recall >= 0.5 (paper: within 4,500 steps)", "4500",
     "c6_m3_s42_role_conv.json", "meta.curve[].tf_M3", lambda: str(first_step_at("c6_m3_s42_role_conv", "tf_M3", 0.5))),
    ("per-line mask, two codes, rank 16, 3k steps", "0.000", "c6_m2_s42_role.r16.json", "lm.2.tf_exact",
     lambda: f'{get("c6_m2_s42_role.r16", "lm", "2", "tf_exact"):.3f}'),
    ("share of errors that are the other stored code, two codes, rank 32 / rank 16 (paper: a third / nearly all)",
     "0.35 / 0.96", "c6_m2_s42_conv{,.r16}.json", "lm.2.greedy_other_code_exact / (1 - lm.2.greedy_exact)",
     lambda: " / ".join(f'{get(t, "lm", "2", "greedy_other_code_exact") / (1 - get(t, "lm", "2", "greedy_exact")):.2f}'
                        for t in ("c6_m2_s42_conv", "c6_m2_s42_conv.r16"))),
    ("slot probe, untrained write head (MLP), two-code runs (paper: 0.22--0.25)", "0.22--0.25",
     "c6_m2_s42_{paper,conv,conv.r16}.json", "probes.2.slot{0,1}_mlp_untrained_head.char_acc",
     lambda: "{:.2f}--{:.2f}".format(*(lambda v: (min(v), max(v)))(
         [get(t, "probes", "2", f"slot{i}_mlp_untrained_head", "char_acc")
          for t in ("c6_m2_s42_paper", "c6_m2_s42_conv", "c6_m2_s42_conv.r16") for i in (0, 1)]))),
    ("one-code recall of the rank-32 two-code model at 6,000 steps (paper: learns one code by ~6k)", "0.953",
     "c6_m2_s42_conv.json", "meta.curve[step=6000].tf_M1",
     lambda: f'{[c for c in load("c6_m2_s42_conv")["meta"]["curve"] if c["step"] == 6000][0]["tf_M1"]:.3f}'),
    ("per-token separation, rank-16 two-code model: token:cos with A/cos with B (paper: four tokens copy A, four copy B, >=0.92 vs <=0.06)", ">=0.92, <=0.06",
     "c6_m2_s42_conv.r16.json", "interference.per_token_cos_with_{A,B}",
     lambda: " ".join(f'{t}:{get("c6_m2_s42_conv.r16", "interference", "per_token_cos_with_A")[t]:.2f}/{get("c6_m2_s42_conv.r16", "interference", "per_token_cos_with_B")[t]:.2f}' for t in (2, 7, 8, 9, 10, 12, 13, 14))),
]


def main():
    rows = summary_rows()
    with open(R / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    cr = curve_rows()
    with open(R / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cr[0]))
        w.writeheader(); w.writerows(cr)
    lines = ["# Binding numbers quoted in the paper, recomputed from `runs/*.json`", "",
             "Generated by `src/latentmem/binding_followup/summarize.py`. "
             "`quoted` is the value as written in the paper or study report; `recomputed` is read from the JSON now.", "",
             "| quantity | quoted | recomputed | file (`runs/`) | JSON field |", "|---|---|---|---|---|"]
    for name, quoted, src, field, fn in CHECKS:
        lines.append(f"| {name} | {quoted} | {fn()} | `{src}` | `{field}` |")
    (R / "PAPER_NUMBERS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote summary.csv ({len(rows)} runs), curves.csv ({len(cr)} rows), PAPER_NUMBERS.md ({len(CHECKS)} checks)")


if __name__ == "__main__":
    main()
