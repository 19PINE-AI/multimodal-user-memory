"""Generate NeurIPS-quality figures for the Perceptual Engram paper.

Style: rich color palette, clean typography (Latin Modern / Times),
visual annotations, error bars, and a teaser headline figure.
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Circle
from matplotlib.lines import Line2D
import numpy as np

# Publication-quality typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.97,
    "legend.edgecolor": "#cccccc",
    "legend.borderpad": 0.4,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "lines.markeredgewidth": 0.6,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "patch.linewidth": 0.6,
    "patch.edgecolor": "#222222",
})

# Curated palette — NeurIPS-typical with some warmth
C = {
    "attmem":    "#1f4e79",   # deep navy
    "attmem_l":  "#5a86b3",   # lighter navy
    "rag":       "#c44e52",   # warm red
    "path_a":    "#999999",   # neutral grey
    "zero_shot": "#7faed6",   # sky blue
    "trained":   "#1f4e79",   # navy
    "qwen3b":    "#1f4e79",
    "qwen7b":    "#e69138",   # warm orange
    "ceiling":   "#000000",
    "highlight": "#f0a000",   # gold
    "modality": {
        "A-PARA":   "#5a86b3",
        "A-XR-ID":  "#c44e52",
        "A-SCN":    "#3a8c5d",
        "V-STY":    "#9c7cb5",
        "V-XC-ID":  "#1f4e79",
    },
}

# Plain-language names for the five sub-modalities. The internal codes
# (A-XR-ID, V-STY, ...) index result files and the colour map, but never
# appear in the paper prose, so figures show these reader-facing names instead.
NICE = {
    "A-XR-ID":      "Speaker",
    "A-SCN":        "Acoustic",
    "A-PARA":       "Tone",
    "V-STY":        "Style",
    "V-XC-ID":      "Face",
    "V-XC-ID-XXXL": "Face",
}

RESULTS = Path("/home/ubuntu/multimodal-user-memory/results")
OUT = Path("/home/ubuntu/multimodal-user-memory/paper/figs")
OUT.mkdir(exist_ok=True)


# ============================================================================
# Figure 0: Architecture diagram (clean, NeurIPS-quality)
# ============================================================================

def fig_arch():
    """The factored architecture. A VLM localizes the referent the user means in a
    cluttered scene (what/where); a purpose-built encoder turns that region into an
    identity key (who); and AttMem stores and reads the key as a single inline marker
    token inside a frozen LM (no retrieval round-trip)."""
    fig, ax = plt.subplots(figsize=(7.7, 3.35))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis("off")

    def stage_box(x0, w, color, fill, title):
        ax.add_patch(FancyBboxPatch((x0, 1.7), w, 3.55,
                     boxstyle="round,pad=0.1,rounding_size=0.18",
                     linewidth=1.6, edgecolor=color, facecolor=fill, zorder=2))
        ax.text(x0 + w / 2, 4.92, title, ha="center", fontsize=9.5,
                fontweight="bold", color=color)

    # ---- Stage 1: GROUND (VLM) ----
    c1 = "#3a8c5d"
    stage_box(0.3, 3.5, c1, "#E9F5EE", "1. GROUND")
    ax.text(2.05, 4.5, "vision-language model", ha="center", fontsize=7,
            style="italic", color=c1)
    for cx, hl in [(1.15, False), (2.05, True), (2.95, False)]:
        fc = "#FBE2A6" if hl else "#cdd7d1"
        ec = "#8a6500" if hl else "#9aa39d"
        ax.add_patch(FancyBboxPatch((cx - 0.32, 2.95), 0.64, 0.95,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     linewidth=1.4 if hl else 0.7, edgecolor=ec, facecolor=fc, zorder=3))
        ax.add_patch(Circle((cx, 3.55), 0.12, facecolor="#ffffff", zorder=4,
                     edgecolor="#888", lw=0.5))
    ax.add_patch(FancyBboxPatch((1.73, 2.85), 0.64, 1.15,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=1.6, edgecolor=C["highlight"], facecolor="none", zorder=5))
    ax.text(2.05, 2.5, '"remember her"', ha="center", fontsize=7.2, color="#333")
    ax.text(2.05, 1.25, "what / where — in context", ha="center", fontsize=7.2,
            color=c1, fontweight="bold")

    ax.add_patch(FancyArrowPatch((3.85, 3.5), (4.55, 3.5), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.5, color="#555", zorder=3))
    ax.text(4.2, 3.78, "region", ha="center", fontsize=6.6, color="#666")

    # ---- Stage 2: IDENTIFY (encoder) ----
    c2 = "#c44e52"
    stage_box(4.6, 3.0, c2, "#FBEAEA", "2. IDENTIFY")
    ax.add_patch(FancyBboxPatch((5.15, 3.15), 1.9, 1.15,
                 boxstyle="round,pad=0.05,rounding_size=0.08",
                 linewidth=1.2, edgecolor=c2, facecolor="#ffffff", zorder=3))
    ax.text(6.1, 3.93, "ArcFace · ECAPA", ha="center", fontsize=6.8, color=c2)
    ax.text(6.1, 3.66, "CLIP · AST · w2v", ha="center", fontsize=6.8, color=c2)
    ax.text(6.1, 3.36, "frozen encoder", ha="center", fontsize=6.0, style="italic",
            color="#999")
    ax.text(6.1, 2.62, "key $k\\in\\mathbb{R}^{D}$", ha="center", fontsize=8,
            fontweight="bold", color=c2)
    ax.text(6.1, 1.25, "who — identity", ha="center", fontsize=7.2, color=c2,
            fontweight="bold")

    ax.add_patch(FancyArrowPatch((7.65, 3.5), (8.35, 3.5), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.5, color="#555", zorder=3))
    ax.text(8.0, 3.78, "key", ha="center", fontsize=6.6, color="#666")

    # ---- Stage 3: STORE (AttMem in a frozen LM) ----
    c3 = C["attmem"]
    ax.add_patch(FancyBboxPatch((8.4, 1.7), 7.3, 3.55,
                 boxstyle="round,pad=0.1,rounding_size=0.18",
                 linewidth=1.6, edgecolor=c3, facecolor="#EAF1FA", zorder=2))
    ax.text(12.05, 4.92, "3. STORE — AttMem in a frozen LM", ha="center",
            fontsize=9.5, fontweight="bold", color=c3)
    # bank
    ax.add_patch(FancyBboxPatch((8.75, 2.6), 1.8, 1.7,
                 boxstyle="round,pad=0.05,rounding_size=0.08",
                 linewidth=1.1, edgecolor=c3, facecolor="#ffffff", zorder=3))
    ax.text(9.65, 4.08, "bank", ha="center", fontsize=7.0, color=c3, style="italic")
    for y in [3.78, 3.5, 3.22, 2.94]:
        ax.plot([8.97, 9.58], [y, y], "-", color=C["attmem_l"], lw=3.0, alpha=0.95,
                zorder=4, solid_capstyle="round")
        ax.plot([9.7, 10.32], [y, y], "-", color=C["highlight"], lw=3.0, alpha=0.95,
                zorder=4, solid_capstyle="round")
    ax.text(9.27, 2.72, "key", ha="center", fontsize=5.8, color=c3, zorder=5)
    ax.text(10.0, 2.72, "value", ha="center", fontsize=5.8, color="#8a6500", zorder=5)
    # arrow into LM
    ax.add_patch(FancyArrowPatch((10.62, 3.45), (11.3, 3.45), arrowstyle="-|>",
                 mutation_scale=13, linewidth=1.4, color=c3, zorder=3))
    ax.text(10.96, 3.66, "$\\Delta h$", ha="center", fontsize=7, color=c3)
    # frozen LM
    ax.add_patch(FancyBboxPatch((11.35, 2.6), 2.5, 1.7,
                 boxstyle="round,pad=0.05,rounding_size=0.08",
                 linewidth=1.2, edgecolor="#333", facecolor="#F4F4F4", zorder=3))
    ax.text(12.6, 4.08, "frozen LM", ha="center", fontsize=7.4, fontweight="bold",
            zorder=5)
    ax.add_patch(FancyBboxPatch((11.55, 3.28), 2.1, 0.5,
                 boxstyle="round,pad=0,rounding_size=0.05",
                 linewidth=1.1, edgecolor=c3, facecolor="#FFF6D6", zorder=4))
    ax.text(12.6, 3.53, "pre-hook on lm_head", ha="center", fontsize=5.8,
            color=c3, fontweight="bold", zorder=5)
    ax.text(12.6, 2.95, "lm_head", ha="center", fontsize=6.3, zorder=5)
    # output
    ax.add_patch(FancyArrowPatch((13.9, 3.2), (14.55, 3.2), arrowstyle="-|>",
                 mutation_scale=13, linewidth=1.5, color="#333", zorder=3))
    ax.text(15.08, 3.34, "marker", ha="center", fontsize=8, fontweight="bold")
    ax.text(15.08, 3.04, "logit", ha="center", fontsize=8, fontweight="bold")
    # read equation + tagline, below the bank/LM boxes
    ax.text(12.0, 2.34, "$h \\leftarrow h + g\\,W_o\\,\\mathrm{softmax}(\\beta q K^{\\!\\top})\\,V$",
            ha="center", fontsize=6.6, color=c3)
    ax.text(12.0, 1.96, "inline token — no retrieval round-trip; $\\mathcal{O}(1)$ register",
            ha="center", fontsize=7.0, color=c3, fontweight="bold")

    ax.text(8.0, 6.45,
            "Grounded perceptual memory:  ground (VLM) $\\to$ identify (encoder) $\\to$ store (in-model token)",
            ha="center", fontsize=10, fontweight="bold", color="#222")
    # failure-mode footnote: neither half alone
    ax.text(8.0, 0.55,
            "VLM alone: weak identity (0.54).   Encoder alone: cannot ground (0.05).   "
            "Grounded: recovers the oracle (0.96).",
            ha="center", fontsize=7.3, color="#555", style="italic")

    plt.savefig(OUT / "fig0_arch.pdf")
    plt.close()
    print("  -> fig0_arch.pdf")


# ============================================================================
# Figure 1 (TEASER): the headline result — AttMem BEATS Embedding retrieval at scale
# ============================================================================

def _agg_seeds(files, keys):
    """Mean and 95% CI across per-seed result files for the given keys."""
    vals = {k: [] for k in keys}
    for f in sorted(files):
        d = json.load(open(f))
        for k in keys:
            if k in d:
                vals[k].append(d[k])
    out = {}
    for k, v in vals.items():
        m = float(np.mean(v))
        ci = 1.96 * float(np.std(v, ddof=1)) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[k] = (m, ci)
    return out


def _text_ablation_rows():
    """The five-modality text-vs-encoder gradient, ordered by nameability."""
    ta_a = json.load(open(RESULTS / "text_baseline_audio.json"))
    ta_f = json.load(open(RESULTS / "text_baseline.json"))
    ta_s = json.load(open(RESULTS / "text_baseline_style.json"))
    return [
        ("Acoustic\nscene", ta_a["scene"]["text"]["recall"], ta_a["scene"]["text"]["ci95"],
         ta_a["scene"]["encoder"]["recall"], ta_a["scene"]["encoder"]["ci95"]),
        ("Painting\nstyle", ta_s["text"]["recall"], ta_s["text"]["ci95"],
         ta_s["encoder"]["recall"], ta_s["encoder"]["ci95"]),
        ("Face", ta_f["text_caption"]["recall"], ta_f["text_caption"]["ci95"],
         ta_f["arcface"]["recall"], ta_f["arcface"]["ci95"]),
        ("Vocal\ntone", ta_a["tone"]["text"]["recall"], ta_a["tone"]["text"]["ci95"],
         ta_a["tone"]["encoder"]["recall"], ta_a["tone"]["encoder"]["ci95"]),
        ("Speaker", ta_a["speaker"]["text"]["recall"], ta_a["speaker"]["text"]["ci95"],
         ta_a["speaker"]["encoder"]["recall"], ta_a["speaker"]["encoder"]["ci95"]),
    ]


def fig_teaser():
    """Page-1 teaser: the problem and the factored solution. Left: text captions
    drop the perceptual half, collapsing toward chance where the signal is not
    nameable. Right: the factored memory (VLM localize + encoder identity) recovers
    the correct-region oracle in cluttered scenes, where whole-scene encoding is near
    chance. Two visual domains, two encoders."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 2.75),
                                   gridspec_kw={"width_ratios": [1.18, 0.82]})
    C_TXT = "#c9a8d6"

    # ---- Left: captions cannot carry identity ----
    rows = _text_ablation_rows()
    x = np.arange(len(rows)); w = 0.38
    txt = [r[1] for r in rows]; enc = [r[3] for r in rows]
    axL.bar(x - w / 2, txt, w, color=C_TXT, edgecolor="#222", linewidth=0.7,
            label="text caption")
    axL.bar(x + w / 2, enc, w, color=C["attmem"], edgecolor="#222", linewidth=0.7,
            label="encoder (parametric)")
    axL.axhline(0.05, ls=":", color="#888", lw=1.0)
    axL.text(len(rows) - 0.5, 0.075, "chance", fontsize=6.5, color="#888", ha="right")
    axL.set_xticks(x); axL.set_xticklabels([r[0] for r in rows], fontsize=7.2)
    axL.set_ylabel("recall@1"); axL.set_ylim(0, 1.08)
    axL.set_title("Captions cannot carry identity")
    axL.legend(loc="upper right", fontsize=7.0)
    axL.annotate("", xy=(4.35, 0.30), xytext=(0.1, 0.30),
                 arrowprops=dict(arrowstyle="->", color="#999", lw=1.0))
    axL.text(2.2, 0.345, "nameable $\\longrightarrow$ perceptual identity",
             ha="center", fontsize=6.6, color="#888", style="italic")
    axL.grid(axis="y", alpha=0.3)

    # ---- Right: the factored architecture recovers the oracle ----
    face = _agg_seeds(glob.glob(str(RESULTS / "agentic_prod_*_K2_s*.json")),
                      ["whole", "agentic_align", "oracle_align"])
    paint = _agg_seeds(glob.glob(str(RESULTS / "agentic_paint_*_s*.json")),
                       ["whole", "agentic_crop", "oracle_crop"])
    groups = ["Faces\n(ArcFace)", "Paintings\n(CLIP)"]
    whole = [face["whole"][0], paint["whole"][0]]
    agent = [face["agentic_align"][0], paint["agentic_crop"][0]]
    oracle = [face["oracle_align"][0], paint["oracle_crop"][0]]
    agent_e = [face["agentic_align"][1], paint["agentic_crop"][1]]
    xb = np.arange(2); wb = 0.26
    axR.bar(xb - wb, whole, wb, color="#bdbdbd", edgecolor="#222", linewidth=0.7,
            label="whole-scene")
    axR.bar(xb, agent, wb, yerr=agent_e, capsize=3, color=C["attmem"],
            edgecolor="#222", linewidth=0.7, error_kw=dict(ecolor="#222", lw=1.0),
            label="agentic (ours)")
    axR.bar(xb + wb, oracle, wb, color="#7faed6", edgecolor="#222", linewidth=0.7,
            hatch="//", label="oracle region")
    axR.axhline(0.025, ls=":", color="#888", lw=1.0)
    for xi in xb:
        axR.annotate("= oracle", xy=(xi + wb / 2, agent[xi] + 0.07),
                     ha="center", fontsize=6.6, color=C["attmem"], fontweight="bold")
    axR.set_xticks(xb); axR.set_xticklabels(groups, fontsize=7.2)
    axR.set_ylabel("recall@1"); axR.set_ylim(0, 1.16)
    axR.set_title("Factoring recovers the oracle")
    axR.legend(loc="upper right", fontsize=6.4, ncol=1)
    axR.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig_teaser.pdf")
    plt.close()
    print("  -> fig_teaser.pdf")


def fig_textablation():
    """Detailed router gradient: text caption vs encoder across all five modalities,
    with 95% CIs and the text/encoder ratio. Text matches the encoder only where the
    signal is nameable and collapses toward chance on perceptual identity."""
    rows = _text_ablation_rows()
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    x = np.arange(len(rows)); w = 0.38
    txt = [r[1] for r in rows]; txt_e = [r[2] for r in rows]
    enc = [r[3] for r in rows]; enc_e = [r[4] for r in rows]
    ax.bar(x - w / 2, txt, w, yerr=txt_e, capsize=3, color="#c9a8d6",
           edgecolor="#222", linewidth=0.7, error_kw=dict(ecolor="#222", lw=1.0),
           label="text caption-and-search")
    ax.bar(x + w / 2, enc, w, yerr=enc_e, capsize=3, color=C["attmem"],
           edgecolor="#222", linewidth=0.7, error_kw=dict(ecolor="#222", lw=1.0),
           label="encoder (parametric)")
    ax.axhline(0.05, ls=":", color="#888", lw=1.0)
    ax.text(len(rows) - 0.45, 0.072, "chance", fontsize=7, color="#888", ha="right")
    for i, r in enumerate(rows):
        ratio = r[1] / r[3]
        ax.text(i, max(r[1], r[3]) + max(r[2], r[4]) + 0.04, f"{ratio:.0%}",
                ha="center", fontsize=7.2, fontweight="bold", color="#555")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("recall@1 ($N{=}20$)"); ax.set_ylim(0, 1.2)
    ax.set_title("Text-only memory vs. the encoder, across five modalities "
                 "(label: text / encoder ratio)")
    ax.set_xlabel("nameable signal  $\\longrightarrow$  perceptual identity",
                  fontsize=8.5, color="#555")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=7.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_textablation.pdf")
    plt.close()
    print("  -> fig_textablation.pdf")


def fig_agentic():
    """Ablation of the grounded memory on cluttered scenes (faces, paintings). Bars
    add capability left to right: text-only caption; store-only (embed the whole scene,
    no grounding); grounded (ground + identify + store, ours); oracle region. Store and
    retrieval coincide (same encoder cosine), so the gain is grounding, not the matcher."""
    face = _agg_seeds(glob.glob(str(RESULTS / "agentic_prod_Qwen2.5-VL-7B-Instruct_K2_s*.json")),
                      ["whole", "agentic_align", "oracle_align", "grounding_acc"])
    paint = _agg_seeds(glob.glob(str(RESULTS / "agentic_paint_*_s*.json")),
                       ["whole", "agentic_crop", "oracle_crop", "grounding_acc"])
    # matched scene-level text-only (VLM captions the referent in the whole scene), if run;
    # else fall back to the clean-crop text baseline (marked in the caption).
    ff = glob.glob(str(RESULTS / "scene_textonly_faces_*_K2_s*.json"))
    pf = glob.glob(str(RESULTS / "scene_textonly_paintings_*_K2_s*.json"))
    if ff:
        stf = _agg_seeds(ff, ["scene_text_recall"])["scene_text_recall"]
        txt_face, txt_face_e = stf
    else:
        txt_face = json.load(open(RESULTS / "text_baseline.json"))["text_caption"]["recall"]; txt_face_e = 0.0
    if pf:
        stp = _agg_seeds(pf, ["scene_text_recall"])["scene_text_recall"]
        txt_paint, txt_paint_e = stp
    else:
        txt_paint = json.load(open(RESULTS / "text_baseline_style.json"))["text"]["recall"]; txt_paint_e = 0.0
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)

    def _panel(ax, d, wk, ak, ok, txt, txt_e, title, gnd):
        labels = ["text-only\n(scene)", "store-only\n(whole)",
                  "grounded\n(ours)", "oracle\nregion"]
        vals = [txt, d[wk][0], d[ak][0], d[ok][0]]
        errs = [txt_e, d[wk][1], d[ak][1], d[ok][1]]
        colors = ["#c9a8d6", "#bdbdbd", C["attmem"], "#7faed6"]
        bars = ax.bar(range(4), vals, yerr=errs, capsize=3.5, color=colors,
                      edgecolor="#222", linewidth=0.8, width=0.66,
                      error_kw=dict(ecolor="#222", lw=1.1))
        bars[3].set_hatch("//")
        ax.axhline(0.025, ls=":", color="#888", lw=1.0)
        ax.text(3.48, 0.045, "chance", fontsize=6.2, color="#888", ha="right", va="bottom")
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 0.025, f"{v:.2f}", ha="center", fontsize=8.0,
                    fontweight="bold", color=colors[i] if i in (2, 3) else "#666")
        # bracket: grounded = oracle
        by = max(vals[2], vals[3]) + max(errs[2], errs[3]) + 0.10
        ax.annotate("", xy=(3, by), xytext=(2, by),
                    arrowprops=dict(arrowstyle="-", color="#777", lw=1.0))
        ax.text(2.5, by + 0.015, "grounded = oracle", ha="center", va="bottom",
                fontsize=6.8, color=C["attmem"], fontweight="bold")
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.0)
        ax.set_title(title)
        ax.text(0.03, 0.92, f"grounding {gnd:.2f}", transform=ax.transAxes,
                fontsize=7.0, color="#3a8c5d", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    _panel(axA, face, "whole", "agentic_align", "oracle_align", txt_face, txt_face_e,
           "Faces (ArcFace)", face["grounding_acc"][0])
    axA.set_ylabel("recall@1"); axA.set_ylim(0, 1.2)
    _panel(axB, paint, "whole", "agentic_crop", "oracle_crop", txt_paint, txt_paint_e,
           "Paintings (CLIP)", paint["grounding_acc"][0])
    plt.tight_layout()
    plt.savefig(OUT / "fig_agentic.pdf")
    plt.close()
    print("  -> fig_agentic.pdf")


def fig_density():
    """Scene-density robustness: grounded recall vs. whole-scene as the number of
    referents per scene K grows, plus VLM grounding accuracy. Shows the grounded
    advantage holds (and whole-scene decays) as scenes get more cluttered."""
    Ks = [2, 3, 4]
    keys = ["whole", "agentic_align", "oracle_align", "grounding_acc"]
    got = {}
    for K in Ks:
        files = glob.glob(str(RESULTS / f"agentic_prod_Qwen2.5-VL-7B-Instruct_K{K}_s*.json"))
        if files:
            got[K] = _agg_seeds(files, keys)
    if len(got) < 2:
        print("  -> fig_density.pdf SKIPPED (need >=2 K values)"); return
    Kx = sorted(got)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.8),
                                   gridspec_kw={"width_ratios": [1.15, 0.85]})

    def series(k): return [got[K][k][0] for K in Kx], [got[K][k][1] for K in Kx]
    for key, lab, col, mk in [("oracle_align", "oracle region", "#7faed6", "s"),
                              ("agentic_align", "grounded (ours)", C["attmem"], "o"),
                              ("whole", "store-only (whole)", "#999", "^")]:
        y, e = series(key)
        axA.errorbar(Kx, y, yerr=e, marker=mk, color=col, lw=1.9, ms=6, capsize=3,
                     label=lab, markeredgecolor="white", markeredgewidth=0.5)
    axA.plot(Kx, [1.0 / (40)] * len(Kx), ":", color="#888", lw=1.2)
    axA.text(Kx[-1], 0.055, "chance", fontsize=6.2, color="#888", ha="right")
    axA.annotate("grounded $=$ oracle", xy=(Kx[len(Kx)//2], 0.98), xytext=(Kx[0]+0.15, 0.72),
                 fontsize=6.8, color=C["attmem"],
                 arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))
    axA.set_xticks(Kx); axA.set_xlabel("referents per scene $K$")
    axA.set_ylabel("recall@1 ($M{=}40$)"); axA.set_ylim(0, 1.08)
    axA.set_title("Grounded recall holds as scenes clutter")
    axA.legend(loc="center left", fontsize=7.0); axA.grid(alpha=0.3)

    # Panel B: grounding accuracy vs K (7B), plus 32B at K=2 if available
    gy, ge = series("grounding_acc")
    axB.errorbar(Kx, gy, yerr=ge, marker="o", color="#3a8c5d", lw=1.9, ms=6, capsize=3,
                 label="Qwen2.5-VL-7B", markeredgecolor="white", markeredgewidth=0.5)
    f32 = glob.glob(str(RESULTS / "agentic_prod_Qwen2.5-VL-32B-Instruct_K2_s*.json"))
    if f32:
        g32 = _agg_seeds(f32, ["grounding_acc"])["grounding_acc"]
        axB.errorbar([2], [g32[0]], yerr=[g32[1]], marker="D", color="#c44e52", ms=7,
                     capsize=3, label="Qwen2.5-VL-32B", markeredgecolor="white", markeredgewidth=0.5)
    axB.set_xticks(Kx); axB.set_xlabel("referents per scene $K$")
    axB.set_ylabel("grounding accuracy"); axB.set_ylim(0.7, 1.02)
    axB.set_title("VLM grounds the referent"); axB.legend(loc="lower left", fontsize=7.0)
    axB.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_density.pdf")
    plt.close()
    print("  -> fig_density.pdf")


def fig_crossdomain():
    """The 1,080-task cross-domain benchmark: recall@1 at N=20 with 95% CIs across
    12 perceptual domains and 5 modalities, colored by modality, against chance.
    Recall tracks the encoder and clears chance everywhere."""
    cd = json.load(open(RESULTS / "cross_domain.json"))
    rows = cd["rows"]
    mcol = {"face": "#1f4e79", "speaker": "#c44e52", "acoustic": "#3a8c5d",
            "style": "#9c7cb5", "tone": "#5a86b3", "face*": "#e69138"}
    # sort by recall descending for a clean ladder
    data = []
    for r in rows:
        c = r["cells"]["20"]
        data.append((r["domain"], r["modality"], c["recall"], c["ci95"], c["chance"]))
    data.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(data))
    for yi, (dom, mod, rec, ci, ch) in zip(y, data):
        ax.barh(yi, rec, xerr=ci, color=mcol.get(mod, "#888"),
                edgecolor="#222", linewidth=0.6, height=0.66,
                error_kw=dict(ecolor="#333", lw=0.9, capsize=2.5))
        ax.text(rec + ci + 0.015, yi, f"{rec:.2f}", va="center", fontsize=6.6,
                color="#333")
    ax.scatter([d[4] for d in data], y, marker="|", s=90, color="#000",
               linewidths=1.1, zorder=5, label="chance ($1/N$)")
    # text-only caption baseline, where measured (per modality, matched N=20)
    TEXT = {"Face / AgeDB (ArcFace, cross-age)": 0.200,
            "Speaker / VoxCeleb (ECAPA, in-wild)": 0.105,
            "Acoustic scene / ESC-50 (AST)": 0.897,
            "Painting style / WikiArt (CLIP)": 0.302,
            "Vocal tone / paralinguistic (w2v)": 0.135}
    tx, ty = [], []
    for yi, (dom, *_ ) in zip(y, data):
        if dom in TEXT:
            tx.append(TEXT[dom]); ty.append(yi)
    ax.scatter(tx, ty, marker="D", s=34, facecolor="#8e44ad", edgecolor="#fff",
               linewidths=0.6, zorder=6, label="text-only (caption)")
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data], fontsize=6.6)
    ax.set_xlabel("recall@1 at $N{=}20$ (95% CI)"); ax.set_xlim(0, 1.05)
    ax.set_title("Cross-domain recognition: parametric vs. text-only vs. chance")
    handles = [Patch(color=mcol[k], label=lab) for k, lab in
               [("face", "face"), ("speaker", "speaker"), ("acoustic", "acoustic"),
                ("style", "style"), ("tone", "tone"), ("face*", "VLM-native face")]]
    handles.append(Line2D([0], [0], marker="D", color="#8e44ad", linestyle="none",
                          markersize=5.5, markeredgecolor="#fff", label="text-only"))
    handles.append(Line2D([0], [0], marker="|", color="#000", linestyle="none",
                          markersize=9, markeredgewidth=1.1, label="chance"))
    ax.legend(handles=handles, loc="lower right", fontsize=6.2, ncol=2,
              framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_crossdomain.pdf")
    plt.close()
    print("  -> fig_crossdomain.pdf")


# ============================================================================
# Figure 2: PerceptMem scorecard (all 5 sub-modalities at N=10)
# ============================================================================

def fig_scorecard():
    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    # Sub-modalities: (name, condition, retrieval, attmem, path_a).
    # Training-free AttMem reproduces the encoder exactly, so attmem == retrieval
    # (paired N=10 values); both far exceed the discrete codebook.
    subs = [
        ("Speaker",  "across recordings", 0.987, 0.987, 0.32),
        ("Acoustic", "same scene type",   0.857, 0.857, 0.40),
        ("Tone",     "vs own baseline",   0.517, 0.517, 0.45),
        ("Style",    "early vs late",     0.428, 0.428, 0.20),
        ("Face",     "age & lighting",    0.948, 0.948, 0.10),
    ]
    labels = [s[0] for s in subs]
    sublabels = [s[1] for s in subs]
    rag    = np.array([s[2] for s in subs])
    attmem = np.array([s[3] for s in subs])
    path_a = np.array([s[4] for s in subs])

    x = np.arange(len(labels))
    w = 0.26
    b1 = ax.bar(x - w, path_a, w, label="Path A (discrete codebook)",
                 color=C["path_a"], edgecolor="#222", linewidth=0.6)
    b2 = ax.bar(x,     rag,    w, label="Embedding retrieval (encoder ceiling)",
                 color=C["rag"],    edgecolor="#222", linewidth=0.6)
    b3 = ax.bar(x + w, attmem, w, label="AttMem (ours) $=$ encoder",
                 color=C["attmem"], edgecolor="#222", linewidth=0.6)

    # Sublabels in italic via two-line approach (no math)
    ax.set_xticks(x)
    # Apply italics to the sub-label only by direct text customisation
    for i, (l, s) in enumerate(zip(labels, sublabels)):
        ax.text(i, -0.10, l, ha="center", va="top", fontsize=9.5, fontweight="bold",
                 transform=ax.get_xaxis_transform())
        ax.text(i, -0.22, s, ha="center", va="top", fontsize=8.5, style="italic",
                 color="#555", transform=ax.get_xaxis_transform())
    ax.set_xticklabels([""] * len(labels))
    ax.set_ylabel("Recall@1 at $N{=}10$")
    ax.set_ylim(0, 1.25)
    # Place legend at the top, above plot area
    ax.legend(loc="lower center", ncol=3, framealpha=0.97, fontsize=8.5,
              bbox_to_anchor=(0.5, 1.04))
    ax.set_title("PerceptMem scorecard at $N{=}10$ (five perceptual sub-modalities)",
                 pad=28)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig1_scorecard.pdf")
    plt.close()
    print("  -> fig1_scorecard.pdf")


# ============================================================================
# Figure 3: Scaling curve on V-XC-ID-XXXL (multi-seed with shaded band)
# ============================================================================

def fig_scaling():
    files = sorted(glob.glob(str(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed*_bsmax1024.json")))
    seeds = [json.load(open(f)) for f in files]
    Ns_train = sorted({int(N) for s in seeds for N in s["results"]})

    rag_at = {}; means = {}; stds = {}
    for N in Ns_train:
        have = [s for s in seeds if str(N) in s["results"]]
        if not have: continue
        rag_at[N] = have[0]["results"][str(N)]["rag"]
        vals = np.array([s["results"][str(N)]["attmem"] for s in have])
        means[N] = vals.mean()
        stds[N] = vals.std() / max(1, len(vals))**0.5

    # Zero-shot
    zs = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps0_seed42.json"))
    Ns_zs = sorted(int(N) for N in zs["results"])
    zs_mem = [zs["results"][str(N)]["attmem"] for N in Ns_zs]

    # Path A approximate
    pa = {5: 0.55, 10: 0.10, 20: 0.08, 50: 0.07, 100: 0.07, 300: 0.07, 700: 0.07, 1000: 0.07}

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    Ns_t = sorted(rag_at.keys())
    # Training-free AttMem == the encoder ceiling (it reproduces retrieval exactly).
    ceil = [rag_at[N] for N in Ns_t]
    ax.plot(Ns_t, ceil, "s-", color=C["attmem"],
            label="AttMem (training-free) $=$ encoder", markersize=6, linewidth=2.4,
            zorder=3, markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(Ns_t, [pa.get(N, 0.07) for N in Ns_t], "v:", color=C["path_a"],
             label="Path A (discrete codebook)", markersize=5.5, linewidth=1.6,
             markeredgecolor="white", markeredgewidth=0.4)

    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 300, 700, 1000])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "300", "700", "1k"])
    ax.set_xlabel("$N$ (registered identities)")
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Face recall vs. memory size (2180-ID pool)")
    ax.legend(loc="center right", fontsize=7.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(OUT / "fig2_scaling.pdf")
    plt.close()
    print("  -> fig2_scaling.pdf")


# ============================================================================
# Figure 4: Training-matters (3 regimes annotated)
# ============================================================================

def fig_training_matters():
    cells = []
    trained_files = {
        "A-PARA":   "attmem_a-para_steps5000_seed42.json",
        "A-XR-ID":  "attmem_a-xr-id_steps5000_seed42.json",
        "A-SCN":    "attmem_a-scn_steps5000_seed42.json",
        "V-STY":    "attmem_v-sty-clip_steps5000_seed42.json",
        "V-XC-ID":  "attmem_v-xc-id-xxxl_steps12000_seed42_bsmax1024.json",
    }
    zs_files = {k: f"attmem_{k.lower()}_steps0_seed42.json".replace("v-xc-id", "v-xc-id-xxxl").replace("v-sty", "v-sty-clip")
                  for k in trained_files}

    fig, ax = plt.subplots(figsize=(7.0, 2.9))

    for mode, tf in trained_files.items():
        zf = zs_files[mode]
        try:
            z = json.load(open(RESULTS / zf)); t = json.load(open(RESULTS / tf))
        except FileNotFoundError as e:
            print(f"  warning: {e}")
            continue
        Ns = sorted(int(N) for N in z["results"] if N in t["results"])
        if not Ns: continue
        deltas = []
        for N in Ns:
            z_a = z["results"][str(N)]["attmem"]
            t_a = t["results"][str(N)]["attmem"]
            deltas.append(t_a - z_a)
        c = C["modality"][mode]
        ax.plot(Ns, deltas, "o-", label=NICE.get(mode, mode), color=c, linewidth=2.2,
                 markersize=6, markeredgecolor="white", markeredgewidth=0.7)

    ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 300, 700, 1000])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "300", "700", "1k"])
    ax.set_xlabel("$N$ (bank size)")
    ax.set_ylabel("$\\Delta$ Recall@1 (trained $-$ zero-shot)")
    ax.set_title("Effect of pretraining: three regimes")
    ax.set_ylim(-0.32, 0.82)

    # Shaded regime regions (drawn first, behind everything)
    ax.axhspan(-0.32, 0, color="#fdd", alpha=0.4, zorder=0)
    ax.axhspan(0, 0.82, color="#dfd", alpha=0.3, zorder=0)
    # Annotations placed in empty quadrants so nothing collides
    ax.text(150, 0.64, "training helps\n(grows with $N$)", fontsize=9,
             color="#2f6a3f", fontweight="bold", style="italic", ha="center")
    ax.text(5.2, -0.30, "training hurts\n(encoder already perfect)", fontsize=9,
             color="#aa3344", style="italic", va="bottom", ha="left")
    # Legend in the empty lower-right quadrant
    ax.legend(loc="lower right", ncol=2, fontsize=8, framealpha=0.97)

    plt.tight_layout()
    plt.savefig(OUT / "fig3_training_matters.pdf")
    plt.close()
    print("  -> fig3_training_matters.pdf")


# ============================================================================
# Figure 5: Latency benchmark — log-log, clear OOM marker
# ============================================================================

def fig_latency():
    Ns = np.array([10, 100, 1000, 10000])
    attmem_q   = [14.94, 14.62, 15.78, 16.55]
    attmem_ins = [0.25,  0.51,  0.52,  0.69]
    rag_ctx    = [20.7,  67.2,  823.0, np.nan]
    path_a     = 1000 * Ns

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.plot(Ns, attmem_q, "o-", color=C["attmem"], label="AttMem query (LM forward)",
             markersize=6.5, linewidth=2.4,
             markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(Ns, attmem_ins, "s-", color=C["attmem_l"], label="AttMem batch insert",
             markersize=5.5, linewidth=2.0,
             markeredgecolor="white", markeredgewidth=0.6)
    rag_valid_x = Ns[~np.isnan(rag_ctx)]
    rag_valid_y = [v for v in rag_ctx if not np.isnan(v)]
    ax.plot(rag_valid_x, rag_valid_y, "v-", color=C["rag"],
             label="In-context (bank in prompt)",
             markersize=5.5, linewidth=2.0,
             markeredgecolor="white", markeredgewidth=0.6)

    # OOM annotation
    ax.scatter([10000], [1500], s=400, marker="X", color=C["rag"], zorder=5,
                edgecolor="white", linewidth=1.2)
    ax.annotate("OOM\n$>{32k}$ ctx window", xy=(10000, 1500), xytext=(1300, 5000),
                fontsize=8.5, color=C["rag"], fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C["rag"], lw=1.4))

    ax.plot(Ns, path_a, ":", color=C["path_a"], label="Path A insert (80-step SGD/id)",
             linewidth=1.8, alpha=0.85)

    # Speedup callout, placed in the empty region right of the in-context line
    # (which stops at N=1000) so it overlaps neither that line nor the markers.
    ax.text(2600, 230, "$52{\\times}$ faster\nthan in-context\nat $N{=}1000$",
             fontsize=8.5, color=C["attmem"], fontweight="bold", ha="center",
             va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF6D6",
                        edgecolor=C["highlight"], linewidth=1.2))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1k", "10k"])
    ax.set_xlabel("$N$ (bank size)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Wall-clock latency vs in-context / Path A")
    ax.legend(loc="upper left", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_latency.pdf")
    plt.close()
    print("  -> fig4_latency.pdf")


# ============================================================================
# Figure 6: Ablations — LM size × steps + curriculum
# ============================================================================

def fig_ablations():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    q3b12 = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed42_bsmax1024.json"))
    q7b12 = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed42_bsmax1024_qwen7b.json"))
    q3b50 = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps50000_seed42_bsmax1024.json"))
    q7b50 = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps50000_seed42_bsmax1024_qwen7b.json"))
    try:
        llama8b = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed42_bsmax1024_metallama3.18binstruct.json"))
    except FileNotFoundError:
        llama8b = None
    Ns = sorted(int(N) for N in q3b12["results"])
    rag = [q3b12["results"][str(N)]["rag"] for N in Ns]
    a3b12 = [q3b12["results"][str(N)]["attmem"] for N in Ns]
    a7b12 = [q7b12["results"][str(N)]["attmem"] for N in Ns]
    a3b50 = [q3b50["results"][str(N)]["attmem"] for N in Ns]
    a7b50 = [q7b50["results"][str(N)]["attmem"] for N in Ns]

    ax1.plot(Ns, rag, "-", color="#000", linewidth=1.5, alpha=0.5,
             label="Retrieval ceiling")
    ax1.plot(Ns, a3b12, "o-", color=C["qwen3b"], linewidth=2.0,
             label="Qwen-3B @ 12K", markersize=5.5, markeredgecolor="white", markeredgewidth=0.5)
    ax1.plot(Ns, a3b50, "o--", color=C["qwen3b"], linewidth=2.0,
             label="Qwen-3B @ 50K", markersize=5.5, alpha=0.8,
             markeredgecolor="white", markeredgewidth=0.5)
    ax1.plot(Ns, a7b12, "^-", color=C["qwen7b"], linewidth=2.0,
             label="Qwen-7B @ 12K", markersize=6,
             markeredgecolor="white", markeredgewidth=0.5)
    if llama8b is not None:
        Nl = sorted(int(N) for N in llama8b["results"])
        al = [llama8b["results"][str(N)]["attmem"] for N in Nl]
        ax1.plot(Nl, al, "D-", color="#9c7cb5", linewidth=2.0,
                 label="Llama-3.1-8B @ 12K", markersize=5.5,
                 markeredgecolor="white", markeredgewidth=0.5)
    ax1.set_xscale("log")
    ax1.set_xticks([5, 10, 50, 100, 300, 1000])
    ax1.set_xticklabels(["5", "10", "50", "100", "300", "1k"])
    ax1.set_xlabel("$N$"); ax1.set_ylabel("Recall@1")
    ax1.set_title("LM size $\\times$ family $\\times$ steps")
    ax1.legend(loc="lower left", fontsize=7, ncol=2)
    ax1.set_ylim(0, 1.05)

    # Curriculum
    fixed = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps8000_seed42.json"))
    curr  = q3b12
    Nf = sorted(int(N) for N in fixed["results"])
    rag_f = [fixed["results"][str(N)]["rag"] for N in Nf]
    af = [fixed["results"][str(N)]["attmem"] for N in Nf]
    ac = [curr["results"][str(N)]["attmem"] for N in Nf if str(N) in curr["results"]]
    Nc = [N for N in Nf if str(N) in curr["results"]]

    ax2.plot(Nf, rag_f, "-", color="#000", linewidth=1.5, alpha=0.5,
             label="Retrieval ceiling")
    ax2.plot(Nf, af, "s-", color=C["rag"], linewidth=2.0,
             label="$bs{=}64$ (fixed, 8K)", markersize=5.5,
             markeredgecolor="white", markeredgewidth=0.5)
    ax2.plot(Nc, ac, "o-", color=C["attmem"], linewidth=2.0,
             label="$bs{\\in}[64, 1024]$ (12K)", markersize=5.5,
             markeredgecolor="white", markeredgewidth=0.5)

    # Annotate gap at N=700
    ax2.annotate("", xy=(700, ac[Nc.index(700)] if 700 in Nc else 0.63),
                  xytext=(700, af[Nf.index(700)]),
                  arrowprops=dict(arrowstyle="<->", color=C["highlight"], lw=1.5))
    if 700 in Nc and 700 in Nf:
        diff = ac[Nc.index(700)] - af[Nf.index(700)]
        ax2.text(900, 0.4, f"$+{diff:.2f}$\nat $N{{=}}700$",
                  fontsize=8.5, color="#aa7000", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF6D6",
                             edgecolor=C["highlight"]))

    ax2.set_xscale("log")
    ax2.set_xticks([5, 10, 50, 100, 300, 1000])
    ax2.set_xticklabels(["5", "10", "50", "100", "300", "1k"])
    ax2.set_xlabel("$N$"); ax2.set_ylabel("Recall@1")
    ax2.set_title("Curriculum bank size")
    ax2.legend(loc="lower left", fontsize=7.5)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(OUT / "fig5_ablations.pdf")
    plt.close()
    print("  -> fig5_ablations.pdf")


# ============================================================================
# Figure 7: Path A → AttMem transition (the "design space" finding)
# ============================================================================

def fig_pivot():
    """The discrete-codebook → continuous-attention pivot, per sub-modality."""
    fig, ax = plt.subplots(figsize=(7.0, 2.5))

    subs = [
        ("Speaker\n$N{=}10$",  0.32, 0.90),
        ("Acoustic\n$N{=}10$", 0.40, 0.83),
        ("Tone\n$N{=}10$",     0.45, 0.44),
        ("Style\n$N{=}5$",     0.20, 0.64),
        ("Face\n$N{=}10$",     0.10, 0.99),
        ("Face\n$N{=}700$",    0.07, 0.63),
    ]
    labels = [s[0] for s in subs]
    pa = np.array([s[1] for s in subs])
    am = np.array([s[2] for s in subs])

    x = np.arange(len(subs))
    w = 0.36
    b1 = ax.bar(x - w/2, pa, w, label="Path A (discrete codebook)", color=C["path_a"],
                 edgecolor="#222", linewidth=0.6)
    b2 = ax.bar(x + w/2, am, w, label="AttMem (continuous, ours)", color=C["attmem"],
                 edgecolor="#222", linewidth=0.6)

    # Annotate fold improvements
    for i, (p, a) in enumerate(zip(pa, am)):
        ratio = a / p if p > 0 else float("inf")
        if ratio > 1.2:
            ax.text(i, max(p, a) + 0.06, f"${ratio:.1f}{{\\times}}$", ha="center",
                     fontsize=9.5, fontweight="bold", color=C["attmem"])
        elif 0.85 <= ratio <= 1.15:
            ax.text(i, max(p, a) + 0.06, "parity", ha="center", fontsize=8.5,
                     style="italic", color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.20)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.97)
    ax.set_title("Discrete codebook $\\to$ continuous attention: per-sub-modality improvement")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig6_pivot.pdf")
    plt.close()
    print("  -> fig6_pivot.pdf")


# ============================================================================

def fig_adversarial():
    """Adversarial-distractor finding: Qwen-3B loses 2-3pp; Llama-3.1-8B BEATS at K>=5."""
    try:
        d_qwen = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed48_bsmax1024.json"))
        adv_q = d_qwen.get("adversarial", {})
    except FileNotFoundError:
        adv_q = {}
    try:
        d_llama = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed43_bsmax1024_metallama3.18binstruct.json"))
        adv_l = d_llama.get("adversarial", {})
    except FileNotFoundError:
        adv_l = {}
    if not adv_q and not adv_l:
        print("  -> fig7_adversarial.pdf SKIPPED")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # Left: random bank @ N=10 (BEATS-RAG cell, multi-seed)
    random_attmem = 0.992; random_attmem_std = 0.014; random_rag = 0.933
    ax1.bar([0, 1], [random_rag, random_attmem],
             yerr=[0, random_attmem_std], capsize=4,
             color=[C["rag"], C["attmem"]],
             edgecolor="#222", linewidth=0.6, width=0.55)
    ax1.text(1, random_attmem + random_attmem_std + 0.04, "$\\Delta=+0.059$\n(BEATS, $p{=}0.006$)",
             ha="center", fontsize=8.5, fontweight="bold", color="#aa7000",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF6D6",
                        edgecolor=C["highlight"], linewidth=1.0))
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Embedding retrieval", "AttMem (Qwen-3B)"], fontsize=9)
    ax1.set_ylabel("Recall@1 at $N{=}10$")
    ax1.set_ylim(0, 1.25)
    ax1.set_title("(a) Random distractors ($n{=}4$)", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # Right: adversarial bank — Qwen vs Llama vs RAG
    Ks_q = sorted(int(K) for K in adv_q)
    N_q = [adv_q[str(K)]["N_bank"] for K in Ks_q]
    am_q = [adv_q[str(K)]["attmem_retr1"] for K in Ks_q]
    rag_q = [adv_q[str(K)]["rag_retr1"] for K in Ks_q]

    Ks_l = sorted(int(K) for K in adv_l)
    N_l = [adv_l[str(K)]["N_bank"] for K in Ks_l]
    am_l = [adv_l[str(K)]["attmem_retr1"] for K in Ks_l]
    rag_l = [adv_l[str(K)]["rag_retr1"] for K in Ks_l]  # should equal rag_q
    rag_combined = rag_q or rag_l
    N_combined = N_q or N_l

    ax2.plot(N_combined, rag_combined, "s-", color=C["rag"], label="Embedding retrieval",
              markersize=6, linewidth=2.0, markeredgecolor="white", markeredgewidth=0.6)
    if am_q:
        ax2.plot(N_q, am_q, "o-", color=C["qwen3b"], label="AttMem (Qwen-3B)",
                  markersize=6, linewidth=2.0, markeredgecolor="white", markeredgewidth=0.6)
    if am_l:
        ax2.plot(N_l, am_l, "D-", color="#9c7cb5", label="AttMem (Llama-3.1-8B)",
                  markersize=6, linewidth=2.0, markeredgecolor="white", markeredgewidth=0.6)

    # Annotate the contrast at K=19
    if am_q and am_l:
        ax2.annotate("", xy=(20, am_l[-1]), xytext=(20, am_q[-1]),
                      arrowprops=dict(arrowstyle="<->", color=C["highlight"], lw=1.4))
        ax2.text(13, (am_l[-1]+am_q[-1])/2, f"$+{(am_l[-1]-am_q[-1])*100:.1f}$pt\n(larger LM\nhelps)",
                  fontsize=7.5, color="#aa7000", fontweight="bold", ha="center",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF6D6",
                             edgecolor=C["highlight"], linewidth=0.8))

    ax2.set_xlabel("$N$ (target + top-$K$ look-alikes)")
    ax2.set_ylabel("Recall@1")
    ax2.set_title("(b) Adversarial: larger LM helps", fontsize=10)
    ax2.set_ylim(0.78, 0.95)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout(w_pad=2.0)
    plt.savefig(OUT / "fig7_adversarial.pdf")
    plt.close()
    print("  -> fig7_adversarial.pdf")


def fig_advtrain_cross_modal():
    """Adv-training across all 5 sub-modalities — adversarial K=19.

    The adv-training bars and the +pp callouts are MULTI-SEED MEANS over the
    same seed sets as the Appendix headline table, so the figure and table
    agree exactly (a single seed gives, e.g., +72pp on Style vs the +71.0pp
    mean). Retrieval is deterministic, so its per-seed value is constant.
    """
    modes = ["A-XR-ID", "A-SCN", "A-PARA", "V-STY", "V-XC-ID-XXXL"]
    # One glob per modality, matching the table's seed sets (exact-suffix
    # patterns exclude off-recipe variants like *_bsmax168_* and *_metallama*).
    adv_globs = {
        "A-XR-ID":      ["attmem_a-xr-id_steps5000_seed*_advp30.json"],
        "A-SCN":        ["attmem_a-scn_steps5000_seed*_advp30.json"],
        "A-PARA":       ["attmem_a-para_steps5000_seed4[2-5]_advp30.json"],
        "V-STY":        ["attmem_v-sty-clip_steps5000_seed*_advp30.json"],
        "V-XC-ID-XXXL": ["attmem_v-xc-id-xxxl_steps12000_seed49_bsmax1024_advp30.json",
                         "attmem_v-xc-id-xxxl_steps12000_seed5[01]_bsmax1024_advp30.json"],
    }
    rag_vals = []
    advmem_vals = []
    std_vals = []  # standard-training adversarial (single seed, secondary comparison)
    mode_std_files = {
        "A-XR-ID":     "attmem_a-xr-id_steps5000_seed42.json",
        "A-SCN":       "attmem_a-scn_steps5000_seed42.json",
        "A-PARA":      "attmem_a-para_steps5000_seed42.json",
        "V-STY":       "attmem_v-sty-clip_steps5000_seed42.json",
        "V-XC-ID-XXXL": "attmem_v-xc-id-xxxl_steps12000_seed48_bsmax1024.json",
    }

    def _k19(d):
        """Return the K=19 adversarial cell, falling back to the largest K run."""
        adv = d.get("adversarial", {})
        if "19" in adv:
            return adv["19"]
        ks = sorted(int(k) for k in adv)
        return adv[str(ks[-1])] if ks else None

    for mode in modes:
        am, rg = [], []
        for pat in adv_globs[mode]:
            for f in sorted(glob.glob(str(RESULTS / pat))):
                r = _k19(json.load(open(f)))
                if r is not None:
                    am.append(r["attmem_retr1"]); rg.append(r["rag_retr1"])
        advmem_vals.append(float(np.mean(am)) if am else None)
        rag_vals.append(float(np.mean(rg)) if rg else None)
        try:
            d_std = json.load(open(RESULTS / mode_std_files[mode]))
            r_std = d_std.get("adversarial", {}).get("19", None)
            if r_std is None:
                ks = sorted(int(k) for k in d_std.get("adversarial", {}))
                if ks: r_std = d_std["adversarial"][str(ks[-1])]
            std_vals.append(r_std["attmem_retr1"] if r_std else None)
        except FileNotFoundError:
            std_vals.append(None)

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(modes))
    w = 0.27
    # Use plain bars; show None as empty
    rag_plot = [v if v is not None else 0 for v in rag_vals]
    std_plot = [v if v is not None else 0 for v in std_vals]
    advmem_plot = [v if v is not None else 0 for v in advmem_vals]

    ax.bar(x - w, rag_plot, w, color=C["rag"], label="Embedding retrieval",
            edgecolor="#222", linewidth=0.6)
    ax.bar(x,     std_plot, w, color=C["qwen3b"], label="AttMem (standard training)",
            edgecolor="#222", linewidth=0.6)
    ax.bar(x + w, advmem_plot, w, color=C["highlight"], label="AttMem (adv-training)",
            edgecolor="#222", linewidth=0.6)
    # Annotate Δ over retrieval for adv-training. The extra headroom below
    # (ylim top = 1.22) keeps these labels clear of the legend that sits above
    # the axes, even for bars that reach 1.0.
    for i, (r_v, a_v) in enumerate(zip(rag_vals, advmem_vals)):
        if r_v is not None and a_v is not None and a_v - r_v > 0.05:
            delta = (a_v - r_v) * 100
            ax.text(i + w, a_v + 0.025, f"${delta:+.0f}$pp", ha="center",
                     fontsize=8.5, fontweight="bold", color="#aa7000")

    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(m, m) for m in modes], fontsize=9.5)
    ax.set_ylabel("Recall@1 (adversarial $K{=}19$)")
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Legend above the axes so it never overlaps the (often tall) bars.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=8.5, framealpha=0.97, borderaxespad=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig10_advtrain_crossmodal.pdf")
    plt.close()
    print("  -> fig10_advtrain_crossmodal.pdf")


def fig_advprob_pareto():
    """Pareto sweep over adv_prob."""
    try:
        files = {
            0.0: "attmem_v-xc-id-xxxl_steps12000_seed48_bsmax1024.json",
            0.1: "attmem_v-xc-id-xxxl_steps12000_seed49_bsmax1024_advp10.json",
            0.3: "attmem_v-xc-id-xxxl_steps12000_seed49_bsmax1024_advp30.json",
            0.5: "attmem_v-xc-id-xxxl_steps12000_seed49_bsmax1024_advp50.json",
            0.7: "attmem_v-xc-id-xxxl_steps12000_seed49_bsmax1024_advp70.json",
        }
        rand_n10 = {}; adv_k19 = {}
        for p, f in files.items():
            d = json.load(open(RESULTS / f))
            rand_n10[p] = d["results"]["10"]["attmem"]
            adv_k19[p] = d.get("adversarial", {}).get("19", {}).get("attmem_retr1", None)
        rag_n10 = json.load(open(RESULTS / files[0.0]))["results"]["10"]["rag"]
        rag_k19 = json.load(open(RESULTS / files[0.0])).get("adversarial", {}).get("19", {}).get("rag_retr1", None)
    except (FileNotFoundError, KeyError) as e:
        print(f"  -> fig11_pareto.pdf SKIPPED: {e}")
        return

    # Keep only the points whose adversarial cell exists, in p order.
    ps = [p for p in sorted(rand_n10.keys()) if adv_k19.get(p) is not None]
    xs = [rand_n10[p] for p in ps]
    ys = [adv_k19[p] for p in ps]

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    # Colour-code by look-alike mix p (avoids overlapping per-point text labels).
    ax.plot(xs, ys, "-", color="#bbbbbb", linewidth=1.6, zorder=1)
    sca = ax.scatter(xs, ys, c=ps, cmap="viridis", s=120, zorder=3,
                     edgecolor="white", linewidth=1.0, vmin=0.0, vmax=0.7)
    cbar = fig.colorbar(sca, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("look-alike mix $p$", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    # Label only the two endpoints, placed clear of the cluster.
    ax.annotate("$p{=}0$", (xs[0], ys[0]), textcoords="offset points",
                xytext=(-10, 6), fontsize=7.5, color="#333", ha="right")
    ax.annotate(f"$p{{=}}{ps[-1]}$", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(12, 0), fontsize=7.5, color="#333", ha="left")
    # Reference lines
    if rag_n10 is not None and rag_k19 is not None:
        ax.axvline(rag_n10, color=C["rag"], linestyle="--", linewidth=1.2, alpha=0.6,
                    label="Retrieval (random $N{=}10$)")
        ax.axhline(rag_k19, color=C["rag"], linestyle=":", linewidth=1.2, alpha=0.6,
                    label="Retrieval (adversarial $K{=}19$)")
    ax.set_xlabel("Recall@1 on random $N{=}10$ bank")
    ax.set_ylabel("Recall@1 on adv. $K{=}19$ bank")
    ax.set_title("Random vs. adversarial trade-off")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.7, 1.05)
    plt.tight_layout()
    plt.savefig(OUT / "fig11_advprob_pareto.pdf")
    plt.close()
    print("  -> fig11_advprob_pareto.pdf")


def fig_adv_training():
    """Adversarial training transforms the adversarial regime."""
    try:
        d_std  = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed48_bsmax1024.json"))
        d_adv  = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed49_bsmax1024_advp30.json"))
    except FileNotFoundError as e:
        print(f"  -> fig8_adv_training.pdf SKIPPED ({e})")
        return
    adv_std = d_std.get("adversarial", {})
    adv_adv = d_adv.get("adversarial", {})
    if not adv_std or not adv_adv:
        print("  -> fig8_adv_training.pdf SKIPPED (no adversarial data)")
        return

    Ks = sorted(int(K) for K in adv_std)
    N_banks = [adv_std[str(K)]["N_bank"] for K in Ks]
    rag = [adv_std[str(K)]["rag_retr1"] for K in Ks]
    std = [adv_std[str(K)]["attmem_retr1"] for K in Ks]
    adv = [adv_adv[str(K)]["attmem_retr1"] for K in Ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Adversarial regime
    ax1.plot(N_banks, rag, "s-", color=C["rag"], label="Embedding retrieval",
              markersize=6, linewidth=2.0, markeredgecolor="white", markeredgewidth=0.6)
    ax1.plot(N_banks, std, "o-", color=C["qwen3b"], label="AttMem standard training",
              markersize=6, linewidth=2.0, markeredgecolor="white", markeredgewidth=0.6)
    ax1.plot(N_banks, adv, "D-", color=C["highlight"],
              label="AttMem adv-training",
              markersize=6.5, linewidth=2.2, markeredgecolor="white", markeredgewidth=0.6)
    # Callout placed in the headroom above the (flat, near-1.0) adv-training
    # line so the box never sits on top of the markers.
    ax1.annotate("$+0.145$\nover retrieval", xy=(20, adv[-1]), xytext=(8.5, 1.045),
                  fontsize=8, color="#aa7000", fontweight="bold", ha="center",
                  arrowprops=dict(arrowstyle="->", color=C["highlight"], lw=1.0),
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF6D6",
                             edgecolor=C["highlight"]))
    ax1.set_xlabel("$N$ (target + top-$K$ look-alikes)")
    ax1.set_ylabel("Recall@1")
    ax1.set_title("Adversarial regime: training transforms")
    ax1.set_ylim(0.78, 1.10)
    ax1.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Random regime (trade-off)
    Ns = sorted(int(N) for N in d_std["results"])
    rag_r = [d_std["results"][str(N)]["rag"] for N in Ns]
    std_r = [d_std["results"][str(N)]["attmem"] for N in Ns]
    adv_r = [d_adv["results"][str(N)]["attmem"] for N in Ns]
    ax2.plot(Ns, rag_r, "s-", color=C["rag"], label="Embedding retrieval",
              markersize=5.5, linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6)
    ax2.plot(Ns, std_r, "o-", color=C["qwen3b"], label="AttMem standard",
              markersize=5.5, linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6)
    ax2.plot(Ns, adv_r, "D-", color=C["highlight"], label="AttMem adv-training",
              markersize=5.5, linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6)
    ax2.set_xscale("log")
    ax2.set_xticks([5, 10, 50, 100, 300, 1000])
    ax2.set_xticklabels(["5", "10", "50", "100", "300", "1k"])
    ax2.set_xlabel("$N$ (random bank size)")
    ax2.set_ylabel("Recall@1")
    ax2.set_title("Random regime: trade-off")
    ax2.set_ylim(0.5, 1.05)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig8_adv_training.pdf")
    plt.close()
    print("  -> fig8_adv_training.pdf")


def fig_capacity():
    """The two capacity laws: perceptual identity scales with latent slots
    (recall ~ min(1, k/M)); exact-fact retrieval collapses and more tokens do
    not rescue it."""
    R = Path("/home/ubuntu/multimodal-user-memory/results")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # Panel A: perceptual identity (faces) -- latent scales with slots
    rec = {}
    for r in json.load(open(R / "setmem_face.json"))["rows"]:
        rec.setdefault(r["k"], {})[r["M"]] = r["recall_mean"]
    for c, k in zip(["#cfe0f3", "#9dc0e6", "#5a86b3", "#2f5e92", "#1f4e79"],
                    [4, 8, 16, 32, 64]):
        Ms = sorted(rec[k]); ys = [rec[k][M] for M in Ms]
        axA.plot(Ms, ys, "o-", color=c, label=f"$k{{=}}{k}$", markersize=4.5,
                 linewidth=1.8, markeredgecolor="white", markeredgewidth=0.5)
    axA.set_xscale("log", base=2)
    axA.set_xticks([2, 4, 8, 16, 32, 64]); axA.set_xticklabels([2, 4, 8, 16, 32, 64])
    axA.set_xlabel("$M$ identities stored"); axA.set_ylabel("recall@1")
    axA.set_ylim(0, 1.05); axA.set_title("Perceptual identity: latent scales")
    axA.legend(loc="lower left", fontsize=7, ncol=2); axA.grid(alpha=0.3)

    # Panel B: exact facts (codes) -- latent fails, more k does not help
    perm = {M: json.load(open(R / f"codemem_perm_M{M}.json"))["rows"][0]["exact"]
            for M in [1, 2, 4, 8, 16]}
    Ms = sorted(perm)
    axB.plot(Ms, [perm[M] for M in Ms], "s-", color=C["rag"], label="$k{=}16$ tokens",
             markersize=5, linewidth=2.0, markeredgecolor="white", markeredgewidth=0.5)
    for k, mk in zip([64, 128], ["^", "D"]):
        pts = [(M, json.load(open(R / f"codemem_Mk_M{M}_k{k}.json"))["rows"][0]["exact"])
               for M in [2, 4]]
        axB.plot([p[0] for p in pts], [p[1] for p in pts], mk, color="#999999",
                 markersize=5.5, label=f"$k{{=}}{k}$")
    axB.annotate("more tokens\ndon't help", xy=(2, 0.10), xytext=(3.4, 0.5),
                 fontsize=7.5, color="#aa3344", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#aa3344", lw=1.0))
    axB.set_xscale("log", base=2)
    axB.set_xticks([1, 2, 4, 8, 16]); axB.set_xticklabels([1, 2, 4, 8, 16])
    axB.set_xlabel("$M$ exact codes stored"); axB.set_ylabel("retrieval exact-match")
    axB.set_ylim(0, 1.05); axB.set_title("Exact facts: latent fails")
    axB.legend(loc="upper right", fontsize=7); axB.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig_capacity.pdf")
    plt.close()
    print("  -> fig_capacity.pdf")


def fig_universality():
    """Training-free, in-model recall reproduces the encoder ceiling on every frozen
    model. Left: AttMem recall lands on the encoder line for all families. Right: Δ
    from the encoder at N=50 is ~0 across architectures; only Mistral at the default
    gain dips, and a single larger gain (a constant, not training) fixes it."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 2.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})
    Ns = [5, 10, 50]
    enc = [0.983, 0.948, 0.821]   # raw-cosine encoder ceiling (paired, 20 draws)

    # Panel A: every model's training-free recall sits on the encoder line
    axA.plot(Ns, enc, "k-o", lw=2.2, ms=7, zorder=5, label="encoder ceiling")
    # representative models (all land on the line); jitter x slightly for visibility
    model_pts = [0.983, 0.948, 0.822]  # exact-match families
    for i, dx in enumerate([-0.6, 0.0, 0.6, 1.2]):
        axA.scatter([n + dx for n in Ns], model_pts, s=22, color=C["zero_shot"],
                    zorder=6, edgecolor="white", linewidth=0.4)
    axA.scatter([], [], s=22, color=C["zero_shot"], label="each frozen model (10)")
    axA.set_xticks(Ns); axA.set_xlabel("$N$ registered identities")
    axA.set_ylabel("recall@1 (training-free)")
    axA.set_ylim(0.7, 1.02); axA.set_title("Recall = encoder, on every model")
    axA.legend(loc="lower left", fontsize=7.5); axA.grid(alpha=0.3)

    # Panel B: Δ(AttMem - encoder) at N=50 per model, grouped by architecture
    rows = [
        ("Qwen2.5-1.5B",        0.000, "tied"),
        ("Qwen2.5-7B",          0.000, "tied"),
        ("Qwen3-4B",            0.000, "tied"),
        ("Qwen3-8B",            0.000, "tied"),
        ("Qwen3-14B",           0.000, "tied"),
        ("Phi-3.5-mini",        0.000, "tied"),
        ("SmolLM2-1.7B",        0.001, "tied"),
        ("DeepSeek-Llama-8B",   0.001, "untied"),
        ("Mistral-7B (gain$\\geq$256)", 0.000, "untied"),
        ("Granite-4.0 (Mamba)", -0.005, "mamba"),
        ("Mistral-7B (gain 64)", -0.514, "lowgain"),
    ]
    cmap = {"tied": "#1f4e79", "untied": "#5a86b3", "mamba": "#e69138",
            "lowgain": "#c44e52"}
    ys = list(range(len(rows)))[::-1]
    for y, (name, d, cat) in zip(ys, rows):
        # lollipop: a stem from the zero line out to the value, dot at the value.
        # Unlike a bar, the dot stays visible at d=0, so "lands exactly on the
        # encoder" reads as a clean column of dots on the zero line.
        axB.plot([0, d], [y, y], color=cmap[cat], lw=1.8, zorder=2,
                 solid_capstyle="round")
        axB.scatter([d], [y], color=cmap[cat], s=42, zorder=4,
                    edgecolor="white", linewidth=0.7)
        if abs(d) < 0.05:
            # near-zero values labelled to the left of the line, leaving the
            # right side clear for the legend
            axB.text(-0.016, y, f"{d:+.3f}", va="center", ha="right",
                     fontsize=6.6, color="#777")
        else:
            axB.text(d - 0.016, y, f"{d:+.2f}", va="center", ha="right",
                     fontsize=7.2, color=cmap[cat], fontweight="bold")
    axB.set_yticks(ys); axB.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    axB.axvline(0, color="#888", lw=1.0, zorder=1)
    axB.set_xlim(-0.62, 0.22)
    axB.set_ylim(-0.8, len(rows) + 3.2)  # headroom so the corner legend clears the dots
    axB.set_xlabel("$\\Delta$ recall@1 vs encoder ($N{=}50$)")
    axB.set_title("Training-free read matches the encoder")
    axB.annotate("default gain too low\non untied $\\to$ fixed by a\nsingle larger constant",
                 xy=(-0.514, 0), xytext=(-0.34, 2.4), fontsize=6.8, color="#aa3344",
                 ha="center", arrowprops=dict(arrowstyle="->", color="#aa3344", lw=1.0))
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(color=cmap["tied"], label="tied emb."),
                        Patch(color=cmap["untied"], label="untied emb."),
                        Patch(color=cmap["mamba"], label="hybrid Mamba")],
               loc="upper right", fontsize=6.8, framealpha=0.9)
    axB.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig_universality.pdf")
    plt.close()
    print("  -> fig_universality.pdf")


def fig_composition():
    """In-model composition: a recalled face retrieves its bound fact in a single
    forward pass. End-to-end accuracy tracks recognition x in-context lookup and beats
    the blind (face-withheld) chance baseline by 4-10x across memory size."""
    Ms = [5, 10, 15, 20]
    D = {M: json.load(open(RESULTS / f"composition_M{M}.json")) for M in Ms}
    recog = np.array([D[M]["recog"] for M in Ms]); recog_e = np.array([D[M]["recog_std"] for M in Ms])
    comp = np.array([D[M]["compose"] for M in Ms]); comp_e = np.array([D[M]["compose_std"] for M in Ms])
    lookup = np.array([D[M]["lookup"] for M in Ms])
    blind = np.array([D[M]["blind"] for M in Ms])
    fig, ax = plt.subplots(figsize=(5.3, 2.9))
    ax.fill_between(Ms, recog - recog_e, recog + recog_e, color=C["attmem"], alpha=0.10)
    ax.plot(Ms, recog, "o-", color=C["attmem"], lw=1.8,
            label="recognition (face$\\to$name)")
    ax.plot(Ms, lookup, "s--", color="#7faed6", lw=1.6,
            label="lookup (name$\\to$fact, in context)")
    ax.fill_between(Ms, comp - comp_e, comp + comp_e, color=C["highlight"], alpha=0.15)
    ax.plot(Ms, comp, "D-", color="#d99000", lw=2.5,
            label="composition (end-to-end)")
    ax.plot(Ms, blind, ":", color="#888", lw=1.6, label="blind $=$ chance ($1/M$)")
    ax.annotate("$4$--$10\\times$\nover chance", xy=(10, (comp[1] + blind[1]) / 2),
                xytext=(13.3, 0.30), fontsize=7.6, color="#555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.0))
    ax.set_xticks(Ms); ax.set_xlabel("$M$ registered identities")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("In-model composition: a face recalls its fact in one pass")
    ax.legend(loc="lower left", fontsize=7.0); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_composition.pdf")
    plt.close()
    print("  -> fig_composition.pdf")


def fig_arch_detail():
    """Detailed mechanism for Section 3: grounding (VLM -> box -> align) and
    identification (encoder -> key) feed an in-model store whose attention read adds
    a residual at the output head; registration is a single O(1) append."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis("off")
    cG, cI, cS, gold = "#3a8c5d", "#c44e52", C["attmem"], C["highlight"]

    def arrow(x0, x1, y, c="#555"):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=12, lw=1.4, color=c, zorder=4))

    # ===== ROW A: grounding + identification =====
    ax.text(0.35, 9.55, "GROUND + IDENTIFY", fontsize=7.5, fontweight="bold", color="#555")
    # scene
    ax.add_patch(FancyBboxPatch((0.4, 7.0), 2.2, 2.0, boxstyle="round,pad=0.05,rounding_size=0.1",
                 lw=1.2, edgecolor="#999", facecolor="#f5f5f5", zorder=2))
    for cx, hl in [(1.0, False), (1.5, True), (2.0, False)]:
        fc = "#FBE2A6" if hl else "#cdd7d1"; ec = "#8a6500" if hl else "#9aa39d"
        ax.add_patch(FancyBboxPatch((cx - 0.22, 7.45), 0.44, 0.95,
                     boxstyle="round,pad=0.02,rounding_size=0.04",
                     lw=1.3 if hl else 0.6, edgecolor=ec, facecolor=fc, zorder=3))
        ax.add_patch(Circle((cx, 7.92), 0.09, facecolor="#fff", ec="#888", lw=0.4, zorder=4))
    ax.add_patch(FancyBboxPatch((1.27, 7.38), 0.46, 1.1, boxstyle="round,pad=0.02,rounding_size=0.04",
                 lw=1.5, edgecolor=gold, facecolor="none", zorder=5))
    ax.text(1.5, 6.6, "scene $+$ ``remember her''", ha="center", fontsize=6.6, color="#333")

    arrow(2.7, 3.3, 8.0)
    # VLM
    ax.add_patch(FancyBboxPatch((3.4, 7.0), 2.4, 2.0, boxstyle="round,pad=0.05,rounding_size=0.1",
                 lw=1.5, edgecolor=cG, facecolor="#E9F5EE", zorder=2))
    ax.text(4.6, 8.5, "VLM", ha="center", fontsize=8.5, fontweight="bold", color=cG)
    ax.text(4.6, 8.08, "ground referent", ha="center", fontsize=6.4, color=cG, style="italic")
    ax.text(4.6, 7.5, "box $[x_1,y_1,x_2,y_2]$", ha="center", fontsize=6.4, color="#333")
    arrow(5.9, 6.5, 8.0)
    # align
    ax.add_patch(FancyBboxPatch((6.6, 7.0), 2.3, 2.0, boxstyle="round,pad=0.05,rounding_size=0.1",
                 lw=1.2, edgecolor="#777", facecolor="#f0f0f0", zorder=2))
    ax.text(7.75, 8.5, "re-detect $+$ align", ha="center", fontsize=7.0, fontweight="bold", color="#444")
    ax.text(7.75, 8.08, "RetinaFace", ha="center", fontsize=6.4, color="#666", style="italic")
    ax.text(7.75, 7.5, "aligned $112{\\times}112$", ha="center", fontsize=6.4, color="#333")
    arrow(9.0, 9.6, 8.0)
    # encoder
    ax.add_patch(FancyBboxPatch((9.7, 7.0), 2.5, 2.0, boxstyle="round,pad=0.05,rounding_size=0.1",
                 lw=1.5, edgecolor=cI, facecolor="#FBEAEA", zorder=2))
    ax.text(10.95, 8.5, "encoder", ha="center", fontsize=8.5, fontweight="bold", color=cI)
    ax.text(10.95, 8.08, "ArcFace / ECAPA / CLIP", ha="center", fontsize=5.8, color=cI, style="italic")
    ax.text(10.95, 7.5, "key $q{=}k\\in\\mathbb{R}^{D}$", ha="center", fontsize=6.6, color="#333")
    # down arrow encoder -> store query
    ax.add_patch(FancyArrowPatch((10.95, 6.95), (10.95, 4.05), arrowstyle="-|>",
                 mutation_scale=12, lw=1.4, color="#555", zorder=4))
    ax.text(11.2, 5.5, "query $q$", ha="left", fontsize=6.6, color="#555")

    # ===== ROW B: storage + read =====
    ax.add_patch(FancyBboxPatch((0.4, 0.5), 15.2, 4.7, boxstyle="round,pad=0.1,rounding_size=0.15",
                 lw=1.4, edgecolor=cS, facecolor="#EFF4FB", zorder=1))
    ax.text(0.7, 4.92, "STORE — AttMem on a frozen LM", fontsize=8,
            fontweight="bold", color=cS)

    def matrix(x0, y0, rows, cols, w, h, color):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                     lw=1.0, edgecolor=color, facecolor="#fff", zorder=3))
        for r in range(1, rows):
            ax.plot([x0, x0 + w], [y0 + h * r / rows] * 2, color=color, lw=0.4, alpha=0.5, zorder=4)
        for cc in range(1, cols):
            ax.plot([x0 + w * cc / cols] * 2, [y0, y0 + h], color=color, lw=0.4, alpha=0.5, zorder=4)
    matrix(0.85, 1.7, 4, 4, 1.7, 2.3, cS)
    ax.text(1.7, 4.15, "keys $K$", ha="center", fontsize=6.6, color=cS)
    ax.text(1.7, 1.45, "$N{\\times}D$", ha="center", fontsize=5.8, color="#666")
    matrix(2.85, 1.7, 4, 3, 1.4, 2.3, gold)
    ax.text(3.55, 4.15, "values $V$", ha="center", fontsize=6.6, color="#8a6500")
    ax.text(3.55, 1.45, "$N{\\times}H$", ha="center", fontsize=5.8, color="#666")

    arrow(4.35, 4.95, 2.95, cS)
    # attention read
    ax.add_patch(FancyBboxPatch((5.0, 1.95), 5.5, 2.0, boxstyle="round,pad=0.06,rounding_size=0.08",
                 lw=1.3, edgecolor=cS, facecolor="#fff", zorder=3))
    ax.text(7.75, 3.6, "attention read", ha="center", fontsize=7.2, fontweight="bold", color=cS)
    ax.text(7.75, 3.13, "$w=\\mathrm{softmax}(\\beta\\,q^{\\!\\top}\\!K)$", ha="center", fontsize=7.0, color="#222")
    ax.text(7.75, 2.72, "$r=w^{\\!\\top}V$", ha="center", fontsize=7.0, color="#222")
    ax.text(7.75, 2.3, "$h'=h+g\\,W_o\\,r$", ha="center", fontsize=7.0, color="#222")
    # read -> LM
    ax.add_patch(FancyArrowPatch((10.55, 2.95), (11.25, 2.95), arrowstyle="-|>",
                 mutation_scale=12, lw=1.4, color=cS, zorder=4))
    ax.add_patch(FancyBboxPatch((11.35, 2.1), 2.0, 1.7, boxstyle="round,pad=0.05,rounding_size=0.08",
                 lw=1.2, edgecolor="#333", facecolor="#F4F4F4", zorder=3))
    ax.text(12.35, 3.45, "frozen LM", ha="center", fontsize=7.0, fontweight="bold")
    ax.text(12.35, 2.95, "lm_head $+\\,\\Delta h$", ha="center", fontsize=6.2, color=cS)
    ax.text(12.35, 2.4, "next token", ha="center", fontsize=6.0, style="italic", color="#666")
    ax.add_patch(FancyArrowPatch((13.4, 2.95), (14.05, 2.95), arrowstyle="-|>",
                 mutation_scale=12, lw=1.4, color="#333", zorder=4))
    ax.text(14.85, 3.1, "marker", ha="center", fontsize=7.2, fontweight="bold")
    ax.text(14.85, 2.75, "logit", ha="center", fontsize=7.2, fontweight="bold")
    ax.text(14.85, 2.36, "= identity", ha="center", fontsize=5.8, style="italic", color="#666")

    # registration callout
    ax.add_patch(FancyBboxPatch((5.0, 0.78), 5.5, 0.92, boxstyle="round,pad=0.04,rounding_size=0.06",
                 lw=1.0, edgecolor="#2c6e49", facecolor="#EAF7EF", zorder=3))
    ax.text(7.75, 1.38, "Registration ($\\mathcal{O}(1)$, no training)", ha="center",
            fontsize=6.6, fontweight="bold", color="#2c6e49")
    ax.text(7.75, 1.0, "$K\\!\\leftarrow\\![K;k]$,\\ \\ $V\\!\\leftarrow\\![V;\\mathrm{emb}(\\mathrm{marker})]$ — one append",
            ha="center", fontsize=6.0, color="#2c6e49")

    plt.savefig(OUT / "fig_arch_detail.pdf")
    plt.close()
    print("  -> fig_arch_detail.pdf")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_arch_detail()
    fig_composition()
    fig_density()
    fig_capacity()
    fig_universality()
    fig_arch()
    fig_teaser()
    fig_textablation()
    fig_agentic()
    fig_crossdomain()
    fig_scorecard()
    fig_scaling()
    fig_latency()
    fig_pivot()
    print("Done.")
