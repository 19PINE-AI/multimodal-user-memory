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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
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

RESULTS = Path("/home/ubuntu/multimodal-user-memory/results")
OUT = Path("/home/ubuntu/multimodal-user-memory/paper/figs")
OUT.mkdir(exist_ok=True)


# ============================================================================
# Figure 0: Architecture diagram (clean, NeurIPS-quality)
# ============================================================================

def fig_arch():
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.6)
    ax.axis("off")

    # Left side: Encoder + bank (the parametric memory)
    bank_box = FancyBboxPatch((0.3, 1.2), 4.2, 4.0,
                               boxstyle="round,pad=0.1,rounding_size=0.2",
                               linewidth=1.8, edgecolor=C["attmem"],
                               facecolor="#E8F0FA", zorder=2)
    ax.add_patch(bank_box)
    ax.text(2.4, 4.85, "AttentionMemory", ha="center", fontsize=11, fontweight="bold",
            color=C["attmem"])
    ax.text(2.4, 4.45, "per-modality bank", ha="center", fontsize=8, color=C["attmem"],
            style="italic")
    # Bank rows visualization
    for i, y in enumerate([3.85, 3.45, 3.05, 2.65]):
        ax.plot([0.7, 2.0], [y, y], "-", color=C["attmem_l"], linewidth=2.5, alpha=0.7)
        ax.plot([2.2, 4.0], [y, y], "-", color=C["highlight"], linewidth=2.5, alpha=0.7)
    ax.text(1.35, 2.25, "key (D)", ha="center", fontsize=8, style="italic",
            color=C["attmem"])
    ax.text(3.1, 2.25, "value (H)", ha="center", fontsize=8, style="italic",
            color="#8a6500")
    ax.text(2.4, 1.75, "$\\mathcal{O}(1)$ append", ha="center", fontsize=7.5,
            color="#444444")
    ax.text(2.4, 1.45, "$\\mathcal{O}(N{\\cdot}D)$ query", ha="center", fontsize=7.5,
            color="#444444")

    # Connector arrow
    arr1 = FancyArrowPatch((4.55, 3.4), (5.85, 3.4),
                            arrowstyle="-|>", mutation_scale=15,
                            linewidth=1.6, color=C["attmem"], zorder=3)
    ax.add_patch(arr1)
    ax.text(5.2, 3.7, "residual $\\Delta h$", ha="center", fontsize=8,
            color=C["attmem"], fontweight="bold")

    # Middle: Frozen LM
    lm_box = FancyBboxPatch((5.95, 1.2), 4.8, 4.0,
                              boxstyle="round,pad=0.1,rounding_size=0.2",
                              linewidth=1.8, edgecolor="#222222",
                              facecolor="#F3F3F3", zorder=2)
    ax.add_patch(lm_box)
    ax.text(8.35, 4.85, "Frozen Qwen2.5-3B", ha="center", fontsize=11, fontweight="bold")
    ax.text(8.35, 4.45, "36 layers · hidden 2048 · vocab 151k", ha="center",
            fontsize=8, style="italic", color="#555555")
    # Internal layers - more spacing
    for y in [3.95, 3.55, 3.15]:
        ax.add_patch(FancyBboxPatch((6.3, y - 0.13), 4.1, 0.26,
                                     boxstyle="round,pad=0,rounding_size=0.05",
                                     linewidth=0.6, edgecolor="#888888",
                                     facecolor="#FFFFFF", zorder=3))
    ax.text(8.35, 3.95, "layer $L_{i}$", ha="center", fontsize=7.5, va="center")
    ax.text(8.35, 3.55, "$\\cdots$", ha="center", fontsize=10, va="center")
    ax.text(8.35, 3.15, "layer $L_{36}$ + norm", ha="center", fontsize=7.5, va="center")
    # Hook (taller, fits both lines)
    ax.add_patch(FancyBboxPatch((6.3, 2.30), 4.1, 0.55,
                                  boxstyle="round,pad=0,rounding_size=0.05",
                                  linewidth=1.4, edgecolor=C["attmem"],
                                  facecolor="#FFF6D6", zorder=3))
    ax.text(8.35, 2.68, "pre-hook on lm\\_head", ha="center", fontsize=7,
            color=C["attmem"], fontweight="bold")
    ax.text(8.35, 2.43, "$h \\leftarrow h + g \\cdot W_o(\\mathrm{softmax}(\\beta qK^{\\!\\top}) V)$",
            ha="center", fontsize=6.5, color=C["attmem"])
    # lm_head
    ax.add_patch(FancyBboxPatch((6.3, 1.78), 4.1, 0.28,
                                  boxstyle="round,pad=0,rounding_size=0.05",
                                  linewidth=0.6, edgecolor="#888888",
                                  facecolor="#FFFFFF", zorder=3))
    ax.text(8.35, 1.92, "lm\\_head", ha="center", fontsize=7.5, va="center",
            family="serif")
    # Output of LM
    arr_out = FancyArrowPatch((10.85, 1.92), (11.95, 1.92),
                               arrowstyle="-|>", mutation_scale=15,
                               linewidth=1.6, color="#222222", zorder=3)
    ax.add_patch(arr_out)

    # Right: logits output label (arrow drawn above with arr_out)
    ax.text(12.85, 2.45, "marker", ha="center", fontsize=10, fontweight="bold")
    ax.text(12.85, 2.10, "logits", ha="center", fontsize=10, fontweight="bold")
    ax.text(12.85, 1.65, "(biased by", ha="center", fontsize=8, style="italic",
            color="#666")
    ax.text(12.85, 1.40, "matching ID)", ha="center", fontsize=8, style="italic",
            color="#666")

    # Input lane at bottom
    ax.text(2.4, 0.7, "encoder(percept)$\\rightarrow$ key $k$",
            ha="center", fontsize=8.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FAFAFA",
                       edgecolor="#cccccc"))
    ax.text(8.35, 0.7, "text + percept positions $\\rightarrow$ inputs\\_embeds",
            ha="center", fontsize=8.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FAFAFA",
                       edgecolor="#cccccc"))
    # vertical arrows
    ax.annotate("", xy=(2.4, 1.15), xytext=(2.4, 0.95),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.0))
    ax.annotate("", xy=(8.35, 1.15), xytext=(8.35, 0.95),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.0))

    # Title
    ax.text(7.0, 6.2, "Bolt-on continuous attention memory for cross-condition perceptual recall",
            ha="center", fontsize=10.5, fontweight="bold", color="#222222")

    plt.savefig(OUT / "fig0_arch.pdf")
    plt.close()
    print("  -> fig0_arch.pdf")


# ============================================================================
# Figure 1 (TEASER): the headline result — AttMem BEATS RAG cosine at scale
# ============================================================================

def fig_teaser():
    """Page-1 teaser: V-XC-ID-XXXL N=10 + V-STY N=5 BEATS-RAG cells with error bars."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.5),
                                     gridspec_kw={"width_ratios": [1.05, 1.0]})

    # Left: V-XC-ID-XXXL (4 seeds) at N=10 — bar with errorbar
    methods = ["Path A\n(discrete\ncodebook)", "RAG\ncosine-NN", "AttMem\n(ours, $n{=}4$)"]
    vals    = [0.10, 0.933, 0.992]
    errs    = [0,    0,     0.014]
    colors  = [C["path_a"], C["rag"], C["attmem"]]
    bars = axL.bar(range(3), vals, yerr=errs, capsize=4, color=colors,
                    edgecolor="#222", linewidth=0.8, width=0.65,
                    error_kw=dict(ecolor="#222", lw=1.2))
    # Annotate the "BEATS" gap
    axL.annotate("", xy=(2, 0.992), xytext=(1, 0.933),
                  arrowprops=dict(arrowstyle="->", color=C["highlight"], lw=2.0,
                                  connectionstyle="arc3,rad=-0.3"))
    axL.text(1.5, 1.07, "BEATS\n$p{=}0.006$", ha="center", fontsize=8.5, color="#aa7000",
              fontweight="bold")
    for i, (v, e) in enumerate(zip(vals, errs)):
        axL.text(i, v + (e if e > 0 else 0) + 0.04, f"{v:.2f}", ha="center",
                  fontsize=9, fontweight="bold", color=colors[i])
    axL.set_xticks(range(3))
    axL.set_xticklabels(methods, fontsize=8.5)
    axL.set_ylabel("retr@1 at N=10")
    axL.set_ylim(0, 1.25)
    axL.set_title("V-XC-ID on 2180-ID face pool")
    axL.grid(axis="y", alpha=0.3)

    # Right: V-STY-CLIP (5 seeds) at N=5 — bar with errorbar; shows 1.6× ratio
    vals    = [0.20, 0.40, 0.640]
    errs    = [0,    0,    0.116]
    colors  = [C["path_a"], C["rag"], C["attmem"]]
    bars = axR.bar(range(3), vals, yerr=errs, capsize=4, color=colors,
                    edgecolor="#222", linewidth=0.8, width=0.65,
                    error_kw=dict(ecolor="#222", lw=1.2))
    axR.annotate("", xy=(2, 0.640), xytext=(1, 0.40),
                  arrowprops=dict(arrowstyle="->", color=C["highlight"], lw=2.0,
                                  connectionstyle="arc3,rad=-0.3"))
    axR.text(1.5, 0.84, "$1.6{\\times}$ over RAG\n$p{=}0.015$", ha="center", fontsize=8.5,
              color="#aa7000", fontweight="bold")
    for i, (v, e) in enumerate(zip(vals, errs)):
        axR.text(i, v + (e if e > 0 else 0) + 0.04, f"{v:.2f}", ha="center",
                  fontsize=9, fontweight="bold", color=colors[i])
    axR.set_xticks(range(3))
    axR.set_xticklabels(["Path A\n(discrete\ncodebook)", "RAG\ncosine-NN",
                          "AttMem\n(ours, $n{=}5$)"], fontsize=8.5)
    axR.set_ylabel("retr@1 at N=5")
    axR.set_ylim(0, 1.0)
    axR.set_title("V-STY (painter style, CLIP-mid encoder)")
    axR.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig_teaser.pdf")
    plt.close()
    print("  -> fig_teaser.pdf")


# ============================================================================
# Figure 2: PerceptMem scorecard (all 5 sub-modalities at N=10)
# ============================================================================

def fig_scorecard():
    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    # Sub-modalities
    subs = [
        ("A-XR-ID",     "speaker identity", 1.000, 0.900, 0.32),
        ("A-SCN",       "acoustic scene",   0.933, 0.833, 0.40),
        ("A-PARA",      "paralinguistic",   0.467, 0.440, 0.45),
        ("V-STY",       "painter style",    0.400, 0.460, 0.20),
        ("V-XC-ID",     "face (cross-cond)", 0.933, 0.992, 0.10),
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
    b2 = ax.bar(x,     rag,    w, label="RAG cosine-NN (encoder ceiling)",
                 color=C["rag"],    edgecolor="#222", linewidth=0.6)
    b3 = ax.bar(x + w, attmem, w, label="AttMem (ours)",
                 color=C["attmem"], edgecolor="#222", linewidth=0.6)

    # Annotate BEATS cells: small "+Δ" label above bar showing the AttMem-vs-RAG gap
    for i, (l, _, r, a, p) in enumerate(subs):
        if a > r:
            delta = a - r
            ax.text(i + w, a + 0.045, f"BEATS\n${{+}}{delta*100:.1f}$pt",
                     ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold", color="#aa7000",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFF6D6",
                                edgecolor=C["highlight"], linewidth=1.0))

    # Sublabels in italic via two-line approach (no math)
    ax.set_xticks(x)
    # Apply italics to the sub-label only by direct text customisation
    for i, (l, s) in enumerate(zip(labels, sublabels)):
        ax.text(i, -0.10, l, ha="center", va="top", fontsize=9.5, fontweight="bold",
                 transform=ax.get_xaxis_transform())
        ax.text(i, -0.22, s, ha="center", va="top", fontsize=8.5, style="italic",
                 color="#555", transform=ax.get_xaxis_transform())
    ax.set_xticklabels([""] * len(labels))
    ax.set_ylabel("retr@1 at $N{=}10$")
    ax.set_ylim(0, 1.4)  # extra room for BEATS labels
    # Place legend at the top, above plot area
    ax.legend(loc="lower center", ncol=3, framealpha=0.97, fontsize=8.5,
              bbox_to_anchor=(0.5, 1.04))
    ax.set_title("PerceptMem v0.2 scorecard at $N{=}10$ (5 perceptual sub-modalities)",
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

    Ns_t = sorted(means.keys())
    m = np.array([means[N] for N in Ns_t])
    s = np.array([stds[N] for N in Ns_t])

    # Shaded band for AttMem (multi-seed SEM)
    ax.fill_between(Ns_t, m - s, m + s, color=C["attmem"], alpha=0.18, zorder=2)
    ax.plot(Ns_t, m, "o-", color=C["attmem"], label="AttMem (trained, mean)",
             markersize=6, linewidth=2.2, zorder=3,
             markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(Ns_t, [rag_at[N] for N in Ns_t], "s-", color=C["rag"],
             label="RAG cosine-NN ceiling", markersize=5.5, linewidth=2.0, zorder=3,
             markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(Ns_zs, zs_mem, "^--", color=C["zero_shot"],
             label="AttMem (zero-shot)", markersize=5, linewidth=1.4,
             markeredgecolor="white", markeredgewidth=0.4)
    ax.plot(Ns_t, [pa.get(N, 0.07) for N in Ns_t], "v:", color=C["path_a"],
             label="Path A (discrete codebook)", markersize=5, linewidth=1.4,
             markeredgecolor="white", markeredgewidth=0.4)

    # Highlight BEATS at N=10
    ax.scatter([10], [means[10]], s=250, facecolor="none",
                edgecolor=C["highlight"], linewidth=2.0, zorder=4)
    ax.annotate("BEATS RAG\n$p{=}0.006$", xy=(10, means[10]), xytext=(30, 1.07),
                fontsize=8.5, color="#aa7000", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C["highlight"], lw=1.2))

    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 300, 700, 1000])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "300", "700", "1k"])
    ax.set_xlabel("$N$ (registered identities)")
    ax.set_ylabel("retr@1")
    ax.set_ylim(0, 1.18)
    ax.set_title("V-XC-ID-XXXL scaling (2180-ID face pool)")
    ax.legend(loc="lower left", fontsize=7.5)
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
        ax.plot(Ns, deltas, "o-", label=mode, color=c, linewidth=2.2,
                 markersize=6, markeredgecolor="white", markeredgewidth=0.7)

    ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 300, 700, 1000])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "300", "700", "1k"])
    ax.set_xlabel("$N$ (bank size)")
    ax.set_ylabel("$\\Delta$ retr@1 (trained $-$ zero-shot)")
    ax.set_title("Effect of pretraining: 3 regimes")
    ax.legend(loc="upper left", ncol=5, fontsize=8.5)
    ax.set_ylim(-0.25, 0.78)

    # Shaded regime regions
    ax.axhspan(-0.25, 0, color="#fdd", alpha=0.4, zorder=0)
    ax.axhspan(0, 0.78, color="#dfd", alpha=0.3, zorder=0)
    ax.text(5.5, -0.20, "training hurts\n(encoder perfect)", fontsize=8.5,
             color="#aa3344", style="italic", va="bottom")
    ax.text(50, 0.62, "training helps\n(grows with N)", fontsize=8.5,
             color="#2f6a3f", fontweight="bold", style="italic")

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
             label="RAG-with-LM-context",
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

    # Speedup callouts
    ax.text(1000, 60, "$52{\\times}$ faster\nthan RAG\nat $N{=}1000$",
             fontsize=8.5, color=C["attmem"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF6D6",
                        edgecolor=C["highlight"], linewidth=1.2))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1k", "10k"])
    ax.set_xlabel("$N$ (bank size)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Wall-clock latency vs RAG / Path A")
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
             label="RAG ceiling")
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
    ax1.set_xlabel("$N$"); ax1.set_ylabel("retr@1")
    ax1.set_title("(a) LM size $\\times$ family $\\times$ steps")
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
             label="RAG ceiling")
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
    ax2.set_xlabel("$N$"); ax2.set_ylabel("retr@1")
    ax2.set_title("(b) Curriculum bank size")
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
        ("A-XR-ID\n$N{=}10$",  0.32, 0.90),
        ("A-SCN\n$N{=}10$",    0.40, 0.83),
        ("A-PARA\n$N{=}10$",   0.45, 0.44),
        ("V-STY\n$N{=}5$",     0.20, 0.64),
        ("V-XC-ID\n$N{=}10$",  0.10, 0.99),
        ("V-XC-ID\n$N{=}700$", 0.07, 0.63),
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
    ax.set_ylabel("retr@1")
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
    ax1.set_xticklabels(["RAG cosine", "AttMem (Qwen-3B)"], fontsize=9)
    ax1.set_ylabel("retr@1 at $N{=}10$")
    ax1.set_ylim(0, 1.25)
    ax1.set_title("(a) Random distractors ($n{=}4$ seeds)")
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

    ax2.plot(N_combined, rag_combined, "s-", color=C["rag"], label="RAG cosine NN",
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

    ax2.set_xlabel("$N$ (target + top-$K$ cosine-similar distractors)")
    ax2.set_ylabel("retr@1")
    ax2.set_title("(b) Adversarial distractors: LM family matters")
    ax2.set_ylim(0.78, 0.95)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig7_adversarial.pdf")
    plt.close()
    print("  -> fig7_adversarial.pdf")


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
    ax1.plot(N_banks, rag, "s-", color=C["rag"], label="RAG cosine NN",
              markersize=6, linewidth=2.0, markeredgecolor="white", markeredgewidth=0.6)
    ax1.plot(N_banks, std, "o-", color=C["qwen3b"], label="AttMem standard training",
              markersize=6, linewidth=2.0, markeredgecolor="white", markeredgewidth=0.6)
    ax1.plot(N_banks, adv, "D-", color=C["highlight"],
              label="AttMem adv-training",
              markersize=6.5, linewidth=2.2, markeredgecolor="white", markeredgewidth=0.6)
    ax1.annotate("$+0.145$\nover RAG", xy=(20, adv[-1]), xytext=(11, 0.97),
                  fontsize=8, color="#aa7000", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=C["highlight"], lw=1.0),
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF6D6",
                             edgecolor=C["highlight"]))
    ax1.set_xlabel("$N$ (target + top-$K$ cosine-similar distractors)")
    ax1.set_ylabel("retr@1")
    ax1.set_title("(a) Adversarial regime: training transforms")
    ax1.set_ylim(0.78, 1.02)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Random regime (trade-off)
    Ns = sorted(int(N) for N in d_std["results"])
    rag_r = [d_std["results"][str(N)]["rag"] for N in Ns]
    std_r = [d_std["results"][str(N)]["attmem"] for N in Ns]
    adv_r = [d_adv["results"][str(N)]["attmem"] for N in Ns]
    ax2.plot(Ns, rag_r, "s-", color=C["rag"], label="RAG cosine NN",
              markersize=5.5, linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6)
    ax2.plot(Ns, std_r, "o-", color=C["qwen3b"], label="AttMem standard",
              markersize=5.5, linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6)
    ax2.plot(Ns, adv_r, "D-", color=C["highlight"], label="AttMem adv-training",
              markersize=5.5, linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6)
    ax2.set_xscale("log")
    ax2.set_xticks([5, 10, 50, 100, 300, 1000])
    ax2.set_xticklabels(["5", "10", "50", "100", "300", "1k"])
    ax2.set_xlabel("$N$ (random bank size)")
    ax2.set_ylabel("retr@1")
    ax2.set_title("(b) Random regime: trade-off")
    ax2.set_ylim(0.5, 1.05)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig8_adv_training.pdf")
    plt.close()
    print("  -> fig8_adv_training.pdf")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_arch()
    fig_teaser()
    fig_scorecard()
    fig_scaling()
    fig_training_matters()
    fig_latency()
    fig_ablations()
    fig_pivot()
    fig_adversarial()
    fig_adv_training()
    print("Done.")
