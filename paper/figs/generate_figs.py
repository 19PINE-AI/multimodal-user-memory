"""Generate all paper figures from the results/ JSONs.

NeurIPS-style: serif fonts, clean grids, color-blind safe palettes,
figure sizes targeted at single-column (3.3") or two-column (6.8") layout.
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# NeurIPS-friendly style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "figure.dpi": 150,
})

# Colour-blind safe palette (Wong)
COLORS = {
    "attmem": "#0072B2",   # blue
    "rag": "#D55E00",      # vermillion
    "path_a": "#999999",   # grey
    "zero_shot": "#56B4E9",  # light blue
    "trained": "#0072B2",  # blue
    "qwen3b": "#0072B2",
    "qwen7b": "#E69F00",   # orange
    "ceiling": "#000000",  # black for reference
}

RESULTS = Path("/home/ubuntu/multimodal-user-memory/results")
OUT = Path("/home/ubuntu/multimodal-user-memory/paper/figs")
OUT.mkdir(exist_ok=True)


# ---------------- Figure 1: PerceptMem scorecard ----------------

def fig_scorecard():
    """Bar chart per sub-modality showing AttMem vs RAG vs Path A at N=10."""
    fig, ax = plt.subplots(figsize=(6.8, 2.6))

    # Sub-modalities and their N=10 results
    subs = [
        ("A-XR-ID",     1.000, 0.900, 0.32),   # (rag, attmem, path_a)
        ("A-SCN",       0.933, 0.833, 0.40),
        ("A-PARA\n(n=5)",  0.467, 0.440, 0.45),
        ("V-XC-ID\n(n=4)", 0.933, 0.992, 0.10),
        ("V-STY\n(n=5)",   0.400, 0.460, 0.20),
    ]
    labels = [s[0] for s in subs]
    rag    = np.array([s[1] for s in subs])
    attmem = np.array([s[2] for s in subs])
    path_a = np.array([s[3] for s in subs])

    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, path_a, w, label="Path A (discrete codebook)", color=COLORS["path_a"], edgecolor="black", linewidth=0.4)
    ax.bar(x,     rag,    w, label="RAG cosine-NN ceiling",      color=COLORS["rag"],    edgecolor="black", linewidth=0.4)
    ax.bar(x + w, attmem, w, label="AttMem (ours)",              color=COLORS["attmem"], edgecolor="black", linewidth=0.4)

    # Annotate AttMem-beats-RAG cells
    for i, (l, r, a, p) in enumerate(subs):
        if a > r:
            ax.text(i + w, a + 0.025, "*", ha="center", va="bottom", fontsize=14, color=COLORS["attmem"], weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("retr@1 at N=10")
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper left", framealpha=0.95, ncol=1, bbox_to_anchor=(0.0, 1.0))
    ax.set_title("PerceptMem v0.2 scorecard — retr@1 at N=10 (* = AttMem BEATS RAG)")
    plt.tight_layout()
    plt.savefig(OUT / "fig1_scorecard.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig1_scorecard.pdf")


# ---------------- Figure 2: V-XC-ID scaling curve ----------------

def fig_scaling():
    """retr@1 vs N for V-XC-ID-XXXL: AttMem trained, AttMem 0-shot, RAG ceiling, Path A flat."""
    # Aggregate multi-seed trained
    files = sorted(glob.glob(str(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed*_bsmax1024.json")))
    seeds = [json.load(open(f)) for f in files]
    Ns_train = sorted({int(N) for s in seeds for N in s["results"]})

    rag_at = {}
    attmem_mean = {}
    attmem_std = {}
    for N in Ns_train:
        have = [s for s in seeds if str(N) in s["results"]]
        if not have: continue
        rag_at[N] = have[0]["results"][str(N)]["rag"]
        vals = np.array([s["results"][str(N)]["attmem"] for s in have])
        attmem_mean[N] = vals.mean()
        attmem_std[N] = vals.std() / max(1, len(vals))**0.5  # std error of mean

    # Zero-shot
    zs = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps0_seed42.json"))
    Ns_zs = sorted(int(N) for N in zs["results"])
    zs_mem = [zs["results"][str(N)]["attmem"] for N in Ns_zs]

    # Path A approximate flat baseline (from sessions 11-16 K-sweep)
    path_a_approx = {5: 0.55, 10: 0.10, 20: 0.08, 50: 0.07, 100: 0.07, 300: 0.07, 700: 0.07, 1000: 0.07}

    fig, ax = plt.subplots(figsize=(3.3, 2.6))

    Ns_t = sorted(attmem_mean.keys())
    means = [attmem_mean[N] for N in Ns_t]
    stds = [attmem_std[N] for N in Ns_t]
    rags = [rag_at[N] for N in Ns_t]
    pas  = [path_a_approx.get(N, 0.07) for N in Ns_t]

    ax.plot(Ns_t, rags, "o-", color=COLORS["rag"], label="RAG cosine-NN ceiling",
            markersize=4.5, linewidth=1.5)
    ax.errorbar(Ns_t, means, yerr=stds, fmt="s-", color=COLORS["attmem"],
                label="AttMem (trained)", markersize=4, capsize=2.5, linewidth=1.6)
    ax.plot(Ns_zs, zs_mem, "^--", color=COLORS["zero_shot"], label="AttMem (zero-shot)",
            markersize=4, linewidth=1.2, alpha=0.85)
    ax.plot(Ns_t, pas, "v:", color=COLORS["path_a"], label="Path A (codebook)",
            markersize=4, linewidth=1.2)

    ax.set_xscale("log")
    ax.set_xlabel("N (bank size / registered identities)")
    ax.set_ylabel("retr@1")
    ax.set_ylim(0, 1.10)
    ax.set_xticks([5, 10, 20, 50, 100, 300, 700, 1000])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "300", "700", "1000"], rotation=0)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=7)
    ax.set_title("V-XC-ID-XXXL scaling (2180-ID face pool)")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_scaling.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig2_scaling.pdf")


# ---------------- Figure 3: Training-matters ablation ----------------

def fig_training_matters():
    """Δ retr@1 (trained - zero-shot) per sub-modality and N."""
    cells = []
    trained_files = {
        "A-PARA":      "attmem_a-para_steps5000_seed42.json",
        "A-XR-ID":     "attmem_a-xr-id_steps5000_seed42.json",
        "A-SCN":       "attmem_a-scn_steps5000_seed42.json",
        "V-STY":       "attmem_v-sty-clip_steps5000_seed42.json",
        "V-XC-ID":     "attmem_v-xc-id-xxxl_steps12000_seed42_bsmax1024.json",
    }
    zs_map = {
        "A-PARA":   "attmem_a-para_steps0_seed42.json",
        "A-XR-ID":  "attmem_a-xr-id_steps0_seed42.json",
        "A-SCN":    "attmem_a-scn_steps0_seed42.json",
        "V-STY":    "attmem_v-sty-clip_steps0_seed42.json",
        "V-XC-ID":  "attmem_v-xc-id-xxxl_steps0_seed42.json",
    }

    fig, ax = plt.subplots(figsize=(6.8, 2.6))

    for i, (mode, tf) in enumerate(trained_files.items()):
        zf = zs_map[mode]
        z = json.load(open(RESULTS / zf))
        t = json.load(open(RESULTS / tf))
        Ns = sorted(int(N) for N in z["results"].keys() if N in t["results"])
        if not Ns: continue
        deltas = []
        for N in Ns:
            za = z["results"][str(N)]["attmem"]
            ta = t["results"][str(N)]["attmem"]
            deltas.append(ta - za)
        c = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"][i]
        ax.plot(Ns, deltas, "o-", label=mode, color=c, linewidth=1.5, markersize=4.5)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("N (bank size)")
    ax.set_ylabel("Δ retr@1 (trained − zero-shot)")
    ax.set_title("Training matters: gain from pretraining vs. zero-shot AttMem")
    ax.set_xticks([5, 10, 20, 50, 100, 300, 700, 1000])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "300", "700", "1000"])
    ax.legend(loc="upper left", ncol=5, fontsize=7, framealpha=0.95)
    ax.set_ylim(-0.2, 0.7)

    # Annotate regimes
    ax.text(20, -0.15, "Training HURTS\n(encoder perfect)", fontsize=7, color="#D55E00", ha="center", style="italic")
    ax.text(500, 0.45, "Training HELPS\n(growing with N)", fontsize=7, color="#0072B2", ha="center", style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "fig3_training_matters.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig3_training_matters.pdf")


# ---------------- Figure 4: Latency ----------------

def fig_latency():
    """Query/insert latency vs N: AttMem vs RAG-with-context."""
    # From results/attmem_latency_benchmark.log
    Ns = [10, 100, 1000, 10000]
    attmem_q = [14.94, 14.62, 15.78, 16.55]
    attmem_ins = [0.25, 0.51, 0.52, 0.69]
    rag_ctx = [20.7, 67.2, 823.0, np.nan]  # NaN = OOM
    path_a_ins_per_n = [1000 * n for n in Ns]  # ~1 s per id

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.plot(Ns, attmem_q, "s-", color=COLORS["attmem"], label="AttMem query",
            markersize=5, linewidth=1.6)
    ax.plot(Ns, attmem_ins, "o-", color=COLORS["zero_shot"], label="AttMem batch insert",
            markersize=5, linewidth=1.6)
    rag_valid = [(n, v) for n, v in zip(Ns, rag_ctx) if not np.isnan(v)]
    ax.plot([n for n, _ in rag_valid], [v for _, v in rag_valid], "v-", color=COLORS["rag"],
            label="RAG-w-LM-context", markersize=5, linewidth=1.6)
    # OOM marker for RAG at 10000
    ax.annotate("OOM\n(>32k context)", xy=(10000, 2000), xytext=(3500, 3000),
                fontsize=7, color=COLORS["rag"], ha="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["rag"], lw=0.8))
    ax.plot(Ns, path_a_ins_per_n, ":", color=COLORS["path_a"],
            label="Path A insert (per-id SGD)", linewidth=1.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (bank size)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency: query/insert vs RAG context-prepend")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1k", "10k"])
    ax.legend(loc="upper left", fontsize=7, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_latency.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig4_latency.pdf")


# ---------------- Figure 5: LM-size + curriculum ablations ----------------

def fig_ablations():
    """Two side-by-side: LM-size (3B vs 7B) at V-XC-ID; curriculum bs effect."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    # LM size
    qwen3b_12k = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed42_bsmax1024.json"))
    qwen7b_12k = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps12000_seed42_bsmax1024_qwen7b.json"))
    qwen3b_50k = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps50000_seed42_bsmax1024.json"))
    qwen7b_50k = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps50000_seed42_bsmax1024_qwen7b.json"))
    Ns = sorted(int(N) for N in qwen3b_12k["results"])
    rag = [qwen3b_12k["results"][str(N)]["rag"] for N in Ns]
    q3b12 = [qwen3b_12k["results"][str(N)]["attmem"] for N in Ns]
    q7b12 = [qwen7b_12k["results"][str(N)]["attmem"] for N in Ns]
    q3b50 = [qwen3b_50k["results"][str(N)]["attmem"] for N in Ns]
    q7b50 = [qwen7b_50k["results"][str(N)]["attmem"] for N in Ns]

    ax1.plot(Ns, rag, "k-", label="RAG ceiling", linewidth=1.4, alpha=0.6)
    ax1.plot(Ns, q3b12, "s-", color=COLORS["qwen3b"], label="3B @ 12K", linewidth=1.4)
    ax1.plot(Ns, q3b50, "s--", color=COLORS["qwen3b"], label="3B @ 50K", linewidth=1.4, alpha=0.7)
    ax1.plot(Ns, q7b12, "o-", color=COLORS["qwen7b"], label="7B @ 12K", linewidth=1.4)
    ax1.plot(Ns, q7b50, "o--", color=COLORS["qwen7b"], label="7B @ 50K", linewidth=1.4, alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("N")
    ax1.set_ylabel("retr@1")
    ax1.set_title("(a) LM size × training steps")
    ax1.set_xticks([5, 10, 50, 100, 300, 1000])
    ax1.set_xticklabels(["5", "10", "50", "100", "300", "1k"])
    ax1.legend(fontsize=7, framealpha=0.95, loc="lower left")
    ax1.set_ylim(0, 1.05)

    # Curriculum: bs=64 fixed (seed 42 from #34) vs bs=64..1024 curriculum (seed 42 from #35)
    fixed = json.load(open(RESULTS / "attmem_v-xc-id-xxxl_steps8000_seed42.json"))  # bs=64 fixed
    curriculum = qwen3b_12k  # bs=64..1024
    Nf = sorted(int(N) for N in fixed["results"])
    rag_f = [fixed["results"][str(N)]["rag"] for N in Nf]
    af = [fixed["results"][str(N)]["attmem"] for N in Nf]
    Nc = sorted(int(N) for N in curriculum["results"])
    ac = [curriculum["results"][str(N)]["attmem"] for N in Nc]
    ax2.plot(Nf, rag_f, "k-", label="RAG ceiling", linewidth=1.4, alpha=0.6)
    ax2.plot(Nf, af, "s-", color="#D55E00", label="bs=64 fixed (8K steps)", linewidth=1.5, markersize=4.5)
    ax2.plot(Nc, ac, "o-", color=COLORS["attmem"], label="bs∈[64,1024] (12K steps)", linewidth=1.5, markersize=4.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("N")
    ax2.set_ylabel("retr@1")
    ax2.set_title("(b) Curriculum bank size")
    ax2.set_xticks([5, 10, 50, 100, 300, 1000])
    ax2.set_xticklabels(["5", "10", "50", "100", "300", "1k"])
    ax2.legend(fontsize=7, framealpha=0.95, loc="lower left")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(OUT / "fig5_ablations.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig5_ablations.pdf")


# ---------------- Figure 6: Architecture overview (TikZ-style schematic with matplotlib) ----------------

def fig_arch():
    """Architecture diagram: bolt-on Qwen + AttMem bank."""
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis("off")

    # Frozen LM box
    from matplotlib.patches import FancyBboxPatch, Arrow, FancyArrowPatch, Rectangle
    lm_box = FancyBboxPatch((1.6, 1.3), 4.4, 2.4, boxstyle="round,pad=0.03",
                              linewidth=1.2, edgecolor="black", facecolor="#EEEEEE")
    ax.add_patch(lm_box)
    ax.text(3.8, 3.5, "Frozen Qwen2.5-3B  (36 transformer layers)", ha="center", fontsize=8, weight="bold")
    ax.text(3.8, 3.15, "input embeds → ... → model.norm → lm_head", ha="center", fontsize=7, style="italic", color="#444")
    # Hook indicator
    ax.text(5.95, 2.4, "HOOK", ha="center", fontsize=7, weight="bold", color="#0072B2",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEEAA", edgecolor="#0072B2"))

    # Bank box
    bank_box = FancyBboxPatch((7.0, 1.3), 2.5, 2.4, boxstyle="round,pad=0.03",
                                linewidth=1.2, edgecolor="#0072B2", facecolor="#E6F0FA")
    ax.add_patch(bank_box)
    ax.text(8.25, 3.5, "AttentionMemory", ha="center", fontsize=8, weight="bold", color="#0072B2")
    ax.text(8.25, 3.15, "per-modality bank", ha="center", fontsize=7, color="#0072B2")
    ax.text(8.25, 2.75, "(key, value)\nrows", ha="center", fontsize=7, color="#333")
    ax.text(8.25, 2.15, "softmax(qK/τ)·V", ha="center", fontsize=7, family="monospace")
    ax.text(8.25, 1.75, "W_o · gain", ha="center", fontsize=7, family="monospace")
    ax.text(8.25, 1.45, "→ residual", ha="center", fontsize=7, family="monospace")

    # Arrow from hook to bank
    ax.annotate("", xy=(7.0, 2.4), xytext=(6.25, 2.4),
                arrowprops=dict(arrowstyle="<->", color="#0072B2", lw=1.5))

    # Inputs at the bottom
    ax.text(2.0, 0.55, "text tokens", fontsize=8, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0F0F0"))
    ax.text(3.8, 0.55, "vision (ArcFace 512-d)", fontsize=8, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#D6E5F4"))
    ax.text(5.7, 0.55, "audio (ECAPA 192-d / wav2vec 1024-d)", fontsize=8, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FBE3CC"))

    ax.annotate("", xy=(2.0, 1.25), xytext=(2.0, 0.85),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate("", xy=(3.8, 1.25), xytext=(3.8, 0.85),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate("", xy=(5.7, 1.25), xytext=(5.7, 0.85),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Output
    ax.annotate("", xy=(3.8, 4.4), xytext=(3.8, 3.85),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.text(3.8, 4.65, "next-token logits  (with marker bias from bank)", ha="center", fontsize=8, weight="bold")

    # Insertion box at right
    ax.text(8.25, 0.95, "Register: O(1)\nQuery: O(N·D)", fontsize=7, ha="center", color="#0072B2",
            style="italic")
    ax.text(8.25, 0.45, "~8M trainable\nover 3.1B frozen", fontsize=7, ha="center", color="#444",
            style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "fig0_arch.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig0_arch.pdf")


# ---------------- main ----------------

if __name__ == "__main__":
    print("Generating paper figures...")
    fig_arch()
    fig_scorecard()
    fig_scaling()
    fig_training_matters()
    fig_latency()
    fig_ablations()
    print("Done.")
