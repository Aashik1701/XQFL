"""Generate Figure 1: XQFL eight-tier architecture diagram.

Faithful to the pipeline described in XQFL_paper.md Section 5.1:
  Tier 1 datasets -> Tier 2 shared PCA(4)+MinMax -> Tier 3 partitioning ->
  Tier 4 per-client encode/train/explain (Q-SHAP) -> Tier 5 FedAvg aggregation ->
  Tier 6 global model + per-round explanation -> Tier 7 Track A/B classification ->
  Tier 8 four evaluation modules (fidelity/stability/divergence/privacy).

Writes results/outputs/figure1_architecture.png (300 dpi, 8x5 in -> 2400x1500),
matching the dimensions of figures 2-5.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(PROJECT_ROOT, "results", "outputs", "figure1_architecture.png")


def main():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=187.5)  # -> 3000 x 1875
    ax.set_xlim(0, 16)
    ax.set_ylim(-1.5, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, fc="#eaf1f8", ec="#2b4a6f", fs=10,
            weight="normal", lw=1.4, align="center", dy=0):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.10,rounding_size=0.16",
                           fc=fc, ec=ec, lw=lw)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2 + dy, text, ha=align, va="center",
                fontsize=fs, fontweight=weight, linespacing=1.25)

    def arrow(x1, y1, x2, y2, color="#2b4a6f", lw=2.0):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            mutation_scale=16, color=color, lw=lw)
        ax.add_patch(a)

    def tier(x, y, text):
        ax.text(x, y, text, fontsize=9, color="#555555", style="italic",
                ha="left", va="center")

    EDGE = "#2b4a6f"
    DATA = "#d9e6f2"
    CLIENT = "#fde2d9"

    # ── Tier 1: Datasets ─────────────────────────────────────────────
    tier(0.25, 9.78, "Tier 1 \u00b7 Data")
    box(0.4, 9.05, 4.3, 0.8, "Wisconsin\n569 \u00d7 30", fc=DATA, ec=EDGE)
    box(11.2, 9.05, 4.3, 0.8, "Pima\n768 \u00d7 8", fc=DATA, ec=EDGE)
    box(5.6, 9.05, 4.2, 0.8, "Binary labels\n85 / 15 class balance", fc=DATA, ec=EDGE)

    # ── Tier 2: Shared preprocessing ─────────────────────────────────
    tier(0.25, 8.5, "Tier 2 \u00b7 Shared preprocessing")
    box(2.8, 7.8, 9.4, 0.75, "PCA \u2192 4 components  \u00b7  MinMax scale to [0, \u03c0]   (fit ONCE on server, broadcast",
        fc=DATA, ec=EDGE, fs=10)
    arrow(2.5, 9.05, 2.5, 8.55)
    arrow(13.4, 9.05, 13.4, 8.55)
    arrow(7.7, 9.05, 7.7, 8.55)

    # ── Tier 3: Partitioning ─────────────────────────────────────────
    tier(0.25, 7.15, "Tier 3 \u00b7 Partitioning")
    box(2.8, 6.4, 9.4, 0.7, "IID   \u00b7   quantity skew   \u00b7   Dirichlet \u03b1=0.5   \u00b7   Dirichlet \u03b1=0.1   \u2192  4 client shards",
        fc=DATA, ec=EDGE, fs=10)
    arrow(7.5, 7.8, 7.5, 7.1)

    # ── Tier 4: Clients ──────────────────────────────────────────────
    tier(0.25, 5.75, "Tier 4 \u00b7 Client-side encode \u00b7 train \u00b7 explain (parallel)")
    clients = [("Client 0", 0.6), ("Client 1", 4.35), ("Client 2", 8.1), ("Client 3", 11.85)]
    for name, x in clients:
        box(x, 4.4, 3.3, 1.25, None, fc=CLIENT, ec=EDGE, fs=9)
        ax.text(x + 1.65, 5.45, name, ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x + 1.65, 5.02, "shard \u2192 encoding\n(angle/amplitude/re-\nupload/IQP) \u2192 VQC\n\u2192 Q-SHAP \u03c6",
                ha="center", va="center", fontsize=8.5, linespacing=1.25)
    for (_, x) in clients:
        arrow(x + 1.65, 6.4, x + 1.65, 5.65)

    # ── Tier 5: FedAvg aggregation ───────────────────────────────────
    tier(0.25, 3.75, "Tier 5 \u00b7 Server aggregation")
    box(2.8, 2.9, 9.4, 0.75, "FedAvg  (sample-weighted mean of client weights)   \u00b7   optional TrimmedMean",
        fc=DATA, ec=EDGE, fs=10)
    for (_, x) in clients:
        arrow(x + 1.65, 4.4, x + 1.65, 3.65)

    # ── Tier 6: Global model + explanation ───────────────────────────
    tier(0.25, 2.2, "Tier 6 \u00b7 Global model")
    box(2.8, 1.45, 9.4, 0.7, "Aggregated global model  \u2192  evaluate + explain EVERY round (incl. round 0)",
        fc=DATA, ec=EDGE, fs=10)
    arrow(7.5, 2.9, 7.5, 2.15)

    # ── Tier 7: Track A / B ──────────────────────────────────────────
    tier(0.25, 0.75, "Tier 7 \u00b7 Collapse gate")
    box(2.8, 0.05, 4.3, 0.7, "Track A (functional)\ntest acc > majority + 2pp", fc="#d9f0dd", ec="#1e6b3a")
    box(8.0, 0.05, 4.2, 0.7, "Track B (collapsed)\nnear-constant class, \u2016\u03c6\u2016 \u2248 0", fc="#f8e3e3", ec="#8a2b2b")
    arrow(7.5, 1.45, 4.95, 0.75)
    arrow(7.5, 1.45, 10.1, 0.75)

    # ── Tier 8: Evaluation modules (consume logged phi vectors) ──────
    tier(0.25, -1.42, "Tier 8 \u00b7 Evaluation on logged \u03c6 vectors")
    rqs = [("RQ1\nFidelity\nJSD(global, central)", 0.6),
           ("RQ2\nStability\nKendall \u03c4 across rounds", 4.15),
           ("RQ3\nDivergence\nJSD(local, global)", 7.7),
           ("RQ4\nPrivacy leak\nMIA AUC \u00b7 DP sweep", 11.25)]
    for label, x in rqs:
        box(x, -1.35, 3.4, 1.05, None, fc="#efe9f7", ec="#5b3a8a", fs=9)
        lines = label.split("\n")
        ax.text(x + 1.7, -0.30, lines[0], ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x + 1.7, -0.62, "\n".join(lines[1:]), ha="center", va="center",
                fontsize=8, linespacing=1.2)
    # evaluation modules are downstream of the Track A/B logged vectors
    for x in (2.3, 5.85, 9.4, 12.95):
        arrow(x, 0.05, x - 1.1, -1.35, lw=1.6)

    fig.suptitle(
        "XQFL eight-tier pipeline: datasets \u2192 shared PCA \u2192 partitioning \u2192 "
        "per-client encode/train/Q-SHAP \u2192 FedAvg \u2192 global model \u2192 Track A/B \u2192 evaluation",
        fontsize=12, y=0.965)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
