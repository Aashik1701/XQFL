#!/usr/bin/env python3
"""Final publication figure set for XQFL: Figures 2-8 plus supplementary S1-S4.

Figure 1 is the hand-designed architecture schematic and is produced by
analysis/gen_figure1.py; this module writes everything else:

  Fig. 2  Experimental coverage + collapse diagnostic
  Fig. 3  RQ1 fidelity heatmap + accuracy-vs-fidelity scatter
  Fig. 4  RQ2 round-to-round attribution stability
  Fig. 5  RQ3 local-to-global divergence + client x round structure
  Fig. 6  RQ4 corrected direct-MIA AUC and per-seed spread
  Fig. 7  Privacy-utility frontier of explanation release
  Fig. 8  Q-SHAP attribution behaviour: evolution and federated/centralized profiles

  Fig. S1 Client class composition under each partition regime
  Fig. S2 The four encoding circuits
  Fig. S3 PCA loadings for both datasets
  Fig. S4 Within-seed accuracy spread before and after the partition-assignment fix

Every plotted value is derived from the committed raw logs, and the aggregate
panels are pulled through analysis/aggregate_results.py -- the same builders that
produce results/outputs/table*.csv -- so figures and tables cannot disagree.

Figure 7 uses utility measured on the same persisted noisy releases as the attack.
S1 and S3 derive partitions and PCA from deterministic seeded paths.

Output: vector PDF + 300-dpi PNG per figure, into results/outputs/final/.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

# The shared visual system (colours, markers, line styles, rcParams) is defined
# once in plots.py; importing it here guarantees every figure in the paper reads
# as one system rather than two.
import plots as P
from plots import (
    ENC_COLOR, ENC_MARKER, ENC_LS, ENC_LABEL,
    DIST_LABEL_1L, DATASET_LABEL, INK, SUBINK, GRIDCOL, CHANCECOL,
)
from aggregate_results import (
    ENC_ORDER, DIST_ORDER, load_explanation_log, build_refs,
    build_rq1_rq3, build_rq3_local_global, build_rq2, build_rq4_mia, build_rq4_dp,
    collect_final_draws,
)
from explain.metrics import js_divergence

DATASETS = ["wisconsin", "pima"]

# "Quantity skew" is too wide for a 4-column heatmap tick row.
DIST_SHORT = {"iid": "IID", "alpha_0.5": r"$\alpha=0.5$",
              "alpha_0.1": r"$\alpha=0.1$", "qty_skew": "Qty skew"}

# Section 5.5: two independently implemented Shapley estimators disagree by this
# much on an identical model. Nothing below it is read as an effect.
ESTIMATOR_FLOOR = 0.171

# Figures are designed at the manuscript's text width and embedded at 1:1.
TEXT_W = 6.9


# ── shared helpers ───────────────────────────────────────────────────────────

def out_dir() -> Path:
    d = PROJECT_ROOT / "results" / "outputs" / "final"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, stem: str, output_dir: Path | None = None):
    d = output_dir or out_dir()
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def panel_tag(fig, ax, letter, dx=-0.058, dy=0.010):
    """(A)/(B) label placed in FIGURE coordinates just above the panel's own
    top-left corner, so it can never collide with the suptitle."""
    pos = ax.get_position()
    fig.text(max(pos.x0 + dx, 0.004), pos.y1 + dy, f"({letter})",
             fontsize=11, fontweight="bold", va="bottom", ha="left", color=INK)


def enc_handles(order=None):
    order = order or ENC_ORDER
    return [Line2D([0], [0], color=ENC_COLOR[e], marker=ENC_MARKER[e],
                   linestyle=ENC_LS[e], markersize=6, linewidth=1.8,
                   label=ENC_LABEL[e]) for e in order]


def numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def final_global_draws(df):
    """Final-round global rows for every completed draw (primary + replication)."""
    return df[(df["client_id"] == "global") & (df["round"] == 15)
              & (df["run_type"].isin(["federated", "replication"]))].copy()


def load_all():
    wis = load_explanation_log(["phase10_log.csv", "phase11_log.csv"])
    ext = load_explanation_log(["rq2_extension_log.csv"])
    pima = load_explanation_log(["phase16_pima_log.csv"])
    for d in (wis, ext, pima):
        numeric(d, ["norm_phi", "margin", "accuracy"])
    return wis, ext, pima


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — experimental coverage and the collapse diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def _coverage_panel(ax, rq3_table):
    """Track-A seed coverage per (dataset x encoding x regime) cell."""
    rows = [(ds, e) for ds in DATASETS for e in ENC_ORDER]
    ax.set_xlim(-0.5, len(DIST_ORDER) - 0.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(len(DIST_ORDER)))
    ax.set_xticklabels([DIST_LABEL_1L[d] for d in DIST_ORDER])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{DATASET_LABEL[ds]} · {ENC_LABEL[e]}" for ds, e in rows][::-1])
    ax.grid(True, color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    def tier(n):
        if n >= 5:
            return "#2f6f3e", 210
        if n >= 3:
            return "#c98a00", 150
        return "#b23b3b", 110

    for yi, (ds, e) in enumerate(rows[::-1]):
        for xi, d in enumerate(DIST_ORDER):
            r = rq3_table[(rq3_table["dataset"] == ds) & (rq3_table["encoding"] == e)
                          & (rq3_table["distribution"] == d)]
            n = int(r.iloc[0]["n_track_a_seeds"]) if not r.empty else 0
            if n == 0:
                ax.plot(xi, yi, marker="x", ms=6, mew=1.5, color="#b23b3b")
            else:
                col, size = tier(n)
                ax.scatter(xi, yi, s=size, color=col, edgecolor="white",
                           linewidth=0.6, zorder=3)
                ax.text(xi, yi, str(n), ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold", zorder=4)

    leg = [Line2D([0], [0], marker="o", ls="", mfc="#2f6f3e", mec="white", ms=10,
                  label=r"sufficient $\geq 5$"),
           Line2D([0], [0], marker="o", ls="", mfc="#c98a00", mec="white", ms=8.5,
                  label=r"limited $3$–$4$"),
           Line2D([0], [0], marker="o", ls="", mfc="#b23b3b", mec="white", ms=6.5,
                  label=r"insufficient $1$–$2$"),
           Line2D([0], [0], marker="x", ls="", mec="#b23b3b", ms=7, mew=1.5,
                  label="none (collapsed)")]
    # Legend sits UNDER the panel: an outside-right legend inflates the saved
    # width well past the manuscript text block.
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.125),
              frameon=False, handletextpad=0.6, ncol=2, columnspacing=1.8,
              borderaxespad=0.0, fontsize=8.5, labelspacing=0.5)
    ax.set_title("Track-A seeds per cell (digit = exact count)",
                 fontsize=9.5, color=SUBINK, fontweight="normal", pad=7)


def _collapse_panel(ax, draws, dataset, show_ylabel):
    """Predictive margin against attribution norm, one point per completed draw."""
    lo = float(draws["margin"].min()) - 0.02
    hi = float(draws["margin"].max()) + 0.02
    ax.axvspan(lo, 0.02, color="#f7eef0", zorder=0)
    ax.axvline(0.02, color="#b23b3b", lw=1.1, ls="--", zorder=2)

    for e in ENC_ORDER:
        sub = draws[draws["encoding"] == e]
        for track, face, alpha in (("A", ENC_COLOR[e], 0.85), ("B", "none", 0.9)):
            s = sub[sub["track_class"] == track]
            if s.empty:
                continue
            ax.scatter(s["margin"], s["norm_phi"], s=26, marker=ENC_MARKER[e],
                       facecolor=face, edgecolor=ENC_COLOR[e],
                       linewidth=1.0, alpha=alpha, zorder=3)

    ax.set_xlabel("Predictive margin over majority baseline", fontsize=9)
    if show_ylabel:
        ax.set_ylabel(r"Attribution norm  $\|\varphi\|_2$")
    ax.set_title(DATASET_LABEL[dataset], fontsize=10, pad=6)
    ax.set_xlim(lo, hi)                      # per-dataset: the two differ a lot
    ax.set_ylim(0, float(draws["norm_phi"].max()) * 1.15)
    # The shaded band is identified in the figure legend instead.


def make_fig02(rq3_table, wis, pima):
    fig = plt.figure(figsize=(TEXT_W, 7.0))
    # left margin sized for panel A's row labels ("Wisconsin - Re-uploading"),
    # which otherwise spill outside the canvas and inflate the tight bbox.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.0], hspace=0.46, wspace=0.26,
                          left=0.275, right=0.985, top=0.905, bottom=0.135)

    axA = fig.add_subplot(gs[0, :])
    _coverage_panel(axA, rq3_table)

    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    _collapse_panel(axB, final_global_draws(wis), "wisconsin", True)
    _collapse_panel(axC, final_global_draws(pima), "pima", False)

    handles = enc_handles() + [
        Line2D([0], [0], marker="o", ls="", mfc="#888888", mec="#888888", ms=6,
               label="Track A (functional)"),
        Line2D([0], [0], marker="o", ls="", mfc="none", mec="#888888", ms=6,
               label="Track B (collapsed)")]
    handles.append(Patch(facecolor="#f7eef0", edgecolor="#b23b3b", ls="--",
                         linewidth=1.0, label="shaded = Track B region"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.004),
               ncol=3, frameon=False, fontsize=8.5, columnspacing=1.3,
               handletextpad=0.5, handlelength=1.8, labelspacing=0.55)

    fig.suptitle("Experimental coverage and the collapse diagnostic",
                 fontsize=11.5, fontweight="bold", x=0.012, ha="left", y=0.985)
    panel_tag(fig, axA, "A", dx=-0.150)
    panel_tag(fig, axB, "B", dx=-0.150)
    save(fig, "fig02_coverage_collapse")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — RQ1 attribution fidelity
# ─────────────────────────────────────────────────────────────────────────────

def _fidelity_matrix(rq1_table, dataset):
    """4 x 4 grid of JSD_cent means; NaN where the cell is not estimated."""
    M = np.full((len(ENC_ORDER), len(DIST_ORDER)), np.nan)
    N = np.zeros_like(M)
    for i, e in enumerate(ENC_ORDER):
        for j, d in enumerate(DIST_ORDER):
            r = rq1_table[(rq1_table["dataset"] == dataset)
                          & (rq1_table["encoding"] == e)
                          & (rq1_table["distribution"] == d)]
            if r.empty:
                continue
            row = r.iloc[0]
            N[i, j] = row["n_track_a"]
            if row["status"] == "ok":
                M[i, j] = row["jsd_cent_mean"]
    return M, N


def _fidelity_heatmap(ax, M, N, dataset, vmin, vmax, cmap, show_ylabel):
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DIST_ORDER)))
    ax.set_xticklabels([DIST_SHORT[d] for d in DIST_ORDER], fontsize=8.5)
    ax.set_yticks(range(len(ENC_ORDER)))
    if show_ylabel:
        ax.set_yticklabels([ENC_LABEL[e] for e in ENC_ORDER], fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_title(DATASET_LABEL[dataset], fontsize=10, pad=6)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#f2f2f2", edgecolor="white",
                                           hatch="///", linewidth=0.8, zorder=2))
                ax.text(j, i, "n/e", ha="center", va="center", fontsize=7.5,
                        color="#8a8a8a", zorder=3)
            else:
                shade = "white" if M[i, j] > 0.5 * (vmin + vmax) else INK
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        fontsize=8.5, color=shade, zorder=3)
                ax.text(j, i + 0.32, f"n={int(N[i, j])}", ha="center", va="center",
                        fontsize=6.5, color=shade, alpha=0.75, zorder=3)
    return im


def _fidelity_scatter(ax, rows):
    """Federated-minus-centralized accuracy against federated-vs-centralized JSD."""
    ax.axhspan(0, ESTIMATOR_FLOOR, color="#eef2f6", zorder=0)
    ax.axhline(ESTIMATOR_FLOOR, color="#5a7d9a", lw=1.1, ls="--", zorder=2)
    ax.axvline(0, color=GRIDCOL, lw=1.0, zorder=1)

    for e in ENC_ORDER:
        s = rows[rows["encoding"] == e]
        if s.empty:
            continue
        for ds, face in (("wisconsin", ENC_COLOR[e]), ("pima", "none")):
            t = s[s["dataset"] == ds]
            if t.empty:
                continue
            ax.scatter(t["d_acc"], t["jsd"], s=30, marker=ENC_MARKER[e],
                       facecolor=face, edgecolor=ENC_COLOR[e], linewidth=1.0,
                       alpha=0.85, zorder=3)

    from scipy.stats import spearmanr
    rho, p = spearmanr(rows["d_acc"], rows["jsd"])
    ax.text(0.015, 0.965,
            f"Spearman $\\rho$ = {rho:+.2f}   (p = {p:.2f},  n = {len(rows)})",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=SUBINK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRIDCOL, lw=0.7))
    ax.text(0.012, ESTIMATOR_FLOOR - 0.008, "estimator-noise floor  (JSD $\\approx$ 0.17)",
            transform=ax.get_yaxis_transform(), ha="left", va="top",
            fontsize=8, color="#41627d")

    ax.set_xlabel("Federated $-$ centralized test accuracy")
    ax.set_ylabel(r"JSD$_{\mathrm{cent}}$")
    ax.set_ylim(0, float(rows["jsd"].max()) * 1.12)


def _per_seed_fidelity(datasets):
    """One row per Track-A draw: accuracy difference and attribution distance."""
    out = []
    for ds, (df, refs) in datasets.items():
        cent_acc = {}
        for _, r in df[(df["run_type"] == "centralized")
                       & (df["client_id"] == "global")].iterrows():
            cent_acc[(r["encoding"], r["seed"])] = float(r["accuracy"])
        for e in ENC_ORDER:
            for d in DIST_ORDER:
                draws = collect_final_draws(df, e, d)
                for _, row in draws[draws["track_class"] == "A"].iterrows():
                    ref = refs.get(e, {}).get(row["seed"], {})
                    cvec, cacc = ref.get("centralized"), cent_acc.get((e, row["seed"]))
                    if cvec is None or cacc is None:
                        continue
                    out.append({
                        "dataset": ds, "encoding": e, "distribution": d,
                        "seed": row["seed"],
                        "d_acc": float(row["accuracy"]) - cacc,
                        "jsd": js_divergence(np.asarray(row["explanation_vector"], float), cvec),
                    })
    return pd.DataFrame(out)


def make_fig03(rq1_table, datasets):
    fig = plt.figure(figsize=(TEXT_W, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.10,
                          left=0.135, right=0.875, top=0.885, bottom=0.145)

    Mw, Nw = _fidelity_matrix(rq1_table, "wisconsin")
    Mp, Np = _fidelity_matrix(rq1_table, "pima")
    vmin = 0.0
    vmax = float(np.nanmax([np.nanmax(Mw), np.nanmax(Mp)]))
    cmap = plt.get_cmap("YlGnBu")

    axA1 = fig.add_subplot(gs[0, 0])
    axA2 = fig.add_subplot(gs[0, 1])
    _fidelity_heatmap(axA1, Mw, Nw, "wisconsin", vmin, vmax, cmap, True)
    im = _fidelity_heatmap(axA2, Mp, Np, "pima", vmin, vmax, cmap, False)

    pos = axA2.get_position()
    cax = fig.add_axes([0.895, pos.y0, 0.020, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"JSD$_{\mathrm{cent}}$", fontsize=9, labelpad=2)
    cb.ax.tick_params(labelsize=7.5)
    cb.ax.axhline(ESTIMATOR_FLOOR, color="#b23b3b", lw=1.6)
    cb.ax.text(-0.45, ESTIMATOR_FLOOR, "0.17",
               transform=cb.ax.get_yaxis_transform(),
               va="center", ha="right", fontsize=7, color="#b23b3b")

    axB = fig.add_subplot(gs[1, :])
    rows = _per_seed_fidelity(datasets)
    _fidelity_scatter(axB, rows)

    handles = enc_handles() + [
        Line2D([0], [0], marker="o", ls="", mfc="#888888", mec="#888888", ms=6,
               label="Wisconsin"),
        Line2D([0], [0], marker="o", ls="", mfc="none", mec="#888888", ms=6,
               label="Pima")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.004),
               ncol=6, frameon=False, fontsize=8.5, columnspacing=1.3,
               handletextpad=0.5, handlelength=1.8)

    fig.suptitle("RQ1  Attribution fidelity under federation",
                 fontsize=11.5, fontweight="bold", x=0.012, ha="left", y=0.985)
    panel_tag(fig, axA1, "A", dx=-0.125)
    panel_tag(fig, axB, "B", dx=-0.125)
    save(fig, "fig03_fidelity")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — RQ2 stability (the validated figure, re-emitted under the final name)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig04(wis, ext, tau_summary):
    """Delegates to the already-validated builder in plots.py so that the
    stability figure is byte-for-byte the one the manuscript was checked
    against; only the filename changes to the final numbering."""
    d = out_dir()
    P.make_figure2(wis, ext, tau_summary, d)
    for ext_ in ("pdf", "png"):
        src = d / f"figure3_stability_over_rounds.{ext_}"
        if src.exists():
            src.replace(d / f"fig04_stability.{ext_}")
    print("  wrote fig04_stability.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — RQ3 local-to-global divergence, aggregate and client x round
# ─────────────────────────────────────────────────────────────────────────────

def _divergence_bars(ax, rq3_table, dataset, show_ylabel):
    slot_x = {"iid": 0.0, "qty_skew": 1.75, "alpha_0.5": 3.6, "alpha_0.1": 5.35}
    order = ["iid", "qty_skew", "alpha_0.5", "alpha_0.1"]
    bar_w, YMAX = 0.30, 0.90

    sub = rq3_table[rq3_table["dataset"] == dataset]
    for ei, e in enumerate(ENC_ORDER):
        offs = (ei - 1.5) * bar_w
        for d in order:
            row = sub[(sub["encoding"] == e) & (sub["distribution"] == d)]
            x = slot_x[d] + offs
            if row.empty:
                continue
            r = row.iloc[0]
            nA = int(r["n_track_a_seeds"])
            if r["status"] != "ok" or not np.isfinite(r["local_global_jsd_mean"]):
                ax.plot(x, 0.012, marker="v", ms=4.5, mfc="white",
                        mec=ENC_COLOR[e], mew=1.1, clip_on=False, zorder=5)
                continue
            m, s = float(r["local_global_jsd_mean"]), float(r["local_global_jsd_std"])
            indicative = nA < 5
            ax.bar(x, m, bar_w, color=ENC_COLOR[e],
                   alpha=0.45 if indicative else 1.0,
                   edgecolor=ENC_COLOR[e], linewidth=0.9,
                   hatch="///" if indicative else None,
                   yerr=s, error_kw={"ecolor": SUBINK, "lw": 0.9, "capsize": 1.8})
            ax.text(x, m + s + 0.012, f"{nA}", ha="center", va="bottom",
                    fontsize=6.2, color=SUBINK)

    ax.axvline(2.68, color="#d5d5d5", lw=0.9)
    ax.set_xticks([slot_x[d] for d in order])
    ax.set_xticklabels([DIST_LABEL_1L[d] for d in order], fontsize=8.5)
    ax.set_title(DATASET_LABEL[dataset], fontsize=10, pad=16)
    ax.set_ylim(0, YMAX)
    ax.set_xlim(-0.8, slot_x["alpha_0.1"] + 0.8)
    ax.grid(axis="x", visible=False)
    if show_ylabel:
        ax.set_ylabel(r"$D_{\mathrm{LG}}$")
    for (a, b, txt) in [(slot_x["iid"], slot_x["qty_skew"], "balanced ratio"),
                        (slot_x["alpha_0.5"], slot_x["alpha_0.1"], "label skew")]:
        ax.plot([a - 0.55, b + 0.55], [0.825, 0.825], color=SUBINK, lw=0.9, clip_on=False)
        ax.text((a + b) / 2, 0.838, txt, ha="center", fontsize=7.3, color=SUBINK)


def _client_round_matrix(df, encoding, distribution, seed):
    """JSD(phi_k^(t), phi_g^(t)) for every logged client-round pair of one run."""
    sub = df[(df["encoding"] == encoding) & (df["distribution"] == distribution)
             & (df["seed"] == seed) & (df["run_type"] == "federated")]
    glob = {int(r["round"]): np.asarray(r["explanation_vector"], float)
            for _, r in sub[sub["client_id"] == "global"].iterrows()}
    rounds = sorted(r for r in glob if r >= 1)
    clients = sorted(c for c in sub["client_id"].unique() if c != "global")
    M = np.full((len(clients), len(rounds)), np.nan)
    for ci, c in enumerate(clients):
        rows = sub[sub["client_id"] == c]
        for _, r in rows.iterrows():
            t = int(r["round"])
            if t in glob and t >= 1:
                M[ci, rounds.index(t)] = js_divergence(
                    np.asarray(r["explanation_vector"], float), glob[t])
    return M, clients, rounds


def _client_round_panel(ax, M, clients, rounds, title, vmin, vmax, cmap, show_ylabel):
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(0, len(rounds), 2))
    ax.set_xticklabels([rounds[i] for i in range(0, len(rounds), 2)], fontsize=8)
    ax.set_yticks(range(len(clients)))
    if show_ylabel:
        ax.set_yticklabels([f"Client {c}" for c in clients], fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Federated round", fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def make_fig05(rq3_table, wis, ref_encoding="amplitude", ref_seed=42):
    fig = plt.figure(figsize=(TEXT_W, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.58, wspace=0.10,
                          left=0.095, right=0.875, top=0.895, bottom=0.175)

    axA1 = fig.add_subplot(gs[0, 0])
    axA2 = fig.add_subplot(gs[0, 1], sharey=axA1)
    _divergence_bars(axA1, rq3_table, "wisconsin", True)
    _divergence_bars(axA2, rq3_table, "pima", False)
    plt.setp(axA2.get_yticklabels(), visible=False)

    Mi, cl_i, rd_i = _client_round_matrix(wis, ref_encoding, "iid", ref_seed)
    Ma, cl_a, rd_a = _client_round_matrix(wis, ref_encoding, "alpha_0.5", ref_seed)
    finite = np.concatenate([Mi[np.isfinite(Mi)], Ma[np.isfinite(Ma)]])
    vmin, vmax = float(finite.min()), float(finite.max())
    cmap = plt.get_cmap("magma_r")

    axB1 = fig.add_subplot(gs[1, 0])
    axB2 = fig.add_subplot(gs[1, 1])
    _client_round_panel(axB1, Mi, cl_i, rd_i, "IID", vmin, vmax, cmap, True)
    im = _client_round_panel(axB2, Ma, cl_a, rd_a, r"$\alpha=0.5$", vmin, vmax, cmap, False)

    pos = axB2.get_position()
    cax = fig.add_axes([0.890, pos.y0, 0.020, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"JSD$(\varphi_k^{(t)},\varphi_g^{(t)})$", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7.5)

    handles = enc_handles() + [Line2D([0], [0], marker="v", ls="", mfc="white",
                                      mec=SUBINK, mew=1.1, ms=6,
                                      label=r"$<3$ Track-A seeds (not estimated)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               ncol=5, frameon=False, handlelength=1.8, columnspacing=1.2, fontsize=8.5)
    fig.text(0.5, 0.012,
             f"Panel B: {ENC_LABEL[ref_encoding]} encoding, Wisconsin, seed {ref_seed}; "
             "round 0 excluded (no local model exists before the first update).",
             fontsize=8, color=SUBINK, ha="center")

    fig.suptitle("RQ3  Local-to-global explanation divergence",
                 fontsize=11.5, fontweight="bold", x=0.012, ha="left", y=0.985)
    panel_tag(fig, axA1, "A", dx=-0.085)
    panel_tag(fig, axB1, "B", dx=-0.085)
    save(fig, "fig05_divergence")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — RQ4 membership leakage through the explanation vectors
# ─────────────────────────────────────────────────────────────────────────────

def _per_seed_attack():
    """Per-seed direct-attack AUC at n = 15 (Wisconsin IID), plus the
    shadow-model control where it was run (n = 5)."""
    base = pd.read_csv(PROJECT_ROOT / "results" / "phase13_mia_log.csv")
    ext = pd.read_csv(PROJECT_ROOT / "results" / "rq4_extension_log.csv")
    direct = pd.concat([
        base[["encoding", "seed", "direct_mia_auc"]],
        ext[["encoding", "seed", "direct_mia_auc"]],
    ], ignore_index=True).drop_duplicates(subset=["encoding", "seed"])
    shadow = base[["encoding", "seed", "shadow_mia_auc"]].dropna()
    return direct, shadow


def _mia_bars(ax, rq4_mia_table):
    sub = rq4_mia_table[(rq4_mia_table["distribution"] == "iid")
                        & (rq4_mia_table["status"] == "ok")].copy()
    sub["_key"] = list(zip(sub["dataset"], sub["encoding"]))
    sub = sub.sort_values("auc_mean")

    xs, labels, colors, hatches = [], [], [], []
    for i, (_, r) in enumerate(sub.iterrows()):
        xs.append(i)
        labels.append(ENC_LABEL[r["encoding"]] +
                      ("\n(Pima)" if r["dataset"] == "pima" else ""))
        colors.append(ENC_COLOR[r["encoding"]])
        hatches.append("//" if r["dataset"] == "pima" else None)

    for x, (_, r), c, h in zip(xs, sub.iterrows(), colors, hatches):
        ax.bar(x, r["auc_mean"], 0.62, color=c, edgecolor=c, linewidth=0.9,
               hatch=h, alpha=0.55 if h else 1.0,
               yerr=r["auc_std"], error_kw={"ecolor": SUBINK, "lw": 0.9, "capsize": 2.2})
        ax.text(x, r["auc_mean"] + r["auc_std"] + 0.012, f"{r['auc_mean']:.3f}",
                ha="center", va="bottom", fontsize=8, color=INK)

    ax.axhline(0.5, color=CHANCECOL, lw=1.2, ls="--", zorder=1)
    ax.text(-0.58, 0.505, "chance", ha="left", va="bottom",
            fontsize=8, color=CHANCECOL)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Direct-attack AUC")
    ax.set_ylim(0.40, 0.90)
    ax.set_xlim(-0.65, len(xs) - 0.35)
    ax.grid(axis="x", visible=False)
    ax.set_title("Mean $\\pm$ 1 SD across seeds", fontsize=9.5,
                 color=SUBINK, fontweight="normal", pad=6)


def _attack_spread(ax, direct, shadow, order):
    rng = np.random.default_rng(0)
    lo, hi = 1.0, 0.0
    for i, e in enumerate(order):
        d = direct[direct["encoding"] == e]["direct_mia_auc"].to_numpy(float)
        s = shadow[shadow["encoding"] == e]["shadow_mia_auc"].to_numpy(float)
        jd = rng.uniform(-0.085, 0.085, len(d))
        js = rng.uniform(-0.085, 0.085, len(s))
        ax.scatter(np.full(len(d), i - 0.19) + jd, d, s=26, marker=ENC_MARKER[e],
                   facecolor=ENC_COLOR[e], edgecolor="white", linewidth=0.5,
                   alpha=0.9, zorder=3)
        ax.scatter(np.full(len(s), i + 0.19) + js, s, s=26, marker=ENC_MARKER[e],
                   facecolor="none", edgecolor="#8a8a8a", linewidth=1.0,
                   alpha=0.95, zorder=3)
        ax.plot([i - 0.33, i - 0.05], [d.mean()] * 2, color=INK, lw=1.6, zorder=4)
        if len(s):
            ax.plot([i + 0.05, i + 0.33], [s.mean()] * 2, color="#8a8a8a", lw=1.6, zorder=4)
        allv = np.concatenate([d, s]) if len(s) else d
        lo, hi = min(lo, float(allv.min())), max(hi, float(allv.max()))

    ax.axhline(0.5, color=CHANCECOL, lw=1.2, ls="--", zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ENC_LABEL[e] for e in order], fontsize=8.5)
    ax.set_ylabel("Attack AUC")
    pad = (hi - lo) * 0.10
    ax.set_ylim(lo - pad, hi + pad)          # no point may be clipped
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_title("Per-seed spread: direct attack (filled, $n=15$) vs. "
                 "shadow-model control (open, $n=5$)",
                 fontsize=9.5, color=SUBINK, fontweight="normal", pad=6)


def make_fig06(rq4_mia_table):
    from analysis.plot_current_rq4 import membership
    membership(out_dir())

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — privacy-utility frontier of explanation release
# ─────────────────────────────────────────────────────────────────────────────

def _release_utility(*args, **kwargs):
    from analysis.current_rq4 import matched_utility
    return matched_utility()

def make_fig07(dp_table, util):
    from analysis.plot_current_rq4 import noise
    noise(out_dir())

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — what the attributions actually look like
# ─────────────────────────────────────────────────────────────────────────────

SHOW_ROUNDS = [0, 1, 2, 5, 10, 15]
PC_LABELS = ["PC1", "PC2", "PC3", "PC4"]


def _track_a_iid_seeds(pool, encoding):
    fin = final_global_draws(pool)
    s = fin[(fin["encoding"] == encoding) & (fin["distribution"] == "iid")
            & (fin["track_class"] == "A")]["seed"]
    return set(s.tolist())


def _evolution_matrix(pool, encoding):
    """Rounds x PCs of mean normalized |phi| for the global model, over
    Track-A IID seeds. Each row sums to 1, so it shows where attribution mass
    sits rather than how large it is."""
    seeds = _track_a_iid_seeds(pool, encoding)
    sub = pool[(pool["encoding"] == encoding) & (pool["distribution"] == "iid")
               & (pool["client_id"] == "global") & (pool["run_type"] == "federated")
               & (pool["seed"].isin(seeds))]
    M = np.full((len(SHOW_ROUNDS), 4), np.nan)
    for ri, t in enumerate(SHOW_ROUNDS):
        rows = sub[sub["round"] == t]
        if rows.empty:
            continue
        mags = []
        for _, r in rows.iterrows():
            a = np.abs(np.asarray(r["explanation_vector"], float))
            tot = a.sum()
            if tot > 0:
                mags.append(a / tot)
        if mags:
            M[ri] = np.mean(mags, axis=0)
    return M, len(seeds)


def _profile_bars(ax, pool, refs_by_enc, encoding, show_ylabel, ymax=None):
    """Mean |phi| per PC for the federated global model and the centralized
    reference, over the same Track-A IID seeds."""
    seeds = _track_a_iid_seeds(pool, encoding)
    fin = final_global_draws(pool)
    fed_rows = fin[(fin["encoding"] == encoding) & (fin["distribution"] == "iid")
                   & (fin["seed"].isin(seeds))]
    fed = np.mean([np.abs(np.asarray(v, float))
                   for v in fed_rows["explanation_vector"]], axis=0)
    cent_vecs = [np.abs(refs_by_enc.get(encoding, {}).get(s, {}).get("centralized"))
                 for s in seeds
                 if refs_by_enc.get(encoding, {}).get(s, {}).get("centralized") is not None]
    cent = np.mean(cent_vecs, axis=0) if cent_vecs else np.full(4, np.nan)

    x = np.arange(4)
    w = 0.38
    ax.bar(x - w / 2, fed, w, color=ENC_COLOR[encoding], edgecolor=ENC_COLOR[encoding],
           linewidth=0.9, label="Federated")
    ax.bar(x + w / 2, cent, w, color="white", edgecolor=ENC_COLOR[encoding],
           linewidth=1.1, hatch="///", label="Centralized")
    ax.set_xticks(x)
    ax.set_xticklabels(PC_LABELS, fontsize=8)
    ax.set_title(ENC_LABEL[encoding], fontsize=9.5, pad=5)
    ax.grid(axis="x", visible=False)
    if ymax is not None:
        ax.set_ylim(0, ymax)                 # one scale: the panels are compared
    if show_ylabel:
        ax.set_ylabel(r"mean $|\varphi_i|$", fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", labelsize=8)
    return float(np.nanmax(fed)), float(np.nanmax(cent))


def make_fig08(wis, ext, wis_refs):
    pool = pd.concat([wis[wis["distribution"] == "iid"], ext], ignore_index=True)

    fig = plt.figure(figsize=(TEXT_W, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.12, 1.0], hspace=0.42, wspace=0.18,
                          left=0.095, right=0.875, top=0.855, bottom=0.115)

    cmap = plt.get_cmap("Blues")
    axesA, im = [], None
    for ci, e in enumerate(ENC_ORDER):
        ax = fig.add_subplot(gs[0, ci])
        M, nseeds = _evolution_matrix(pool, e)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(4))
        ax.set_xticklabels(PC_LABELS, fontsize=7.5)
        ax.set_yticks(range(len(SHOW_ROUNDS)))
        ax.set_yticklabels([f"R{t}" for t in SHOW_ROUNDS] if ci == 0 else [],
                           fontsize=8)
        ax.set_title(f"{ENC_LABEL[e]}\n$n$={nseeds} seeds", fontsize=9, pad=5)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]) and M[i, j] >= 0.16:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.4,
                            color="white" if M[i, j] > 0.55 else INK)
        axesA.append(ax)
    axesA[0].set_ylabel("Federated round", fontsize=9)

    posA = axesA[3].get_position()
    cax = fig.add_axes([0.893, posA.y0, 0.020, posA.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"share of $|\varphi|$ mass", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7.5)

    # One shared y-scale across the four encodings, else the panels invite a
    # cross-encoding comparison the axes do not actually support.
    peaks = []
    for e in ENC_ORDER:
        seeds = _track_a_iid_seeds(pool, e)
        fin = final_global_draws(pool)
        rows = fin[(fin["encoding"] == e) & (fin["distribution"] == "iid")
                   & (fin["seed"].isin(seeds))]
        peaks.append(np.max(np.mean([np.abs(np.asarray(v, float))
                                     for v in rows["explanation_vector"]], axis=0)))
        cv = [np.abs(wis_refs.get(e, {}).get(sd, {}).get("centralized"))
              for sd in seeds
              if wis_refs.get(e, {}).get(sd, {}).get("centralized") is not None]
        if cv:
            peaks.append(np.max(np.mean(cv, axis=0)))
    ymax = float(max(peaks)) * 1.15

    axesB = []
    for ci, e in enumerate(ENC_ORDER):
        ax = fig.add_subplot(gs[1, ci])
        _profile_bars(ax, pool, wis_refs, e, ci == 0, ymax=ymax)
        axesB.append(ax)

    fig.legend(handles=[Patch(facecolor="#888888", label="Federated"),
                        Patch(facecolor="white", edgecolor="#888888", hatch="///",
                              label="Centralized")],
               loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=2,
               frameon=False, fontsize=8.5, columnspacing=1.6)

    fig.suptitle("Q-SHAP attribution behaviour  (Wisconsin, IID)",
                 fontsize=11.5, fontweight="bold", x=0.012, ha="left", y=0.985)
    panel_tag(fig, axesA[0], "A", dx=-0.085)
    panel_tag(fig, axesB[0], "B", dx=-0.085)
    save(fig, "fig08_attribution")


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary S1 — client class composition under each partition regime
# ─────────────────────────────────────────────────────────────────────────────

GHOST_THRESHOLD = 20   # Section 6.2: shards below this are "ghost clients"


def make_figS1(seed=42, n_clients=4, *, numbered=False, output_dir=None):
    from data.loaders import load_breast_cancer_data
    from data.partition import iid_split, dirichlet_split, quantity_skew_split

    X_tr, _, y_tr, _ = load_breast_cancer_data(random_state=seed)
    regimes = [
        ("IID", iid_split(X_tr, y_tr, n_clients, seed=seed)),
        ("Quantity skew", quantity_skew_split(X_tr, y_tr, n_clients, 0.5, seed=seed)),
        (r"$\alpha=0.5$", dirichlet_split(X_tr, y_tr, n_clients, 0.5, seed=seed)),
        (r"$\alpha=0.1$", dirichlet_split(X_tr, y_tr, n_clients, 0.1, seed=seed)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, 3.5), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.775, bottom=0.215, wspace=0.16)

    C0, C1 = "#4c72b0", "#dd8452"
    for ax, (name, parts) in zip(axes, regimes):
        idx = np.arange(n_clients)
        n0 = np.array([int((p[1] == 0).sum()) for p in parts], float)
        n1 = np.array([int((p[1] == 1).sum()) for p in parts], float)
        tot = n0 + n1
        ax.bar(idx, n0, 0.66, color=C0, edgecolor="white", linewidth=0.7)
        ax.bar(idx, n1, 0.66, bottom=n0, color=C1, edgecolor="white", linewidth=0.7)
        for i, t in enumerate(tot):
            ghost = t < GHOST_THRESHOLD
            lbl = f"{int(t)}\nghost" if ghost else f"{int(t)}"
            ax.text(i, t + 5, lbl, ha="center", va="bottom", fontsize=7.5,
                    linespacing=1.15,
                    color="#b23b3b" if ghost else SUBINK,
                    fontweight="bold" if ghost else "normal")
        ax.set_xticks(idx)
        ax.set_xticklabels([f"C{i}" for i in idx], fontsize=8.5)
        ax.set_title(name, fontsize=10, pad=6)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.65, n_clients - 0.35)

    axes[0].set_ylabel("Training samples on client")
    biggest = max(max(int((p[1] == 0).sum()) + int((p[1] == 1).sum()) for p in parts)
                  for _, parts in regimes)
    axes[0].set_ylim(0, biggest * 1.22)          # room for the count labels
    fig.legend(handles=[Patch(facecolor=C0, label="Class 0 (benign)"),
                        Patch(facecolor=C1, label="Class 1 (malignant)"),
                        Line2D([0], [0], color="#b23b3b", lw=0, marker="s", ms=0,
                               label=f"ghost client: $<${GHOST_THRESHOLD} samples")],
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3,
               frameon=False, fontsize=8.5)
    fig.suptitle(f"{'' if numbered else 'S1  '}Client class composition by partition regime "
                 f"(Wisconsin, seed {seed})",
                 fontsize=11, fontweight="bold", x=0.012, ha="left", y=0.975)
    save(fig, "fig09_partitions" if numbered else "figS1_partitions", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary S2 — the four encoding circuits
# ─────────────────────────────────────────────────────────────────────────────

GATE_FC = {"RY": "#E69F00", "RZ": "#0072B2", "H": "#009E73",
           "ZZ": "#CC79A7", "SP": "#6f7d8c"}


def _wires(ax, n=4, x0=0.0, x1=10.0):
    for q in range(n):
        y = n - 1 - q
        ax.plot([x0, x1], [y, y], color="#9a9a9a", lw=1.0, zorder=1)
        ax.text(x0 - 0.32, y, rf"$q_{q}$", ha="right", va="center", fontsize=8.5,
                color=SUBINK)


def _box(ax, x, y, label, kind, w=0.86, h=0.52):
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=GATE_FC[kind],
                               edgecolor="white", linewidth=1.0, zorder=3, alpha=0.92))
    ax.text(x, y, label, ha="center", va="center", fontsize=7.2, color="white",
            zorder=4, fontweight="bold")


def _cnot(ax, x, yc, yt):
    ax.plot([x, x], [yc, yt], color=INK, lw=1.1, zorder=3)
    ax.plot(x, yc, marker="o", ms=4.5, color=INK, zorder=4)
    ax.add_patch(plt.Circle((x, yt), 0.16, facecolor="white", edgecolor=INK,
                            linewidth=1.1, zorder=4))
    ax.plot([x - 0.16, x + 0.16], [yt, yt], color=INK, lw=1.0, zorder=5)
    ax.plot([x, x], [yt - 0.16, yt + 0.16], color=INK, lw=1.0, zorder=5)


def _zz(ax, x, y1, y2, label):
    ax.plot([x, x], [y1, y2], color=GATE_FC["ZZ"], lw=1.6, zorder=3)
    for y in (y1, y2):
        ax.plot(x, y, marker="o", ms=5.0, color=GATE_FC["ZZ"], zorder=4)
    ax.text(x, (y1 + y2) / 2, label, ha="center", va="center", fontsize=6.2,
            color=GATE_FC["ZZ"], zorder=5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


def _circuit_axes(ax, title, subtitle, xmax):
    ax.set_xlim(-0.75, xmax)
    ax.set_ylim(-0.75, 3.75)
    ax.axis("off")
    # Two stacked lines in axes coordinates; set_title would land on the
    # subtitle because an axis-off panel has no tick space to separate them.
    ax.text(0.0, 1.26, title, transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0.0, 1.06, subtitle, transform=ax.transAxes, fontsize=7.8,
            color=SUBINK, ha="left", va="bottom")


def make_figS2(*, numbered=False, output_dir=None):
    fig, axes = plt.subplots(4, 1, figsize=(TEXT_W, 7.8))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.905, bottom=0.075, hspace=0.95)

    # Angle: one RY per feature, single pass.
    ax = axes[0]
    _circuit_axes(ax, "Angle", "one RY rotation per feature — a product state, single embedding pass", 6.0)
    _wires(ax, x1=5.6)
    for q in range(4):
        _box(ax, 1.2, 3 - q, rf"$R_Y(x_{q})$", "RY")
    ax.text(3.2, 1.5, "no feature products\nno entanglement in the embedding",
            ha="center", va="center", fontsize=7.6, color=SUBINK, style="italic")

    # Amplitude: state preparation over all four wires.
    ax = axes[1]
    _circuit_axes(ax, "Amplitude", r"feature vector loaded into the $2^4$ state amplitudes after zero-pad and $L_2$ normalisation", 6.0)
    _wires(ax, x1=5.6)
    ax.add_patch(plt.Rectangle((0.72, -0.34), 1.5, 3.68, facecolor=GATE_FC["SP"],
                               edgecolor="white", linewidth=1.0, zorder=3, alpha=0.92))
    ax.text(1.47, 1.5, "StatePrep\n" + r"$|\psi\rangle = x/\|x\|$", ha="center", va="center",
            fontsize=7.6, color="white", zorder=4, fontweight="bold")
    ax.text(3.7, 1.5, "accesses the full Hilbert space;\ndepth 1 on a simulator only",
            ha="center", va="center", fontsize=7.6, color=SUBINK, style="italic")

    # Re-uploading: two blocks of RY + CNOT chain.
    ax = axes[2]
    _circuit_axes(ax, "Data re-uploading",
                  "the same features are re-injected before every variational layer (2 uploads)", 10.6)
    _wires(ax, x1=10.2)
    for blk in range(2):
        bx = 1.2 + blk * 4.6
        for q in range(4):
            _box(ax, bx, 3 - q, rf"$R_Y(x_{q})$", "RY")
        for i in range(3):
            _cnot(ax, bx + 1.15 + i * 0.62, 3 - i, 2 - i)
        ax.text(bx + 1.15 + 0.62, -0.58, f"upload {blk + 1}", ha="center", va="top",
                fontsize=7.4, color=SUBINK)

    # IQP: H, RZ(x_i), IsingZZ(x_i x_j) over all 6 pairs, H.
    ax = axes[3]
    _circuit_axes(ax, "IQP",
                  r"$H \rightarrow R_Z(x_i) \rightarrow \mathrm{IsingZZ}(x_i x_j)$ for all 6 pairs $\rightarrow H$", 10.6)
    _wires(ax, x1=10.2)
    for q in range(4):
        _box(ax, 0.75, 3 - q, "H", "H", w=0.6)
        _box(ax, 1.85, 3 - q, rf"$R_Z(x_{q})$", "RZ")
        _box(ax, 9.3, 3 - q, "H", "H", w=0.6)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for k, (i, j) in enumerate(pairs):
        _zz(ax, 3.0 + k * 1.02, 3 - i, 3 - j, rf"$x_{i}x_{j}$")

    fig.legend(handles=[Patch(facecolor=GATE_FC["RY"], label=r"$R_Y$ rotation"),
                        Patch(facecolor=GATE_FC["RZ"], label=r"$R_Z$ rotation"),
                        Patch(facecolor=GATE_FC["H"], label="Hadamard"),
                        Patch(facecolor=GATE_FC["ZZ"], label="IsingZZ coupling"),
                        Patch(facecolor=GATE_FC["SP"], label="State preparation"),
                        Line2D([0], [0], color=INK, marker="o", ms=4, lw=1.1,
                               label="CNOT")],
               loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=6,
               frameon=False, fontsize=8, columnspacing=1.2)
    fig.suptitle(("" if numbered else "S2  ") + "The four data-encoding circuits (4 qubits; features pre-scaled to $[0,\\pi]$)",
                 fontsize=11, fontweight="bold", x=0.012, ha="left", y=0.985)
    save(fig, "fig10_circuits" if numbered else "figS2_circuits", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary S3 — PCA loadings for both datasets
# ─────────────────────────────────────────────────────────────────────────────

def _pca_loadings(loader, seed=42, n_components=4):
    """Re-derive the server-fit basis exactly as the pipeline does: standardise
    on the training split, then fit PCA to the qubit count."""
    from sklearn.decomposition import PCA
    X_tr, _, _, _ = loader(random_state=seed)
    p = PCA(n_components=n_components, random_state=seed).fit(X_tr)
    return p.components_, p.explained_variance_ratio_


def make_figS3(seed=42, *, numbered=False, output_dir=None):
    from data.loaders import load_breast_cancer_data, load_pima_data
    from sklearn.datasets import load_breast_cancer

    wis_names = list(load_breast_cancer().feature_names)
    pima_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigree", "Age"]

    Cw, Vw = _pca_loadings(load_breast_cancer_data, seed)
    Cp, Vp = _pca_loadings(load_pima_data, seed)

    fig = plt.figure(figsize=(TEXT_W, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.58], hspace=1.05,
                          left=0.135, right=0.885, top=0.915, bottom=0.215)

    lim = max(np.abs(Cw).max(), np.abs(Cp).max())
    cmap = plt.get_cmap("RdBu_r")
    im = None
    for gi, (C, V, names, title, fs) in enumerate([
            (Cw, Vw, wis_names, "Wisconsin (30 features)", 5.4),
            (Cp, Vp, pima_names, "Pima (8 features)", 6.6)]):
        ax = fig.add_subplot(gs[gi])
        im = ax.imshow(C, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(C.shape[1]))
        ax.set_xticklabels(names, fontsize=fs, rotation=90)
        ax.set_yticks(range(C.shape[0]))
        ax.set_yticklabels([f"PC{i+1}  ({V[i]*100:.0f}%)" for i in range(C.shape[0])],
                           fontsize=8.5)
        ax.set_title(title, fontsize=9.5, pad=5)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    pos = gs[0].get_position(fig)
    cax = fig.add_axes([0.90, pos.y0, 0.022, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("loading", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(("" if numbered else "S3  ") + "PCA loadings — what the four attribution components contain",
                 fontsize=11, fontweight="bold", x=0.012, ha="left", y=0.978)
    fig.text(0.5, 0.010,
             "Percentages are explained variance. The Q-SHAP components are these\n"
             "directions, not original clinical variables; no clinical reading is "
             "claimed from them.",
             fontsize=8, color=SUBINK, ha="center", va="bottom", linespacing=1.4)
    save(fig, "fig11_pca_loadings" if numbered else "figS3_pca_loadings", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary S4 — within-seed accuracy spread before/after the partition fix
# ─────────────────────────────────────────────────────────────────────────────

ARCHIVE = "prefix_archive_20260906T194218"


def _within_seed_ranges(df):
    """Per (encoding, regime, seed): the range of final-round global accuracy
    across repeated draws. Only cells with >= 2 draws can show a range."""
    fin = final_global_draws(df)
    out = []
    for (e, d, s), g in fin.groupby(["encoding", "distribution", "seed"]):
        if len(g) < 2:
            continue
        a = g["accuracy"].astype(float)
        out.append({"encoding": e, "distribution": d, "seed": s,
                    "range_pp": float((a.max() - a.min()) * 100), "n_draws": len(g)})
    return pd.DataFrame(out)


def make_figS4(wis_after, *, numbered=False, output_dir=None):
    arch_dir = PROJECT_ROOT / "results" / ARCHIVE
    before_frames = []
    for name in ("phase10_log.csv", "phase11_log.csv"):
        p = arch_dir / name
        if not p.exists():
            continue
        import csv as _csv, json as _json
        with open(p, newline="") as fh:
            rows = list(_csv.DictReader(fh))
        d = pd.DataFrame(rows)
        d["seed"] = d["seed"].astype(int)
        d["round"] = d["round"].astype(int)
        d["accuracy"] = d["accuracy"].astype(float)
        d["explanation_vector"] = d["explanation_vector"].apply(_json.loads)
        before_frames.append(d)
    if not before_frames:
        print("  S4 skipped: pre-fix archive not found")
        return
    before = _within_seed_ranges(pd.concat(before_frames, ignore_index=True))
    after = _within_seed_ranges(wis_after)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TEXT_W, 3.6),
                                   gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.32})
    fig.subplots_adjust(left=0.095, right=0.985, top=0.80, bottom=0.155)

    rng = np.random.default_rng(1)
    for i, (label, dfr, col) in enumerate([("Before fix", before, "#b23b3b"),
                                           ("After fix", after, "#2f6f3e")]):
        v = dfr["range_pp"].to_numpy(float)
        axL.scatter(np.full(len(v), i) + rng.uniform(-0.13, 0.13, len(v)), v,
                    s=26, facecolor=col, edgecolor="white", linewidth=0.5, alpha=0.85,
                    zorder=3)
        axL.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=1.8, zorder=4)
        axL.text(i, max(v) + 1.4, f"max {max(v):.1f} pp", ha="center", va="bottom",
                 fontsize=8, color=col, fontweight="bold")
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["Before fix", "After fix"], fontsize=9.5)
    axL.set_ylabel("Within-seed final-accuracy range (pp)")
    axL.set_ylim(bottom=0)
    axL.set_xlim(-0.55, 1.55)
    axL.grid(axis="x", visible=False)
    axL.set_title("Every repeated cell (bar = median)", fontsize=9.5,
                  color=SUBINK, fontweight="normal", pad=6)

    bins = np.linspace(0, max(before["range_pp"].max(), after["range_pp"].max()) + 1, 22)
    axR.hist(before["range_pp"], bins=bins, color="#b23b3b", alpha=0.55,
             label=f"Before ($n$={len(before)})", edgecolor="white", linewidth=0.5)
    axR.hist(after["range_pp"], bins=bins, color="#2f6f3e", alpha=0.75,
             label=f"After ($n$={len(after)})", edgecolor="white", linewidth=0.5)
    axR.set_xlabel("Within-seed range (pp)")
    axR.set_ylabel("Cells")
    axR.legend(frameon=False, fontsize=8.5)
    axR.set_title("Distribution", fontsize=9.5, color=SUBINK,
                  fontweight="normal", pad=6)

    fig.suptitle(("" if numbered else "S4  ") + "Within-seed accuracy spread before and after the "
                 "client-to-partition fix",
                 fontsize=11, fontweight="bold", x=0.012, ha="left", y=0.975)
    save(fig, "fig12_partition_fix" if numbered else "figS4_partition_fix", output_dir)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    d = out_dir()
    print(f"Output directory: {d}\n")

    wis, ext, pima = load_all()
    wis_refs, pima_refs = build_refs(wis), build_refs(pima)
    datasets = {"wisconsin": (wis, wis_refs), "pima": (pima, pima_refs)}

    rq1_table = build_rq1_rq3(datasets)
    rq3_table = build_rq3_local_global(datasets)
    rq2_table = build_rq2(wis, ext, pima)
    rq4_mia_table = build_rq4_mia()
    rq4_dp_table = build_rq4_dp()
    dp_log = None  # corrected persisted release records are read by current_rq4

    tau_summary = {r["encoding"]: (r["tau_mean"], r["tau_std"])
                   for _, r in rq2_table[(rq2_table["dataset"] == "wisconsin")
                                         & (rq2_table["distribution"] == "iid")].iterrows()}

    print("Figure 2  coverage + collapse ...");   make_fig02(rq3_table, wis, pima)
    print("Figure 3  RQ1 fidelity ...");          make_fig03(rq1_table, datasets)
    print("Figure 4  RQ2 stability ...");         make_fig04(wis, ext, tau_summary)
    print("Figure 5  RQ3 divergence ...");        make_fig05(rq3_table, wis)
    print("Figure 6  RQ4 membership ...");        make_fig06(rq4_mia_table)
    print("Figure 7  privacy-utility ...")
    util = _release_utility(wis, ext, dp_log)
    make_fig07(rq4_dp_table, util)
    print("Figure 8  attribution behaviour ..."); make_fig08(wis, ext, wis_refs)
    print("Figure S1 partitions ...");            make_figS1()
    print("Figure S2 circuits ...");              make_figS2()
    print("Figure S3 PCA loadings ...");          make_figS3()
    print("Figure S4 partition fix ...");         make_figS4(wis)

    util.to_csv(d / "fig07_utility_derived.csv", index=False)
    print(f"\nAll figures written to {d}")


if __name__ == "__main__":
    main()
