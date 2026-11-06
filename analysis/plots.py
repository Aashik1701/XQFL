#!/usr/bin/env python3
"""Publication figures for XQFL (Figures 2-6).

  Figure 2 - Evidential coverage of the experimental space (Track-A seeds per cell)
  Figure 3 - RQ2 stability over rounds        (Wisconsin IID, n=15)
  Figure 4 - RQ3 local-to-global divergence    (mean_{k,t} JSD(phi_k, phi_g))
  Figure 5 - RQ4 membership-inference signal    (Wisconsin IID n=15 + Pima amplitude)
  Figure 6 - RQ4 calibrated Laplace-noise sweep (Wisconsin, 4 eps, categorical axis)

Every plotted value is pulled through analysis/aggregate_results.py (the same
builders that produce results/outputs/table*.csv), so figures and tables cannot
silently disagree. No pre-fix / archived logs are read. Figure 4 uses the
corrected RQ3 metric build_rq3_local_global -> rq3_local_global_cell
(local client Q-SHAP vs. same-round global Q-SHAP, Track-A seeds, round 0
excluded).

Output: vector PDF (embedded in the manuscript) + 300-dpi PNG (for standalone
viewing), same basename.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_results import (
    ENC_ORDER, DIST_ORDER, get_out_dir, load_explanation_log, build_refs,
    build_rq3_local_global, build_rq4_mia, build_rq4_dp,
)
from explain.metrics import kendalls_tau

# ─────────────────────────────────────────────────────────────────────────────
# ONE visual system for every figure.
# ─────────────────────────────────────────────────────────────────────────────
#
# Each encoding has a fixed identity: colour + marker + line style, so it reads
# the same in Figures 2-6 and stays distinguishable in grayscale.
#   Angle         orange          circle    solid
#   Amplitude     blue            square    solid
#   Re-uploading  bluish-green    triangle  dashed
#   IQP           reddish-purple  diamond   dotted
ENC_COLOR  = {"angle": "#E69F00", "amplitude": "#0072B2",
              "reuploading": "#009E73", "iqp": "#CC79A7"}   # Okabe-Ito: orange/blue/green/reddish-purple
ENC_MARKER = {"angle": "o", "amplitude": "s", "reuploading": "^", "iqp": "D"}
ENC_LS     = {"angle": "-", "amplitude": "-", "reuploading": "--", "iqp": ":"}
ENC_LABEL  = {"angle": "Angle", "amplitude": "Amplitude",
              "reuploading": "Re-uploading", "iqp": "IQP"}
DIST_LABEL = {"iid": "IID", "alpha_0.5": r"$\alpha=0.5$",
              "alpha_0.1": r"$\alpha=0.1$", "qty_skew": "Quantity\nskew"}
DIST_LABEL_1L = {"iid": "IID", "alpha_0.5": r"$\alpha=0.5$",
                 "alpha_0.1": r"$\alpha=0.1$", "qty_skew": "Quantity skew"}
DATASET_LABEL = {"wisconsin": "Wisconsin", "pima": "Pima"}

INK       = "#1a1a1a"
SUBINK    = "#5a5a5a"
GRIDCOL   = "#d9d9d9"
CHANCECOL = "#999999"

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#7a7a7a",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": SUBINK,
    "ytick.color": SUBINK,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.color": GRIDCOL,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def _save(fig, out_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def _enc_handles():
    """Proxy legend handles carrying the full colour+marker+linestyle identity."""
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=ENC_COLOR[e], marker=ENC_MARKER[e],
                   linestyle=ENC_LS[e], markersize=6, linewidth=1.8,
                   label=ENC_LABEL[e]) for e in ENC_ORDER]


# ── Figure 2 : RQ2 stability over rounds ─────────────────────────────────────

def _round_pair_series(df, encoding):
    """Per round-pair r=1..15: mean +/- std of Kendall tau between the
    consecutive-round global attribution vectors, over Track-A-final seeds."""
    sub = df[(df["encoding"] == encoding) & (df["distribution"] == "iid")
             & (df["client_id"] == "global") & (df["run_type"] == "federated")]
    per_seed = []
    for _seed, g in sub.groupby("seed"):
        g = g.sort_values("round")
        if len(g) < 2 or g.iloc[-1]["track_class"] != "A":
            continue
        vecs, rounds = list(g["explanation_vector"]), list(g["round"])
        per_seed.append({rounds[i]: kendalls_tau(vecs[i - 1], vecs[i])
                         for i in range(1, len(vecs))})
    rr = sorted(set().union(*[set(d) for d in per_seed])) if per_seed else []
    m = np.array([np.nanmean([d[r] for d in per_seed if r in d]) for r in rr])
    s = np.array([np.nanstd([d[r] for d in per_seed if r in d], ddof=1) for r in rr])
    return np.array(rr), m, s


def make_figure2(wis_df, wis_ext_df, tau_summary, out_dir):
    iid = pd.concat([wis_df[wis_df["distribution"] == "iid"], wis_ext_df],
                    ignore_index=True) if not wis_ext_df.empty \
        else wis_df[wis_df["distribution"] == "iid"]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [2.3, 1.0], "wspace": 0.42})

    # -- left: mean tau vs round (mean lines only; the right panel carries uncertainty) --
    axL.axvspan(0.5, 3, color="#eef0f2", zorder=0)
    axL.annotate("early dominant-component lock-in", xy=(1.75, 1.005), xytext=(1.75, 1.11),
                 ha="center", fontsize=7.8, color=SUBINK, style="italic",
                 arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.8))
    axL.text(9, 1.02, "high-$\\tau$ saturation", fontsize=7.8, color=SUBINK, style="italic")
    for e in ENC_ORDER:
        rr, m, _s = _round_pair_series(iid, e)
        if len(rr) == 0:
            continue
        axL.plot(rr, m, color=ENC_COLOR[e], marker=ENC_MARKER[e], ls=ENC_LS[e],
                 ms=4, lw=1.7)
    axL.set_xlabel("Federated round $r$")
    axL.set_ylabel(r"mean Kendall $\tau\,(\varphi_g^{(r-1)},\,\varphi_g^{(r)})$")
    axL.set_xlim(0.5, 15.5)
    axL.set_ylim(0, 1.18)
    axL.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axL.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    axL.legend(handles=_enc_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.42),
               ncol=4, frameon=False, handlelength=2.2, columnspacing=1.4)

    # -- right: mean tau_stab point estimate +/- std (no bar-length illusion) --
    order = ["amplitude", "angle", "reuploading", "iqp"]
    y = np.arange(len(order))[::-1].astype(float)
    means = [float(tau_summary[e][0]) for e in order]
    stds  = [float(tau_summary[e][1]) for e in order]
    for yi, m, s, e in zip(y, means, stds, order):
        axR.errorbar(m, yi, xerr=s, fmt=ENC_MARKER[e], color=ENC_COLOR[e],
                     ms=7, capsize=3, elinewidth=1.2, mec="white", mew=0.6)
        axR.text(m, yi + 0.34, f"{m:.3f}", ha="center", fontsize=8, color=INK)
    axR.set_yticks(y)
    axR.set_yticklabels([ENC_LABEL[e] for e in order])
    axR.set_ylim(-0.7, 3.7)
    axR.set_xlim(0.74, 1.02)
    axR.set_xticks([0.8, 0.9, 1.0])
    axR.set_xlabel(r"mean $\tau_{\mathrm{stab}}\pm$ std  ($n=15$)")
    axR.grid(axis="y", visible=False)
    axR.plot([1.0, 1.0], [y[0] + 0.35, y[1] - 0.35], color=SUBINK, lw=0.9)
    axR.plot([1.0, 1.0], [y[2] + 0.35, y[3] - 0.35], color=SUBINK, lw=0.9)
    axR.text(1.008, (y[0] + y[1]) / 2, "not\nresolved", va="center", ha="left",
             fontsize=6.8, color=SUBINK)
    axR.text(1.008, (y[2] + y[3]) / 2, "vs. Angle:\nresolved", va="center", ha="left",
             fontsize=6.8, color=SUBINK)

    fig.suptitle("RQ2 — round-to-round attribution stability (Wisconsin, IID)",
                 fontsize=11, y=1.03, x=0.02, ha="left")
    _save(fig, out_dir, "figure3_stability_over_rounds")


# ── Figure 3 : RQ3 local-to-global divergence ───────────────────────────────

def make_figure3(rq3_table, out_dir):
    from matplotlib.lines import Line2D
    # "balanced-ratio" block (IID, Quantity skew) | gap | "label-skew" block
    slot_x = {"iid": 0.0, "qty_skew": 1.75, "alpha_0.5": 3.6, "alpha_0.1": 5.35}
    order  = ["iid", "qty_skew", "alpha_0.5", "alpha_0.1"]
    bar_w  = 0.30
    YMAX   = 0.86
    sep_x  = 2.68

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), sharey=True,
                             gridspec_kw={"wspace": 0.07})
    for ax, ds in zip(axes, ["wisconsin", "pima"]):
        sub = rq3_table[rq3_table["dataset"] == ds]
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
        ax.axvline(sep_x, color="#d5d5d5", lw=0.9)
        ax.set_xticks([slot_x[d] for d in order])
        ax.set_xticklabels([DIST_LABEL_1L[d] for d in order], fontsize=8.5)
        ax.set_title(DATASET_LABEL[ds])
        ax.set_ylim(0, YMAX)
        ax.set_xlim(-0.8, slot_x["alpha_0.1"] + 0.8)
        ax.grid(axis="x", visible=False)
        # regime brackets
        for (a, b, txt) in [(slot_x["iid"], slot_x["qty_skew"], r"balanced ratio  (IID $\approx$ Qty)"),
                            (slot_x["alpha_0.5"], slot_x["alpha_0.1"], "label skew")]:
            ax.plot([a - 0.55, b + 0.55], [0.80, 0.80], color=SUBINK, lw=0.9, clip_on=False)
            ax.text((a + b) / 2, 0.815, txt, ha="center", fontsize=7.3, color=SUBINK)

    axes[0].set_ylabel(r"$D_{\mathrm{LG}}=\mathrm{mean}_{k,t}\ \mathrm{JSD}(\varphi_k^{(t)},\varphi_g^{(t)})$")
    handles = _enc_handles() + [Line2D([0], [0], marker="v", ls="", mfc="white",
                                       mec=SUBINK, mew=1.1, ms=6,
                                       label=r"$<3$ Track-A seeds (not estimated)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=5, frameon=False, handlelength=2.0, columnspacing=1.4, fontsize=8.5)
    fig.suptitle("RQ3 — local-to-global explanation divergence by data regime",
                 fontsize=11, y=1.0, x=0.02, ha="left")
    _save(fig, out_dir, "figure4_divergence_by_distribution")


# ── Figure 4 : RQ4 membership-inference signal ──────────────────────────────

def make_figure4(rq4_mia_table, out_dir):
    from analysis.plot_current_rq4 import membership
    membership(out_dir, 'figure5_mia_auc_comparison', panels=False)

def make_figure5(rq4_dp_table, out_dir):
    from analysis.plot_current_rq4 import noise
    noise(out_dir, 'figure6_dp_auc_vs_epsilon', utility=False)

def make_figure6_coverage(rq3_table, out_dir):
    """Track-A seed coverage per (dataset x encoding x distribution) cell, from
    the RQ3 table's distinct-seed count (n_track_a_seeds). Dot glyph encodes the
    evidential tier; the number is the actual Track-A seed count."""
    dists = DIST_ORDER
    rows = [(ds, e) for ds in ["wisconsin", "pima"] for e in ENC_ORDER]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.set_xlim(-0.5, len(dists) - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels([DIST_LABEL_1L[d] for d in dists])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{DATASET_LABEL[ds]} · {ENC_LABEL[e]}" for ds, e in rows][::-1])
    ax.grid(True, color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    def tier(n):
        if n >= 5:  return ("#2f6f3e", 230, "sufficient")      # filled, dark green
        if n >= 3:  return ("#c98a00", 165, "limited")         # amber
        return ("#b23b3b", 120, "insufficient")                # red, small

    for yi, (ds, e) in enumerate(rows[::-1]):
        for xi, d in enumerate(dists):
            r = rq3_table[(rq3_table["dataset"] == ds) & (rq3_table["encoding"] == e)
                          & (rq3_table["distribution"] == d)]
            n = int(r.iloc[0]["n_track_a_seeds"]) if not r.empty else 0
            col, size, _ = tier(n)
            if n == 0:
                ax.plot(xi, yi, marker="x", ms=6, mew=1.4, color="#b23b3b")
            else:
                ax.scatter(xi, yi, s=size, color=col, edgecolor="white", linewidth=0.6, zorder=3)
                ax.text(xi, yi, str(n), ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold", zorder=4)

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", ls="", mfc="#2f6f3e", mec="white", ms=11,
                  label="sufficient  ($\\geq 5$ Track-A seeds)"),
           Line2D([0], [0], marker="o", ls="", mfc="#c98a00", mec="white", ms=9,
                  label="limited  ($3$–$4$)"),
           Line2D([0], [0], marker="o", ls="", mfc="#b23b3b", mec="white", ms=6,
                  label="insufficient  ($1$–$2$)"),
           Line2D([0], [0], marker="x", ls="", mec="#b23b3b", ms=7, mew=1.4,
                  label="none  (all seeds collapsed)")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, handletextpad=0.6, labelspacing=0.9)

    fig.suptitle("Evidential coverage of the experimental space",
                 fontsize=11, y=1.0, x=0.02, ha="left")
    ax.set_title("digit = distinct Track-A seeds contributing to that cell "
                 "(RQ3 basis; 15 seeds drawn for IID, 5+ elsewhere)",
                 fontsize=8.5, color=SUBINK, fontweight="normal", pad=8)
    _save(fig, out_dir, "figure2_experimental_coverage")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    out_dir = get_out_dir()
    print(f"Output directory: {out_dir}\n")

    wis_df = load_explanation_log(["phase10_log.csv", "phase11_log.csv"])
    wis_ext_df = load_explanation_log(["rq2_extension_log.csv"])
    pima_df = load_explanation_log(["phase16_pima_log.csv"])
    wis_refs, pima_refs = build_refs(wis_df), build_refs(pima_df)

    rq3_table = build_rq3_local_global({
        "wisconsin": (wis_df, wis_refs), "pima": (pima_df, pima_refs)})
    rq4_mia_table = build_rq4_mia()
    rq4_dp_table = build_rq4_dp()

    # RQ2 tau_stab summary, computed the same way build_rq2 does (Table 2)
    from aggregate_results import build_rq2
    rq2_tbl = build_rq2(wis_df, wis_ext_df, pima_df)
    tau_summary = {r["encoding"]: (r["tau_mean"], r["tau_std"])
                   for _, r in rq2_tbl[(rq2_tbl["dataset"] == "wisconsin")
                                       & (rq2_tbl["distribution"] == "iid")].iterrows()}

    print("Figure 2 ..."); make_figure2(wis_df, wis_ext_df, tau_summary, out_dir)
    print("Figure 3 ..."); make_figure3(rq3_table, out_dir)
    print("Figure 4 ..."); make_figure4(rq4_mia_table, out_dir)
    print("Figure 5 ..."); make_figure5(rq4_dp_table, out_dir)
    print("Figure 6 ..."); make_figure6_coverage(rq3_table, out_dir)
    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()
