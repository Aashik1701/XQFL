#!/usr/bin/env python3
"""Produce the four write-up-ready figures identified in the completion audit
(results/XQFL_completion_audit.md, Step 6):

  Figure 2 — stability over rounds     (RQ2, Wisconsin IID n=15)
  Figure 3 — local-to-global divergence by distribution
             (RQ3, mean_{k,t} JSD(phi_k, phi_g), Wisconsin + Pima)
  Figure 4 — MIA AUC comparison        (RQ4, Wisconsin IID n=15 + Pima/amplitude)
  Figure 5 — MIA AUC under calibrated Laplace noise vs epsilon   (RQ4, Wisconsin)

(Numbered to match the audit's Figure 1 = existing architecture diagram, so
Figures 2-5 here line up with the audit's own numbering.)

Reuses the data loading / aggregation / exclusion logic in
analysis/aggregate_results.py rather than recomputing it, so the tables and
figures can never silently disagree with each other.

Color: a fixed Okabe-Ito (colorblind-safe) categorical assignment, one color
per encoding, held constant across every figure so the same encoding always
reads as the same color. No dual axes, no rainbow colormaps.
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
    PROJECT_ROOT, RESULTS_DIR, ENC_ORDER, DIST_ORDER, IQP_SEED7,
    MIN_VALID_DRAWS, get_out_dir, load_explanation_log, build_refs,
    build_rq1_rq3, build_rq3_local_global, build_rq4_mia, build_rq4_dp, _read_csv,
)
from explain.metrics import kendalls_tau

# Okabe-Ito palette, fixed assignment — never cycled, never reused for a
# different encoding within or across figures.
ENC_COLOR = {
    "angle":       "#E69F00",  # orange
    "amplitude":   "#0072B2",  # blue
    "reuploading": "#009E73",  # green
    "iqp":         "#D55E00",  # vermillion
}
ENC_LABEL = {
    "angle": "Angle", "amplitude": "Amplitude",
    "reuploading": "Re-uploading", "iqp": "IQP",
}
DIST_LABEL = {
    "iid": "IID", "alpha_0.5": "α=0.5", "alpha_0.1": "α=0.1",
    "qty_skew": "Qty. skew",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, **kw)
    return leg


def _titled(fig, ax, main: str, sub: str):
    """Bold main title (suptitle) + smaller gray detail line (axes title),
    sized to avoid the clipping a single long two-line ax.set_title() causes
    at default figsize."""
    fig.suptitle(main, fontsize=13, y=0.985)
    ax.set_title(sub, fontsize=9.5, color="#555555", pad=8)


# ── Figure 2: stability over rounds (RQ2, Wisconsin IID n=15) ──────────────

def round_pair_series(df: pd.DataFrame, encoding: str, exclude_seeds=frozenset()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per round-pair (1..15), mean +/- std of Kendall's tau between
    consecutive-round global explanation vectors, across all included seeds
    with a Track-A final outcome."""
    sub = df[(df["encoding"] == encoding) & (df["distribution"] == "iid")
             & (df["client_id"] == "global") & (df["run_type"] == "federated")]
    per_seed_taus = []
    for seed, g in sub.groupby("seed"):
        if seed in exclude_seeds:
            continue
        g = g.sort_values("round")
        if len(g) < 2 or g.iloc[-1]["track_class"] != "A":
            continue
        vecs = list(g["explanation_vector"])
        rounds = list(g["round"])
        taus = {}
        for i in range(1, len(vecs)):
            taus[rounds[i]] = kendalls_tau(vecs[i - 1], vecs[i])
        per_seed_taus.append(taus)

    all_rounds = sorted(set().union(*[set(t) for t in per_seed_taus])) if per_seed_taus else []
    means, stds = [], []
    for r in all_rounds:
        vals = [t[r] for t in per_seed_taus if r in t]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return np.array(all_rounds), np.array(means), np.array(stds)


def make_figure2_stability(wisconsin_df, wisconsin_ext_df, out_dir: Path):
    iid_combined = pd.concat([
        wisconsin_df[wisconsin_df["distribution"] == "iid"], wisconsin_ext_df,
    ], ignore_index=True) if not wisconsin_ext_df.empty else wisconsin_df[wisconsin_df["distribution"] == "iid"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for enc in ENC_ORDER:
        exclude = set()   # MN-09 retired — IQP/seed=7 no longer excluded
        rounds, means, stds = round_pair_series(iid_combined, enc, exclude)
        if len(rounds) == 0:
            continue
        ax.plot(rounds, means, color=ENC_COLOR[enc], linewidth=2,
                marker="o", markersize=4, label=ENC_LABEL[enc])
        ax.fill_between(rounds, means - stds, means + stds, color=ENC_COLOR[enc], alpha=0.12, linewidth=0)

    ax.set_xlabel("Federation round")
    ax.set_ylabel("Kendall's τ (round $r{-}1$ vs. round $r$)")
    _titled(fig, ax, "RQ2 — Explanation stability over training rounds",
            "Wisconsin, IID, n=15 seeds (all encodings)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, 16))
    _legend(ax, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "figure2_stability_over_rounds.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ── Figure 3: divergence by distribution (RQ3, JSD_cent) ───────────────────

def make_figure3_divergence(rq3_table: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    dists = DIST_ORDER
    x = np.arange(len(dists))
    width = 0.19

    for ax, dataset in zip(axes, ["wisconsin", "pima"]):
        sub = rq3_table[rq3_table["dataset"] == dataset]
        for i, enc in enumerate(ENC_ORDER):
            means, stds = [], []
            for d in dists:
                row = sub[(sub["encoding"] == enc) & (sub["distribution"] == d)]
                if row.empty or row.iloc[0]["status"] != "ok":
                    means.append(np.nan); stds.append(0.0)
                else:
                    means.append(row.iloc[0]["local_global_jsd_mean"])
                    stds.append(row.iloc[0]["local_global_jsd_std"])
            offs = (i - 1.5) * width
            ax.bar(x + offs, means, width, yerr=stds, color=ENC_COLOR[enc],
                   label=ENC_LABEL[enc], capsize=2,
                   error_kw={"linewidth": 1, "ecolor": "#555555"})
        ax.set_xticks(x)
        ax.set_xticklabels([DIST_LABEL[d] for d in dists])
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Distribution")

    axes[0].set_ylabel(r"mean$_{k,t}$ JSD($\varphi_k$, $\varphi_g$)  (local vs. global)")
    fig.suptitle("RQ3 — Local-to-global explanation divergence by distribution\n"
                  "Track-A seeds only; missing bars = insufficient Track-A data or collapse (see Table 3 for rates)")
    _legend(axes[1], loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(rect=(0, 0, 0.86, 0.92))
    path = out_dir / "figure3_divergence_by_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ── Figure 4: MIA AUC comparison (RQ4) ──────────────────────────────────────

def make_figure4_mia(rq4_mia_table: pd.DataFrame, out_dir: Path):
    iid = rq4_mia_table[(rq4_mia_table["distribution"] == "iid") & (rq4_mia_table["dataset"] == "wisconsin")]
    pima_row = rq4_mia_table[(rq4_mia_table["dataset"] == "pima")]

    labels, means, stds, colors, ns = [], [], [], [], []
    for enc in ENC_ORDER:
        row = iid[iid["encoding"] == enc]
        if row.empty:
            continue
        labels.append(ENC_LABEL[enc]); colors.append(ENC_COLOR[enc])
        means.append(row.iloc[0]["auc_mean"]); stds.append(row.iloc[0]["auc_std"])
        ns.append(int(row.iloc[0]["n"]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, means, 0.55, yerr=stds, color=colors, capsize=3,
           error_kw={"linewidth": 1, "ecolor": "#555555"})
    for xi, m, n in zip(x, means, ns):
        ax.text(xi, m + 0.03, f"n={n}", ha="center", fontsize=9, color="#444444")

    wisconsin_n = int(iid[iid["encoding"] == "amplitude"].iloc[0]["n"]) if not iid.empty else 15
    pima_underpowered = False
    if not pima_row.empty and pima_row.iloc[0]["status"] == "ok":
        pr = pima_row.iloc[0]
        pima_n = int(pr["n"])
        pima_underpowered = pima_n < wisconsin_n
        xi = len(labels)
        ax.bar(xi, pr["auc_mean"], 0.55, yerr=pr["auc_std"], color=ENC_COLOR["amplitude"],
               hatch="//", edgecolor="white", capsize=3,
               error_kw={"linewidth": 1, "ecolor": "#555555"})
        n_label = f"n={pima_n}" + (" (preliminary)" if pima_underpowered else "")
        ax.text(xi, pr["auc_mean"] + 0.03, n_label, ha="center", fontsize=9, color="#444444")
        labels.append("Amplitude\n(Pima)")
        x = np.arange(len(labels))

    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.text(len(labels) - 0.5, 0.505, "random guess", fontsize=9, color="#888888", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Direct-MIA AUC")
    ax.set_ylim(0, 1.0)
    pima_note = "hatched = Pima check (preliminary — see bar label for n)" if pima_underpowered else "hatched = Pima check (n-matched)"
    _titled(fig, ax, "RQ4 — Membership-inference leakage by encoding",
            f"Wisconsin IID n={wisconsin_n} (all encodings); {pima_note}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out_dir / "figure4_mia_auc_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ── Figure 5: DP AUC vs epsilon (RQ4) ───────────────────────────────────────

def make_figure5_dp(rq4_dp_table: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for enc in ENC_ORDER:
        sub = rq4_dp_table[rq4_dp_table["encoding"] == enc].sort_values("epsilon")
        if sub.empty:
            continue
        ax.errorbar(sub["epsilon"], sub["auc_mean"], yerr=sub["auc_std"],
                    color=ENC_COLOR[enc], linewidth=2, marker="o", markersize=5,
                    capsize=3, label=ENC_LABEL[enc])

    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.text(0.11, 0.51, "random guess", fontsize=9, color="#888888")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.5, 1.0, 5.0])
    ax.set_xticklabels(["0.1", "0.5", "1.0", "5.0"])
    ax.set_xlabel("Privacy budget ε (log scale; lower = more private)")
    ax.set_ylabel("Direct-MIA AUC")
    ax.set_ylim(0.25, 0.85)
    _titled(fig, ax, "RQ4 — MIA AUC under calibrated Laplace noise vs. budget",
            "Wisconsin, 4 encodings × 5 seeds")
    _legend(ax, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "figure5_dp_auc_vs_epsilon.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = get_out_dir()
    print(f"Output directory: {out_dir}\n")

    print("Loading raw logs ...")
    wisconsin_df = load_explanation_log(["phase10_log.csv", "phase11_log.csv"])
    wisconsin_ext_df = load_explanation_log(["rq2_extension_log.csv"])
    pima_df = load_explanation_log(["phase16_pima_log.csv"])
    wisconsin_refs = build_refs(wisconsin_df)
    pima_refs = build_refs(pima_df)

    print("\nFigure 2 — stability over rounds ...")
    make_figure2_stability(wisconsin_df, wisconsin_ext_df, out_dir)

    print("\nFigure 3 — local-to-global divergence by distribution ...")
    rq3_table = build_rq3_local_global({
        "wisconsin": (wisconsin_df, wisconsin_refs),
        "pima": (pima_df, pima_refs),
    })
    make_figure3_divergence(rq3_table, out_dir)

    print("\nFigure 4 — MIA AUC comparison ...")
    rq4_mia_table = build_rq4_mia()
    make_figure4_mia(rq4_mia_table, out_dir)

    print("\nFigure 5 — DP AUC vs epsilon ...")
    rq4_dp_table = build_rq4_dp()
    make_figure5_dp(rq4_dp_table, out_dir)

    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()
