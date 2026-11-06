#!/usr/bin/env python3
"""Aggregate XQFL results into the five write-up-ready tables identified in
the completion audit (results/XQFL_completion_audit.md, Step 6):

  Table 1 (RQ1) — fidelity: JSD_cent by encoding x distribution x dataset
  Table 2 (RQ2) — stability: round-to-round Kendall's tau / cosine / L2 at
                  n=15 IID (Wisconsin), plus per-distribution where enough
                  Track-A trajectories exist to support a mean
  Table 3 (RQ3) — local-to-global divergence: mean over clients k and rounds t
                  of JSD(phi_k^(t), phi_g^(t)), computed directly from the
                  per-client and global explanation rows, distribution-first
                  with a delta-vs-IID column. This is the metric the Methods
                  define for RQ3 and is distinct from Table 1's JSD_cent
                  (federated-vs-centralized final attribution = RQ1).
  Table 4 (RQ4a) — MIA: direct-MIA AUC at n=15 IID (Wisconsin), non-IID
                  (qty_skew, alpha_0.1, Wisconsin), and Pima/amplitude/IID
                  when that run has completed
  Table 5 (RQ4b) — DP: direct-MIA AUC vs epsilon, 4 encodings x 4 epsilon
                  (Wisconsin)

All five tables are computed directly from the raw per-round / per-seed CSVs
in results/ — nothing here is a hand-typed number. Where a rule below
reproduces a number that also appears in analysis/methods_notes.md, it was
checked against that number during development (see git history / session
notes); the computation is the source of truth, not the cross-check.

Exclusion rules applied throughout
-----------------------------------
* NO per-seed anomaly exclusions. The corrected federated recompute
  (2026-09-07, results/CORRECTED_RERUN_AUDIT.md) showed all three pre-fix
  anomalies (MN-09 IQP/seed=7; MN-08 reuploading/alpha_0.5/seed=999 and
  amplitude/alpha_0.5/seed=123) to be artifacts of the node_id%n_clients
  partition-assignment bug, not real instability. Every seed now enters
  every aggregation; no seed value is substituted. IQP Wisconsin IID is
  therefore n=15 (RQ2/RQ4) and the MN-09 3-draw-mean MIA substitution is
  disabled.
* Track-B (collapsed) draws are excluded from every divergence/stability
  mean; the collapse rate (B / total draws) is reported alongside every
  cell so a near-zero JSD from a dead model is never misread as "good
  fidelity" (see methods_notes.md's silent-failure-mode note).
* Any cell with fewer than MIN_VALID_DRAWS (=3) valid (Track-A,
  post-exclusion) draws is reported as "insufficient" rather than as a
  computed mean — still reports n and the collapse rate. This is the rule
  the task calls out by name for reuploading/Pima/alpha_0.1 (2/15 draws).

Outputs
-------
CSV + Markdown for every table, written to OUT_DIR (see get_out_dir() below
— tries /mnt/user-data/outputs first, since that's where the task asked for
them; falls back to results/outputs/ when that path doesn't exist on this
filesystem, which is the case on macOS).
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT))

from explain.metrics import kendalls_tau, cosine_similarity, l2_distance, js_divergence

MIN_VALID_DRAWS = 3   # below this, report "insufficient" rather than a mean

# Canonical distribution labels — Wisconsin logs use "quantity", Pima logs
# use "qty_skew" for the same condition; normalize both to "qty_skew".
DIST_CANON = {
    "iid": "iid",
    "alpha_0.5": "alpha_0.5",
    "alpha_0.1": "alpha_0.1",
    "quantity": "qty_skew",
    "qty_skew": "qty_skew",
}
DIST_ORDER = ["iid", "alpha_0.5", "alpha_0.1", "qty_skew"]
ENC_ORDER = ["angle", "amplitude", "reuploading", "iqp"]

# ── global exclusion rules ────────────────────────────────────────────────
#
# The corrected federated recompute (2026-09-07, results/CORRECTED_RERUN_AUDIT.md)
# eliminated all three pre-fix seed anomalies — they were artifacts of the
# node_id%n_clients partition-assignment bug, not real instability:
#   * MN-09  IQP/seed=7  : now Track-A ~0.868, mid-pack, stable across all
#                          distributions (was "bimodal 0.63 <-> 0.90").
#   * MN-08  reuploading/alpha_0.5/seed=999 : ||phi|| now 0.41/0.41/0.43
#                          (was ~0.07 "near-collapse").
#   * MN-08  amplitude/alpha_0.5/seed=123   : JSD_cent now 0.97 sigma above the
#                          leave-one-out mean (was 4.4 sigma).
# None of the three is excluded any longer; every seed enters every aggregation.
IQP_SEED7_EXCLUDE_DATASET = None          # was "wisconsin" (MN-09) — retired
IQP_SEED7 = 7
RQ3_EXTRA_EXCLUSIONS: dict = {}           # was {reup/a0.5:999, amp/a0.5:123} — retired


def _exclude_iqp_seed7(dataset: str, encoding: str) -> bool:
    return (IQP_SEED7_EXCLUDE_DATASET is not None
            and dataset == IQP_SEED7_EXCLUDE_DATASET and encoding == "iqp")


def get_out_dir() -> Path:
    primary = Path("/mnt/user-data/outputs")
    try:
        primary.mkdir(parents=True, exist_ok=True)
        return primary
    except OSError:
        fallback = PROJECT_ROOT / "results" / "outputs"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"NOTE: /mnt/user-data/outputs is not available on this filesystem "
              f"(that path is a Linux-sandbox convention). Writing outputs to "
              f"{fallback} instead.")
        return fallback


# ── generic CSV loading ─────────────────────────────────────────────────────

def _read_csv(name: str) -> list[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_explanation_log(names: list[str]) -> pd.DataFrame:
    """Load one or more phase10/11/16-style per-round explanation CSVs into
    a single typed, normalized DataFrame."""
    rows = []
    for name in names:
        rows.extend(_read_csv(name))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype(int)
    df["round"] = df["round"].astype(int)
    df["accuracy"] = df["accuracy"].astype(float)
    df["explanation_vector"] = df["explanation_vector"].apply(json.loads)
    df["distribution"] = df["distribution"].map(DIST_CANON).fillna(df["distribution"])
    return df


# ── reference vectors (centralized + IID-federated R15) ────────────────────

def build_refs(df: pd.DataFrame) -> dict:
    """refs[encoding][seed] = {'centralized': arr, 'iid_fed': arr}"""
    refs = defaultdict(dict)
    for _, r in df.iterrows():
        enc, seed = r["encoding"], r["seed"]
        d = refs[enc].setdefault(seed, {})
        if r["run_type"] == "centralized" and r["client_id"] == "global":
            d["centralized"] = np.array(r["explanation_vector"], dtype=float)
        elif (r["run_type"] == "federated" and r["distribution"] == "iid"
              and r["client_id"] == "global" and r["round"] == 15):
            d["iid_fed"] = np.array(r["explanation_vector"], dtype=float)
    return refs


# ── draw collection (final-round global rows, primary + replication) ───────

def collect_final_draws(df: pd.DataFrame, encoding: str, distribution: str) -> pd.DataFrame:
    """One row per completed draw (primary or replication) for this
    encoding x distribution: final-round (R15) global explanation vector,
    accuracy, and track class."""
    sub = df[(df["encoding"] == encoding)
             & (df["distribution"] == distribution)
             & (df["client_id"] == "global")
             & (df["round"] == 15)
             & (df["run_type"].isin(["federated", "replication"]))]
    return sub[["seed", "track_class", "accuracy", "explanation_vector"]].reset_index(drop=True)


# ── RQ1 / RQ3: JSD_cent fidelity/divergence table ───────────────────────────

def jsd_cell(dataset: str, df: pd.DataFrame, refs: dict, encoding: str, distribution: str) -> dict:
    draws = collect_final_draws(df, encoding, distribution)
    n_total = len(draws)
    n_b = int((draws["track_class"] == "B").sum())
    collapse_rate = (n_b / n_total) if n_total else float("nan")

    track_a = draws[draws["track_class"] == "A"].copy()

    excluded_note = []
    if _exclude_iqp_seed7(dataset, encoding):
        track_a = track_a[track_a["seed"] != IQP_SEED7]
        excluded_note.append("IQP/seed=7 excluded (MN-09)")
    extra = RQ3_EXTRA_EXCLUSIONS.get((encoding, distribution))
    if dataset == "wisconsin" and extra:
        before = len(track_a)
        track_a = track_a[~track_a["seed"].isin(extra)]
        if len(track_a) < before:
            excluded_note.append(f"seed(s) {sorted(extra)} excluded (RQ3 anomaly resolution)")

    jsds = []
    for _, row in track_a.iterrows():
        ref = refs.get(encoding, {}).get(row["seed"], {})
        cvec = ref.get("centralized")
        if cvec is None:
            continue
        vec = np.array(row["explanation_vector"], dtype=float)
        jsds.append(js_divergence(vec, cvec))

    n_valid = len(jsds)
    if n_valid >= MIN_VALID_DRAWS:
        mean, std = float(np.mean(jsds)), float(np.std(jsds, ddof=1))
        status = "ok"
    elif n_valid > 0:
        mean, std = float("nan"), float("nan")
        status = "insufficient"
    else:
        mean, std = float("nan"), float("nan")
        status = "absent" if n_total == 0 else "insufficient"

    return {
        "dataset": dataset, "encoding": encoding, "distribution": distribution,
        "n_total_draws": n_total, "n_track_a": len(track_a) if n_total else 0,
        "n_valid_jsd": n_valid, "collapse_rate": collapse_rate,
        "jsd_cent_mean": mean, "jsd_cent_std": std,
        "status": status, "note": "; ".join(excluded_note),
    }


def build_rq1_rq3(datasets: dict) -> pd.DataFrame:
    rows = []
    for dataset, (df, refs) in datasets.items():
        if df.empty:
            continue
        combos = df[(df["client_id"] == "global") & (df["round"] == 15)
                    & (df["run_type"] == "federated")][["encoding", "distribution"]].drop_duplicates()
        for _, c in combos.iterrows():
            rows.append(jsd_cell(dataset, df, refs, c["encoding"], c["distribution"]))
    out = pd.DataFrame(rows)
    out["_enc_ord"] = out["encoding"].map({e: i for i, e in enumerate(ENC_ORDER)})
    out["_dist_ord"] = out["distribution"].map({d: i for i, d in enumerate(DIST_ORDER)})
    return out.sort_values(["dataset", "_enc_ord", "_dist_ord"]).drop(columns=["_enc_ord", "_dist_ord"]).reset_index(drop=True)


def rq3_local_global_cell(dataset: str, df: pd.DataFrame, encoding: str, distribution: str) -> dict:
    """RQ3 as defined in the Methods: the mean over clients k and rounds t of
    JSD(phi_k^(t), phi_g^(t)) — the per-round divergence between each client's
    local Q-SHAP vector and the global (post-aggregation) Q-SHAP vector.

    This is NOT JSD_cent (federated-vs-centralized final-round attribution, which
    is RQ1). It is computed directly from the per-client and global explanation
    rows logged every round.

    Rules, consistent with the rest of this script:
      * primary `federated` runs only, one draw per seed;
      * a seed contributes only if its final-round (R15) global model is Track A
        (a collapsed global model's near-constant vectors make local-to-global
        divergence meaningless); Track-B seeds are counted, not averaged;
      * NO per-seed anomaly exclusions. After the 2026-09-07 corrected rerun
        the module-level exclusion constants are disabled
        (IQP_SEED7_EXCLUDE_DATASET is None, RQ3_EXTRA_EXCLUSIONS is empty), so
        the _exclude_iqp_seed7() branch below never fires and every seed enters
        the mean. The pre-fix MN-09 (IQP/seed=7) and MN-08
        (amplitude/alpha_0.5/seed=123, reuploading/alpha_0.5/seed=999)
        exclusions were shown to be partition-bug artifacts and retired;
      * round 0 (initial broadcast, all clients == global) is skipped.
    """
    sub = df[(df["encoding"] == encoding)
             & (df["distribution"] == distribution)
             & (df["run_type"] == "federated")]
    n_total_seeds = int(sub["seed"].nunique()) if not sub.empty else 0

    excluded_note = []
    per_seed_means = []
    n_track_b = 0
    for seed, g in sub.groupby("seed"):
        if _exclude_iqp_seed7(dataset, encoding) and seed == IQP_SEED7:
            if "IQP/seed=7 excluded (MN-09)" not in excluded_note:
                excluded_note.append("IQP/seed=7 excluded (MN-09)")
            continue

        gfin = g[(g["client_id"] == "global") & (g["round"] == 15)]
        if gfin.empty or gfin.iloc[-1]["track_class"] != "A":
            n_track_b += 1
            continue

        gvecs = {int(r["round"]): np.asarray(r["explanation_vector"], dtype=float)
                 for _, r in g[g["client_id"] == "global"].iterrows()}
        clients_by_round = defaultdict(dict)
        for _, r in g[g["client_id"] != "global"].iterrows():
            clients_by_round[int(r["round"])][str(r["client_id"])] = \
                np.asarray(r["explanation_vector"], dtype=float)

        jsds = []
        for rnd, gvec in gvecs.items():
            if rnd == 0:
                continue
            for _, cvec in clients_by_round.get(rnd, {}).items():
                jsds.append(js_divergence(cvec, gvec))
        if jsds:
            per_seed_means.append(float(np.mean(jsds)))

    n = len(per_seed_means)
    collapse_rate = (n_track_b / n_total_seeds) if n_total_seeds else float("nan")
    if n >= MIN_VALID_DRAWS:
        mean, std, status = float(np.mean(per_seed_means)), float(np.std(per_seed_means, ddof=1)), "ok"
    elif n > 0:
        mean, std, status = float("nan"), float("nan"), "insufficient"
    else:
        mean, std, status = float("nan"), float("nan"), "absent" if n_total_seeds == 0 else "insufficient"

    return {
        "dataset": dataset, "encoding": encoding, "distribution": distribution,
        "n_total_seeds": n_total_seeds, "n_track_a_seeds": n, "n_track_b_seeds": n_track_b,
        "collapse_rate": collapse_rate,
        "local_global_jsd_mean": mean, "local_global_jsd_std": std,
        "status": status, "note": "; ".join(excluded_note),
    }


def build_rq3_local_global(datasets: dict) -> pd.DataFrame:
    """RQ3 table: mean_{k,t} JSD(phi_k, phi_g) by dataset x encoding x
    distribution, with a delta-vs-IID column so 'does local-to-global divergence
    grow with client heterogeneity?' reads directly off the table."""
    rows = []
    for dataset, (df, _refs) in datasets.items():
        if df.empty:
            continue
        combos = df[(df["run_type"] == "federated")][["encoding", "distribution"]].drop_duplicates()
        for _, c in combos.iterrows():
            rows.append(rq3_local_global_cell(dataset, df, c["encoding"], c["distribution"]))
    out = pd.DataFrame(rows)
    iid_means = (out[out["distribution"] == "iid"]
                 .set_index(["dataset", "encoding"])["local_global_jsd_mean"])
    out["delta_vs_iid"] = out.apply(
        lambda r: (r["local_global_jsd_mean"] - iid_means.get((r["dataset"], r["encoding"]), float("nan")))
        if pd.notna(r["local_global_jsd_mean"]) else float("nan"), axis=1)
    out["_enc_ord"] = out["encoding"].map({e: i for i, e in enumerate(ENC_ORDER)})
    out["_dist_ord"] = out["distribution"].map({d: i for i, d in enumerate(DIST_ORDER)})
    return (out.sort_values(["dataset", "_dist_ord", "_enc_ord"])
               .drop(columns=["_enc_ord", "_dist_ord"]).reset_index(drop=True))


# ── RQ2: round-to-round stability ───────────────────────────────────────────

def stability_cell(dataset: str, df: pd.DataFrame, encoding: str, distribution: str) -> dict:
    sub = df[(df["encoding"] == encoding) & (df["distribution"] == distribution)
             & (df["client_id"] == "global") & (df["run_type"] == "federated")]

    per_seed = {}
    excluded_note = []
    for seed, g in sub.groupby("seed"):
        if _exclude_iqp_seed7(dataset, encoding) and seed == IQP_SEED7:
            excluded_note.append("IQP/seed=7 excluded (MN-09)")
            continue
        g = g.sort_values("round")
        # Classify the seed by its FINAL-round track class (consistent with
        # collect_final_draws elsewhere in this script). A collapsed model's
        # near-constant vectors make round-to-round stability meaningless, so
        # Track-B *final* outcomes are excluded — but a transient dip through
        # the collapse threshold at an early round while the model is still
        # converging is not itself a collapse, so intermediate rounds are not
        # filtered (this matches the validated methodology in
        # experiments/phase10_explanation_sweep.py's print_rq2, which does
        # not filter by round-level track class for IID).
        if len(g) < 2 or g.iloc[-1]["track_class"] != "A":
            continue
        vecs = list(g["explanation_vector"])
        taus, coss, l2s = [], [], []
        for i in range(1, len(vecs)):
            taus.append(kendalls_tau(vecs[i - 1], vecs[i]))
            coss.append(cosine_similarity(vecs[i - 1], vecs[i]))
            l2s.append(l2_distance(vecs[i - 1], vecs[i]))
        per_seed[seed] = (np.nanmean(taus), np.nanmean(coss), np.nanmean(l2s))

    n = len(per_seed)
    if n >= MIN_VALID_DRAWS:
        taus = [v[0] for v in per_seed.values()]
        coss = [v[1] for v in per_seed.values()]
        l2s = [v[2] for v in per_seed.values()]
        status = "ok"
        result = dict(tau_mean=float(np.mean(taus)), tau_std=float(np.std(taus, ddof=1)),
                      cos_mean=float(np.mean(coss)), cos_std=float(np.std(coss, ddof=1)),
                      l2_mean=float(np.mean(l2s)), l2_std=float(np.std(l2s, ddof=1)))
    else:
        status = "insufficient" if n > 0 else "absent"
        result = dict(tau_mean=float("nan"), tau_std=float("nan"),
                      cos_mean=float("nan"), cos_std=float("nan"),
                      l2_mean=float("nan"), l2_std=float("nan"))

    n_total_seeds = sub["seed"].nunique()
    return {
        "dataset": dataset, "encoding": encoding, "distribution": distribution,
        "n_final_track_a_seeds": n, "n_total_seeds": n_total_seeds,
        **result, "status": status, "note": "; ".join(excluded_note),
    }


def build_rq2(wisconsin_df: pd.DataFrame, wisconsin_ext_df: pd.DataFrame, pima_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # n=15 IID Wisconsin: combine primary (phase10) + extension (rq2_extension) seeds
    iid_combined = pd.concat([
        wisconsin_df[wisconsin_df["distribution"] == "iid"],
        wisconsin_ext_df,
    ], ignore_index=True) if not wisconsin_ext_df.empty else wisconsin_df[wisconsin_df["distribution"] == "iid"]

    for enc in ENC_ORDER:
        rows.append(stability_cell("wisconsin", iid_combined, enc, "iid"))

    # Per-distribution, Wisconsin, wherever primary sweep data exists
    for enc in ENC_ORDER:
        for dist in ["alpha_0.5", "alpha_0.1", "qty_skew"]:
            if ((wisconsin_df["encoding"] == enc) & (wisconsin_df["distribution"] == dist)).any():
                rows.append(stability_cell("wisconsin", wisconsin_df, enc, dist))

    # Pima IID, amplitude + reuploading (the two encodings with Track-A IID data)
    if not pima_df.empty:
        for enc in ["amplitude", "reuploading", "angle", "iqp"]:
            if ((pima_df["encoding"] == enc) & (pima_df["distribution"] == "iid")).any():
                rows.append(stability_cell("pima", pima_df, enc, "iid"))

    out = pd.DataFrame(rows)
    out["_enc_ord"] = out["encoding"].map({e: i for i, e in enumerate(ENC_ORDER)})
    out["_dist_ord"] = out["distribution"].map({d: i for i, d in enumerate(DIST_ORDER)})
    return out.sort_values(["dataset", "_dist_ord", "_enc_ord"]).drop(columns=["_enc_ord", "_dist_ord"]).reset_index(drop=True)


# ── RQ4a: MIA ────────────────────────────────────────────────────────────────

def build_rq4_mia() -> pd.DataFrame:
    from analysis.current_rq4 import mia_table
    return mia_table()

# ── RQ4b: DP sweep ──────────────────────────────────────────────────────────

def build_rq4_dp() -> pd.DataFrame:
    from analysis.current_rq4 import noise_table
    return noise_table()

# ── output helpers ───────────────────────────────────────────────────────────

def _fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, title: str):
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)

    cols = list(df.columns)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {csv_path.name}  ({len(df)} rows)")


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

    print("\nBuilding Table 1/3 (RQ1 fidelity / RQ3 divergence) ...")
    datasets = {
        "wisconsin": (wisconsin_df, wisconsin_refs),
        "pima": (pima_df, pima_refs),
    }
    rq1_table = build_rq1_rq3(datasets)
    rq3_table = build_rq3_local_global(datasets)
    write_table(rq1_table, out_dir, "table1_rq1_fidelity",
                "Table 1 — RQ1 Fidelity: JSD_cent by Encoding x Distribution x Dataset")
    write_table(rq3_table, out_dir, "table3_rq3_divergence",
                "Table 3 — RQ3 Local-to-Global Divergence: mean_{k,t} JSD(phi_k, phi_g) by Distribution (delta vs IID)")

    print("\nBuilding Table 2 (RQ2 stability) ...")
    rq2_table = build_rq2(wisconsin_df, wisconsin_ext_df, pima_df)
    write_table(rq2_table, out_dir, "table2_rq2_stability",
                "Table 2 — RQ2 Stability: Round-to-Round tau / cosine / L2")

    print("\nBuilding Table 4 (RQ4 MIA) ...")
    rq4_mia_table = build_rq4_mia()
    write_table(rq4_mia_table, out_dir, "table4_rq4_mia",
                "Table 4 — RQ4 Membership Inference: Direct-MIA AUC")

    print("\nBuilding Table 5 (RQ4 DP) ...")
    rq4_dp_table = build_rq4_dp()
    write_table(rq4_dp_table, out_dir, "table5_rq4_dp",
                "Table 5 — RQ4 Calibrated Laplace-Noise Evaluation: Direct-MIA AUC vs epsilon")

    print(f"\nAll tables written to {out_dir}")

    pima_mia_status = rq4_mia_table[(rq4_mia_table["dataset"] == "pima")]["status"].iloc[0]
    if pima_mia_status == "absent":
        print("\nNOTE: Pima/amplitude/IID MIA not yet available — re-run this script "
              "after experiments/rq4_pima_mia_amplitude.py completes to include it "
              "in Table 4.")


if __name__ == "__main__":
    main()
