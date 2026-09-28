# XQFL — Final Manuscript Audit (post corrected recompute)

Manuscript: `XQFL_paper.md`. Authoritative outputs: `results/outputs/` (regenerated
`analysis/aggregate_results.py` + `analysis/plots.py`). Recompute audit:
`results/CORRECTED_RERUN_AUDIT.md`. Pre-fix archive (historical only):
`results/prefix_archive_20260906T194218/`.

The corrected pipeline also **retires the three pre-fix per-seed anomaly
exclusions** (MN-09 IQP/seed 7; MN-08 re-uploading/α=0.5/seed 999; MN-08
amplitude/α=0.5/seed 123): they did not reproduce once the partition-assignment
defect was fixed. `aggregate_results.py` now applies no exclusions, so every
condition mean uses all seeds, and IQP Wisconsin IID is n = 15.

---

## A. Numerical consistency — Tables 1–5

Every "mean ± std" cell in each manuscript table was matched **verbatim** against
its source CSV by script (`results/outputs/table*.csv`).

| Table | Source | Cells checked | Result |
|---|---|---|---|
| Table 1 (RQ1 JSD_cent) | `table1_rq1_fidelity.csv` | 30 rows (22 "ok" values + 8 "insufficient") | **PASS** — all 22 numeric cells present verbatim; all 8 insufficient rows flagged as such with matching n |
| Table 2 (RQ2 τ) | `table2_rq2_stability.csv` | Wisconsin IID 4 + Pima IID 3 | **PASS** — τ and n match (IQP now n = 15) |
| Table 3 (RQ3 local-to-global JSD) | `table3_rq3_divergence.csv` | 23 "ok" rows + Δ-vs-IID | **PASS** — every mean ± std present verbatim |
| Table 4 (RQ4 MIA) | `table4_rq4_mia.csv` | 5 table rows (4 Wisconsin IID + Pima amplitude); 7 non-IID cells quoted in §7.4 prose | **PASS** — table rows verbatim; §7.4 prose means (0.556 / 0.643 / 0.700 / 0.700 qty; 0.573 / 0.701 / 0.651 α=0.1) all match |
| Table 5 (RQ4 DP) | `table5_rq4_dp.csv` | all 16 cells | **PASS** — all verbatim, IQP now n = 5 |

Prose-only numbers verified against source: RQ2 gaps/SE (amplitude +0.1 pp tied;
re-uploading −4.9 pp / 2.83 SE resolved; IQP −7.7 pp / 2.92 SE resolved); RQ4 IID
gaps-vs-angle (amplitude +8.7 pp / 2.5 SE; IQP +11.4 pp / 2.5 SE; re-uploading
+13.3 pp / 3.2 SE — all resolved) and gaps-vs-chance (angle 1.6 SE not resolved;
amplitude 4.2, IQP 3.6, re-uploading 4.7 SE resolved); DP monotonic 19/20 pairs,
Δf = 0.229 / 0.262 / 0.344 / 0.287; shadow-MIA means 0.455–0.554; §6.4
learning-curve (mean gap −0.030, std 0.072, |gap| 0.045; round-1 −0.21 to −0.29;
round-15 within 0–1.8 pp); §6.4 amplitude IID FedAvg 95.44 ± 1.56, centralized
86.84 % (angle) / 92.81 ± 1.30 % (amplitude); §7.5 within-seed ranges max 6.1 pp
(Wisconsin) / 3.2 pp (Pima); §8.2 classical control 0.186 ± 0.145 (retained),
quantum IID band 0.121–0.152, gaps 1.66 / 0.84 SE.

Figures 2–5 regenerated from the same aggregation functions; Figure 3 now plots
`local_global_jsd_mean` (the RQ3 metric), not `jsd_cent`.

## B. Claim consistency

| Claim | Authoritative evidence | Manuscript location | Status |
|---|---|---|---|
| Federation preserves attributions under balanced data | Wisconsin IID JSD_cent 0.12–0.15 all encodings; qty_skew 0.08–0.20 | Abstract, §7.1, §8.1, §9 | SUPPORTED |
| Amplitude strongest two-dataset fidelity (Pima IID JSD_cent 0.201 ± 0.101) | `table1_rq1_fidelity.csv` pima/amplitude/iid | Abstract, C2, §7.1, §8.1, §9, Table 6 | SUPPORTED |
| Re-uploading two-dataset, higher Pima IID divergence (0.335 ± 0.087) | `table1` pima/reuploading/iid | C2, §7.1, Table 6 | SUPPORTED |
| Angle collapses on Pima at this circuit size (0/5 Track-A everywhere) | `table1` pima/angle/* all insufficient, collapse_rate 1.0 | Abstract, C2, §7.1, §8.3, §8.6, §9, Table 6 | SUPPORTED |
| IQP functional on Pima **only under IID** (3/5 Track-A, JSD 0.257 ± 0.062) | `table1` pima/iqp: iid ok; α0.5/α0.1/qty insufficient | Abstract, C2, §7.1, §8.3, §8.6, Table 6 | SUPPORTED WITH QUALIFICATION (IID-only) — replaces "angle and IQP collapse on Pima" |
| RQ2 two-tier ranking amplitude ≈ angle (τ 0.94) > re-uploading (0.886) ≈ IQP (0.858), n = 15 | `table2` wisconsin/*/iid | C3, §7.2, Table 2, §8.3, Table 6 | SUPPORTED (IQP now n = 15, not n = 14) |
| Kendall's τ four-feature resolution ceiling | §7.2 para 2 (unchanged, methodological) | C3, §3.5, §7.2 | SUPPORTED |
| RQ3 = mean_{k,t} JSD(φ_k, φ_g); α = 0.5 largest Wisconsin local-to-global divergence (+0.17–0.18 angle/amplitude); quantity skew ≈ IID (±0.03) | `table3_rq3_divergence.csv` (new metric) | C5, §1.4, §4.4, §6.5, §7.3, Table 3, §8.1, Table 6 | SUPPORTED — first time on the Methods-defined metric |
| RQ3 Pima amplitude flat / inconclusive at 3 Track-A seeds (does not reproduce Wisconsin ordering) | `table3` pima/amplitude/* (Δ_vs_IID −0.04 to −0.08, n = 3) | C5, §7.3, §8.6, Table 6 | SUPPORTED (reported as a limitation, not a contradiction) |
| RQ4 IID MIA: re-uploading highest (0.665), angle closest to chance (0.533), not resolved above chance (1.6 SE) | `table4` wisconsin/*/iid; SE computed from std | Abstract, C4, §7.4, Table 4, §8.1, §9, Table 6 | SUPPORTED |
| RQ4 IID MIA: amplitude, IQP, re-uploading each resolvably above angle (2.5–3.2 SE) | gap/SE from `table4` | C4, §7.4, Table 4 | SUPPORTED — **changed** from "only re-uploading resolvably above angle" (now all three, at n = 15 with all seeds) |
| RQ4 DP: no single ε brings all encodings to near-chance; ε = 1.0 → angle 0.415, others 0.56–0.59; ε = 0.1 for all four | `table5_rq4_dp.csv` | Abstract, C4, §7.4, Table 5, §8.4, §9, Table 6 | SUPPORTED |
| RQ4 DP mechanism is calibrated Laplace, empirically calibrated Δf, **not certified (ε, δ)** | §6.5b (retained), §8.6 | §6.5b, §7.4, §8.4, §9, captions | SUPPORTED — no "protects / guaranteed protection" language remains |
| RQ4 non-IID: quantity skew leaks more than α = 0.1 for amplitude (+7.0 pp) and re-uploading (+4.9 pp) but **not IQP (−0.1 pp)** | `table4` non-IID cells | §7.4 | SUPPORTED WITH QUALIFICATION — **changed**: IQP effect removed |
| RQ4 non-IID: IQP and re-uploading tied at 0.700 under quantity skew | `table4` qty cells | §7.4, Table 6 | SUPPORTED — **changed** from "IQP leads at 0.722" |
| RQ4 non-IID: re-uploading α = 0.1 MIA now measurable (4/5, 0.651 ± 0.199) | `table4` wisconsin/reuploading/alpha_0.1 | §7.4 | SUPPORTED — **changed** from "0/5 Track-A, absent" |
| Pima amplitude IID MIA 0.564 ± 0.109, above chance, tied with Wisconsin | `table4` pima/amplitude/iid (n = 15) | §7.4, Table 4, §8.6, Table 6 | SUPPORTED |
| 20–30 pp "attractor-bifurcation" scheduler-stochasticity regime | corrected within-seed ranges max 6.1 pp (Wisconsin) / 3.2 pp (Pima) | (was C5, §7.5, §8.6) | **REMOVED** — was a partition-bug artifact; §7.5 now states the corrected ~1–6 pp residual and explicitly disavows the old regime |
| MN-09 IQP/seed 7 bimodal instability / exclusion | seed 7 Wisconsin IID Track-A 0.868, mid-pack; stable in all non-IID conditions | (was §6.6, §7.2, §7.4, §8.3) | **REMOVED** — bug artifact; no exclusion, no correction anywhere |
| MN-08 re-uploading/α = 0.5/seed 999 near-zero ‖φ‖ (0.07) | corrected ‖φ‖ 0.41 / 0.41 / 0.43; JSD_cent ≈ leave-one-out mean | (was §7.3) | **REMOVED** — bug artifact |
| MN-08 amplitude/α = 0.5/seed 123 anomaly (4.4 σ) | corrected 0.97 σ above leave-one-out mean | (was §7.3, Table 1) | **REMOVED** — did not reproduce |
| dominant-client class purity predicts collapse (MN-12) | `angle_falsification` (manual-partition, pre-fix pipeline) not re-run | (was C5, §7.5) | **REMOVED** — evidence rests on the uncorrected pipeline; not reinstated |
| round-0 initialization-vs-training collapse distinction (MN-13) | phase-log-derived, pre-fix | (was §7.5) | **REMOVED** — not re-verified |
| Classical control (RQ1 circularity, Pima capacity) | `rq_classical_baseline_extended.csv` (not federated → unaffected, retained) | §6.4b, §8.2 | SUPPORTED — comparison re-stated against corrected quantum band (1.66 / 0.84 SE, directional, not resolved) |
| Q-SHAP estimator cross-checks (8-vs-24 perm; `shap` KernelExplainer) | centralized-model checks, unaffected | §5.5 | SUPPORTED (retained) |

**Unresolved:** none. Every claim is either supported by a regenerated output,
supported with an explicit qualification, or removed.

## C. Pre-fix contamination check

Full-text scan of `XQFL_paper.md` for the pre-fix numeric fingerprints
(`0.207 ± 0.111`, `0.311`, `0.237`, `0.224`, `0.299 ± 0.181`, `0.343`, `0.329`,
`0.939`, `0.933`, `0.879 ± 0.049`, `0.868`, `0.545 ± 0.088`, `0.619 ± 0.115`,
`0.612 ± 0.171`, `0.644 ± 0.134`, `0.421`, `0.722`, `22.81`, `26.3`, `16.7`,
`4.4 σ`, `0.073`, `n = 14`) and phrases (`MN-0x`/`MN-1x`, `bimodal`,
`attractor-bifurcation` except the corrected disavowal in §7.5,
`scheduler/scheduling stochasticity`, `every encoding leaks`, `protects` /
`guaranteed protection`, `harmless`, `divergence-versus-centralized` as an RQ3
label, `formal table pending`, `two internal tasks`, `actor RNG`,
`replicated before being accepted`):

**No pre-fix federated numerical result remains as current evidence.** The only
retained pre-fix numbers are the classical control (`0.186 ± 0.145`, the 0.49
collapse seed, Pima +5.0 pp / 5-of-15), all explicitly labelled as the
non-federated control (§8.2), and the estimator cross-checks in §5.5
(centralized models). `0.311` and `0.619` still occur but only as **new**
values (DP Table 5 angle ε = 0.1 = 0.311 ± 0.027; RQ3 Table 3
Wisconsin/angle/α = 0.5 = 0.619 ± 0.129).

## D. RQ validation

| RQ | Metric used | Status |
|---|---|---|
| RQ1 Fidelity | final-round JSD(φ_g^T, φ_cent), Track-A | **supported** — Table 1 fully re-quoted; headline (amplitude two-dataset) intact; Pima IQP wording corrected to IID-only |
| RQ2 Stability | mean consecutive-round Kendall's τ of φ_g | **supported** — two-tier ranking and 2-SE gap survive; IQP n = 15 |
| RQ3 Divergence | mean_{k,t} JSD(φ_k^t, φ_g^t) | **supported with Pima limitation** — Wisconsin pattern (α = 0.5 peak, quantity skew ≈ IID) now on the correct metric; Pima amplitude flat and inconclusive at 3 Track-A seeds |
| RQ4 Privacy | 5-fold CV LR MIA AUC; calibrated Laplace ε-sweep | **supported with revised non-IID claims** — IID MIA and DP fully support the conclusions; non-IID: IQP quantity-skew "leadership" and "quantity > α = 0.1 for IQP" removed; re-uploading α = 0.1 now measurable; Pima amplitude IID MIA added |

## E. Anomaly validation

| Anomaly | Pre-fix | Corrected | Disposition |
|---|---|---|---|
| MN-08 re-uploading/α = 0.5/seed 999 | near-zero ‖φ‖ ≈ 0.07, "stochastically persistent near-collapse" | ‖φ‖ 0.41 / 0.41 / 0.43; JSD_cent 0.20 / 0.20 / 0.24 ≈ LOO mean 0.206 | **no longer an anomaly** — seed included |
| MN-08 amplitude/α = 0.5/seed 123 | 4.4 σ above leave-one-out mean | 0.97 σ above leave-one-out mean | **no confirmed anomaly** — seed included |
| MN-09 IQP/seed 7 (Wisconsin) | bimodal 0.63 ↔ 0.90, 50 % anomaly rate | Track-A 0.868, mid-pack; stable across all four distributions | **no longer an anomaly** — seed included, no MIA substitution |

## F. Contribution validation

Contributions rebuilt to five (§1.5):

| # | Contribution | Corrected evidence |
|---|---|---|
| C1 | From-scratch feature-level Q-SHAP pipeline for federated VQCs (per-round global + per-client logging, Track A/B, deterministic aggregation, per-round partition-integrity check) | §5, §6.7 |
| C2 | Two-dataset fidelity result (amplitude lowest, Pima IID JSD_cent 0.201 ± 0.101) | Table 1 |
| C3 | Stability ranking at 15 seeds (amplitude ≈ angle > re-uploading ≈ IQP) | Table 2 |
| C4 | Privacy characterization (encoding-dependent MIA; calibrated Laplace ε-sweep) | Tables 4–5 |
| C5 | Heterogeneity → local-to-global explanation divergence (α = 0.5 peak on Wisconsin; quantity skew ≈ IID; Pima inconclusive) | Table 3 |

**The scheduler-stochasticity contribution is removed.** No contribution about
actor-scheduling variance, attractor bifurcation, or a mandatory replication
protocol remains. The former "failure-aware validation protocol" contribution is
folded into C1 (Track A/B + collapse diagnostics), without the
anomaly-detection-through-repeated-trials framing that rested on the bug-induced
anomalies.

## Quality gate (§26)

- [x] Abstract matches corrected results
- [x] Introduction / RQs match actual analyses (RQ3 question is local-vs-global)
- [x] Methods match implementation (§6.7 documents the partition fix)
- [x] RQ1 uses federated-vs-centralized JSD
- [x] RQ2 uses consecutive-round stability
- [x] RQ3 uses true local-to-global JSD (Table 3 title + §7.3 + Figure 3)
- [x] RQ4 uses corrected MIA / DP results; Pima amplitude MIA (n = 15) included
- [x] Table 1 regenerated and fully re-quoted (all non-IID cells)
- [x] Table 2 corrected (IQP n = 15; gap −4.9 pp)
- [x] Table 3 is the genuine RQ3 metric
- [x] Table 4 includes the final Pima amplitude MIA row
- [x] Table 5 corrected DP values (IQP n = 5)
- [x] Figures 2–5 regenerated from the same aggregation code
- [x] 22–30 pp scheduler claim removed (§7.5, §8.6, C5)
- [x] MN-08 (×2) and MN-09 exclusions removed (§6.6, §7.3, Table 1, §6.5b)
- [x] IQP-Pima wording corrected to "functional only under IID"
- [x] RQ4 non-IID claims corrected (IQP qty-skew leadership; qty > α = 0.1 for IQP)
- [x] Empirical DP wording (no "protects" / "guaranteed protection"; "not a certified (ε, δ) guarantee" retained)
- [x] No "every encoding leaks" overclaim
- [x] No "quantity skew is harmless" overclaim
- [x] No stale pre-fix numbers (Section C)
- [x] No "formal table pending" / internal unfinished-work language (§10, Table 6)
- [x] Conclusion matches evidence
- [x] Future work is genuinely future work (adds mitigation directions; removes internal tasks)
- [x] This audit generated

## Files changed

- `XQFL_paper.md` — Abstract, §1.5 (contributions rebuilt 6 → 5), §4.4/§6.5 (RQ3 wording already correct — verified), §5.4 (trimmed-mean stale numbers removed), §6.4 (learning curve + FedAvg accuracy), §6.5b (Δf), §6.6 (exclusions retired), **§6.7 added** (partition-assignment disclosure), §7.1 + Table 1, §7.2 + Table 2 + Fig 2 caption, §7.3 + Table 3 + Fig 3 caption (full RQ3 rewrite), §7.4 + Tables 4–5 + Figs 4–5 captions, §7.5 (retitled, scheduler claim removed), §8.1, §8.2 (classical comparison re-stated), §8.3, §8.4, §8.6, §9, §10, Appendix Table 6.
- `analysis/aggregate_results.py` — `rq3_view()` removed; `build_rq3_local_global()` / `rq3_local_global_cell()` added; all per-seed exclusions retired (`_exclude_iqp_seed7` gated on a now-`None` constant; `RQ3_EXTRA_EXCLUSIONS` emptied; MN-09 MIA substitution disabled); docstring updated.
- `analysis/plots.py` — Figure 3 now consumes the RQ3 local-to-global table; IQP/seed-7 exclusion removed from Figure 2; captions de-MN-09'd; Figure 5 title verb changed from "DP protection" to "MIA AUC under calibrated Laplace noise".
- `federation/run_federation.py` — partition id from `context.node_config["partition-id"]` + range check.
- `federation/server.py` — per-round assertion that trained partitions == {0..n-1}, with logging.
- `experiments/` — `phase11` `ENCODING_IID_STD` and `rq4_dp_extension` `SENSITIVITY` now auto-derive from the corrected `phase10_log.csv`; orchestrators `rerun_corrected_stage1.py` / `stage2.py` added.
- `results/outputs/table1..5_*.{csv,md}` and `figure2..5_*.png` — regenerated.
- `results/CORRECTED_RERUN_AUDIT.md`, `results/FINAL_MANUSCRIPT_AUDIT.md` — added.

**No `git commit` / `git push` run.**

---

# Phase 8 — Final Publication-Readiness Consistency Gate (2026-09-07, second pass)

Run after the hostile-review pass (`results/FINAL_HOSTILE_REVIEW.md`). No experiments
were re-run; the corrected `results/` logs and `results/outputs/` tables from the
2026-09-07 recompute remain authoritative.

## New findings this pass

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| P8-1 | §6.4 quoted "86.84%" for Wisconsin **angle centralized accuracy**; this is the value for 3 of the 5 seeds, not the 5-seed mean. The adjacent amplitude figure was already a mean ± std. | MINOR (real inconsistency) | Fixed: now "87.37 ± 0.78%" (5-seed mean ± std from `phase10_log.csv`; identical in the pre-fix archive — centralized runs were never affected by the partition bug). |
| P8-2 | RQ1 estimator-noise floor (§5.5, JSD ≈ 0.17) was measured but not applied to §7.1 / §8.6 / Appendix Table 6. | MAJOR (MC-1) | Fixed: floor added to §7.1 (new paragraph), §8.6 (new bullet), Abstract, §1.5 C2, §9, Appendix Table 6 RQ1 cell. RQ1 interpretation restricted to the tight-band observation + the Pima amplitude-vs-re-uploading gap (0.134, clears the floor). |
| P8-3 | RQ3 α=0.5 > α=0.1 non-monotonicity not attributed to Track-A survivorship. | MAJOR (MC-2) | Fixed: §7.3 (new sentences), Abstract, §1.5 C5, Appendix Table 6 RQ3 cell. Claim scoped to "moderate label skew is the most divergent functional condition; no monotone dose-response". |
| P8-4 | Direct-MIA: standardization fit on pooled member/non-member set before CV; member/non-member explanations use train/test backgrounds. | MAJOR construct-validity (MC-3) | Disclosed, not recomputed: §7.4 (new paragraph), §8.6 (new bullet). Ranking and angle-at-chance are robust to both; paper now states direct-MIA AUC is a signal-presence proxy. |
| P8-5 | Collapse threshold is per-test-split, presented as fixed (65.16% / 66.94%). | MINOR (mc-5) | Fixed: §4.3 notes the threshold moves ≈ ±1.5 pp across seeds; quoted values are the seed-42 split. |
| P8-6 | RQ2 ranking rests on a τ that saturates by round 2. | MINOR (mc-4) | Fixed: §7.2 reframes the ranking as dominant-component lock-in speed/cleanliness. |
| P8-7 | n ≤ 3 cells printed with 3-dp means. | MINOR (mc-7) | Fixed: Table 1 and Table 3 captions flag n = 3 cells as indicative only. |
| P8-8 | `aggregate_results.py` / `rq4_dp_extension.py` docstrings described retired exclusions / stale Δf as active. | MINOR (mc-6) | Fixed: both docstrings updated to current behaviour. No logic change — CSVs regenerate byte-identical (verified). |
| P8-9 | The revision brief's Phase-1 "freeze" list gives IQP IID MIA = 0.632 and IQP DP ε=1.0 = 0.551. Authoritative CSVs + raw logs give **0.647** (`phase13` + `rq4_extension`, 15-seed mean 0.64693) and **0.588** (`rq4_dp_extension`, ε=1.0 mean 0.5884). | MINOR (documentation) | **No manuscript action** — the manuscript already follows the CSVs. Flagged so a later pass does not "correct" the paper backwards toward the brief. |
| P8-10 | §6.6 2-SE resolution rule not family-wise controlled across ~30 Table-1 / ~26 Table-3 cells. | MINOR | Disclosed: §6.6 now says per-cell comparisons are not multiple-testing corrected. |

## Numerical consistency — re-verified

- `python analysis/aggregate_results.py` re-run: `table{1..5}_*.csv` regenerate **byte-identical** to the committed versions (docstring-only code edits).
- Automated scan: 95 `X ± Y` pairs in `XQFL_paper.md`; 93 match a `results/outputs/table*.csv` cell to 2–3 dp. The 2 non-table pairs (`92.81 ± 1.30`, `95.44 ± 1.56`) are §6.4 accuracy reference points — verified directly against `phase10_log.csv` (amplitude centralized 5-seed 92.81 ± 1.30; amplitude IID FedAvg R15 15-seed 95.44 ± 1.56).
- Load-bearing spot checks (all PASS): Pima amplitude/re-uploading IID JSD_cent (0.2007 / 0.3349); Wisconsin amplitude/IQP RQ2 τ (0.9363 / 0.8578); Wisconsin angle α=0.5 / amplitude qty RQ3 (0.6190 / 0.2917); Wisconsin IQP IID MIA (0.6469); IQP DP ε=1.0 (0.5884).
- Stale-term grep (`scheduler.stochastic`, `attractor.bifurcat`, `MN-0[89]`, `MN-1[23]`, `0.722`, `every encoding leaks`, `\bprotects\b`, `guaranteed protection`): only match is the §7.5 sentence that explicitly disavows the old regime. Clean.

## RQ metric validity — re-confirmed

- RQ1 (`jsd_cell`): final-round `JSD(φ_g^(15), φ_cent)`, Track-A, on the shared test set. ✔
- RQ2 (`stability_cell`): mean consecutive-round Kendall τ of `φ_g`, Track-A by final-round class, cosine + L2 alongside. ✔
- RQ3 (`rq3_local_global_cell`): `mean` over rounds `t ≥ 1` and clients `k` of `JSD(φ_k^(t), φ_g^(t))`, per-seed then mean-of-means, Track-A only, round 0 skipped. This is the §4.4 metric and is distinct from RQ1. `rq3_view()` (the RQ1-resort) is gone. ✔
- RQ4 (`build_rq4_mia` / `build_rq4_dp`): 5-fold CV LR AUC on 50 + 50 `φ` vectors; DP = Laplace(0, Δf/ε) with Δf = runtime-derived mean ‖φ‖. ✔

## Federated validity — re-confirmed in source

- `run_federation.py:227` — `partition_id = int(context.node_config["partition-id"])`, `KeyError`/`TypeError`/`ValueError` → `RuntimeError`; range check `0 ≤ partition_id < n_clients`.
- `server.py:39` — `DeterministicFedAvg.aggregate_fit` collects `client_id`s, prints them, raises unless `== list(range(min_fit_clients))`. Live on every logged run (`LoggingFedAvg` → `super()` chain).
- `CORRECTED_RERUN_AUDIT.md`: the assertion never fired across the full recompute.

## PDF

- `XQFL_paper.pdf` rebuilt by `build_pdf.sh` (pandoc 3.11 → tectonic, `fix_approx.lua`, STIX Two Text/Math). 29 pages, no missing-glyph warnings.
- Removed the duplicate H1 title and 5 redundant manual `*Figure N.*` caption paragraphs (pandoc now emits one numbered caption per figure from the image alt text). `numbersections: false` (headings carry manual numbers).
- Visually inspected pp. 1–2 (title/ToC/abstract), 4 (RQs/contributions), 10 (Fig 1), 13 (methods), 15–20 (Tables 1–5, Figs 2–3), 27–29 (Appendix Table 6 + references). All tables and figures render; Greek/math symbols correct.
- Known cosmetic residue (not blocking): `figure1_architecture.png` has overlapping tier/box labels (defect in `analysis/gen_figure1.py`, not the PDF); Appendix Table 6's IQP column wraps heavily; the manuscript uses plain-text math notation (`φ_g^(t)`, `JSD_cent`) rather than typeset math — consistent and legible but a full LaTeX-math pass is a separate task.

## Reproducibility package

- `README.md` — rewritten from a 7-byte stub; answers the 12 required questions, maps the repo layout.
- `reproducibility/README.md` — Tier A (rebuild tables/figures from committed logs, seconds), Tier B (rebuild PDF), Tier C (regenerate all raw logs, ~12 h, via the two stage orchestrators). Per-table source-CSV map; determinism notes; what-was-corrected note.
- `references.bib` — 37 entries mirroring the manuscript's hand-numbered list (`note = {[n]}`), for mechanical checking / a future BibLaTeX build.
- `build_pdf.sh` — committed.
- `requirements.txt` — already a full pinned freeze (`flwr==1.32.1` etc.); no change needed.

## Verdict

**READY WITH MINOR FIXES.** No result was overturned or recomputed. All Phase-8
manuscript changes are added caveats, scope corrections, and one accuracy-figure
fix (P8-1). Remaining editorial items are the plain-text-math typesetting decision
and the Figure 1 regeneration — both optional, neither affecting a claim.

**No `git commit` / `git push` run.**

---

# Phase 9 — Mathematical Formulation Pass (2026-09-07, third pass)

Formalization only: no experiment re-run, no numerical result changed. `analysis/aggregate_results.py`
re-run → `table{1..5}_*.csv` byte-identical.

## Equations added (12, numbered in reading order)

| # | Where | Object |
|---|-------|--------|
| 1 | §3.1 | Sample-weighted FedAvg aggregation |
| 2 | §3.4 | Q-SHAP value function (mean-imputation from background) |
| 3 | §3.4 | Exact Shapley value + permutation-sampling estimator ($M=8$) |
| 4 | §3.5 | Attribution magnitude normalization ($\delta=10^{-10}$) |
| 5 | §3.5 | Base-2 Jensen–Shannon **distance** (the metric the code computes) |
| 6 | §4.1 | Explanation operator $\Phi$ and the sample-averaged $\phi_g^{(t)},\phi_k^{(t)},\phi_{\mathrm{cent}}$ |
| 7 | §4.3 | Track A / Track B indicator ($\mathrm{Acc}(f_{\theta_g^{(T)}}) > \rho_{\mathrm{maj}}+0.02$) |
| 8 | §4.4 | RQ1 metric $\mathrm{JSD}_{\mathrm{cent}}=\mathrm{JSD}(\phi_g^{(T)},\phi_{\mathrm{cent}})$ (boxed) |
| 9 | §4.4 | RQ2 metric $\tau_{\mathrm{stab}}=\frac1T\sum_{t=1}^{T}\tau(\phi_g^{(t-1)},\phi_g^{(t)})$ (boxed) |
| 10 | §4.4 | RQ3 metric $D_{\mathrm{LG}}=\mathrm{mean}_{s\in\mathcal A}[\frac{1}{|\mathcal P_s|}\sum_{(t,k)}\mathrm{JSD}(\phi_{k,s}^{(t)},\phi_{g,s}^{(t)})]$ (boxed) |
| 11 | §6.5 | Direct-MIA AUC (5-fold CV logistic attack on $\hat\phi$) |
| 12 | §6.5b | Laplace release mechanism + empirical scale $\Delta f_{\mathrm{emp}}$ |

Equations before this pass: 0 numbered (all methodology in prose). After: 12.

## Sections modified

§3.1, §3.2, §3.3, §3.4, §3.5, §4.1, §4.3, §4.4, §5.5, §6.5, §6.5b, §7.2, §7.3, §7.4 (equation refs).
No results/discussion/conclusion/appendix content changed beyond adding "(Eq. N)" pointers in §7.2–§7.4.

## Notation changes (notation-only, no result change)

- Global parameters: `θ^(t)` → `$\theta_g^{(t)}$` (subscript $g$) everywhere in §3–§4; client `$\theta_k^{(t)}$`, centralized `$\theta_{\mathrm{cent}}$`.
- Data-embedding unitary renamed `S(x)` → `$U_{\mathrm{enc}}(x)$` (§3.2, §4.1) so `$S$` is free for Shapley coalitions `$S\subseteq F$`.
- JSD stabilizer renamed `ε = 1e-10` → `$\delta = 10^{-10}$` (§3.5, §5.5) to free `$\epsilon$` for the DP budget.
- Model `f(·,θ_k)` → `$f_{\theta_k}$`; feature set `$F=\{1,2,3,4\}$`, `$|F|=4$`; total samples `$N=\sum_k n_k$`; `$T=15$` rounds.
- `$K, n_k, E, M, B, \bar b, X_{\mathrm{expl}}, \mathcal A, \mathcal P_s, \rho_{\mathrm{maj}}, \Delta f_{\mathrm{emp}}$` introduced and used consistently.
- Isolated plain-unicode `θ`, `2^q` in §3.2–§3.3 converted to inline math.

## Implementation-vs-formula inconsistency discovered

**The reported "JSD" is the Jensen–Shannon _distance_, not the divergence.** `explain/metrics.py::js_divergence`
calls `scipy.spatial.distance.jensenshannon(p, q, base=2)`, which returns
$\sqrt{\tfrac12 D_{\mathrm{KL}}(p\|m)+\tfrac12 D_{\mathrm{KL}}(p'\|m)}$ — the square root of the divergence.
The pre-pass prose (§3.5) described it as "the symmetrized, bounded form of KL divergence", which reads as
the divergence itself.

- **Classification:** notation/definition only. `js_divergence` is the single chokepoint for every JSD in the
  paper (RQ1 `JSD_cent`, RQ3 `D_LG`, the §5.5 estimator cross-checks, the §7.1 estimator floor). All of them
  are this distance. **No reported number changes**; the metric is internally consistent, just mislabeled.
- **Fix:** Eq. (5) now defines the quantity with the explicit square root and states it is
  `scipy.spatial.distance.jensenshannon(·,·,base=2)`, the JS distance; a sentence in §3.5 says every JSD value
  in the paper (incl. `JSD_cent`, `D_LG`) is this distance. The `[0,1]` bound and "0 iff identical" both still
  hold, so no downstream claim is affected.

Two further places where I followed the code over the prompt's generic template (not inconsistencies in the paper):
- RQ2 Eq. (9): the code's `stability_cell` compares rounds $0\!\to\!1,\dots,14\!\to\!15$ (15 pairs, **includes**
  the init→round-1 step); Eq. (9) uses $\frac1T\sum_{t=1}^{T}$ accordingly, not the template's $\frac{1}{T-1}\sum_{t=2}^{T}$.
- RQ3 Eq. (10): the code iterates rounds $t=1,\dots,15$ (round 0 skipped, no per-client vectors there); Eq. (10)
  uses $1\le t\le T$, not the template's $t=1,\dots,T-1$.

## Confirmations

- No numerical result, mean, std, Track-A count, seed set, architecture value, or RQ definition changed.
- No new experiment, dataset, or run introduced.
- No theoretical guarantee, bound, convergence/privacy proof, or statistical test introduced.
- Withdrawn scheduler-stochasticity claim not reintroduced; retired seed-anomaly claims not reintroduced;
  RQ3 metric remains the local-to-global JSD (Eq. 10), not RQ1's `JSD_cent`.
- DP: Eq. (12) explicitly labels $\Delta f_{\mathrm{emp}}$ non-certified and states no $(\epsilon,\delta)$-DP is claimed.
- §5.5 estimator-noise floor (JSD $\approx 0.17$) preserved as an empirical reference, not reinterpreted as a bound.
- PDF rebuilt (`build_pdf.sh`), 31 pages, all 12 equations render, no missing-glyph / undefined-control-sequence warnings.

## Remaining mathematical weakness (for later, optional)

- `\boldsymbol{\phi}` renders as plain italic $\phi$ under STIX Two Math (no bold-italic phi); vectors are visually
  identical to scalars. Cosmetic; a `\vec` or `\mathbf` convention could be adopted paper-wide.
- Eq. (3) and Eq. (12) are wide; they fit the text block but a two-line `aligned` split would be cleaner.
- The per-seed vs. seed-averaged distinction for Eqs. (8)/(9) is handled with one sentence in §4.4 rather than
  by writing the outer $\mathrm{mean}_{s\in\mathcal A}$ into every boxed metric (done only for Eq. 10, where
  ghost-client bookkeeping makes it necessary).

**No `git commit` / `git push` run.**

---

# Phase 10 — Publication Figure Overhaul (2026-09-07, fourth pass)

Visualization only. No numerical result, metric, RQ definition, Track-A rule, seed, or
claim changed. `python analysis/aggregate_results.py` re-run after the figure work →
`table{1..5}_*.csv` **byte-identical**. Archived pre-fix logs were not read.

## Figures

Final count: **6** (was 5). Figure 6 added: an evidential-coverage matrix
(dataset × encoding × regime → distinct Track-A seed count), placed in §6.6 and
therefore numbered **Figure 2** in document order; the four RQ result figures shift to
Figures 3–6. Rationale for adding it: the design is 2 datasets × 4 encodings × 4
regimes × Track-A filtering, currently legible only by cross-reading Tables 1 and 3;
one dot-matrix makes "what has enough functional evidence" readable at a glance, and
makes explicit that every gap is model collapse, not a skipped run. It is a
coverage map, not a reproduced table.

All six figures rebuilt under one visual system (`analysis/plots.py` rewrite +
`analysis/gen_figure1.py` rewrite): fixed per-encoding identity
(Angle = orange/circle/solid, Amplitude = blue/square/solid,
Re-uploading = green/triangle/dashed, IQP = reddish-purple/diamond/dotted — Okabe–Ito,
colour + marker + line style so the series survive greyscale), consistent labels
("Angle / Amplitude / Re-uploading / IQP"; "IID / α = 0.5 / α = 0.1 / Quantity skew";
"Wisconsin / Pima"), shared typography, chance line style, and error convention
(mean ± std, matching the tables). Output is now vector **PDF** (embedded) + 300-dpi PNG.
IQP was recoloured vermilion → reddish-purple this pass for better separation from
Angle's orange in the bar figures.

| Fig | Redesigned? | What changed |
|---|---|---|
| 1 Framework | Yes (from scratch) | Was an "eight-tier" stack with overlapping tier/box labels. Now a three-zone diagram — LOCAL (shared PCA → per-client encode/VQC/Q-SHAP), SERVER (FedAvg → global model + Track A/B gate → global Q-SHAP; parallel centralized model → φ_cent), ANALYSIS (each RQ written as the vector pair it compares). Zone labels rotated on the margin (no overlap). Adds the centralized RQ1 reference path, which the old figure omitted. §5.1 prose rewritten from "eight tiers" to the three zones; "Tier 6" ref in §6.5b updated. |
| 2 Coverage | New | Track-A seed count per cell; ● ≥5 / amber 3–4 / small red 1–2 / × all collapsed, with the digit shown. From `table3_rq3_divergence.csv` `n_track_a_seeds`. |
| 3 RQ2 stability | Yes | Was one saturated line plot titled "everything is stable". Now two panels: mean τ vs round with the rounds 1–3 lock-in region shaded and annotated, plus a point-estimate ± std summary (dot, not bar, to avoid a truncated-baseline length illusion) with an explicit "Amplitude vs Angle: not resolved / vs Re-uploading, IQP: resolved" bracket. No significance stars. |
| 4 RQ3 divergence | Yes | Regimes grouped **balanced ratio (IID, Quantity skew) | label skew (α=0.5, α=0.1)** with a visual gap, so the x-axis carries no IID→α=0.5→α=0.1 severity gradient. Track-A seed count printed above every bar; n<5 bars pale/hatched (indicative); an open ▽ at the axis for <3 Track-A seeds (never a zero bar). Caption states α=0.1 is not an ordered extension of α=0.5 and carries ghost-client degeneracy. |
| 5 RQ4 MIA | Yes | Prominent dashed chance line labelled "chance (AUC = 0.5)"; bars ordered by mean AUC; error bars = ±std (Table 4 convention, not invented CIs); "not resolved above chance" annotation on Angle; title/caption say "membership-inference signal" / "signal-presence proxy", never "attacker success". |
| 6 RQ4 DP sweep | Yes | x-axis is now **categorical** (ε ∈ {0.1, 0.5, 1.0, 5.0}, evenly spaced, "not to scale" stated) instead of log — only four points were tested. ±std bands. Chance line. Caption: Δf_emp is an empirical scale, not a certified bound; no (ε, δ)-DP claimed. No "protects"/"guaranteed". |

## Data provenance (Figures 3–6; Figure 2 is coverage)

| Figure | Script → source | Metric | Filtering | n |
|---|---|---|---|---|
| 2 Coverage | `plots.py::make_figure6_coverage` ← `build_rq3_local_global` ← `phase10/phase11/phase16` logs | count of distinct Track-A seeds | final-round global Track A | 15 (IID) / 5+ elsewhere; the plotted digit is the surviving count |
| 3 RQ2 stability | `plots.py::make_figure2` ← `phase10_log.csv` + `rq2_extension_log.csv`; summary via `build_rq2` | per-round Kendall τ of φ_g; mean τ_stab (Eq. 9) | Wisconsin, IID, `client_id=global`, final-round Track A | 15 |
| 4 RQ3 divergence | `plots.py::make_figure3` ← `build_rq3_local_global` ← `rq3_local_global_cell` | D_LG = mean_{k,t} JSD(φ_k, φ_g) (Eq. 10), round 0 excluded, same-round client vs global | Track-A seeds; `status==ok` (≥3) plotted, else ▽ | 3–5 per cell (digit shown); IID Wisconsin 5 |
| 5 RQ4 MIA | `plots.py::make_figure4` ← `build_rq4_mia` ← `phase13_mia_log.csv` + `rq4_extension_log.csv` + `rq4_pima_mia_amplitude_log.csv` | direct-MIA AUC (Eq. 11), mean ± std | Wisconsin IID (+ Pima amplitude) | 15 |
| 6 RQ4 DP | `plots.py::make_figure5` ← `build_rq4_dp` ← `rq4_dp_extension_log.csv` | direct-MIA AUC after Laplace(0, Δf_emp/ε) (Eq. 12), mean ± std | Wisconsin IID | 5 per (encoding, ε) |

**Figure 4 RQ3 check (mandatory):** confirmed it consumes `build_rq3_local_global` →
`rq3_local_global_cell`, which computes `js_divergence(client_vec, global_vec)` for the
same round, Track-A seeds, round 0 skipped — the corrected local-to-global metric, not
RQ1's `JSD_cent`.

## Scientific validation

- No numerical result, mean, std, Track-A count, Track-B classification, seed, RQ
  definition, or confidence grade changed; tables regenerate byte-identical.
- No experiment added; no archived pre-fix result used.
- Withdrawn scheduler-stochasticity / attractor-bifurcation claim not reintroduced
  (the only surviving mention is the §7.5 sentence that disavows it).
- No figure implies a monotone skew→divergence relationship (Figure 4 grouping);
  no figure implies formal DP (Figure 6 wording); no figure implies realistic
  attacker success (Figure 5 wording); insufficient cells are marked, never zero.

## Visual QA (manuscript PDF, 33 pp, rebuilt)

- Figure 1 label overlap fixed; three zones clearly separated by rotated margin
  labels; client→server weight flow and Q-SHAP placement explicit; RQ mapping shown.
- All six figures render as embedded vector PDF; captions are single "Figure N:";
  numbering is sequential and every in-text reference (Figures 1, 2, 4) matches.
- No missing glyphs / undefined control sequences in the build.
- Figure caption notation aligned with the body ($\alpha$, $\tau_{\mathrm{stab}}$,
  $D_{\mathrm{LG}}$, $\Delta f_{\mathrm{emp}}$, $(\epsilon,\delta)$).

## Remaining figure weaknesses (minor, non-blocking)

- Figure 1: the per-client weight arrows visually pass through the local-Q-SHAP boxes
  (arrowhead touches the φ_k label) at two columns; the "client weights → FedAvg"
  label sits close to the VQC row. Legible but not perfectly clean.
- Figure 1 uses a sans-serif face (DejaVu Sans) while the body is serif (STIX) — a
  standard journal convention (figures sans, text serif), kept deliberately.
- Figure 6 (DP): the ±std bands for amplitude / re-uploading / IQP overlap heavily
  because those three encodings' means nearly coincide in the data; the mean lines
  are still separable.
- Figure 2 (coverage) floats to the page after its §6.6 reference when it does not
  fit at the foot of the reference page — normal LaTeX float behaviour, adjacent.

**No `git commit` / `git push` run.**

---

# Phase 11 — Final Hostile Hardening Pass (2026-09-07, fifth pass)

Consistency + wording only. No experiment, dataset, seed, RQ, Track-A/B rule, metric,
or numerical result changed. `python analysis/aggregate_results.py` re-run → all five
`table*.csv` **byte-identical**; manuscript table cells unchanged (spot-checked).

## Changes made

| # | File / location | Change | Reason |
|---|---|---|---|
| 1 | `XQFL_paper.md` Abstract, §1.5 C2, §7.1, §8.6, §9, Appendix Table 6 (6 spots) | "the one fidelity gap that **clears / exceeds** the estimator-noise floor" → "the one between-encoding fidelity gap this study **resolves at two standard errors**, though **no larger than** the estimator's ≈0.17 cross-implementation noise". | The gap is 0.134 < 0.17; a raw difference smaller than the floor cannot "clear" it. The gap *is* resolved against sampling uncertainty (≈2.3 SE, the paper's own bar), so that is the defensible statement. (Prompt §11.) |
| 2 | Figure files `results/outputs/figure{2..6}_*` | Renamed so the file number = the figure's displayed number: `figure2_experimental_coverage` (was `figure6_*`), `figure3_stability_over_rounds` (was `figure2_*`), `figure4_divergence_by_distribution` (was `figure3_*`), `figure5_mia_auc_comparison` (was `figure4_*`), `figure6_dp_auc_vs_epsilon` (was `figure5_*`). Updated `analysis/plots.py` `_save` stems + header docstring, the 6 `![](path)` in `XQFL_paper.md`, and `README.md`. | Figure 6 (coverage) was inserted before the RQ result figures, so pandoc's document-order numbering (correct in the PDF) no longer matched the filenames. A maintainer opening `figure2_stability_*` and finding it is "Figure 3" is a real wart. |
| 3 | `XQFL_paper.md` §5.1 | "(Eqs. 8–11)" → "(Eqs. 8–12)". | RQ4 is formalized in Eqs. 11 **and** 12. |
| 4 | `XQFL_paper.md` §3.5 | "the **divergence** reported throughout is the base-2 Jensen–Shannon **distance**" → "the **quantity** reported throughout as \"JSD\" is the base-2 Jensen–Shannon distance". | "the divergence … is the distance" was self-contradictory; the metric is the JS distance (√ of the divergence, as returned by `scipy.spatial.distance.jensenshannon`). |
| 5 | `XQFL_paper.md` §8.1 | "Amplitude encoding keeps its attributions **closest** to the centralized reference across two datasets and stable across rounds." → "Amplitude encoding **has the lowest two-dataset federated-vs-centralized divergence** and is **among the most stable** across rounds (tied with angle)." | "closest" is a superlative not resolved on Wisconsin; "among the most stable (tied with angle)" matches Table 2. (Prompt §11–12.) |
| 6 | `XQFL_paper.md` §8.6 | "Re-uploading's Pima local-to-global divergence at α=0.1 is **uncharacterizable**" → "Re-uploading's Pima local-to-global divergence **(RQ3)** at α = 0.1 **is not estimated** … its RQ1 fidelity at α = 0.1 does reach seven Track-A draws and is reported." | Scope the "cannot say" strictly to the RQ3 metric (2/5 Track-A, below the 3-seed minimum); RQ1 at that cell *is* characterized. (Prompt §17.) |
| 7 | `XQFL_paper.md` §7.2, §7.4 | Added "Figure 3 shows the per-round τ trajectories …"; "(Table 4, Figure 5)"; "(Table 5, Figure 6)". | Every RQ result figure is now referenced from its Results subsection. (Prompt §28.) |
| 8 | `XQFL_paper.md` prose + figure captions | Normalized "α=0.5 / α=0.1" → "α = 0.5 / α = 0.1" in prose and captions (3 lines); table data cells keep the compact form. | Terminology consistency. (Prompt §25.) |
| 9 | `analysis/gen_figure1.py` | Weight-flow cleanup: the weights bus moved to just below the VQC row with short blue stubs from each VQC; a separate grey arrow now carries each model down to its local Q-SHAP box (no arrow through the φ_k boxes); the "client weights θ_k" label moved off the VQC row into the open band above FedAvg. Q-SHAP boxes nudged down 0.1 for spacing. | The one remaining Figure 1 visual collision. Precision cleanup, not a redesign. (Prompt §29.) |
| 10 | `README.md` | "Figures 2–5" → "Figures 2–6"; figure glob → `figure{2..6}_*.{pdf,png}`; the §12 limitation "clears it" wording aligned with change #1. | Documentation consistency. |

Note: two lines in the Phase-D section of this file ("Figures 2–5 regenerated …")
are historical records of a five-figure draft and are left as written; the current
figure set and numbering are as in Phase 10 + change #2 above.

## RQ3 verification (mandatory, re-done)

`analysis/aggregate_results.py::rq3_local_global_cell`: for each Track-A-final seed it
takes the per-round global vector `gvecs[rnd]` and each logged client vector
`clients_by_round[rnd][cid]`, skips `rnd == 0`, and calls
`js_divergence(cvec, gvec)`; per-seed mean, then mean over Track-A seeds.
`explain/metrics.py::js_divergence` = `scipy.spatial.distance.jensenshannon(p, q, base=2)`
on the |·|-normalized (+δ) vectors — the **Jensen–Shannon distance**. Client vectors
are logged pre-aggregation by `LoggingFedAvg.aggregate_fit` before `super().aggregate_fit`.
This is `D_LG` (Eq. 10), **not** `JSD_cent` (Eq. 8). Confirmed.

## Equation audit (all 12 vs implementation)

| Eq | Object | Matches code? |
|---|---|---|
| 1 | Sample-weighted FedAvg | ✓ `DeterministicFedAvg` / `FedAvg` weighted mean |
| 2 | Q-SHAP value function, mean-imputation | ✓ `_subset_predict` in `q_shap.py` |
| 3 | Exact Shapley + permutation estimator, M = 8 | ✓ `q_shap_values`, `N_PERMUTATIONS = 8` |
| 4 | Magnitude normalization, δ = 1e-10 | ✓ `js_divergence` `epsilon=1e-10` |
| 5 | Base-2 JS **distance** (√ form) | ✓ `scipy...jensenshannon(base=2)` |
| 6 | Φ = mean of per-sample Q-SHAP over Xexpl | ✓ `qshap.mean(axis=0)`, `n_global_explain=10`, `n_global_bg=20` |
| 7 | Track A = 1[Acc(f_θg(T)) > ρ_maj + 0.02] | ✓ `phase10…:99` `bincount(y_test).max()/len + 0.02`; final-round class used for filtering |
| 8 | RQ1 JSD_cent = JSD(φ_g^(T), φ_cent) | ✓ `jsd_cell` final-round global vs centralized |
| 9 | RQ2 τ_stab = (1/T)Σ_{t=1..T} τ(φ_g^(t-1), φ_g^(t)), incl. init→round-1 | ✓ `stability_cell` iterates all 16 logged global rounds → 15 consecutive pairs |
| 10 | RQ3 D_LG, rounds 1..T, round 0 excluded | ✓ `rq3_local_global_cell`, `if rnd == 0: continue` |
| 11 | AUC_MIA, 5-fold CV logistic on φ, 50+50 | ✓ `run_mia_attack` `cv=5`, `phase13…` `N_MIA_SAMPLES=50` |
| 12 | Laplace(0, Δf_emp/ε) per component; Δf_emp = mean‖φ_g^(t)‖ over IID Track-A rounds ≥1 | ✓ `dp_noise.add_dp_noise` `scale = s/ε`; `rq4_dp_extension._compute_sensitivity` |

## Historical-contamination sweep (manuscript)

Absent: `MN-08/09/12/13`, `dominant-client purity`, `0.722`, `eight-tier` / `Tier N`,
"scheduler stochasticity" as a contribution, "attractor bifurcation" except the §7.5
disavowal, `seed 999` / `seed 123` except the §6.6 "did not reproduce" disclosure.
Present and intentional: §6.6 lists the three retired seed exclusions as *not
reproduced*; §7.5 discloses that the "20–30 point" within-seed regime was a
partition-bug artifact; §7.4 reports "re-uploading, seed 7, quantity skew = 0.916" as
"a seed-level observation rather than an encoding-level ranking". All three are
transparency disclosures, not surviving claims.

## Verdict

**READY TO FREEZE.** No numerical result, RQ, Track-A/B rule, or scientific conclusion
was touched. All changes are wording precision, figure-file naming, one Figure 1
arrow-routing cleanup, and documentation. The "clears the floor" overstatement — the
one genuine wording defect a hostile reviewer would catch on RQ1 — is corrected in all
six locations.

**No `git commit` / `git push` run.**

---

# Phase 12 — Figure 1 replaced with the hand-designed diagram (2026-09-08)

The user supplied a polished, hand-designed Figure 1 (`XQFL Architecture.png`).
It was verified box-by-box against the manuscript and adopted in place of the
matplotlib schematic. No experiment, result, table, or equation changed.

## Verification of the supplied diagram (all accurate)

| Element in the diagram | Matches |
|---|---|
| DATA: Wisconsin 569×30, Pima 768×8 → Shared PCA→4, MinMax [0,π], "fit once on the server, broadcast unchanged" → "common 4-D space" → "Partition into 4 client shards: IID · Quantity skew · α = 0.5 · α = 0.1" | §6.1, §5.2, §4.2 |
| LOCAL: Client 0–3, each Encoding (angle/amplitude/re-uploading/IQP) → VQC 4 qubits, 2 layers → local Q-SHAP φ_0^(t)…φ_3^(t) | §5.3, §6.3, Eq. 6 |
| Weight flow (blue) VQCs → "client weights ϑ_k^(t) → FedAvg" | Eq. 1 |
| SERVER: FedAvg (Eq. 1) "sample-weighted mean of client weights" → Global VQC ϑ_g^(t) "Track A / B gate (Eq. 7)" → global Q-SHAP φ_g^(t) "every round t = 1…15" | §5.4, Eq. 7, §5.1 |
| Centralized model, pooled data · 45 epochs → reference φ_cent; "not federated" / grey dashed from Shared PCA | §6.4, §6.4b (same shared PCA, not federated) |
| Orange dotted "local φ_k(t) logged for all clients, never aggregated" | §5.1 ("only weights cross… local explanations logged for offline analysis") |
| ANALYSIS cards: RQ1 φ_g^(T)↕φ_cent JSD (Eq. 8); RQ2 φ_g^(t)↕φ_g^(t−1) Kendall τ (Eq. 9); RQ3 φ_k^(t)↕φ_g^(t) JSD (Eq. 10); RQ4 φ(member)↕φ(non-member) MIA AUC (Eq. 11) | §4.4, Eqs. 8–11 |
| Legend: weight/explanation/logged/data flows; "φ — Q-SHAP vector over the 4 PCA features, estimator Eqs. 2–6"; "T = 15" | Eqs. 2–6, §3.1 |
| Bottom line: "same shared PCA(4) space with the same Q-SHAP estimator (Eqs. 2–6): comparability is established before any research-question comparison is made" | thesis, §4.4 |

**One notation mismatch, handled:** the diagram writes the model-parameter vector
as ϑ (vartheta); the manuscript uses θ. Same Greek letter, different typographic
variant. The caption now states "The figure writes the weight symbol as ϑ; it is
the model-parameter vector θ of the text."

**One routing nuance, handled in the caption:** RQ4's member/non-member vectors
are per-sample Q-SHAP from the *global* model, not the per-client logged vectors.
The RQ cards state their own operands, and the caption now spells out each RQ's
input pair and adds "the calibrated-noise sweep of Eq. 12 is applied to these
[RQ4] vectors."

## Changes made

| File | Change |
|---|---|
| `paper_assets/figure1_architecture_source.png` | **new** — master copy of the supplied diagram |
| `results/outputs/figure1_architecture.png` | replaced: the master with its baked-in "Figure 1. …" title band cropped off (top 92 px) + a small white margin; 300 dpi, 2687×1378 |
| `results/outputs/figure1_architecture.pdf` | **deleted** — the stale matplotlib version |
| `analysis/gen_figure1.py` | rewritten: was a matplotlib schematic; now crops the `paper_assets/` master → `results/outputs/figure1_architecture.png` (keeps `python analysis/gen_figure1.py` as a deterministic repro step) |
| `XQFL_paper.md` §5.1 prose | "three zones" → "four zones (data, local, server, analysis)"; wording aligned to the diagram (per-client φ_k^(t), φ_cent on the same PCA space, per-client explanations "logged … but never aggregated") |
| `XQFL_paper.md` Figure 1 caption | rewritten to the four-zone diagram: DATA/LOCAL/SERVER/ANALYSIS contents, the four RQ operand pairs with equation numbers, the four arrow colours, the Eq. 12 note, and the ϑ = θ clarification; image path `…figure1_architecture.pdf` → `.png` |
| `README.md`, `reproducibility/README.md` | Figure 1 described as a hand-designed schematic prepared by `gen_figure1.py` (crop of the `paper_assets/` master) |

## Verification

- Tables 1–5 regenerate **byte-identical** (`aggregate_results.py` re-run).
- PDF rebuilt: 34 pages, figure captions 1–6 sequential, no missing-glyph / undefined-control-sequence warnings.
- Figure 1 renders at full text width with a single pandoc "Figure 1:" caption (no title duplication); internal text legible on screen and in print (≈410 ppi at text width); the four zones, legend, and flow colours are clear.
- No `.pdf` reference to `figure1_architecture` remains anywhere.

**No `git commit` / `git push` run.**
