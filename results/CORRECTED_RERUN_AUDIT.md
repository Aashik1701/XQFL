# XQFL — Corrected Federated Recompute: Audit

**Trigger.** `federation/run_federation.py` assigned client data partitions with
`int(context.node_id) % n_clients`. `context.node_id` is a large random integer,
so this was not a bijection onto `{0,1,2,3}`: **every federated run trained on
only ~3 of the 4 partitions — one dropped, one double-weighted in FedAvg —
nondeterministically per run.** IID results were largely immune (partitions
exchangeable); every non-IID result, RQ3 in full, and the scheduler-stochasticity
methodological contribution were compromised.

**Fix.** Partition id now comes solely from `context.node_config["partition-id"]`
(hard range check). `federation/server.py` asserts the set of trained partitions
equals `{0..n-1}` exactly every round and logs it; any drop/dup/failure raises.
`analysis/aggregate_results.py`: `rq3_view()` (which merely re-sorted `JSD_cent`)
removed and replaced with `build_rq3_local_global()` computing the Methods'
definition, `mean_{k,t} JSD(phi_k^(t), phi_g^(t))`, from the per-client rows.

Pre-fix logs archived to `results/prefix_archive_20260906T194218/` (full snapshot).
Centralized baselines and the classical control were **not** re-run (unaffected)
and were preserved by stripping `phase10_log.csv` / `phase16_pima_log.csv` to
their centralized rows before the federated re-run.

---

## 1. Rerun conditions / seeds

| sweep | new federated runs | seeds |
|---|---|---|
| `phase10` (Wisconsin IID all-enc + amplitude non-IID) | 35 | 5 (IID + non-IID) |
| `rq2_seed_extension` (Wisconsin IID) | 40 | 10 (11–47) → n=15 with phase10 |
| `phase11` (Wisconsin angle/reup/IQP × 3 non-IID) | 45 primary + 48 replication | 5 |
| `phase16` (Pima, 4 enc × 4 dist) | 80 primary + 52 replication | 5 |
| `phase13_mia` (Wisconsin IID direct+shadow MIA) | 20 | 5 |
| `rq4_seed_extension` (Wisconsin IID direct MIA) | 40 | 10 → n=15 with phase13 |
| `rq4_noniid_mia` (Wisconsin α=0.1 + qty MIA) | 40 | 5 |
| `rq4_dp_extension` (Wisconsin IID, 4 ε) | 20 retrains × 4 ε = 80 | 5 |
| `rq4_pima_mia_amplitude` (Pima IID MIA) | 15 | 15 |
| **total** | **≈ 435 federated model trainings** (~26 000 local fits + Q-SHAP) | |

Centralized (40 runs: 20 Wisconsin + 20 Pima) and the classical control (30 runs)
reused unchanged.

Per-encoding constants that were hard-coded from the pre-fix runs are now derived
at import time from the corrected `phase10_log.csv`: `phase11` Trigger-S
`ENCODING_IID_STD`, `rq4_dp_extension` Laplace `SENSITIVITY` (Δf = mean‖φ‖).

## 2. Wall-clock

| stage | productive | notes |
|---|---|---|
| Stage 1 (phase10 → rq2_ext → phase11) | 4.3 h | ran overnight |
| — phase16 first attempt | (6.5 h lost) | crashed: `OSError 28 No space left on device` (host disk 99 % full, unrelated data); no corruption, resumed from checkpoint |
| Stage 1 tail (phase16 resume) | 3.8 h | |
| Stage 2 (phase13 → rq4_seed → rq4_noniid → rq4_dp) | 2.9 h | MIA sweeps ~5× faster than their docstring estimates |
| rq4_pima_mia_amplitude | ~0.5 h | |
| **productive total** | **≈ 12 h** | ≈ 18 h including the crashed attempt |

## 3. Partition-integrity confirmation

- **No partition assertion ever fired** across every rerun log (Stage 1 + Stage 2).
- `[round N] partitions trained: [0, 1, 2, 3]` logged for **every round of every
  federation** (~416 occurrences per round index).
- Log-level check: 144 / 144 Wisconsin explanation cells have all 15 rounds with
  exactly `{0,1,2,3}`; Pima 90 / 106 identical, the other 16 are α=0.1 **ghost
  clients** (0-sample partitions at seeds 1/999 — expected, documented; the guard
  still sees all four partitions, the ghost simply logs no explanation).
- **Zero** short/odd cells.

## 4. Conclusions that SURVIVED

| # | claim | pre-fix | post-fix | verdict |
|---|---|---|---|---|
| S1 | **Federation does not degrade attributions under balanced data** | Wisconsin IID JSD_cent 0.13–0.16 all enc | 0.12–0.15 all enc (within ~1 SE) | **holds — unchanged** |
| S2 | **Amplitude two-dataset fidelity — the paper's headline** | Pima IID JSD_cent 0.207 ± 0.111 | **0.201 ± 0.101** | **holds — essentially identical** |
| S3 | **RQ2 stability two-tier ranking**: amplitude ≈ angle (τ ≈ 0.94) > reuploading ≈ IQP (τ ≈ 0.87) | amp 0.939 / ang 0.933 / reup 0.879 / iqp 0.868 (n=15) | amp 0.936 / ang 0.935 / reup 0.886 / iqp 0.865 | **holds — ranking and two-SE gap intact** |
| S4 | **Angle collapses on Pima** at this circuit size | 0 Track-A all distributions | 0 Track-A all distributions (IID acc ≈ 60.6 %, below majority) | **holds** |
| S5 | **RQ4 MIA ranking**: reuploading highest, angle lowest ≈ chance; only reuploading resolvably above angle | ang 0.545 / amp 0.619 / iqp 0.612 / reup 0.644 | ang 0.533 / amp 0.620 / iqp 0.632 / reup 0.665 | **holds — slightly sharper** |
| S6 | **Angle's MIA is not resolved above chance** at 2 SE | 1.97 SE | 1.60 SE | **holds — more so** |
| S7 | **RQ4 DP: no single ε brings all encodings to near-chance**; ε=1.0 → angle ≈ 0.42, others ≈ 0.55–0.58; near-chance for all needs ε=0.1 | | angle 0.415 / amp 0.562 / reup 0.577 / iqp 0.551 at ε=1.0; all ≈ 0.31–0.33 at ε=0.1 | **holds — curve within ~0.01 everywhere** |
| S8 | **RQ3: moderate label skew (α=0.5) drives the largest local-to-global divergence; quantity skew negligible** | (measured on the wrong metric) | on the **correct** metric: Wisconsin Δ_vs_IID α=0.5 = +0.17–0.18 (angle, amplitude), qty = −0.03 … +0.02 all enc | **holds — and now legitimately measured** |
| S9 | **Quantity skew is the mildest non-IID condition for divergence** | | qty Δ_vs_IID ≈ 0 for every encoding, 0 Track-B | **holds** |
| S10 | **Centralized / classical-control conclusions** (RQ1 circularity check, Pima capacity story) | unaffected | not re-run | **holds by construction** |

## 5. Conclusions that CHANGED

| # | claim | pre-fix | post-fix | change |
|---|---|---|---|---|
| C1 | **Scheduler-stochasticity contribution (methodological contribution #5)**: "actor-level scheduling stochasticity … within-seed accuracy ranges of 22 to 30 percentage points … requiring a formal replication protocol" | 22.81 pp (amp/α=0.5/s7, 4 draws); "16.7–26.3 pp" angle bifurcation | **max within-seed range 6.1 pp** across 24 Wisconsin non-IID multi-draw cells (median 0.9 pp); Pima max 3.2 pp. angle/α=0.5/seed7: 2.6 pp | **The 20–30 pp "attractor-bifurcation regime" was mostly the partition bug.** Residual scheduler noise is ~1–6 pp, consistent with the paper's own ±1.75 pp IID floor. This contribution must be rewritten or withdrawn. |
| C2 | **MN-09 — IQP/seed=7 excluded from all Wisconsin aggregations** ("bimodally unstable across all 4 distributions, 50 % anomaly rate") | seed 7 bimodal 0.63 ↔ 0.90 | seed 7 Wisconsin IID acc **0.868, Track A**, mid-pack; non-IID α=0.1 0.912/0.912/0.904 all Track A | **The seed-7 instability was a bug artifact.** MN-09 exclusion no longer justified; including seed 7 now changes τ / JSD_cent by < 0.005. No MN-09 correction needed anywhere. |
| C3 | **MN-08 — reuploading/α=0.5/seed=999 excluded** ("near-zero ‖φ‖ = 0.073 despite Track A, stochastically persistent near-collapse") | ‖φ‖ ≈ 0.07 | 3 draws, ‖φ‖ = 0.41 / 0.41 / 0.43 (normal), JSD_cent 0.20 / 0.20 / 0.24 ≈ LOO mean 0.206 | **No anomaly.** Exclusion no longer justified. |
| C4 | **MN-08 — amplitude/α=0.5/seed=123 excluded** ("JSD_cent 4.4 σ above leave-one-out mean, reproducible in 3 draws") | 4.4 σ | 1 draw, JSD_cent 0.227 vs LOO 0.187 ± 0.041 = **0.97 σ** | Anomaly largely gone (1 draw; a replication pass would confirm). |
| C5 | **"IQP produces no / near-no functional Pima data"** | Pima IID 1 Track-A seed (insufficient) | Pima IID **3 / 5 Track-A**, JSD_cent 0.258 ± 0.062, τ 0.800; still collapses under α=0.5 and α=0.1 | Downgrade "angle **and IQP** collapse on Pima" → "angle collapses; IQP is marginal on Pima (functional IID only, 3/5)". |
| C6 | **"Re-uploading's Pima α=0.1 divergence is uncharacterizable (2/15 Track-A)"** — a stated limitation | 2 Track-A draws | **7 Track-A**, JSD_cent 0.353 ± 0.041; RQ3 local-global 2/5 (still thin) | Limitation partially lifted for RQ1; RQ3 cell still insufficient. |
| C7 | **RQ4 non-IID: "quantity skew leaks more than α=0.1"** (amplitude +9.0 pp, IQP +4.8 pp) | | amplitude qty 0.643 vs α=0.1 0.573 = **+7.0 pp**; IQP qty 0.700 vs α=0.1 0.701 = **−0.1 pp** | Amplitude effect holds (smaller); **IQP effect disappears** — qty ≈ α=0.1 for IQP now. |
| C8 | **RQ4 non-IID: "IQP leads the quantity-skew encoding mean (0.722)"** | IQP 0.722 (seed-7 corrected) > reup 0.692 | IQP 0.700 **= reuploading 0.700** | IQP no longer clearly leads qty-skew; tied with reuploading. |
| C9 | **RQ4 non-IID: reuploading α=0.1 MIA "0/5 Track-A, absent"** | absent | **4/5 Track-A, AUC 0.651 ± 0.199** | Now measurable. |
| C10 | **Wisconsin non-IID RQ1 fidelity, several cells** | angle α=0.5 JSD_cent 0.210 (n=13); IQP α=0.1 0.190 (n=3) | angle α=0.5 **0.154** (n=9); IQP α=0.1 **0.295** (n=10) | Non-IID JSD_cent numbers moved 0.03–0.10 in several cells and Track-A counts shifted substantially (IQP α=0.1 collapse rate 0.67 → 0.13). Every non-IID row of Table 1 must be re-read from the new outputs. |
| C11 | **Wisconsin amplitude IID FedAvg accuracy** | 94.21 ± 2.20 pp | 95.44 ± 1.56 pp (now above centralized 92.81) | Minor; update the §6.4 reference number. |

## 6. Anomalies that STILL reproduce

- **None of the three documented seed exclusions (MN-08 ×2, MN-09) reproduce** on
  the corrected runs — see C2–C4. They were bug artifacts.
- The **RQ3 α-ordering discrepancy between datasets persists**: Wisconsin shows
  α=0.5 as the peak local-to-global divergence, but Pima amplitude is roughly
  flat (α=0.5 and α=0.1 slightly *below* IID, n=3). Consistent with the pre-fix
  observation that the Wisconsin α-pattern does not cleanly replicate on Pima;
  now on the correct metric and still small-n.
- **Pima IQP is genuinely unstable across distributions** (IID 3/5 Track-A, α=0.5
  0/5, α=0.1 0/5, qty 2/5) — not a seed artifact, a real capacity-boundary
  effect, and now backed by more data than pre-fix.

## 7. Can RQ1–RQ4 be supported from authoritative outputs?

| RQ | status |
|---|---|
| **RQ1 Fidelity** | **Yes.** `table1_rq1_fidelity.csv` regenerated from corrected runs. Headline (amplitude two-dataset, Pima IID JSD 0.201 ± 0.101) intact; Wisconsin IID intact; **all non-IID cells and Track-A counts must be re-quoted from the new table** (C10), and the angle/IQP-Pima wording softened (C5). |
| **RQ2 Stability** | **Yes.** Two-tier τ ranking and 2-SE gap survive essentially unchanged; n=15 IID confirmed for all four encodings. Pima IQP now has a (thin) stability number. |
| **RQ3 Divergence** | **Yes — for the first time.** `table3_rq3_divergence.csv` now holds the Methods-defined `mean_{k,t} JSD(phi_k, phi_g)`, not re-sorted `JSD_cent`. Wisconsin: α=0.5 peak, quantity skew ≈ IID — the qualitative claim is now legitimately supported. Pima amplitude is flat/inconclusive (n=3); report as such. |
| **RQ4 Privacy** | **Mostly.** IID MIA (n=15) and the DP ε-sweep are stable and fully support the pre-fix conclusions (S5–S7). Non-IID MIA conclusions need revision (C7–C9). `rq4_pima_mia_amplitude` re-running now to refresh the one Pima MIA cell. |

**Not re-run (out of scope / superseded):**
`rq3_anomaly_replication` (MN-08 amplitude/α=0.5/seed123 extra draws — now ~1 σ, low value),
`angle_falsification` / `phase12` / `phase9b` (MN-12 dominant-client-purity and
Ray-determinism analyses that feed §7.5's *secondary* claims). Given C1 already
overturns the primary §7.5 contribution, these should be re-scoped rather than
re-run as-is.

---

## Bottom line

The corrected recompute **confirms every load-bearing scientific claim** —
amplitude's two-dataset fidelity, the stability ranking, the MIA ranking with
angle at chance, the DP-per-encoding result, and RQ3's "α=0.5 worst / quantity
skew benign" on a now-correct metric. It **overturns**: (i) methodological
contribution #5 (the 20–30 pp scheduler-stochasticity regime — really the bug),
(ii) all three replication-protocol seed exclusions (MN-08 ×2, MN-09 — bug
artifacts), (iii) several RQ4 non-IID sub-findings (IQP qty-skew leadership,
qty > α=0.1 for IQP), and (iv) the "IQP produces no Pima data" claim. Every
Table-1-through-5 number and the abstract/contributions/conclusion must be
re-quoted from the regenerated `results/outputs/` before the manuscript is
touched.
