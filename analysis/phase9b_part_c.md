# Phase 9b Part C — Non-IID Distribution Robustness and Aggregation Strategy

**Governing framework:** MN-02 (noise floor ±1.75pp), MN-03a (encoding×partition interaction),
MN-04 (α=0.1 scope limitation, collapse predictor, behavioral taxonomies), MN-05 (per-condition
reporting mode), MN-08 (actor-RNG-determined outcomes near attractor boundaries — applies to
Wisconsin/α=0.5/seed=7 and all α=0.1 attractor cases on **both** Wisconsin and Pima; see §4.1
for the unified account superseding any per-dataset "distinct mechanism" framing below). All
five-seed means carry ≥3pp meaningful-difference threshold.

---

## Overview

Part C evaluates amplitude encoding — the strongest performer in Parts A and B — under
three non-IID data distributions and one alternative aggregation strategy. The amplitude
encoding baseline (IID FedAvg, 5 seeds) is:

| Dataset | IID FedAvg (ref) | Centralised (ref) |
|---|---|---|
| Wisconsin | 94.21% ± 2.20pp | 86.84% |
| Pima | 71.30% ± 0.96pp | 69.48% (PolyMLP proxy) |

---

## 1. Wisconsin — Distribution Robustness

### 1.1 Moderate label skew (α=0.5)

| Seed | Final acc | Note |
|---|---|---|
| 1 | 96.49% | |
| 7 | 78.95% | Notable outlier |
| 42 | 91.23% | |
| 123 | 92.11% | |
| 999 | 83.33% | |
| **Mean±std** | **88.42% ± 7.11pp** | |

**vs IID: −5.79pp.** Substantially degraded but still above the centralised baseline
(86.84%). The wide standard deviation (7.11pp) is driven primarily by seed=7 (78.95%
in the Phase 9b single run), a 15–26pp drop depending on the run (see replication below).

**⚠ Seed=7 replication (MN-08, 2026-07-30) — 3 additional runs:**

| Run | Final acc | Drop from IID |
|---|---|---|
| Phase 9b single run | 78.95% | 18.42pp |
| Rep 1 | 71.05% | 26.32pp |
| Rep 2 | 93.86% | 3.51pp |
| Rep 3 | 87.72% | 9.65pp |
| **Mean ± std (all 4)** | **82.90% ± 8.65pp** | **range: 22.81pp** |

Partition (same all 4 runs): C0=83@2%c1, C1=65@78%c1, C2=111@100%c1, C3=196@62%c1.
R0=64.04% in all 4 runs (deterministic by init seed, per MN-04c finding).

**MN-03a attribution revised:** The partition IS adversarial for seed=7 — C2 is 100%
class-1, C0 is 98% class-0, creating conflicting gradient signals that place the model
near an attractor boundary. MN-03a correctly identifies this as difficult geometry for
amplitude encoding. However, the specific 78.95% Phase 9b value is not the stable
outcome of this partition — it is one draw from a 22.81pp-range distribution.
The MN-03a encoding×partition interaction is a **risk indicator** (adversarial partition
→ high variance outcome) not a deterministic cause. The actual outcome per run is
actor-RNG-determined (MN-08). This extends MN-08 from the α=0.1 regime into α=0.5:
adversarial partition geometry at α=0.5 can also produce attractor-boundary sensitivity,
albeit less severely than α=0.1 cases (MN-08 instances at α=0.1 showed 21–30pp ranges;
this α=0.5 case shows 22.81pp — comparable in magnitude).

**Measured α=0.5 within-seed noise for adversarial partitions:** 22.81pp range (replaces
MN-06's '~7–10pp (est.)' estimate, which was calibrated across seeds, not within an
attractor-sensitive seed). The 7–10pp estimate remains valid for the non-adversarial
seeds 1, 42, 123, 999 at this condition.

Seeds 1, 42, 123, 999 (91–97%) show no replication anomalies — their 5-seed mean±std
adequately summarises a unimodal distribution. Treat seed=7 as a separate
actor-RNG-sensitive cluster with effective accuracy 71–94% and report accordingly.

### 1.2 Severe label skew (α=0.1) — per-seed taxonomy required

α=0.1 routinely produces 1–2 ghost clients (<20 samples) in the Wisconsin 455-sample
training set. For 3 of 5 seeds, this reduces the effective federation to ~2 clients
(see MN-04, partition table). Reported outcomes conflate label heterogeneity with
client-count degeneracy.

| Seed | Dominant client (weight, purity) | Final acc | Pattern |
|---|---|---|---|
| 1 | C0: 44.8%, 100% benign | 89.47% | Clean 4-client escape |
| 7 | C3: 51.0%, 67% benign | 90.35% | Clean ~3-client escape |
| 42 | C0: 61.1%, 100% benign | 64.04% | ~~Partition-driven collapse~~ → **Bifurcation-sensitive** (single draw; MN-04a retraction, MN-08) |
| 123 | C0: 59.1%, 100% benign | 85.96% | Late 2-client escape — borderline threshold |
| 999 | C1: 60.7%, 100% benign | 64.04% | ~~Partition-driven collapse~~ → **Bifurcation-sensitive** (single draw; MN-04a retraction, MN-08) |
| **Mean±std** | | **78.77% ± 13.55pp** | Bimodal — do not interpret as unimodal distribution |

**Collapse pattern (seeds 42, 999):** Both collapsed to exactly 64.04% — the Wisconsin
all-benign majority-class baseline — in these single Phase 9b runs. Round logs show:
round 0 at 36.84% (anti-correlated start) → round 1 at 65.79% (collapse to majority
class) → rounds 2–15 flat at 63–65%.

**⚠ Phase 10 replication (MN-04a retraction, 2026-07-30):** Seeds 42 and 999 are NOT
deterministically collapsed. Phase 10 RQ3 found seed=42 with 0/15 B rounds (FUNCTIONAL)
and seed=999 with 2/15 B rounds (MIXED). 3-rep replication: seed=42 = 2 COLLAPSED +
1 FUNCTIONAL + 1 MIXED; seed=999 = 1 COLLAPSED + 3 MIXED. The 64.04% Phase 9b values
are single draws from a stochastic distribution. The MN-04a predictor is a high-risk
indicator, not a deterministic rule. See MN-08 for unified treatment.

**Wisconsin borderline case (seed=123):** Seed=123 (59.1%, 100% benign) sits just under
the collapse threshold — the predictor narrowly misses, and seed=123 escapes to 85.96%
via the second-largest client (C3, 38.2% weight) eventually counteracting. This is
consistent with the threshold being approximate (5-seed Wisconsin observation, not a
calibrated rule). The out-of-sample test of this predictor applied to Pima is in
Section 2.2.

**Escape pattern (seeds 1, 7):** Seeds with sub-majority dominant-client weight (<52%)
or mixed purity achieve 89–90%, close to the IID mean. At α=0.1, these seeds are
effectively 3–4-client federations despite the extreme Dirichlet concentration.

**Scope note (MN-04):** The α=0.1 Wisconsin results describe two qualitatively different
regimes in the same table — effective 4-client federation (seeds 1, 7) and degenerate
1–2-client federation (seeds 42, 123, 999). The 5-seed mean (78.77%) conflates these
regimes and should not be interpreted as the expected outcome of 4-client FL under
severe label skew.

### 1.3 Quantity skew

| Seed | Final acc |
|---|---|
| 1 | 92.11% |
| 7 | 94.74% |
| 42 | 95.61% |
| 123 | 90.35% |
| 999 | 92.98% |
| **Mean±std** | **93.16% ± 2.09pp** |

**vs IID: −1.05pp (*NF*).** Quantity skew with skew_factor=0.5 (exponential ramp,
4 clients) produces no meaningful accuracy degradation relative to IID. Amplitude
encoding is robust to client-size imbalance when label distributions remain balanced.
This contrasts sharply with label-skew conditions, confirming that the heterogeneity
type (label vs quantity) is the primary driver of performance degradation.

---

## 2. Pima — Distribution Robustness

**Critical context:** Pima/amplitude exhibits attractor trapping (MN-04b, MN-04c). The
VQC initial parameters land near the all-class-0 (65% accuracy) or all-class-1 (35%
accuracy) prediction surface. The starting attractor is determined by the VQC
initialization seed alone — not by the partition draw (MN-04c, confound-breaking
experiment, confirmed across 12 replicates). Whether training subsequently escapes that
starting attractor is **not** primarily a function of partition quality: the
confound-breaking experiment shows within-cell final-accuracy ranges of 21–30pp from
identical init+partition, meaning actor-RNG mini-batch stochasticity — not partition
structure — dominates escape outcomes (MN-08). This is the same unified mechanism that
governs Wisconsin's α=0.1 collapse cases (§4.1); initialization sets the *starting*
point in both datasets, but the *final* outcome near an attractor boundary is
actor-RNG-determined in both. Per MN-05, per-seed behavioral reporting is required for
α=0.5 and α=0.1.

### 2.1 Moderate label skew (α=0.5) — per-seed taxonomy required

All five seeds have functional 4-client partitions (no ghost clients). Attractor failures
at α=0.5 cannot be attributed to partition degeneracy.

| Seed | R0 | Attractor start | R15 final (orig.) | Pattern |
|---|---|---|---|---|
| 1 | 64.94% | class-0 | 70.78% | Moderate clean learning — oscillates 68–71% |
| 7 | 36.36% | class-1 | 64.29% | Escape from class-1 in R1; peaks ~67–69%; mild late-round decline to near-baseline |
| 42 | 64.94% | class-0 | 65.58% | Near-stuck in class-0 (+0.64pp over 15 rounds) |
| 123 | 35.06% | class-1 | 68.18% | Fast class-1 escape in R1 to 70.13%; settles at 67–69% |
| 999 | 64.94% | class-0 | 67.53% | Slow class-0 escape |
| **Mean±std** | | | **67.27% ± 2.50pp** | Masks qualitative diversity — per-seed taxonomy required |

**vs IID: −4.03pp.** The mean decline is above the ≥3pp reliability threshold but the
distribution is not unimodal: seeds 7 and 42 end near or at the majority-class baseline
while seed=1 achieves 70.78%. The 2.50pp standard deviation understates within-condition
variability.

**Seed=7 replication note (MN-04d, 3 reps):** The trajectory shape — escape from class-1
in round 1, peak at ~67–69%, mild late decline — reproduces consistently across all 3
replicates (range 2.6pp, mean 66.02%). The original "below baseline" endpoint (64.29%)
is the tail of this tight distribution; 3 replicates all end at or above 64.94%. Revised
characterization: **seed=7 at α=0.5 consistently escapes the class-1 attractor but
achieves only partial recovery (~65–68%), not a return to near-IID performance.**
The α=0.5 noise regime is normal (~2.6pp range), confirming no bifurcation-level
sensitivity here.

**Attractor vs partition contributions:** MN-04c confirms that initialization sets the
starting attractor (R0 is deterministic by init seed, confirmed across 12 replicates).
Whether training escapes is a more complex question: the confound experiment shows
within-cell final-accuracy ranges of 21–30pp from identical init+partition, meaning
actor-RNG stochasticity is a primary determinant of escape at α=0.1. At α=0.5, the
same two seeds (7, 123) start in the class-1 attractor but reach different final
states (seed=7 regresses to below baseline; seed=123 escapes cleanly). These different
partition structures (seed=123 has a large mixed-class client, seed=7's is more extreme)
are consistent with partition-quality influencing escape probability, but the α=0.1
replicate data warns against treating partition as the sole or primary determinant.
Escape at α=0.5 may also be partly stochastic.

### 2.2 Severe label skew (α=0.1) — per-seed taxonomy

| Seed | Eff. clients | R0 | Final acc (orig.) | Pattern — replicate status |
|---|---|---|---|---|
| 1 | ~1/4 (C0=99.3% weight) | 64.94% | 65.58% | Variable escape: 3 reps show 64.9–70.8% finals, 3 distinct trajectory shapes (see MN-04b) |
| 7 | ~3/4 | 36.36% | 39.61% | Stochastic bifurcation: 3 reps span 44.8–65.6%, original 39.61% is low-end draw (see MN-04b) |
| 42 | ~3/4 | 64.94% | 64.94% | All-class-0 collapse — perfectly flat all 15 rounds |
| 123 | ~1/4 (C0=96.1% weight) | 35.06% | 35.06% | All-class-1 collapse — perfectly flat all 15 rounds |
| 999 | ~2/4 | 64.94% | 64.94% | All-class-0 collapse — perfectly flat all 15 rounds |
| **Mean±std** | | | **54.03% ± 15.32pp** | Deeply misleading — single-draw per seed; do not use as a summary statistic |

**vs IID: −17.27pp.** The mean is not interpretable. Seeds 42, 123, 999 are confirmed
repeatable (flat collapse trajectories). Seeds 1 and 7 are stochastically variable
across their ~21–30pp and ~6pp ranges respectively; their single-run values from the
original sweep are individual draws, not stable characterizations.

**Out-of-sample test of collapse predictor (MN-04a applied to Pima):**
The Wisconsin collapse predictor (dominant weight ≥60% AND 100% class purity → collapse)
was designed on Wisconsin data. Applied to Pima:

| Seed | Dominant client | Weight | Purity | Predicted | Actual |
|---|---|---|---|---|---|
| 42 | C0: 208@100%c1 | ~33.8% | 100% | Collapse unlikely (weight <60%) | Collapse (all-class-0, init-driven) |
| 999 | C1: 232@0%c1 + C3: 382@56%c1 | — | Mixed | Collapse unlikely | Collapse (all-class-0, init-driven) |
| 123 | C0: 590@32%c1 | ~96.1% | Mixed (32%) | Partial escape possible | Collapse (all-class-1, init-driven) |
| 7 | C1: 386@3%c1 | ~62.9% | Nearly pure class-0 | Collapse? | Anti-correlated (class-1 init, partially pushed) |
| 1 | C0: 610@35%c1 | ~99.3% | Mixed (35%) | Collapse possible | Transient escape then plateau |

**The Wisconsin collapse predictor does NOT transfer to Pima** — meaning the specific
risk-flagging condition (partition weight ≥60% + purity) misfires on Pima partition
geometry. However, the original framing that this reflects two mechanistically distinct
phenomena ("partition-driven" for Wisconsin, "initialization-driven" for Pima) is
incorrect.

**⚠ Revised framing (MN-08, 2026-07-30):** Both Wisconsin and Pima exhibit the same
underlying mechanism — actor-RNG-determined basin selection near attractor boundaries
(MN-08). Wisconsin Phase 10 replication shows seeds 42 and 999 are equally stochastic
at the boundary (2 COLLAPSED / 1 FUNCTIONAL / 1 MIXED for seed=42 across 4 runs).
The predictor "doesn't transfer" because the risk-flagging condition is dataset-specific,
not because the mechanism differs. The correct conclusion is: the Wisconsin collapse
predictor was never a deterministic rule for Wisconsin either — it identifies high-risk
partition geometry for both datasets, with the outcome stochastically determined by
actor-RNG mini-batch ordering in both cases.

**Scope note (MN-04):** Seeds 1 and 123 are de-facto single-client (C0 holds 96–99.3%
of FedAvg weight). Their outcomes (transient escape then plateau, and complete class-1
collapse) are artifacts of near-zero federation structure, not labels-kew behavior.
Seeds 7, 42, 999 have 2–3 functional clients and show genuine (if pathological)
multi-client dynamics.

### 2.3 Quantity skew

| Seed | Final acc |
|---|---|
| 1 | 70.13% |
| 7 | 70.78% |
| 42 | 68.83% |
| 123 | 70.78% |
| 999 | 69.48% |
| **Mean±std** | **70.00% ± 0.85pp** |

**vs IID: −1.30pp (*NF*).** Identical finding to Wisconsin: quantity skew does not
meaningfully degrade Pima performance. The tight standard deviation (0.85pp) and
absence of attractor failures confirms that client-size imbalance (with balanced label
distributions) does not trigger the initialization×partition pathology.

---

## 3. Aggregation Strategy — TrimmedMean vs FedAvg (IID)

Both strategies tested under IID conditions only. Trimming fraction = 0.10
(remove top and bottom 10% of clients by L2 norm before averaging).

| Dataset | FedAvg IID | TrimmedMean IID | Δ | Verdict |
|---|---|---|---|---|
| Wisconsin | 94.21% ± 2.20pp | 95.26% ± 2.95pp | **+1.05pp (*NF*)** | No meaningful difference |
| Pima | 71.30% ± 0.96pp | 70.39% ± 1.87pp | **−0.91pp (*NF*)** | No meaningful difference |

Both deltas are below the ≥3pp reliable-difference threshold. TrimmedMean's increased
Wisconsin std (2.95pp vs 2.20pp) is driven by single-seed variation; neither dataset
shows a consistent directional benefit.

**Interpretation:** At IID with n_clients=4 and trim fraction=0.10, TrimmedMean has
insufficient clients to make the trimming consequential. With 4 clients, trimming 10%
means removing 0.4 clients per side — in practice the bottom and top clients are
partially downweighted but rarely excluded. The strategy reduces to near-FedAvg in this
regime. A meaningful comparison would require n_clients ≥ 10 or a Byzantine-fault
injection scenario.

---

## 4. Cross-Cutting Analysis

### 4.1 The attractor failure mechanism — unified account

**⚠ Revised (MN-08, 2026-07-30):** The original framing of "two mechanistically
distinct collapse patterns" (Wisconsin = partition-driven; Pima = initialization-driven)
is retracted. Phase 10 replication shows Wisconsin/amplitude/α=0.1 is equally
stochastic at the boundary (MN-04a retraction). Both datasets share a single underlying
mechanism. The original Wisconsin vs Pima contrast was an artifact of single-run Phase
9b measurement.

**Unified mechanism — actor-RNG basin selection at attractor boundaries (MN-08):**
When extreme non-IID partitioning (or VQC initialization near a trivial fixed point)
places the model at an attractor boundary, the outcome — which basin the model converges
to — is selected by stochastic mini-batch ordering across Ray actor subprocesses (MN-05),
not by the partition structure or initialization alone.

- **In Wisconsin/α=0.1:** The ≥60% dominant-weight + 100% class-purity condition
  places the model near the majority-class attractor boundary. Phase 10 replication
  of seeds 42 and 999 produces COLLAPSED, FUNCTIONAL, and MIXED outcomes across runs
  — the same partition geometry gives different final states each time. The predictor
  identifies high-risk geometry, not a guaranteed outcome.

- **In Pima/α=0.1:** The VQC initializes near trivial fixed points (all-class-0 or
  all-class-1). The confound-breaking experiment (MN-04c, 3 replicates per cell) shows
  within-cell final-accuracy ranges of 21–30pp from identical init+partition — actor-RNG
  stochasticity dominates, and no partition or initialization signal is detectable above
  the noise at 3 replicates per cell.

- **What initialization does determine:** Round-0 accuracy tracks init_seed exclusively,
  confirmed in all 12 MN-04c replicates. Initialization sets which attractor basin
  training begins in, but not where it ends.

**Partition geometry as risk indicator (not outcome predictor):**
The ≥60% + purity condition (Wisconsin) and the near-trivial-fixed-point start (Pima)
correctly flag conditions that place the model near an attractor boundary. They do not
determine whether the model falls into the collapse or functional basin in any given run.
Mitigation directions: (a) seeding Ray actor RNG to eliminate per-run stochasticity;
(b) running ≥3 replicates per high-risk condition before labeling an outcome;
(c) variance-scaled VQC initialization; (d) weight-capping in FedAvg to reduce
dominant-client effect.

### 4.2 Encoding×partition interaction at non-IID (MN-03a context)

MN-03a identified that seed=7's IID partition creates an encoding-specific effect —
angle and IQP are hurt (z ≈ −1.7), amplitude is helped (z = +1.4), reuploading is
neutral. The non-IID Part C data (amplitude only) cannot confirm this encoding
specificity under label skew, since only amplitude is swept in Part C. The cross-encoding
non-IID comparison is not available from this sweep design. MN-03a's "single-pass vs
repeated/full-Hilbert" hypothesis remains flagged as PENDING CONFIRMATION.

### 4.3 Summary: degradation by distribution type and dataset

| Condition | Wisconsin Δ vs IID | Pima Δ vs IID |
|---|---|---|
| α=0.5 | −5.79pp (moderate; seed=7 attractor-boundary sensitive — 22.81pp within-seed range, MN-08; seeds 1,42,123,999 stable 91–97%) | −4.03pp (masked; per-seed taxonomy required) |
| α=0.1 | −15.44pp (attractor-boundary regime; seeds 42, 999 stochastic per-run — MN-08; seeds 1, 7 functional; seed 123 borderline) | −17.27pp (attractor-boundary regime; all 5 seeds distinct; within-cell variance 21–30pp — MN-08) |
| Quantity skew | −1.05pp (*NF*) | −1.30pp (*NF*) |
| TrimmedMean/IID | +1.05pp (*NF*) | −0.91pp (*NF*) |

**Key finding:** Quantity skew and TrimmedMean produce negligible change relative to
IID in both datasets (all differences *NF*). Label skew produces substantial degradation
in both datasets, increasing nonlinearly from α=0.5 to α=0.1 due to partition degeneracy
(effective client count drops). The Wisconsin and Pima degradations at α=0.1 are both
driven by the same underlying mechanism — actor-RNG-determined basin selection at
attractor boundaries (MN-08) — triggered by different dataset-specific conditions
(dominant-gradient partition in Wisconsin; VQC near trivial fixed point in Pima). The
collapse predictor (MN-04a) does not transfer its specific risk-flagging condition
between datasets, but both datasets exhibit equally stochastic outcomes at the boundary
(MN-04a retraction; MN-08).

---

## 5. Limitations

1. **Seed confound:** init_seed and partition_seed are the same integer in the original
   sweep. The confound-breaking experiment (MN-04c) confirms initialization sets the
   starting attractor (R0, robust across 12 replicates) but cannot isolate partition
   effects on final accuracy because within-cell actor-RNG variance (21–30pp range) is
   far larger than any detectable between-cell signal. A full 5×5 factorial design at
   ≥10 replicates per cell would be required, but is unlikely to be interpretable unless
   the actor-RNG sensitivity is first reduced (e.g. by seeding the actor RNG).

2. **Actor-RNG sensitivity in attractor regime:** The MN-02 noise floor (±1.75pp
   half-range, IID calibration) does not apply to Pima/α=0.1. Within-cell variance is
   21–30pp range from identical init+partition — stochastic mini-batch order alone
   selects the final attractor. The single-seed point values in the original sweep
   (e.g., seed=7 → 39.61%) each carry ~21pp uncertainty, not ±1.75pp. All Pima/α=0.1
   results should be treated as single draws from a high-variance attractor-sensitive
   distribution. Per-seed behavioral taxonomies describe observed draws, not stable
   outcomes.

3. **α=0.1 regime conflation:** For most seeds, α=0.1 with n_clients=4 produces
   de-facto 1–2-client federation. Results labeled "α=0.1" describe partition degeneracy
   more than label heterogeneity per se. A larger n_clients or larger dataset would
   separate these effects.

4. **Part C is amplitude-only:** Non-IID per-encoding comparisons are not available from
   this sweep. The encoding×partition interaction hypothesis (MN-03a) cannot be
   confirmed or denied for non-IID conditions from the current data.

5. **TrimmedMean at n_clients=4:** The trim fraction (10%) has negligible effect at
   this scale. The strategy comparison is inconclusive.
