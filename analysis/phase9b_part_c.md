# Phase 9b Part C — Non-IID Distribution Robustness and Aggregation Strategy

**Governing framework:** MN-02 (noise floor ±1.75pp), MN-03a (encoding×partition interaction),
MN-04 (α=0.1 scope limitation, collapse predictor, behavioral taxonomies), MN-05 (per-condition
reporting mode). All five-seed means carry ≥3pp meaningful-difference threshold.

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
(86.84%). The wide standard deviation (7.11pp) is driven primarily by seed=7 (78.95%),
a 15pp drop from that seed's IID result (97.37%). Seed=7's outlier behavior is
consistent with MN-03a: this seed×partition combination produces difficult data geometry
for amplitude encoding under heterogeneous conditions, even though amplitude was
seed=7's best IID result. The remaining four seeds (91–97%) indicate that five-seed
mean±std adequately summarises the distribution for Wisconsin/α=0.5 once the outlier
is noted.

Applying the ≥1.75pp noise floor: seeds 1 and 7 are separated by 17.5pp — well above
the noise floor. Seeds 42 and 123 differ by only 0.88pp (*NF*). Treat the group as
two clusters: {7, 999} (low-to-mid, 79–83%) and {1, 42, 123} (high, 91–97%).

### 1.2 Severe label skew (α=0.1) — per-seed taxonomy required

α=0.1 routinely produces 1–2 ghost clients (<20 samples) in the Wisconsin 455-sample
training set. For 3 of 5 seeds, this reduces the effective federation to ~2 clients
(see MN-04, partition table). Reported outcomes conflate label heterogeneity with
client-count degeneracy.

| Seed | Dominant client (weight, purity) | Final acc | Pattern |
|---|---|---|---|
| 1 | C0: 44.8%, 100% benign | 89.47% | Clean 4-client escape |
| 7 | C3: 51.0%, 67% benign | 90.35% | Clean ~3-client escape |
| 42 | C0: 61.1%, 100% benign | 64.04% | **Partition-driven collapse** (majority-class, all-benign) |
| 123 | C0: 59.1%, 100% benign | 85.96% | Late 2-client escape — borderline threshold |
| 999 | C1: 60.7%, 100% benign | 64.04% | **Partition-driven collapse** (majority-class, all-benign) |
| **Mean±std** | | **78.77% ± 13.55pp** | Bimodal — do not interpret as unimodal distribution |

**Collapse pattern (seeds 42, 999):** Both collapsed to exactly 64.04% — the Wisconsin
all-benign majority-class baseline. The collapse predictor (MN-04a) is confirmed:
dominant client weight ≥60% AND class purity ≈100% → collapse within 1 round. Round
logs show: round 0 at 36.84% (anti-correlated start) → round 1 at 65.79% (collapse to
majority class) → rounds 2–15 flat at 63–65%.

**Out-of-sample test of collapse predictor on Pima:** Seed=123 in the Wisconsin table
(59.1%, 100% benign) is a borderline case — the predictor narrowly misses, and seed=123
does escape to 85.96% via the second-largest client (C3, 38.2% weight) eventually
counteracting. This is consistent with the threshold being approximate (5-seed observation,
not a calibrated rule).

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

**Critical context:** Pima/amplitude exhibits initialization-driven attractor trapping
(MN-04b, MN-04c). The VQC initial parameters land near the all-class-0 (65% accuracy)
or all-class-1 (35% accuracy) prediction surface, and training may or may not escape
depending on partition quality. The starting attractor is determined by the VQC
initialization seed alone — not by the partition draw (MN-04c, confound-breaking
experiment). Per MN-05, per-seed behavioral reporting is required for α=0.5 and α=0.1.

### 2.1 Moderate label skew (α=0.5) — per-seed taxonomy required

All five seeds have functional 4-client partitions (no ghost clients). Attractor failures
at α=0.5 cannot be attributed to partition degeneracy.

| Seed | R0 | Attractor start | R15 final | Pattern |
|---|---|---|---|---|
| 1 | 64.94% | class-0 | 70.78% | Moderate learning — oscillates 68–71%, clean |
| 7 | 36.36% | class-1 | 64.29% | Partial class-1 escape (peaks ~67% at R6), regresses **below baseline** |
| 42 | 64.94% | class-0 | 65.58% | Near-stuck in class-0 (+0.64pp over 15 rounds) |
| 123 | 35.06% | class-1 | 68.18% | Fast class-1 escape in R1 to 70.13%; settles at 67–69% |
| 999 | 64.94% | class-0 | 67.53% | Slow class-0 escape |
| **Mean±std** | | | **67.27% ± 2.50pp** | Masks qualitative diversity — misleading |

**vs IID: −4.03pp.** The mean decline is above the ≥3pp reliability threshold but the
distribution is not unimodal — seed=7 ends below the 64.94% majority-class baseline,
while seed=1 achieves 70.78% (close to IID). The 2.50pp standard deviation severely
understates within-condition variability.

**Seed=7 below-baseline outcome:** Starting in the class-1 attractor (R0=36.36%),
seed=7 partially escapes to 67.53% by round 6 but then regresses steadily to 64.29%
at round 15 — 0.65pp below the non-diabetic majority-class floor. This is the only
seed in the entire Phase 9b sweep to finish below its majority-class baseline. A
full 4-client α=0.5 partition cannot sustain the escape achieved in early rounds.

**Attractor vs partition contributions:** The confound-breaking experiment (MN-04c)
established that initialization sets the starting attractor. The α=0.5 data reveals
that the partition quality then controls whether training escapes: seed=123 (class-1
start) escapes immediately in round 1, while seed=7 (also class-1 start) only partially
escapes before regressing. The two partitions are structurally different for these seeds
(seed=123 has C1=380 samples at 26%c1, a large mixed-class client; seed=7's partition
is more extreme), consistent with partition-quality-driven escape.

### 2.2 Severe label skew (α=0.1) — per-seed taxonomy

| Seed | Eff. clients | R0 | Final acc | Pattern |
|---|---|---|---|---|
| 1 | ~1/4 (C0=99.3% weight) | 64.94% | 65.58% | Transient escape to 70.78% (R3) then decline; net +0.64pp |
| 7 | ~3/4 | 36.36% | 39.61% | Anti-correlated attractor — slow rise from class-1, never escapes |
| 42 | ~3/4 | 64.94% | 64.94% | All-class-0 collapse — perfectly flat all 15 rounds |
| 123 | ~1/4 (C0=96.1% weight) | 35.06% | 35.06% | All-class-1 collapse — perfectly flat all 15 rounds |
| 999 | ~2/4 | 64.94% | 64.94% | All-class-0 collapse — perfectly flat all 15 rounds |
| **Mean±std** | | | **54.03% ± 15.32pp** | Deeply misleading — do not use as a summary statistic |

**vs IID: −17.27pp.** Largest degradation of any condition across both datasets. The
mean is not interpretable: it averages three separate trivial fixed points (all-class-0
at 64.94%, all-class-1 at 35.06%, and the anti-correlated attractor at 39.61%) with a
transient partial escape (65.58%).

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

**The Wisconsin collapse predictor does NOT transfer to Pima.** The Wisconsin rule
(partition-driven: dominant gradient overwhelms in round 1) is a different mechanism
from Pima's initialization-driven attractor trapping. Seeds 42, 999, and 123 collapse
due to VQC initialization landing in a trivial attractor, independent of partition weight.
Seed=7's anti-correlated outcome is driven by the class-1 initialization interacting with
a conflicting-gradient partition. The collapse predictor requires separate calibration per
dataset and should not be generalized across datasets.

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

### 4.1 The two attractor failure mechanisms

Two mechanistically distinct collapse patterns appear across Phase 9b Part C:

**Mechanism A — Partition-driven gradient overwhelm (Wisconsin, α=0.1):**
When a single client holds ≥60% FedAvg weight with near-100% class purity, its gradient
dominates round 1 aggregation and pulls the global model to the majority class in a
single step. The model never recovers. This is a training-dynamics failure — the model
IS learning (from the dominant client), but what it learns is wrong. Mitigation: reduce
maximum client weight (e.g. clip by FedAvg weight cap) or increase n_clients.

**Mechanism B — Initialization-driven attractor trapping (Pima, both α):**
The VQC's 13 trainable parameters initialize near the all-class-0 or all-class-1 trivial
fixed point. Training gradients from the federation cannot dislodge the model from this
fixed point — either because the gradients are too small relative to the attractor basin
depth, or because conflicting client gradients cancel rather than combine. This failure
occurs even with full 4-client functional partitions at α=0.5. Mitigation: improved VQC
initialization (e.g. variance-scaled initialization that avoids trivial fixed points),
or data re-encoding that changes the loss landscape on Pima.

**Confound note (MN-04c):** In the current sweep, partition_seed and init_seed are
confounded (same integer). The confound-breaking experiment (Pima/α=0.1, 4 runs crossing
init/partition seeds) confirms both mechanisms act independently:
- Round-0 accuracy tracks init_seed exclusively (R0=64.94% for init=42 regardless of
  partition; R0=36.36% for init=7 regardless of partition).
- Final accuracy depends on the partition: partition=42 preserves both attractors
  unchanged; partition=7 pushes both inits toward a confused intermediate state.

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
| α=0.5 | −5.79pp (moderate, driven by seed=7) | −4.03pp (masked; per-seed taxonomy required) |
| α=0.1 | −15.44pp (bimodal; 2/5 collapse, 3/5 near-IID) | −17.27pp (deeply bimodal; all 5 seeds distinct) |
| Quantity skew | −1.05pp (*NF*) | −1.30pp (*NF*) |
| TrimmedMean/IID | +1.05pp (*NF*) | −0.91pp (*NF*) |

**Key finding:** Quantity skew and TrimmedMean produce negligible change relative to
IID in both datasets (all differences *NF*). Label skew produces substantial degradation
in both datasets, increasing nonlinearly from α=0.5 to α=0.1 due to partition degeneracy
(effective client count drops). The Wisconsin and Pima degradations at α=0.1 are driven
by fundamentally different mechanisms (gradient overwhelm vs attractor trapping
respectively), and the collapse predictor does not transfer between datasets.

---

## 5. Limitations

1. **Seed confound:** init_seed and partition_seed are the same integer in the original
   sweep. The confound-breaking experiment (MN-04c) provides partial resolution but a
   full 5×5 factorial design (5 partition seeds × 5 init seeds) would be required to
   completely characterise the interaction.

2. **α=0.1 regime conflation:** For most seeds, α=0.1 with n_clients=4 produces
   de-facto 1–2-client federation. Results labeled "α=0.1" describe partition degeneracy
   more than label heterogeneity per se. A larger n_clients or larger dataset would
   separate these effects.

3. **Part C is amplitude-only:** Non-IID per-encoding comparisons are not available from
   this sweep. The encoding×partition interaction hypothesis (MN-03a) cannot be
   confirmed or denied for non-IID conditions from the current data.

4. **TrimmedMean at n_clients=4:** The trim fraction (10%) has negligible effect at
   this scale. The strategy comparison is inconclusive.

5. **Actor-RNG non-determinism:** Within-seed spread ≈ ±1.75pp means individual seed
   results carry this uncertainty (MN-02). All effects reported as *NF* (not found)
   reflect this noise floor.
