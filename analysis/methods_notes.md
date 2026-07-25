# XQFL Methods Notes

This file documents methodological findings that affect how results should be
interpreted or reported. It is updated as the project progresses through phases.

---

## MN-01: Tau Saturation at n\_features = 4

**Status:** Confirmed (Phase 6, 2026-07-22)
**Affects:** All stability analyses using Kendall's τ or Spearman ρ on quantum feature attributions.

### Finding

In experiments with n\_qubits = 4 PCA features, Kendall's τ between consecutive
global explanation rounds saturates at τ = +1.000 by round 2, even while
attribution magnitudes are still growing by ≈ 0.020 per round (7× total increase
from round 1 to round 15 under IID FedAvg with angle encoding).

### Root cause

Kendall's τ for a length-4 vector has only **three distinct non-trivial values**:

| Configuration | τ |
|---|---|
| Identical ranking | +1.000 |
| One adjacent swap | +0.333 |
| Two swaps (or one non-adjacent) | −0.333 |
| Fully reversed ranking | −1.000 |

Once PC0 dominates (|φ\_PC0| >> |φ\_PC1, PC2, PC3|), any gradient step that
leaves PC0 in first place reports τ = +1.000, regardless of how much the
magnitudes change. This happens by round 1 in the IID baseline; the metric then
carries no information for the remaining 14 rounds.

Spearman ρ has the same pathology.

### Evidence

Phase 6 experiment: angle encoding, IID FedAvg, 4 clients, 15 rounds.

| Round | τ (r→r+1) | cos sim (r→r+1) | L2 dist (r→r+1) | \|\|φ\|\| |
|------:|----------:|----------------:|----------------:|---------:|
| 0→1   | +0.667    | +0.166          | 0.04254         | 0.00162 → 0.04278 |
| 1→2   | **+1.000**| +0.9999         | 0.03521         | 0.04278 → 0.07798 |
| 7→8   | **+1.000**| +1.0000         | 0.01854         | 0.22220 → 0.24071 |
| 14→15 | **+1.000**| +1.0000         | 0.00557         | 0.29383 → 0.29936 |

- L2 distance is non-zero in every pair, including plateau rounds (12–15, accuracy = 84.21%).
- \|\|φ\|\| grows monotonically from 0.043 (round 1) to 0.299 (round 15): a 7× increase
  while τ reads +1.000 the entire time.
- Cosine similarity reaches ≥ 0.9999 at round 1→2 and stays there: **direction
  locks in very early**, but magnitude does not.

This rules out gradient vanishing (which would freeze both direction and magnitude).
The model knows which features matter almost immediately; gradient updates continue
to strengthen those attributions throughout training.

### Protocol (applies from Phase 7 onward)

**Never report τ or ρ alone for n\_features ≤ 4.**  Report all five items:

1. **Kendall's τ** — rank stability (context: expected to saturate early)
2. **Spearman ρ** — rank stability (alternative; same saturation risk)
3. **Cosine similarity** — direction stability, magnitude-agnostic
4. **L2 distance** — absolute change; sensitive to both direction and magnitude
5. **\|\|φ\|\| per round** — attribution magnitude growth curve

Use `stability_report(a, b)` from `explain/metrics.py` for items 1–4 in a single call.

### Interpretation guide

| τ | cos | L2 | Interpretation |
|---|---|---|---|
| 1.0 | ~1.0 | ≈ 0 | Genuine convergence (direction and magnitude both frozen) |
| 1.0 | ~1.0 | > 0 | Rank saturated; direction fixed; magnitude still training **(most common case at n=4)** |
| < 1.0 | < 1.0 | > 0 | Explanation still unstable (expected in early rounds or heavy non-IID) |
| 1.0 | low | > 0 | Ranking preserved but direction rotating — unlikely at n=4, check data |

### When this matters most

- Comparing stability across **distribution types** (IID vs non-IID): use cosine
  similarity to measure how quickly direction locks in, not τ alone.
- Reporting **convergence** of explanation vectors: use L2 distance as the
  stopping criterion, not τ.
- **Phase 7+**: stability tables must include cosine sim and L2 alongside τ.

---

---

## MN-02: Tied Accuracy at Small Test Sets — Hard Core vs Stochastic Tail

**Status:** Confirmed (Phase 9a checks, 2026-07-22)
**Affects:** Accuracy comparisons when test-set n is small enough that 1 sample ≈ 1 pp.

### Finding

In Phase 9a, Wisconsin/angle encoding hit 84.21% (96/114 correct) across 4 of 5 seeds
(seeds 1, 42, 123, 999). With 114 test samples, each sample is worth 0.877 pp.  Naively
this looks like a saturation ceiling. The per-sample overlap check revealed a more
precise picture.

### Evidence

Misclassification overlap (reruns for seeds 1, 42, 123, 999):

| Pair     | Shared errors | Only A | Only B | Jaccard |
|----------|--------------|--------|--------|---------|
| 1↔42     | 15           | 3      | 4      | 0.682   |
| 1↔123    | 17           | 1      | 4      | 0.773   |
| 1↔999    | 16           | 2      | 0      | 0.889   |
| 42↔123   | 14           | 5      | 7      | 0.539   |
| 42↔999   | 13           | 6      | 3      | 0.591   |
| 123↔999  | 16           | 5      | 0      | 0.762   |

- Mean pairwise Jaccard: **0.706**
- Samples wrong in ALL 4 seeds: **13** (indices 6, 11, 20, 23, 33, 35, 37, 44, 53, 61, 84, 93, 105)
- Total unique error positions across all 4 seeds: **26**

### Verdict: PARTIAL OVERLAP — hard core + stochastic tail

The 13 always-wrong samples are a **genuine structurally hard subset** at this
architecture (4 PCA components, angle encoding). The remaining ~5 error slots per seed
are allocated stochastically — different samples are misclassified depending on the
random partition and initialisation. Neither (a) pure identical-ceiling nor (b) pure
discrete-artifact; it is both simultaneously.

**Correct framing:** "Angle encoding on Wisconsin converges to approximately 84%
accuracy. There is a hard core of 13 test samples that all seeds fail on; an additional
13 test positions are variably wrong depending on the seed."

### Additional finding: Ray simulation non-determinism (root cause: scheduling, not aggregation order)

The rerun (phase9a_checks.py) produced different final accuracies than the original
Phase 9a sweep for the same seeds:

| Seed | Phase 9a acc | Rerun acc | Δ |
|------|-------------|-----------|---|
| 1    | 84.21%      | 84.21%    | 0 |
| 42   | 84.21%      | 83.33%    | −0.88pp |
| 123  | 84.21%      | 81.58%    | −2.63pp |
| 999  | 84.21%      | 85.96%    | +1.75pp |

**Root cause investigation (Phase 9b pre-check, item 3):** A `DeterministicFedAvg` subclass
was added to `federation/server.py` that sorts results by integer CID before calling
`aggregate_inplace`. This eliminates floating-point non-associativity from aggregation
order — but it did NOT fix the non-determinism (seed=42 × 5 back-to-back runs: range
3.51pp, 0.8421/0.8070/0.8421/0.8772/0.8421).

**True root cause (confirmed via sequential-actor test, Phase 9b pre-check):**
Forcing `num_cpus=1` (1-actor pool → all 4 clients sequential within each round)
still produced range=3.51pp across 5 back-to-back runs of seed=42 — identical to
the 8-actor case. Scheduling order is therefore NOT the root cause.

The actual source is the **unseeded PyTorch RNG inside the Ray actor process**.
`torch.manual_seed(seed)` is called in the main process but does not propagate into
the Ray actor subprocess. The actor's `torch.randperm` calls for mini-batch shuffling
within `XQFLClient.fit()` draw from an unseeded actor-local RNG (initialized from
OS entropy), producing different mini-batch orders each run and thus different local
parameter trajectories, even when starting from identical global weights.

**Design decision — intentionally left unseeded (Phase 9b onward):**
Seeding the actor-local RNG (e.g. `torch.manual_seed(client_id * 1000 + server_round)`
inside `XQFLClient.fit()`) would make mini-batch order fully deterministic at the cost
of eliminating the realistic per-client stochasticity that real FL participants exhibit.
Clients in production federated learning never have coordinated mini-batch seeds; fixing
the actor RNG would make the simulation artificially more reproducible than the setting
it models. The decision is therefore to **preserve the unseeded actor RNG** as the
realistic choice, accept ±3.5pp within-seed spread as a structural property of the
simulation, and report all results as distributions (mean ± std across ≥5 seeds) rather
than as point values. This is not a deferred bug — it is a deliberate trade-off of
bit-level reproducibility for external validity.

**Implication:** ±3.5pp within-seed spread is irreducible with current setup.
Within-seed std ≈ 1.2pp; across-seed std ≈ 1.8pp (Phase 9a), so within-seed noise
is a substantial fraction of total observed variance.

**Statistical protocol for Phase 9b+ (ACTION B):**
1. Report mean ± std across ≥5 seeds as the primary table entry.
2. For any effect of interest ≥ 4pp: confirmed (well above within-seed noise floor).
3. For any effect < 4pp: unconfirmed — increase seed count or rerun 5× per seed before
   interpreting the effect.
4. Do not report individual seed point values as exact numbers — always note ±1.75pp
   uncertainty (half the observed within-seed range).
5. When comparing two conditions (e.g. angle vs amplitude), the difference must exceed
   2× the within-seed noise floor (≈ 2 × 3.5pp = 7pp) to be considered reliable from
   a single-seed measurement. With 5-seed means, this threshold drops to ~7/√5 ≈ 3pp.

All federated results must be reported as ranges/means, not as exact point values.

### Protocol

When reporting tied accuracies across seeds at small test sets (n < 200):
1. Always check Jaccard overlap of misclassified samples before attributing to "ceiling".
2. Mean Jaccard < 0.5 → discrete-accuracy artifact (report as count-level coincidence).
3. Mean Jaccard > 0.9 → genuine hard subset (report as structural failures).
4. Mean Jaccard 0.5–0.9 → mixed: report hard core count + stochastic tail separately.
5. For federated experiments: treat point estimates as ±1.75pp uncertain; use ≥5-seed means.

See MN-01 for the analogous metric-resolution issue in Kendall's tau at n=4 features.

---

## MN-03: Retroactive Noise-Floor Reassessment (Phase 9b)

**Status:** Added Phase 9b, 2026-07-23
**Affects:** Phase 9a findings referenced below.

The confirmation that within-seed spread ≈ ±1.75pp (half-range of 3.51pp) requires
reassessing Phase 9a findings that were within or near this noise floor.

### MN-03a: Wisconsin/angle seed=7 — evidence re-opening, pending Phase 9b completion

**Original claim (Phase 9a):** Seed=7 achieved 79.8% vs 84.2% for the other 4 seeds
— a 4.4pp gap, described as "the only seed-level outlier."

**Initial reassessment (Phase 9b pre-checks):** 4.4pp is close to the within-seed
noise floor (3.51pp range, 1.2pp std per seed). Downgraded to UNCONFIRMED on the
grounds that actor-RNG mini-batch variance alone could produce a gap this size.

**New evidence (Phase 9b Part A, first independent replication):**
In Phase 9b's independent sweep execution, Wisconsin/angle seed=7 landed at **78.9%**
— within 0.9pp of the Phase 9a value (79.8%), and again the lowest of all seeds
observed so far (Phase 9b seeds 1=86.8%, 42=84.2% are both higher).
This is the second independent run where seed=7 lands substantially below the other
seeds. If actor-RNG noise were the sole cause, seed=7 landing low should be a coin
flip each execution — there would be no reason to expect the same seed to consistently
occupy the bottom of the distribution across independent runs. Two hits at seed=7,
each from a different execution context, is evidence of a genuine seed-level effect.

**What to check once Phase 9b Part A Wisconsin/angle (seeds 1,7,42,123,999) completes:**

1. **If seed=7 is lowest across ALL 4 encodings (angle, amplitude, reuploading, IQP):**
   This points to seed=7's IID data *partition* being harder than average — a subset of
   training samples less representative of the test set, independent of the model.
   The effect is real but not encoding-specific; it says something about the data, not
   the model comparison.

2. **If seed=7 is low only for angle encoding (not amplitude/reuploading/IQP):**
   This is the more interesting finding — an interaction between this specific partition
   and angle encoding's more limited representational capacity. Angle can't recover from
   a harder partition; richer encodings can. This would make the seed=7 pattern
   scientifically meaningful for the encoding comparison, not just a nuisance.

3. **If seed=7 is not consistently lowest once all 5 Phase 9b seeds complete:**
   The two-hit observation may be a small-sample coincidence (probability ~0.4 for
   two independent draws from a 5-seed uniform distribution to both give the min).
   In that case revert to UNCONFIRMED.

**Final verdict from Phase 9b Part A (all 20 Wisconsin IID runs complete):**

Five-seed statistics (Wisconsin, IID, FedAvg):

| Encoding | Mean | Std | Seed=7 acc | z(seed=7) | Interpretation |
|---|---|---|---|---|---|
| angle | 84.03% | 3.06% | 78.95% | **−1.66** | Substantially hurt |
| amplitude | 94.21% | 2.20% | 97.37% | **+1.44** | Actively helped |
| reuploading | 94.74% | 0.88% | 93.86% | −1.00 | Neutral (within expected range for lowest of 5) |
| IQP | 86.67% | 3.94% | 79.82% | **−1.74** | Substantially hurt |

**Three-way split (z-score based, not raw delta):**
1. **Hurt** (z ≈ −1.7): angle and IQP — both single-pass structured encodings
2. **Neutral** (z = −1.0): reuploading — z-score is within what any one of 5 seeds would
   normally produce in a tight distribution (std=0.88%); 0.88pp below mean is not anomalous
3. **Helped** (z = +1.4): amplitude — seed=7 is amplitude's highest result across all 5 seeds

**Current best hypothesis for the three-way split:**
The structural distinction is between *fixed-structure single-pass* encodings (angle:
RY per qubit; IQP: H+RZ+ZZ+H, both one-shot) and *repeated or full-Hilbert* encodings
(reuploading: data re-embedded per layer; amplitude: StatePrep into full 2^4-dim Hilbert
space). Seed=7's IID partition may create a data geometry that fixed-structure encodings
cannot adapt to in a single pass, while richer or repeated encodings compensate.

This hypothesis is explicitly flagged as **CURRENT BEST HYPOTHESIS — PENDING CONFIRMATION.**
Reuploading uses repeated RY+CNOT layers without entanglement in the encoding step, yet is
unaffected. IQP uses entangling IsingZZ gates but is still hurt. This rules out a simple
"entanglement vs no-entanglement" distinction and is not yet fully explained. The hypothesis
will require cross-checking against the non-IID per-encoding results once Part C's
remaining encoding conditions are swept (Part C uses amplitude only; per-encoding non-IID
data is not yet available). Until that confirmation, treat this as an observed pattern, not
a closed mechanistic finding.

**What MN-03a is now closed on:**
- The Wisconsin/angle seed=7 "outlier" is a genuine seed-level effect, not actor-RNG noise
  alone (confirmed by two independent sweeps both placing seed=7 lowest, and by the z-score
  analysis showing other encodings are not similarly affected)
- The effect is encoding-specific (angle and IQP are hurt; amplitude and reuploading are not)
- Therefore it is an encoding×partition interaction, not a data-partition difficulty effect

### MN-03b: Pima Item-1 collapse-vs-boundary finding — add actor-RNG caveat

**Original claim (Phase 9b Item 1):** Standalone training experiments showed classical
MLP collapses to exactly 64.9% (all-zeros output) in some seeds while quantum angle
never collapses, instead producing a sub-baseline anti-correlated boundary (~60–62%).
This difference was attributed to structural differences in the loss landscape.

**Caveat:** The standalone experiments (phase9b_investigations.py) used explicit
`torch.manual_seed(seed)` in the main process before each model's training call.
Unlike federated training, standalone models do NOT run inside Ray actors, so the
main-process seed IS respected. The collapse-vs-non-collapse signature observed in
standalone training is therefore NOT contaminated by the unseeded actor-RNG effect.

However, the specific seeds that collapsed (seed=7 and seed=42 for classical MLP in
Item 2 of the same script) varied depending on the call order and intervening random
state consumption (e.g. whether the PolyMLP was trained first). The exact collapse/
non-collapse assignment per seed is context-dependent and should not be cited as
"seed X always collapses" without controlling the call sequence.

**What remains robust:** The structural distinction — classical MLP CAN collapse to
an all-zeros fixed point (loss → 0.647, train_acc → 65.15%), while quantum angle
CANNOT reach an equivalent trivial fixed point due to the VQC architecture — is
robust to seeding and call order. It is a property of the model family, not of a
specific seed. The per-seed behavioral assignments are not robust.

**Protocol:** Report the collapse/non-collapse distinction as a model-family property.
Do not cite specific seeds as "collapsed" or "succeeded" without the full experimental
context (call order, intervening state).

---

## MN-04: α=0.1 Scope Limitation — Dirichlet Degeneracy at Small n_clients

**Status:** Added Phase 9b, 2026-07-24
**Affects:** All α=0.1 non-IID findings in Phase 9b Part C.

### Design constraint: α=0.1 conflates heterogeneity severity with effective client count

At α=0.1 with n_clients=4 and moderate dataset sizes (Wisconsin: 455 training, Pima: 614
training), the Dirichlet allocation routinely produces partitions in which 1–2 clients
receive near-zero samples. This reduces the effective federation size below the intended
4-client structure. Results across both datasets:

**Wisconsin α=0.1 — per-client sample counts and effective structure:**

| Seed | C0 (n, %c1) | C1 (n, %c1) | C2 (n, %c1) | C3 (n, %c1) | Eff. clients | Final acc |
|---|---|---|---|---|---|---|
| 1 | 204, 100% | 45, 0% | 47, 70% | 159, 30% | 4/4 | 89.47% |
| 7 | 94, 0% | 8*, 100% | 121, 100% | 232, 67% | ~3/4 | 90.35% |
| 42 | 278, 100% | 3*, 67% | 157, 2% | 17*, 12% | ~2/4 | 64.04% |
| 123 | 269, 100% | 3*, 67% | 9*, 0% | 174, 8% | ~2/4 | 85.96% |
| 999 | 8*, 100% | 276, 100% | 170, 1% | 1*, 0% | ~2/4 | 64.04% |

*ghost client (<20 samples; contributes negligible FedAvg weight).

Note: all seeds have aggregate class-1 fraction ~62.6–62.8% (Dirichlet preserves global
class frequencies). Collapse cannot be attributed to aggregate imbalance.

**Pima α=0.1 — per-client structure:**

| Seed | C0 (n, %c1) | C1 (n, %c1) | C2 (n, %c1) | C3 (n, %c1) | Eff. clients | Final acc |
|---|---|---|---|---|---|---|
| 1 | 610, 35% | 0**, 0% | 2*, 0% | 2*, 50% | ~1/4 | 65.58% |
| 7 | 196, 87% | 386, 3% | 30, 100% | 2*, 50% | ~3/4 | 39.61% |
| 42 | 208, 100% | 40, 3% | 17*, 24% | 349, 0% | ~3/4 | 64.94% |
| 123 | 590, 32% | 3*, 100% | 19*, 100% | 2*, 50% | ~1/4 | 35.06% |
| 999 | 0**, nan | 232, 0% | 0**, nan | 382, 56% | ~2/4 | 64.94% |

*ghost (<20 samples). **zero samples; excluded from FedAvg by zero weight.

**Scope-limitation statement:**

At α=0.1 with n_clients=4, what is being studied for most seeds is NOT a 4-client
federated learning system under severe label heterogeneity. It is a 1-client or 2-client
federated system with a degenerate dominant partition. These are qualitatively different
regimes:

- A 4-client federation under heterogeneity preserves the benefits of multi-client
  gradient diversity while testing robustness to label skew.
- A 1-client or 2-client effective federation eliminates gradient diversity entirely —
  the model's behavior is then determined by the class composition of 1–2 dominant
  clients, not by the federation protocol.

The heterogeneity effect (label skew) and the client-count effect (diversity loss) are
confounded in these results and cannot be disentangled within the current experiment
design. This is a structural property of α=0.1 with small n_clients and moderate dataset
sizes — not a partition code bug.

**Practical implication:** The α=0.1 results describe a qualitatively different regime
from α=0.5, not merely a quantitatively more severe version of the same regime. α=0.5
tests heterogeneity within a functional multi-client structure (all 5 Wisconsin seeds
retain ≥3 effective clients). α=0.1 tests robustness to partition degeneracy, i.e., what
happens when FedAvg's implicit assumption of meaningful client diversity is violated.

### MN-04a: Wisconsin α=0.1 collapse predictor

Collapse at Wisconsin occurs when: a single client holds ≥60% FedAvg weight with ≈100%
class purity (benign), giving its gradient signal enough weight to overwhelm the
counterweight from the second client.

| Seed | Dominant client weight | Purity | Outcome |
|---|---|---|---|
| 1 | 44.8% (C0, 100% class-1) | pure | Clean escape — below majority weight |
| 7 | 51.0% (C3, 67% class-1) | mixed | Clean escape — not pure |
| 42 | 61.1% (C0, 100% class-1) | pure | **Collapse** — ≥60% + pure |
| 123 | 59.1% (C0, 100% class-1) | pure | Late escape — borderline; C3 (38.2%) eventually wins |
| 999 | 60.7% (C1, 100% class-1) | pure | **Collapse** — ≥60% + pure |

The ~60% weight threshold is dataset-and-hyperparameter-specific (n_rounds, local_epochs,
learning rate all affect how quickly a dominant gradient signal overwhelms the minority).
Treat this as a 5-seed observation, not a general rule.

### MN-04b: Pima α=0.1 behavioral taxonomy — four distinct patterns

Pima/amplitude/α=0.1 is NOT well-described by mean±std (54.0%±13.7% is misleading). The
five seeds represent four qualitatively distinct behavioral regimes:

1. **All-class-0 collapse** (seeds 42, 999): stuck at 64.94% (= 100/154 = predict
   non-diabetic for all). In both cases, a dominant client with 0% class-1 and
   sufficient weight drives the model to the all-class-0 attractor. Perfectly flat
   accuracy from round 0 through round 15.

2. **All-class-1 collapse** (seed 123): stuck at 35.06% (= 54/154 = predict diabetic for
   all) from round 0 through round 15, perfectly flat. Seed=123 is a de-facto 1-client
   federation (C0=96.1% weight, 32% class-1). The amplitude VQC's initial parameters
   happen to land on an all-class-1 prediction, and no gradient signal is sufficient to
   move a near-single-client model off this fixed point over 15 rounds.

3. **Stochastic bifurcation** (seed 7): original sweep: 39.61% after 15 rounds
   (starts at 36.36%, rises slowly). Multi-client structure (C0=196@87%c1,
   C1=386@3%c1) creates conflicting gradient signals near the attractor boundary.

   **⚠ Replicate check (MN-04c ref-seed-7, 3 reps):** The 39.61% outcome did NOT
   reproduce in 2/3 reps — those ended at 65.58% and 64.29% (class-0 territory).
   One rep stayed in intermediate range (44.81%). Within-cell range: 21pp. The
   original "anti-correlated attractor" label described one draw from a high-variance
   distribution. **Revised characterization: seed=7 at α=0.1 is stochastically
   bifurcation-sensitive; final state spans ~36–66% depending on actor-RNG mini-batch
   order. Do not report as a named behavioral pattern.**

4. **Stochastic attractor escape** (seed 1): de-facto 1-client federation (C0=99.3%
   weight, 35% class-1). Original sweep: rises from 64.94% (R0) to 70.78% (R3),
   then declines to 65.58% by R15 ("transient escape → regression" shape).

   **⚠ Replicate check (3 reps, 2026-07-24):** Three distinct trajectory shapes observed:
   - Rep 1: escapes to 71.4% at R4, **holds 69–71% through R15, final=70.78%** — sustained escape, no regression
   - Rep 2: near-flat 64.9–65.6% all 15 rounds, **final=65.58%** — near-stuck, no escape peak
   - Rep 3: brief escape to 69.5% at R2, declines to 64.9% at R15 — closest to original shape
   Finals: 70.78%, 65.58%, 64.94% — mean=67.10%, range=5.8pp (smaller than seeds 7/42).

   The range (5.8pp) is smaller than the α=0.1 bifurcation-regime cases (21–30pp),
   consistent with seed=1's near-single-client structure (C0=99.3% weight) dominating
   the gradient regardless of mini-batch order. However the trajectory SHAPES vary
   substantially — the original "transient escape then regression" is one of three
   distinct trajectory patterns. **Revised characterization: seed=1 at α=0.1 shows
   variable escape magnitude (65–71% final); the specific trajectory shape is not
   reliably reproducible from a single run.**

   **Sign/magnitude reconciliation (Item 2 check):**

   Post-peak deltas (rounds 3→15, 12 transitions): 6 negative (−6.50pp total),
   2 positive (+1.30pp total), 4 zero. One outlier step: −2.60pp at round 5→6
   (3-sample loss); removing it leaves 11 transitions with mean −0.35pp.

   - *Sign test*: including zeros as "non-positive" gives 10/12 non-positive
     (p ≈ 2% under null), marginally significant. Excluding zeros gives 6/8
     (p ≈ 14.5%), not significant. Result is sensitive to treatment of flat rounds.
   - *Magnitude test*: mean delta = −0.43pp, sample SE ≈ 0.26pp (from the actual
     distribution of deltas), t ≈ −1.69, p ≈ 6% — marginal, not conclusive.
     Using the actor-RNG noise floor (±1.75pp) as SE proxy gives t ≈ −0.86,
     p ≈ 21% — not significant. The two estimates disagree because within-run
     round-to-round variance is much smaller than across-run between-seed variance;
     the noise-floor SE is inappropriate for within-run drift assessment.
   - *Reconciliation*: both tests are marginal. Neither conclusively confirms
     nor rules out systematic drift. The dominant visual feature — the −5.20pp
     decline from round 3 to round 15 — is driven disproportionately by two steps
     (−2.60pp and −1.30pp), each resolving to test-set resolution (0.65pp = 1 sample
     on 154 samples). The overall decline is real by inspection but not statistically
     robust at this sample size.

   **Cross-trajectory comparison (does the pattern recur?):**
   - Pima/α=0.1 seed=123 (other single-client case): flat at 35.06% from round 0 —
     no post-peak, no regression. Pattern does NOT appear.
   - Wisconsin/α=0.1 seeds 1 and 123: escape trajectories (rising to 89%, 86%) —
     no regression. Pattern does NOT appear.

   **Verdict:** Downgrade from "systematic drift" to **observed in one trajectory,
   not corroborated**. The mechanistic explanation (actor-RNG amplification, lack of
   cross-client averaging) remains plausible but unconfirmed. Report this seed as
   "transient escape then plateau/mild decline" — do not assert systematic drift.

**Protocol:** Report Pima/α=0.1 as a behavioral taxonomy (4 distinct patterns) rather
than a distribution statistic. Explicitly separate the single-client artifact cases
(seeds 1 and 123) from the genuine multi-client results (seeds 7, 42, 999) when
drawing conclusions about FL behavior under extreme skew.

---

### MN-04c: Seed confound experiment — initialization vs partition for Pima attractor

**Experiment design:** The sweep seeds both the Dirichlet partition draw
(`np.random.default_rng(seed)` in `partition.py`) and the VQC weight initialization
(`torch.rand(n_layers, n_qubits)` after `torch.manual_seed(seed)` in the sweep script)
with the same value. To disentangle them, four runs were executed with Pima/amplitude/α=0.1:

| Label | init_seed | partition_seed | R0 acc | Final acc |
|---|---|---|---|---|
| ref seed=42 | 42 | 42 | 64.94% (class-0) | 64.94% |
| ref seed=7  | 7  | 7  | 36.36% (class-1) | 46.10% |
| Test A: p=42, i=7  | 7  | 42 | **36.36%** (class-1) | 35.06% |
| Test B: p=7,  i=42 | 42 | 7  | **64.94%** (class-0) | 41.56% |

**Finding 1 — initialization determines attractor start (definitive):**
Round-0 accuracy is evaluated before any training and depends only on the initial VQC
weights. R0 tracks the init_seed exactly: init=7 gives R0=36.36% (class-1) regardless
of partition; init=42 gives R0=64.94% (class-0) regardless of partition. The partition
plays no role in attractor initialization.

**Finding 2 — partition quality determines final outcome:**

| Partition | init=42 final | init=7 final |
|---|---|---|
| partition=42 | 64.94% (stuck class-0) | 35.06% (stuck class-1) |
| partition=7  | 41.56% (confused intermediate) | 46.10% (partial escape, confused) |

Partition=42 (C0=208@100%c1 dominant, C3=349@0%c1 dominant) cannot pull either init
out of its starting attractor — both end in the attractor they started in. Partition=7
(C0=196@87%c1, C1=386@3%c1 — conflicting gradient signals) disrupts both attractors
but pushes both to an intermediate confused state (41–46%), not productive accuracy.

**Conclusion:** The observed behavior is a product of BOTH factors:
- **Initialization sets the starting attractor** — which trivial fixed point the VQC
  begins at (class-0 or class-1) is solely determined by torch initialization weights.
- **Partition quality determines escape/trajectory** — whether training moves away from
  the starting attractor, stays in it, or converges to a confused intermediate state
  depends on gradient diversity across clients.

The two mechanisms are separable and independently consequential. The original confounded
data cannot distinguish them; future experiments should cross init_seed and partition_seed
explicitly when investigating attractor behavior.

**Previously stated "initialization-driven" framing was incomplete.** The correct
framing is: initialization-seeded attractors, partition-determined outcomes.

**Multi-replicate results (3 reps per cell, 2026-07-24):**

Each cell was run 3 times with identical `torch.manual_seed(init_seed)` (same VQC init)
but unseeded Ray actor RNG (different mini-batch trajectories each run). Accuracies below
are on the 0–100% scale.

| Label | Init | Partition | R0 | Rep finals | Mean | Std | Range |
|---|---|---|---|---|---|---|---|
| ref seed=42 | 42 | 42 | 64.94% ×3 | 59.09%, 35.06%, 64.94% | 53.03% | **16pp** | **30pp** |
| ref seed=7  | 7  | 7  | 36.36% ×3 | 65.58%, 64.29%, 44.81% | 58.23% | **12pp** | **21pp** |
| Test A (p=42, i=7) | 7 | 42 | 36.36% ×3 | 35.06%, 64.94%, 64.29% | 54.76% | **17pp** | **30pp** |
| Test B (p=7, i=42) | 42 | 7 | 64.94% ×3 | 64.94%, 44.16%, 51.30% | 53.46% | **11pp** | **21pp** |

**Finding 1 — CONFIRMED and strengthened:** R0 is perfectly deterministic in all 12
runs. init=42 → R0=64.94% in all 6 runs; init=7 → R0=36.36% in all 6 runs. The
partition seed has zero effect on the starting attractor.

**Finding 2 — WITHDRAWN:** Within-cell final-accuracy range is 21–30pp. This dwarfs
the 3.51pp IID noise-floor range from MN-02 by an order of magnitude. The four cell
means (53.03%, 58.23%, 54.76%, 53.46%) are statistically indistinguishable — all
within a 5pp band with individual run ranges of 21–30pp. No partition signal is
detectable above the noise. The original single-run claim that "partition=42 locks
inits in their attractors while partition=7 pushes them to a confused intermediate"
was an artifact of small-n sampling from a very high-variance distribution.

**New finding — extreme actor-RNG sensitivity at Pima/α=0.1:** Within-cell variance
is 11–17pp (std), range 21–30pp, for the SAME initialization and partition. Individual
runs land in the class-0 attractor (~65%), the class-1 attractor (~35%), or an
intermediate state (~44–59%), depending solely on the actor-RNG mini-batch ordering.
The MN-02 noise floor (±1.75pp half-range, calibrated on Wisconsin/angle/IID) does not
apply to attractor-sensitive regimes. In these regimes, mini-batch order alone determines
which basin the model converges to — the model is effectively at a bifurcation point
where stochastic perturbations select the final attractor.

**Implication for ref seed=7 discrepancy:** The original concern that 46.10% (first
confound run) vs 39.61% (original sweep) constitutes a 6.49pp gap "above the noise
floor" is resolved: the noise floor here is ~21pp range, within which both values fall.
The 3-rep mean for the same config (init=7, partition=7) is 58.23%, with reps spanning
44.81%–65.58%. Both the original sweep value (39.61%) and the first confound value
(46.10%) are simply draws from this ~21pp distribution.

**Revised framing for MN-04c:** Only Finding 1 is robust. The attractor-start is
initialization-determined (confirmed, 12/12). The final-accuracy outcome at
Pima/amplitude/α=0.1 is not systematically controlled by either initialization or
partition under the current unseeded actor-RNG setup — it is dominated by stochastic
mini-batch trajectories. With 3 replicates per cell, no partition or initialization
signal for final accuracy is detectable above within-cell noise.

---

### MN-04d: Pima α=0.5 behavioral taxonomy — attractor failures persist at moderate skew

**Status:** Added Phase 9b, 2026-07-24

The five Pima/amplitude/α=0.5 seeds show qualitatively distinct behavioral patterns,
making mean±std (67.27%±2.50%) an inadequate summary.

**Key observation:** The same seeds (7, 123) start in the class-1 attractor at α=0.5
as at α=0.1. Seeds 1, 42, 999 start in the class-0 attractor at α=0.5 as at α=0.1.
This confirms that round-0 attractor assignment is initialization-driven (same torch
seed → same VQC weights → same starting attractor), as established by MN-04c.

| Seed | R0 | Attractor start | R15 final (orig.) | Pattern |
|---|---|---|---|---|
| 1 | 64.94% | class-0 | 70.78% | Moderate clean learning — oscillates 68–71% |
| 7 | 36.36% | class-1 | 64.29% | Escape from class-1 in R1; peaks ~67–69%; mild late regression; see replicate note |
| 42 | 64.94% | class-0 | 65.58% | Near-stuck in class-0 despite full 4-client structure (+0.64pp from R0) |
| 123 | 35.06% | class-1 | 68.18% | Fast class-1 escape to 70.13% in round 1; settles at 67–69% |
| 999 | 64.94% | class-0 | 67.53% | Slow class-0 escape over 15 rounds |

**Seed=7 replicate check (3 reps, 2026-07-24):**

Trajectory shape IS consistently reproduced: all 3 reps show escape from class-1
attractor in round 1 (R1 jumps to 63–67%), peak around R4 (66.9–68.8%), then mild
late-round decline. The trajectory shape is reliable.

Finals across 3 reps: 64.94%, 67.53%, 65.58% — mean=66.02%, range=**2.6pp**.
Original sweep value: 64.29% (the only observation below the 64.94% baseline).

The "regression to BELOW baseline" claim applies to 1 out of 4 total observations
(original + 3 reps). The remaining 3 end at or above 64.94%. The 2.6pp range confirms
α=0.5 is NOT in the high-variance bifurcation regime (unlike α=0.1's 21–30pp) — the
behavior is tightly constrained once the model escapes class-1. The original sweep's
64.29% is a tail draw from a distribution centered ~66%.

**Revised characterization for seed=7 at α=0.5:** Consistently escapes class-1
attractor in round 1 and peaks at 67–69%; undergoes mild late-round decline ending at
65–67%. Occasionally (1/4 obs.) touches or falls marginally below the majority-class
baseline. The "below baseline" framing overstates the severity — reframe as "partial
regression, final accuracy ≈ baseline" rather than "regression below baseline."

**Scope:** Unlike α=0.1, all five α=0.5 seeds have functional 4-client partitions
(no ghost clients). The attractor failures at α=0.5 therefore cannot be attributed to
partition degeneracy — they are genuine attractor-trapping failures under moderate
label heterogeneity. This is a dataset×encoding incompatibility, not an α=0.1 artifact.

**Protocol:** Report Pima/amplitude/α=0.5 as a 5-seed behavioral taxonomy, not as
mean±std. The same per-seed structure applies as for α=0.1, with the distinction that
α=0.5 outcomes are "softer" (partial escapes and regressions rather than complete
collapse) but still qualitatively heterogeneous.

---

## MN-05: Governing Framework for Part C Reporting

**Status:** Established Phase 9b, 2026-07-24
**Applies to:** All Phase 9b Part C result tables and narrative.

### Per-condition reporting mode

| Dataset | Distribution | Reporting mode | Rationale |
|---|---|---|---|
| Wisconsin | IID | mean±std | Clean learning, no pathologies |
| Wisconsin | α=0.5 | mean±std, flag seed=7 outlier | Seed=7 outlier (78.95% vs 91.23% for others); encoding×partition interaction (MN-03a), not attractor failure |
| Wisconsin | α=0.1 | per-seed taxonomy | Bimodal collapse/escape; see MN-04a |
| Wisconsin | quantity | mean±std | Clean learning, no pathologies |
| Pima | IID | mean±std | No attractor failures at IID |
| **Pima** | **α=0.5** | **per-seed taxonomy** | 3/5 seeds show attractor pathology at full 4-client (MN-04d); seed=7 "below baseline" revised to "partial regression, ≈baseline" after 3-rep check |
| **Pima** | **α=0.1** | **per-seed taxonomy (with caveats)** | Seeds 42/123/999 collapse patterns reliable; seeds 1 and 7 are stochastically variable (MN-04b, MN-06) — report as "variable escape" not named patterns |
| Pima | quantity | mean±std | Clean learning, no pathologies |
| Wisconsin | TrimmedMean/IID | mean±std | No attractor failures |
| Pima | TrimmedMean/IID | mean±std | No attractor failures |

### Noise floor and significance

The appropriate noise floor is **regime-dependent** (see MN-06):
- IID and clean multi-client non-IID: apply ≥1.75pp noise floor per seed (MN-02).
  For 5-seed means, >3pp threshold (≈ 7pp/√5).
- Pima/amplitude near attractor bifurcation (α=0.1, and possibly α=0.5): within-seed
  range is 21–30pp (MN-04c, MN-06). Individual seed values carry this uncertainty.
  Per-seed taxonomies describe observed draws from a high-variance distribution, not
  stable fixed outcomes. Do not report single draws as definitive patterns.

### Key confound disclaimer for Pima attractor findings

Results from the original sweep confound init_seed and partition_seed (same integer
seeds both). The confound-breaking experiment (MN-04c) confirms initialization
determines attractor start (R0, robust); final accuracy at α=0.1 is dominated by
actor-RNG stochasticity, not init or partition (Finding 2 withdrawn). All Pima
attractor behavioral claims should note the single-draw uncertainty and the regime-
specific noise amplification documented in MN-06.

---

## MN-06: Actor-RNG Noise is Non-Uniform Across the Experiment Space

**Status:** Added Phase 9b, 2026-07-24
**Affects:** All accuracy comparisons, behavioral taxonomies, and stability claims.

### Finding

Actor-RNG noise (from unseeded Ray actor mini-batch shuffling, see MN-02) is not
a fixed property of the simulation — it varies by more than an order of magnitude
depending on the model's position in the loss landscape.

**Observed noise levels by regime:**

| Regime | Example condition | Within-seed range | Within-seed std | Noise source |
|---|---|---|---|---|
| IID, clean multi-client | Wisconsin/angle/IID | ~3.5pp | ~1.2pp | Mini-batch order variation on smooth loss surface |
| Non-IID, functional multi-client | Wisconsin/amplitude/α=0.5 | ~7–10pp (est.) | ~3pp | Increased gradient heterogeneity amplifies batch-order effects |
| **Attractor bifurcation** | **Pima/amplitude/α=0.1** | **21–30pp** | **11–17pp** | Mini-batch order selects which attractor basin the model converges to |

The attractor-bifurcation regime values come from the MN-04c confound experiment
(3 replicates per cell, identical init+partition, only actor-RNG varies):
- ref seed=42: finals 59.09%, 35.06%, 64.94% — range 30pp
- ref seed=7:  finals 65.58%, 64.29%, 44.81% — range 21pp
- Test A (p=42,i=7): finals 35.06%, 64.94%, 64.29% — range 30pp
- Test B (p=7,i=42): finals 64.94%, 44.16%, 51.30% — range 21pp

### Mechanism

On a smooth loss surface (IID, well-conditioned model), mini-batch ordering changes
the precise gradient update at each step but the model converges to the same basin
regardless. The accumulated effect of different orderings is small (≈3.5pp).

Near a loss-surface bifurcation — where two or more attractor basins are separated by
a low saddle — small perturbations from mini-batch ordering can tip the training
trajectory into different basins. Each replicate's actor-RNG produces a different
mini-batch sequence, and whichever sequence crosses the saddle first commits the model
to that basin. The Pima/amplitude configuration appears to have such a geometry: the
VQC with 13 parameters sits near a saddle between the all-class-0 (accuracy ≈ 65%)
and all-class-1 (accuracy ≈ 35%) trivial fixed points, and training can converge to
either, or to an intermediate state, from the same starting point.

### Implication for behavioral taxonomies

Any behavioral pattern observed in a **single run** in an attractor-sensitive regime
(Pima/amplitude/α=0.1, possibly α=0.5) has noise equivalent to 21–30pp. A taxonomy
entry like "seed=7 shows anti-correlated attractor at 39.61%" is a description of one
draw from a distribution with ~21pp range — the same seed and partition produced
65.58% and 64.29% in two out of three replicates. Single-draw patterns should not be
elevated to "named behaviors" without replicate confirmation.

**Taxonomy replication outcomes (2026-07-24):**

- **MN-04b seed=7 α=0.1 (anti-correlated attractor, 39.61%):** NOT reproduced.
  MN-04c replicates: 65.58%, 64.29%, 44.81% — original 39.61% is one draw from a
  21pp-range distribution. Renamed to "stochastic bifurcation" — no fixed label
  assignable. (See MN-04b item 3.)

- **MN-04b seed=1 α=0.1 (transient escape → regression, 65.58%):** Shape NOT
  reliably reproduced. Three reps show three distinct trajectory patterns (sustained
  escape 70.78%, near-stuck 65.58%, brief escape then regression 64.94%). Range
  5.8pp — smaller than full bifurcation regime but trajectory shape is variable.
  Renamed to "stochastic attractor escape." (See MN-04b item 4.)

- **MN-04d seed=7 α=0.5 (regression below baseline, 64.29%):** Trajectory SHAPE
  reproduced (class-1 escape + mild late decline) but "below baseline" endpoint
  is NOT reliable — 3 reps give 64.94%, 67.53%, 65.58% (mean 66.02%, range 2.6pp).
  The 2.6pp range confirms α=0.5 is NOT in the high-variance bifurcation regime.
  Revised to "partial regression, final ≈ baseline." (See MN-04d.)

### Protocol for attractor-sensitive regimes

1. Never report a single-run result from Pima/amplitude/α=0.1 (or any condition
   suspected of attractor bifurcation) as a fixed behavioral characterization.
2. Run ≥3 replicates per seed before naming a behavioral pattern.
3. Report the distribution of final accuracies across replicates, not just the shape
   of one trajectory.
4. If a pattern reproduces consistently across ≥3 replicates (e.g. always ends
   class-0), it may be named. If the final state varies by >10pp across replicates,
   describe the seed as "stochastically attractor-sensitive" rather than assigning a
   fixed behavioral label.
5. The ±1.75pp noise floor from MN-02 applies only outside the bifurcation regime.
   Within the bifurcation regime, report replicate range as the effective uncertainty.

### What is NOT affected

- Wisconsin/amplitude results (all α levels): Wisconsin does not exhibit the VQC
  attractor geometry. Its noise is in the normal 3–10pp range.
- Pima/amplitude/IID and quantity-skew: clean learning, no attractor failures
  observed, normal noise range.
- Round-0 accuracy for any seed: R0 is pre-training and fully deterministic given
  the torch init seed and test set. Not subject to actor-RNG noise.
- MN-03a encoding×partition finding (z-score analysis on Wisconsin/IID): based on
  5-seed means, well above the IID noise floor.
- MN-04a Wisconsin collapse predictor: partition-driven, not actor-RNG-sensitive.
