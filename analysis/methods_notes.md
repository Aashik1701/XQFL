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

### MN-04a: Wisconsin α=0.1 collapse predictor — RETRACTED AS DETERMINISTIC (2026-07-30)

**⚠ Retraction notice:** The original predictor claimed "partition-driven, not
actor-RNG-sensitive." Phase 10 replication (3 reps × 2 seeds) refutes this.
Seeds 42 and 999 — the two "confirmed collapse" cases — produce COLLAPSED,
FUNCTIONAL, and MIXED outcomes in independent reruns under identical partition and
initialization. The predictor is a risk indicator, not a deterministic rule. See MN-08.

Original predictor finding: collapse at Wisconsin occurs when a single client holds
≥60% FedAvg weight with ≈100% class purity (benign), giving its gradient signal
enough weight to overwhelm the counterweight from the second client.

| Seed | Dominant client weight | Purity | Original outcome | Replication status |
|---|---|---|---|---|
| 1 | 44.8% (C0, 100% class-1) | pure | Clean escape — below majority weight | Not retested (escape, no predictor trigger) |
| 7 | 51.0% (C3, 67% class-1) | mixed | Clean escape — not pure | Not retested (escape, no predictor trigger) |
| 42 | 61.1% (C0, 100% class-1) | pure | ~~Collapse~~ → **Bifurcation-sensitive** | Phase 10 RQ3: 0/15 B rounds (FUNCTIONAL); 3 reps: 2 COLLAPSED + 1 FUNCTIONAL + 1 MIXED |
| 123 | 59.1% (C0, 100% class-1) | pure | Late escape — borderline; C3 (38.2%) eventually wins | Not retested |
| 999 | 60.7% (C1, 100% class-1) | pure | ~~Collapse~~ → **Bifurcation-sensitive** | Phase 10 RQ3: 2/15 B rounds (MIXED); 3 reps: 1 COLLAPSED + 2 MIXED |

**Revised interpretation:** The ≥60% weight + 100% purity condition identifies
high-risk partition geometry — it places the model near an attractor boundary where
actor-RNG mini-batch ordering selects the outcome. It does not guarantee collapse.
The 64.04% "collapse" accuracy reported for seeds 42 and 999 in Phase 9b was a
single draw from a stochastic distribution spanning COLLAPSED through FUNCTIONAL.

The ~60% weight threshold remains a useful heuristic for flagging high-risk
partitions. Treat as a risk indicator only, not a deterministic rule. At least 3
replicates per seed are required before labeling a run as "confirmed collapse."

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
| Non-IID, non-adversarial partition | Wisconsin/amplitude/α=0.5 seeds 1,42,123,999 | ~7–10pp (est.) | ~3pp | Increased gradient heterogeneity amplifies batch-order effects |
| Non-IID, adversarial partition (α=0.5) | Wisconsin/amplitude/α=0.5 seed=7 | **22.81pp (measured, 4 runs)** | ~8.65pp | Actor-RNG basin selection — MN-08 extends to α=0.5 for adversarial partitions |
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
- MN-04a Wisconsin collapse predictor: RETRACTED as deterministic (see MN-04a and MN-08). The ≥60% + purity condition is a high-risk flag, not a deterministic rule — seeds 42 and 999 show COLLAPSED/FUNCTIONAL/MIXED outcomes across replicates.

---

## MN-07: Explanation Sweep — Runtime Estimate and Design Decisions

**Status:** Pre-sweep design note (2026-07-26)
**Affects:** Phase 10 (targeted explanation sweep for RQ1, RQ2, RQ3).

### 1. Runtime calibration from Phase 6

Phase 6 (angle encoding, IID, 4 clients, 15 rounds, Wisconsin,
`n_permutations=8`, `n_global_explain=10`, `n_explain=5`) completed in **97 seconds
(1.6 min)** measured from log timestamps.

Per-round breakdown (from phase6_log.csv timestamps):
- Round 1: ~18s (includes PennyLane JIT compilation — one-time overhead)
- Rounds 2–15: ~5–6s each (training + Q-SHAP + aggregation)
- Warm-up-corrected throughput: ~79s over 14 post-JIT rounds ≈ 5.6s/round

Q-SHAP forward-pass count per run:
```
n_permutations × (1 + n_qubits) × (n_global_explain + n_clients × n_explain) × n_rounds
= 8 × 5 × (10 + 4×5) × 15 = 8 × 5 × 30 × 15 = 18,000 passes / run
```

Cost scales linearly with `n_permutations × n_explain_samples`. With n_qubits=4
(fixed), there is no exponential component within the sweep design.

### 2. Feasibility of proposed explanation sweep

**Minimum viable explanation sweep (RQ1 + RQ2):**
- 4 encodings × 1 distribution (IID) × 1 seed (seed=42)
- Same n_permutations=8, n_global_explain=10, n_explain=5 as Phase 6
- Estimated: 4 × ~100s ≈ **7 minutes** (amplitude may add ~15–20% circuit depth
  overhead on default.qubit; within 4 qubits this is dominated by Python overhead)
- Centralized baseline (no federation, 1 model per encoding):
  ~30–60s extra (no Ray overhead, single Q-SHAP call on full test set) → negligible

**RQ3 add-on (non-IID divergence):**
- 1 encoding (amplitude) × 4 distributions × 1 seed each
- Estimated: 4 × ~100s ≈ **7 minutes**

**Total combined sweep (RQ1 + RQ2 + RQ3):** ~15–20 minutes. No checkpointing
or resumable structure required — fits comfortably in one sitting.

If 5-seed replication is added for stability estimates (RQ2), the estimate scales
to 4 encodings × 5 seeds × IID ≈ 33 minutes. Still a single sitting; no
checkpoint needed.

**Decision:** Do not add checkpointing. Run as a single blocking script with
progress prints. If a run fails, re-run that condition only (identified by
encoding × distribution × seed label in stdout).

### 3. Centralized baseline design (RQ1)

For each encoding, run one centralized training baseline:
- `HybridQuantumClassifier.fit(X_train_full, y_train, epochs=45, lr=0.05, batch_size=32)`
  (45 epochs ≈ 15 federated rounds × 3 local epochs — matching total gradient steps)
- Shared PCA + scaler fitted identically to federated setup (same `_fit_shared_preprocessor`)
- Q-SHAP on `X_test[:10]` with background `X_train[:20]`, `n_permutations=8`

Comparison metrics against round-15 global federated Q-SHAP:
- Kendall's τ (rank agreement)
- Cosine similarity (direction agreement)
- L2 distance (magnitude difference)

This directly answers RQ1: does federation alter the explanation distribution
beyond the accuracy degradation, or do the two converge to the same attractor?

### 4. RQ3 framing: collapsed-model explanations

**Decision: Two-track conditional reporting.**

Collapsed models (accuracy ≤ majority-class baseline + 2pp threshold,
checked per round) produce Q-SHAP vectors that degenerate toward zero magnitude:
when the model predicts a constant output regardless of input, all marginal
contributions are near zero. Computing local-global JSD in this state produces
a number but it carries no interpretable signal about explanation alignment.

| Track | Condition | Analysis |
|-------|-----------|----------|
| **Track A** — functional | round accuracy > majority baseline + 2pp | Compute local-global JSD and Kendall's τ per round; report trend vs. non-IID level |
| **Track B** — collapsed | round accuracy ≤ majority baseline + 2pp | Report that Q-SHAP ||φ|| → 0; note that explanation divergence is undefined; treat as correlated failure (both accuracy and interpretability collapse together) |

Collapse threshold values:
- Wisconsin: majority class ≈ 62.1% (class-0 = no cancer); threshold = 64.1%
- Pima: majority class = 64.94% (class-0 = no diabetes); threshold = 66.9%

The Track B finding is itself reportable for RQ3: "In cases where FL accuracy
collapses under extreme non-IID, explanations also degenerate, meaning the two
failure modes are correlated — non-IID does not merely degrade accuracy, it
eliminates both predictive and explanatory utility."

**Round-level vs run-level classification:** Apply the threshold per round per
client, not just to the global final accuracy. A model may collapse in early
rounds and recover (or the reverse). Log both the per-round Track A/B
classification and the final classification for summary reporting.

**Seeds to exclude from Track A divergence analysis (known or high-risk collapses):**
- Wisconsin/amplitude/α=0.1: seed=123 (borderline — check per-round; late escape in Phase 9b)
- Wisconsin/amplitude/α=0.1: seeds 42, 999 — bifurcation-sensitive; classify per-run by per-round Track A/B. Do NOT pre-label as "confirmed collapse" — both seeds produce FUNCTIONAL outcomes in some runs (MN-04a retraction; see MN-08)
- Pima/amplitude/α=0.1: seeds 7, 123 (class-1 attractor, flat trajectory)
- Pima/amplitude/α=0.1: seeds 42, 999 (class-0 attractor, flat trajectory)

These exclusions should be stated explicitly in the explanation sweep results
(not silently dropped) so the denominator of the Track A analysis is clear.

---

## MN-08: Actor-RNG-determined outcomes near attractor boundaries — unified phenomenon

**Status:** 2026-07-30
**Supersedes:** MN-04a deterministic collapse claim; Pima "doesn't transfer" out-of-sample
conclusion in phase9b_part_c.md §2.2 and §4.1; IQP seed=7 wrong-attractor characterization.

### 1. General phenomenon

This quantum-federated system exhibits **actor-RNG-determined outcomes near attractor
boundaries**, independent of encoding, dataset, or which specific partition geometry
creates the boundary. When extreme non-IID partitioning places the model at an attractor
boundary — where the loss surface has two near-equally-attractive basins — the outcome
(which basin the model converges to) is selected by the stochastic mini-batch ordering
across Ray actor subprocesses (MN-05), not by the partition structure or initialization
alone.

**Four independent instances confirm this as a general phenomenon:**

| Instance | Encoding | Dataset | Condition | Evidence |
|---|---|---|---|---|
| Wisconsin/α=0.1 seeds 42, 999 | amplitude | Wisconsin | ≥60% dominant weight + 100% purity | Phase 10 RQ3 + 3 reps each: COLLAPSED/FUNCTIONAL/MIXED outcomes across runs |
| Pima/α=0.1 (all seeds) | amplitude | Pima | Extreme Dirichlet + VQC near trivial fixed point | MN-04c: 21–30pp within-cell range, 4 cell means statistically indistinguishable |
| Wisconsin/α=0.5/seed=7 | amplitude | Wisconsin | C2=100% class-1, C0=98% class-0 partition conflict | 3 reps: 71.05%, 93.86%, 87.72%; Phase 9b+reps range=22.81pp; MN-03a attribution downgraded to risk indicator |
| IQP/IID/seed=7 | IQP | Wisconsin | Single sweep run with cos≈0.07 (wrong attractor) | 3 reps all return cos≥0.99 (NORMAL attractor): original was a noisy draw, not bifurcation |

Note: the IQP seed=7 instance is a single noisy draw, not a reproducible bifurcation —
it confirms that actor-RNG noise can produce wrong-attractor one-off outcomes even under
IID, at lower probability than the extreme non-IID cases.

Note: the Wisconsin/α=0.5/seed=7 instance extends MN-08 beyond α=0.1. Adversarial
partition geometry at α=0.5 (conflicting-gradient clients) is sufficient to create
attractor-boundary sensitivity, producing a within-seed range (22.81pp) comparable to
the α=0.1 cases. The α=0.5 condition is not generally affected — seeds 1, 42, 123, 999
show no replication anomalies. It is the specific partition structure (C2=100% class-1)
that creates the boundary, not α=0.5 per se.

### 2. Two dynamical modes at the boundary

Phase 10 replication of Wisconsin/amplitude/α=0.1/seed=999 (three MIXED runs)
reveals two qualitatively distinct trajectory patterns at the boundary:

**Mode 1 — Early lock-in (most common):** The actor-RNG selects a basin within
the first 1–2 rounds, and training locks in. Subsequent rounds are stable within the
chosen basin. Examples: Phase 10 RQ3 run (2/15 B rounds: B at R1-2, stable A from R3
onward at 0.86–0.88); seed=999 Rep 3 (3/15 B rounds: B at R1-2, A at R3-14, single
slip at R15).

**Mode 2 — Ongoing boundary oscillation:** The competition between attractors
continues throughout all 15 rounds, with individual round outcomes determined by each
round's mini-batch sampling. Accuracy oscillates between B-classified (0.6404) and
A-classified (0.6579) trajectories — a gap of 1.75pp (numerically coincident with the
MN-02 IID noise-floor estimate, though MN-02 was calibrated on an unrelated regime and
this agreement should not be taken as a mechanistic link) — crossing the threshold 3
times across 15 rounds. Example: seed=999 Rep 1 (10/15 B
rounds: B at R1-6, A at R7-11, B at R12-15). This mode occurs when the partition
geometry places the model exactly at the threshold between basins, so no accumulation
of gradient steps is sufficient to escape the boundary region.

**Practical characterization:** The actor-RNG *primarily* selects the outcome in
early rounds. When the partition geometry is close enough to the basin boundary, the
selection may not finalize until later rounds or may never finalize within n_rounds=15.

### 3. What the Wisconsin collapse predictor (MN-04a) now means

The ≥60% dominant-weight + 100% class-purity condition correctly identifies **high-risk
partition geometry** that places the model near the attractor boundary. It does NOT
determine the outcome. The correct interpretation is:

> Partitions satisfying ≥60% + pure **increase the probability of collapse** by
> placing the model at or near the attractor boundary. Whether collapse actually occurs
> in any given run depends on the actor-RNG mini-batch ordering for that run.

This distinction has a direct practical consequence: in a deployed system, the same
partition geometry will produce collapse in some runs and functional learning in others,
with no deterministic guarantee from the partition structure alone.

### 4. Revised framing for "Pima doesn't transfer to Wisconsin"

The original phase9b_part_c.md §2.2 conclusion — "The Wisconsin collapse predictor does
NOT transfer to Pima" — was correct in the literal sense (the predictor misfires on Pima
partition weights) but incorrect in its mechanistic interpretation. It implied Wisconsin
had a separate, deterministic partition-driven mechanism that was absent from Pima. Phase
10 replication shows that Wisconsin is equally stochastic at the boundary.

**Revised framing:** The Wisconsin collapse predictor and Pima attractor trapping are
not two mechanistically distinct phenomena — they are the same phenomenon (actor-RNG
basin selection at attractor boundaries) triggered by different conditions (dominant-
gradient partitions in Wisconsin; VQC initialization near trivial fixed points in Pima).
The predictor "does not transfer" not because the mechanisms differ, but because the
risk-flagging condition (partition weight vs VQC initialization geometry) is
dataset-specific. Both datasets share the same underlying stochastic attractor mechanism.

### 5. Scope and limits of this finding

- The phenomenon requires partition geometry that places the model near an attractor
  boundary. IID conditions are not affected (MN-02 noise floor applies, ±1.75pp).
  α=0.5 is affected for seeds with adversarial partition structure (e.g. Wisconsin/seed=7,
  22.81pp within-seed range); α=0.5 seeds with non-adversarial partitions are unaffected.
  Treat α=0.5 as "generally safe but check partition geometry" rather than categorically safe.
- The specific boundary conditions are dataset- and encoding-specific (n_qubits, n_layers,
  amplitude vs IQP encoding, class balance all affect the loss surface geometry).
- This finding does not invalidate Wisconsin/amplitude/α=0.1 seed=123 (borderline case
  that escapes via second client), Wisconsin/amplitude/α=0.5 seeds 1, 42, 123, 999
  (no replication anomalies), or any IID results.
- Mitigation directions: (a) seeding the Ray actor RNG to eliminate per-run stochasticity;
  (b) running ≥3 replicates per condition flagged as high-risk; (c) variance-scaled VQC
  initialization to avoid near-trivial-fixed-point starts.

### 6. MN-08c: Within-collapse bifurcation (Phase 11, 2026-08-02)

**New mechanism distinct from Modes 1 and 2.**  Modes 1 and 2 describe actor-RNG selection
at the Track A/B boundary — whether the model ends up functional or collapsed.  MN-08c
describes actor-RNG selection *within* Track B: the model collapses in every draw, but the
*specific collapsed state* is actor-RNG-sensitive, alternating between two distinct attractors
that both lie on the B side of the threshold.

**Evidence:** Phase 11 replication of angle encoding / Wisconsin / α=0.1 / seeds 42 and 999
(2 geometry-triggered replication reps each, G1 flag: ≥60% FedAvg weight + ≥90% purity):

| Seed | Primary | Rep 1 | Rep 2 | Range | All Track B? |
|---|---|---|---|---|---|
| 42 | 63.16% | 63.16% | **46.49%** | 16.7pp | ✓ all B |
| 999 | 63.16% | **36.84%** | **36.84%** | 26.3pp | ✓ all B |

Seeds 1, 7, 123 show single-attractor collapse: all draws land at ~61–63% with ≤0pp spread.
Seeds 42 and 999 show within-B bifurcation: the model alternates between the majority-class
predictor (~63.16% accuracy, always predicting class-0) and the anti-majority predictor
(~36.84% accuracy, always predicting class-1).  These are complement accuracies
(63.16% + 36.84% = 100%), confirming that both are constant-output states rather than
partially-functional models.

**Mechanism:** The angle encoding VQC under extreme Dirichlet skew (α=0.1) has two
distinct collapsed fixed points — one for each constant-output class — in addition to (or
instead of) functional attractors.  Which collapsed attractor the model reaches is
determined by the actor-RNG mini-batch ordering in early rounds, exactly as in Modes 1/2,
but the competition is between two collapsed states rather than between a collapsed and a
functional state.

**Implication for reporting:** Track A/B classification alone is insufficient to
characterize collapsed-regime behaviour.  For encodings and conditions where MN-08c applies,
two runs that both score "Track B" may have reached qualitatively different model states
(opposite constant-output predictions).  Any collapsed-regime comparison that pools Track B
results without examining the specific accuracy values — or the predicted class distribution —
risks treating these two states as equivalent.

**Phase 12 amplitude data (2026-08-02) — amplitude comparison gap closed:**

3 independent draws logged for amplitude/Wisconsin/α=0.1/seeds 42 and 999
(results/phase12_log.csv, run keys `amplitude|alpha_0.1|{seed}|draw{n}`):

| Seed | Draw 1 | Draw 2 | Draw 3 | Range | A/B |
|---|---|---|---|---|---|
| 42 | 45.6%(B) φ=0.056 | 55.3%(B) φ=0.092 | 55.3%(B) φ=0.092 | 9.7pp | 0A/3B |
| 999 | 45.6%(B) φ=0.082 | 65.8%(A) φ=0.085 | 74.6%(A) φ=0.252 | 29.0pp | 2A/1B |

**Cross-encoding comparison — all four encodings at α=0.1, seeds 42 and 999
(Phase 11 replication data + Phase 12; per-draw ‖φ‖ from logged CSV):**

| Encoding | Seed | Draw 1 | Draw 2 | Draw 3 | Range | A/B |
|---|---|---|---|---|---|---|
| angle | 42 | 63.2%(B) φ=0.011 | 63.2%(B) φ=0.008 | 46.5%(B) φ=0.048 | 16.7pp | **0A/3B** |
| angle | 999 | 63.2%(B) φ=0.022 | 36.8%(B) φ=0.003 | 36.8%(B) φ=0.002 | 26.3pp | **0A/3B** |
| reuploading | 42 | 73.7%(A) φ=0.207 | 38.6%(B) φ=0.068 | 87.7%(A) φ=0.125 | 49.1pp | **2A/1B** |
| reuploading | 999 | 64.9%(B) φ=0.118 | 63.2%(B) φ=0.015 | 44.7%(B) φ=0.074 | 20.2pp | 0A/3B |
| IQP | 42 | 63.2%(B) φ=0.003 | 39.5%(B) φ=0.058 | 57.9%(B) φ=0.119 | 23.7pp | 0A/3B |
| IQP | 999 | 63.2%(B) φ=0.000 | 53.5%(B) φ=0.148 | 75.4%(A) φ=0.279 | 21.9pp | **1A/2B** |
| amplitude | 42 | 45.6%(B) φ=0.056 | 55.3%(B) φ=0.092 | 55.3%(B) φ=0.092 | 9.7pp | 0A/3B |
| amplitude | 999 | 45.6%(B) φ=0.082 | 65.8%(A) φ=0.085 | 74.6%(A) φ=0.252 | 29.0pp | **2A/1B** |

*Table verification (2026-08-02, programmatic):* All 24 accuracy/Track-label pairs
cross-checked against the exact collapse threshold (36/57 + 0.02 = 65.1579%).
Zero mismatches.  Exact per-draw margins logged; narrowest Track A margin is amplitude/
seed=999/draw2 at +0.63pp above threshold.

**Final scope determination (Phase 11 + Phase 12, 2026-08-02):**

*What is confirmed:* Multi-attractor collapse-regime sensitivity under α=0.1 G1-geometry
is present across all four encodings — all show draw-to-draw variance of 10–49pp in the
collapsed regime.  This is the general finding.

*What is NOT confirmed:* An encoding-expressiveness split between "strict within-B" and
"A/B-crossing" encodings.  The reuploading/seed=42 result refutes this framing directly:
reuploading shows the largest A/B bifurcation in the dataset (49.1pp, 2A/1B with a draw
reaching 87.7%) at seed=42, while showing all-B behaviour at seed=999.  A/B-crossing is
therefore partition-dependent for all non-angle encodings, not encoding-specific.

Reuploading/seed=999 also deserves note: all three draws are Track B, but the primary
draw's ‖φ‖=0.118 is comparable to IQP's rep1 ‖φ‖=0.148 (which corresponds to a 53.5%
Track B draw) and non-trivially larger than angle's collapsed phi (max 0.048).  The 0A/3B
classification for reuploading/seed=999 reflects where these draws fall relative to the
65.16% threshold, not a qualitative difference in model behaviour.

*Amplitude/seed=42 range (9.7pp):* With only 3 draws, this is consistent with the
10–49pp ranges seen for other encodings/seeds; treat it as modest-to-clear within-B
variance, not as a categorically smaller effect.  All four encodings show within-B variance
of roughly the same order of magnitude when considered across the two tested seeds.

*The one consistent finding — angle uniquely all-B (⚠ under-replicated claim):*

All angle draws collected across Phase 11 under G1/G2-flagged conditions are Track B with
‖φ‖ ≤ 0.083, consistent with constant-output predictors.  The full inventory of
geometry-flagged angle conditions in the Phase 11 log is:

| Condition | Flag type | Draws | All B? | Max ‖φ‖ |
|---|---|---|---|---|
| α=0.1 / seed=42 | G1+G2 | 3 (1 prim + 2 rep) | ✓ | 0.048 |
| α=0.1 / seed=123 | G2 only | 3 (1 prim + 2 rep) | ✓ | 0.015 |
| α=0.1 / seed=999 | G1+G2 | 3 (1 prim + 2 rep) | ✓ | 0.022 |
| α=0.5 / seed=999 | G1+G2 | 3 (1 prim + 2 rep)† | ✓ | 0.083 |

†α=0.5/seed=999 has 6 CSV rows due to the Phase 11 Ray-deadlock restart (all six are
Track B; the valid checkpointed set is 1 primary + 2 reps).

Total: 12 confirmed draws across 3 G1-proper (G1+G2) conditions and 1 G2-only condition,
all Track B, max ‖φ‖=0.083.  Contrast: angle under non-flagged conditions (α=0.5/seeds
1,7,42,123; α=0.1/seed=1) reaches Track A with ‖φ‖ up to 0.315 — collapse is not
intrinsic to angle encoding, only to G1/G2-adversarial partitions.

⚠ **This is the project's most under-replicated surviving claim.**  12 draws across 4
conditions on a single dataset (Wisconsin) with a single architecture (4-qubit, 2-layer
BasicEntanglerLayers) is not sufficient evidence for a strong encoding-level statement.
The sample of G1-proper seeds is 3 out of 5 (seeds 42 and 999 at α=0.1, seed=999 at
α=0.5), which is not comprehensive.  The Pima dataset, which has a different majority
class, feature distribution, and class balance, has not been tested under G1 geometry for
any encoding.  **If compute time allows before final write-up, the first priority should
be 3 additional draws for angle/Wisconsin/α=0.1/seed=1 or /seed=7 under a manually
constructed G1 partition, or equivalently for angle/Pima/α=0.1 under the same G1-flagged
seeds.** A single Track A draw would falsify the current "uniquely all-B" claim as stated.

*Does not affect:* IID conditions, quantity-skew conditions where no collapse was observed.

---

## Phase 13 — Cross-Encoding MIA Sweep (RQ4 extension)

**Objective:** Extend Phase 8's single-seed angle MIA to all 4 encodings × 5 seeds (IID, Wisconsin), closing RQ4's coverage gap from "1 encoding × 1 seed" to "4 encodings × IID × 5 seeds".

**Architecture note:** Phase 10's IID explanation vectors (phase10_log.csv) are global background-averaged Q-SHAP (one 4-D vector per round per client) — not the per-sample Q-SHAP vectors required for MIA attack features.  Phase 10 model weights were not persisted.  Phase 13 retrains each of the 20 IID federated models using `model_out=[]`, reconstructs the model, then computes per-sample Q-SHAP for 50 training members and 50 test non-members using `_qshap_for_samples()` (n_bg=20, n_permutations=8), matching the Phase 8 protocol exactly.  Shadow model MIA uses 3 shadows × 15 local epochs each (local `.fit()`, no federation/Ray).

**Protocol:** Same as Phase 8 — direct CV-LR MIA (cv=5) + ShadowModelMIA (n_shadow=3, n_explain=20, shadow_split=0.5).  Script: `experiments/phase13_mia_sweep.py`.  Results: `results/phase13_mia_log.csv` (20 rows).

**Wall-clock:** All 20 runs completed in ~50 min total (vs 2 h estimate); per-run range 85–184 s; no timeouts, no errors.

### Per-encoding direct MIA AUC (mean ± std, 5 seeds, IID Wisconsin)

| Encoding | Direct MIA AUC | Shadow MIA AUC | Shadow CV AUC |
|---|---|---|---|
| angle | 0.551 ± 0.097 | 0.548 ± 0.077 | 0.484 ± 0.046 |
| amplitude | 0.628 ± 0.139 | 0.499 ± 0.108 | 0.487 ± 0.085 |
| reuploading | 0.698 ± 0.144 | 0.459 ± 0.158 | 0.470 ± 0.055 |
| IQP | 0.637 ± 0.204 | 0.475 ± 0.104 | 0.465 ± 0.080 |

### Per-seed breakdown

| Encoding | Seed | Acc | Direct | Shadow | ShCV |
|---|---|---|---|---|---|
| angle | 1 | 0.7456 | 0.5220 | 0.5148 | 0.5575 |
| angle | 7 | 0.7281 | 0.5940 | 0.4456 | 0.4392 |
| angle | 42 | 0.8421 | 0.6540 | 0.5984 | 0.5117 |
| angle | 123 | 0.7807 | 0.3760 | 0.5144 | 0.4358 |
| angle | 999 | 0.8246 | 0.6100 | 0.6680 | 0.4742 |
| amplitude | 1 | 0.9561 | 0.6120 | 0.5888 | 0.5633 |
| amplitude | 7 | 0.9386 | 0.8060 | 0.2936 | 0.3717 |
| amplitude | 42 | 0.9298 | 0.7540 | 0.5764 | 0.5508 |
| amplitude | 123 | 0.9649 | 0.4240 | 0.4968 | 0.3967 |
| amplitude | 999 | 0.9649 | 0.5440 | 0.5408 | 0.5542 |
| reuploading | 1 | 0.8684 | 0.5780 | 0.5696 | 0.5775 |
| reuploading | 7 | 0.8772 | 0.9300 | 0.1804 | 0.4433 |
| reuploading | 42 | 0.9474 | 0.7840 | 0.6052 | 0.4517 |
| reuploading | 123 | 0.9737 | 0.5360 | 0.3848 | 0.4200 |
| reuploading | 999 | 0.9474 | 0.6620 | 0.5532 | 0.4583 |
| IQP | 1 | 0.8860 | 0.4220 | 0.5180 | 0.5600 |
| IQP | 7 | 0.8070 | 0.8560 | 0.3008 | 0.3600 |
| IQP | 42 | 0.8684 | 0.8540 | 0.5748 | 0.4608 |
| IQP | 123 | 0.9211 | 0.3800 | 0.4164 | 0.3958 |
| IQP | 999 | 0.9035 | 0.6740 | 0.5652 | 0.5508 |

### Findings

#### (A) Cross-encoding ranking — not statistically reliable

All four encodings show mean direct AUC > 0.5, confirming Q-SHAP explanations carry some membership signal across all encodings.  **The correct primary finding is: all four encodings leak; no reliable ranking of leakage magnitude between encodings can be claimed at 5 seeds.**

The between-encoding mean spread (0.551–0.698, range 14.7pp) is larger than the 3pp project threshold, but within-encoding standard deviations (0.097–0.204) are of comparable magnitude, so ±1σ ranges overlap substantially for all pairs.  The same logic that prevents citing RQ2's encoding-stability differences as reliable applies here.  Any write-up stating "reuploading leaks more than angle" would overstate what 5 seeds can support.

#### (B) Consistency check with Phase 8

angle/seed=42 direct AUC=0.654 (Phase 13) vs 0.640 (Phase 8) — within 1.4pp on independent reruns.  Phase 8's result was not a one-off artefact.

#### (C) Seed=123 below-random AUC — replication check and downgrade

Direct AUC < 0.5 for all four encodings at seed=123 in Phase 13: 0.376 (angle), 0.424 (amplitude), 0.536 (reuploading), 0.380 (IQP).

**Pipeline composition check (Phase 13):** Attack split composition is not degenerate — 50 members / 50 non-members for all seeds; seed=123 class balance (66%/72% class 1) is within the range of other seeds. CV fold viability: `min(5, 50, 50) = 5`. Not a pipeline artifact.

**Replication attempt (Phase 13b, script `experiments/phase13b_seed123_resample.py`):** Applied the same replication standard used for every other single-draw anomaly in this project. Two checks:

1. *Draw 1 re-run:* Retrain all 4 models for seed=123 (actor-RNG generates new training trajectory) and recompute Q-SHAP for the same [0:50] attack samples. Results: angle=0.452, amplitude=0.344, reuploading=0.316, IQP=0.562. All four values differ from Phase 13 by >3pp (the project-standard threshold). IQP flipped from 0.380 → 0.562 (below to above 0.5). This demonstrates the [0:50] draw result is sensitive to actor-RNG variation in the training trajectory.

2. *Draw 2 (resample):* Same models (from Draw 1 re-run), different attack samples: training[50:100] as members, test[50:100] as non-members. Background held constant (training[0:20], same as Phase 13). Results: angle=0.544, amplitude=0.588, reuploading=0.680, IQP=0.606. **All four encodings above 0.5. 0/4 encodings replicate.**

**Verdict: downgraded.** The below-random AUC at seed=123 does not survive resampling (0/4 replication) and is unstable across training runs (all four values shift >3pp under actor-RNG). Correct treatment: *observed in one training-trajectory × attack-draw combination; not confirmed.* The same treatment applied to all other single-draw anomalies in this project.

**Cross-check in Phase 10 RQ1/RQ2/RQ3 data:** No corroborating signal. Seed=123 IID federated models show normal R15 accuracy, explanation norms, Kendall's tau stability, and JSD fidelity across all four encodings — within the range of other seeds, with no systematic indication of "unusually generic representation learning."

**Consequence for RQ4 reporting:** Do not cite seed=123 as evidence of partition-dependent privacy resistance. The observation is noted in the data but carries no evidential weight given the failed replication.

#### (D) Shadow attack failure — mechanism corrected

Initial explanation: "shadow models trained locally diverge in training dynamics from the federated target."

**This explanation is not supported by the data.**  Tested via Pearson and Spearman correlation between target_acc (proxy for target model quality) and shadow_mia_auc across all 20 runs:
- Pearson r = 0.032, p = 0.893
- Spearman r = 0.017, p = 0.945

No correlation.  Per-encoding correlation is marginally significant for angle (r=0.882, p=0.048, n=5) but non-significant for amplitude, reuploading, and IQP.  With n=5 per encoding, even significant correlations have no interpretive weight.

The decisive evidence against "diverging dynamics": seed=7 achieves the highest direct MIA AUC for three of four encodings — reuploading=0.930, IQP=0.856, amplitude=0.806 — confirming abundant membership signal exists in those models' explanations.  Yet at the same seed, shadow MIA achieves the lowest AUCs for those same encodings: reuploading=0.180, IQP=0.301, amplitude=0.294.  The shadow attack fails most severely precisely where the membership signal is strongest, which eliminates "not enough signal in the target" as the cause.

**Supported explanation:** The failure is in shadow→target transfer of the learned attack pattern.  Shadow models are trained as single, locally-fitted classifiers (15 local epochs, 50% data split) while the target is a federated model (15 rounds × 3 local epochs × FedAvg aggregation × 4 clients).  These produce structurally different explanation geometries regardless of convergence level.  Phase 8 originally flagged n_shadow=3 as a potential limitation; this remains a contributing factor alongside the structural mismatch.  The shadow MIA AUC results should not be interpreted as evidence about the amount of leakage in the target — they reflect an attack that is not well-matched to the federated setting.

#### Summary for RQ4 write-up

1. **Primary finding:** All four quantum encodings tested (angle, amplitude, reuploading, IQP) produce Q-SHAP explanation vectors that leak membership information under direct MIA (mean direct AUC 0.55–0.70 across encodings, all > 0.5).  Leakage magnitude is not reliably ranked between encodings at 5 seeds.

2. **Partition dependence:** Leakage appears partition-dependent across seeds but no confirmed pattern has been identified.  Seed=123 showed below-random AUC across all four encodings in Phase 13 Draw 1; Phase 13b resample showed 0/4 replication and actor-RNG instability in draw 1 re-run (all four values shifted >3pp).  Downgraded: observed in one training-trajectory × attack-draw combination; not confirmed.  (See Section (C) above and the Replication Discipline section for the full retraction.)

3. **Shadow attack:** Not effective in the federated Q-SHAP setting, with the failure attributable to structural mismatch between locally-trained shadow models and federated target explanation geometry, not to absence of membership signal.  Direct MIA is the reliable attack for this setting.

4. **DP defence:** angle/seed=42 only (Phase 8); ε=1.0 Laplace → AUC=0.296.  Not tested across encodings or seeds.

**RQ4 coverage after Phase 13:** 4 encodings × IID × 5 seeds (Wisconsin).  Non-IID MIA, Pima MIA, and cross-encoding DP remain untested.

---

## Replication Discipline

**Standing rule:** No unusual single-run finding is reported as a confirmed result without at least one independent replication.  "Independent" means a different random draw of one of the sources of non-determinism: a different attack-split sample, a different Ray actor-RNG trajectory from a fresh training run, or a different random seed.  Replication is not required for expected behavior (e.g., a Track-B collapse in a condition where collapse is the documented norm) or for reproducing a previously confirmed finding.

This rule exists because this project has a documented history of single-draw anomalies that did not survive replication — every one of the retracted findings below passed "looks suspicious" on first observation and failed on the first independent draw.  Later RQ sections cite this standard rather than re-establishing it.

### Confirmed retractions

**1. Wisconsin/angle/seed=7 accuracy outlier framing (MN-03a, 2026-07-30)**
*Original:* Phase 7 report described Wisconsin/angle/seed=7 as "the only seed-level outlier" (78.95% vs ~91% for others under IID/α=0.5), and attributed this to encoding-specific sensitivity specific to the angle encoding at that partition.
*Why it was retracted:* MN-03a z-score analysis across all four encodings revealed this is an encoding × partition interaction, not an angle-specific phenomenon.  Amplitude/seed=7 is that encoding's *highest* result (z=+1.4), not an outlier.  The "outlier" framing implied angle was uniquely disadvantaged by seed=7's partition; the correct finding is that encodings respond to the same partition geometry in qualitatively different directions.  Additionally, MN-08 extended the explanation to actor-RNG boundary sensitivity: Wisconsin/amplitude/α=0.5/seed=7 shows a 22.81pp range across 3 independent runs, confirming the stochastic dimension.
*Correct treatment:* Seed=7 partition creates an adversarial gradient conflict geometry (C2=100% class-1 vs C0=98% class-0); which encoding escapes or collapses depends on expressive capacity, not a fixed labelling of "outlier."

**2. Phase 7 label-skew accuracy ordering (multi-seed validation, 2026-07-22)**
*Original:* Phase 7 (single seed, seed=42) showed α=0.5 accuracy (57.0%) < α=0.1 accuracy (63.2%), implying "moderate non-IID is worse than extreme non-IID."  This counter-intuitive monotonicity violation was treated as a finding.
*Why it was retracted:* Multi-seed check (seeds 1, 7, 42, 123) showed the α=0.5 < α=0.1 ordering holds only 1/4 seeds.  Seed=42 is the outlier; in the other three seeds, α=0.5 outperforms α=0.1 (expected direction).
*Correct treatment:* Do not cite monotonicity violation as a general finding.  Seed=42's result is noted with its seed label.  The robust claim is that quantity skew shows better accuracy and lower JSD than either label-skew condition.

**3. MN-04a deterministic collapse predictor (MN-04a retraction, 2026-07-30)**
*Original:* Collapse at Wisconsin/amplitude/α=0.1 was claimed to be deterministically predicted by the rule "dominant client ≥60% FedAvg weight + 100% class purity."  Seeds 42 and 999 were labelled as "confirmed collapse."
*Why it was retracted:* Phase 10 replication (3 runs each, seeds 42 and 999) produced COLLAPSED, FUNCTIONAL, and MIXED outcomes from identical partitions and initializations.  Seed=42 Run 1 (Phase 10 RQ3): 0/15 Track-B rounds (fully functional).  Seed=999: 3 independent outcomes including one fully collapsed and two MIXED.
*Correct treatment:* The ≥60%+purity rule identifies high-risk partition geometry (proximity to an attractor boundary) where actor-RNG mini-batch ordering selects the outcome.  It is a risk indicator, not a deterministic rule.  See MN-08 for the unified framework.

**4. Reuploading 'strict within-B' claim (MN-08c retraction, 2026-07-30)**
*Original:* After documenting that angle encoding collapses to Track B on all tested adversarial partitions, an implicit encoding-expressiveness taxonomy emerged: angle encoding as "always-B" vs reuploading as exhibiting only modest within-B variance, suggesting reuploading is categorically more robust in the collapsed-state dimension.
*Why it was retracted:* MN-08c table shows reuploading/seed=42 with a 49.1pp across-run range (2A/1B: two Track-A outcomes and one Track-B), the largest A/B spread in the dataset.  This directly refutes "strict within-B" for reuploading.  Reuploading is bifurcation-sensitive at adversarial partitions, the same as amplitude.  The angle vs others split is preserved (angle is the only encoding that produces all-B outcomes across tested conditions), but the within-reuploading characterization was wrong.
*Correct treatment:* All four encodings show within-condition variance under bifurcation-sensitive geometry.  The claim that reuploading is categorically "within-B" is retired; per-run Track classification is required.

**5. IQP/seed=7 'wrong attractor' (MN-08, 2026-07-30)**
*Original:* A single Phase 10 IID run for IQP encoding at seed=7 produced cosine similarity ≈ 0.07 between the federated explanation and the centralized baseline (near-orthogonal, suggesting a structurally different attractor).  This was described as "IQP seed=7 lands in a wrong attractor under IID."
*Why it was retracted:* Three independent Phase 10 replications of the same condition (IQP/IID/seed=7) all returned cosine similarity ≥ 0.99 — normal attractor, close to centralized baseline.  The original was a single noisy draw.
*Correct treatment:* The original observation documents that actor-RNG noise can produce a qualitatively wrong attractor outcome even under IID conditions (at low probability).  It is not a reproducible bifurcation phenomenon specific to IQP.  Note: IQP/seed=7 continues to show anomalies in Phase 11 non-IID conditions (see RQ3 section flagged candidates below) — these have not yet been replicated and are currently treated as provisional.

**6. Seed=123 MIA below-random AUC (Phase 13b retraction, 2026-08-02)**
*Original:* Phase 13 direct MIA showed below-random AUC (< 0.5) for all four encodings at seed=123 under IID Wisconsin.  Composition check ruled out degenerate attack split.  Initial framing: "genuine partition-dependent privacy resistance; a new axis for RQ4."
*Why it was retracted:* Phase 13b (script `experiments/phase13b_seed123_resample.py`) applied two checks.  (1) Draw 1 re-run: all four encoding AUCs shifted >3pp relative to Phase 13 (IQP flipped from 0.380 to 0.562 across the 0.5 boundary), demonstrating the result is sensitive to actor-RNG variation in the training trajectory.  (2) Draw 2 resample (attack samples [50:100] instead of [0:50]): 0/4 encodings below 0.5 (range 0.544–0.680).  Cross-check of seed=123 in Phase 10/11 RQ1/2/3 data showed no corroborating anomalies.
*Correct treatment:* Observed in one training-trajectory × attack-draw combination; not confirmed.  Do not cite as evidence of partition-dependent privacy resistance.

---

## Phase 11 Data: RQ3 Divergence-by-Distribution Summary

**Status:** 2026-08-02
**Data sources:** Phase 10 log (amplitude non-IID; all centralized and IID-federated baselines) + Phase 11 log (angle, reuploading, IQP non-IID).
**Matrix:** 4 encodings × 3 distributions × 5 seeds = 60 cells (complete).
**Distributions:** α=0.5 (moderate label skew), α=0.1 (severe label skew), quantity (IID class distribution, unequal sample counts).

### JSD computation

`js_divergence()` in `explain/metrics.py`: takes absolute values → normalises to sum-1 → Jensen-Shannon divergence base-2.  Returns values in [0, 1]; higher = more divergence.

Two reference baselines per cell:
- **JSD vs Centralized**: divergence between the non-IID federated global explanation (round 15) and the centralized model trained on the full dataset.  Measures the combined effect of federating + non-IID data distribution.
- **JSD vs IID-Federated**: divergence between the non-IID federated global explanation and the IID-federated global explanation at the same seed.  Isolates the effect of data distribution skew alone (subtracting the federation overhead already present under IID).

Both baselines come from Phase 10 (centralized: 20 rows at round=15; IID-federated: 20 rows at round=15, global).

### ⚠ Track-B artifact warning

When a model collapses (Track B), its explanation vector is near-zero (‖φ‖ ≈ 0).  The JSD between two near-zero vectors is also near-zero — so collapsed models look as though they have excellent fidelity.  This is a documented silent failure mode (see MN-04a, MN-07).  Cells with Track B labels and low JSD values should NOT be interpreted as "low divergence = good"; they reflect a model that has stopped producing meaningful explanations.  The condition mean JSD for heavy-collapse conditions (angle/α=0.1: 4/5 Track-B; IQP/α=0.1: 3/5 Track-B) is dominated by this artifact and should be excluded or annotated in any written comparison.

Where multiple federated runs exist for the same (encoding, distribution, seed) — which occurs for angle/α=0.5 (all 5 seeds) and angle/α=0.1/seed=1 due to replications run to characterise actor-RNG variance — the primary table uses the earliest-timestamp 'federated' run.

### Summary table: mean ± std across 5 seeds

| Encoding | Distribution | JSD vs Cent | JSD vs IID-Fed | Mean Acc | #Track-B |
|---|---|---|---|---|---|
| angle | α=0.5 | 0.212 ± 0.077 | 0.195 ± 0.123 | 0.684 ± 0.152 | 1/5 |
| angle | α=0.1 | 0.092 ± 0.062 † | 0.150 ± 0.037 † | 0.658 ± 0.053 | **4/5** |
| angle | quantity | 0.103 ± 0.044 | 0.057 ± 0.026 | 0.847 ± 0.025 | 0/5 |
| amplitude | α=0.5 | 0.188 ± 0.061 § | 0.100 ± 0.069 § | 0.888 ± 0.105 | 0/5 |
| amplitude | α=0.1 | 0.193 ± 0.099 | 0.237 ± 0.126 | 0.802 ± 0.107 | 0/5 |
| amplitude | quantity | 0.088 ± 0.019 | 0.076 ± 0.014 | 0.937 ± 0.013 | 0/5 |
| reuploading | α=0.5 | 0.212 ± 0.027 ‡ | 0.119 ± 0.015 ‡ | 0.923 ± 0.037 ‡ | 0/5 |
| reuploading | α=0.1 | 0.201 ± 0.081 | 0.190 ± 0.109 | 0.819 ± 0.109 | 1/5 |
| reuploading | quantity | 0.187 ± 0.050 | 0.105 ± 0.039 | 0.942 ± 0.011 | 0/5 |
| IQP | α=0.5 | 0.224 ± 0.060 | 0.333 ± 0.253 ‡ | 0.795 ± 0.084 | 0/5 |
| IQP | α=0.1 | 0.353 ± 0.135 † | 0.466 ± 0.153 † | 0.670 ± 0.188 | **3/5** |
| IQP | quantity | 0.219 ± 0.152 ‡ | 0.198 ± 0.200 ‡ | 0.863 ± 0.033 | 0/5 |

† Mean is dominated by near-zero Track-B explanations — do not interpret as "low divergence."  Exclude Track-B seeds before comparing across conditions.
‡ Seed-specific outlier excluded: reuploading/α=0.5 excludes seed=999 (2/3 draws anomalous ‖φ‖, MN-08 boundary-sensitive — see item 3 resolution); IQP/quantity excludes seed=7 (50% anomaly rate, MN-09 — see Phase 14); reported stats are for the remaining 4 seeds.  Corrected means/std are lower and more reliable than all-5-seed figures.
§ amplitude/α=0.5 excludes seed=123 (JSD_cent elevation reproducible across 3 draws at 4.4σ above LOO mean — see item 4 resolution).  All-5-seed mean would be 0.218 ± 0.078; corrected 4-seed mean is 0.188 ± 0.061.

### Per-seed detail table (round 15, global explanation vs baselines)

| Encoding | Dist | Seed | JSD_cent | JSD_iid | Acc | Track | ‖φ‖ |
|---|---|---|---|---|---|---|---|
| angle | α=0.5 | 1 | 0.1921 | 0.1137 | 0.7719 | A | 0.2542 |
| angle | α=0.5 | 7 | 0.1112 | 0.0982 | 0.8421 | A | 0.3147 |
| angle | α=0.5 | 42 | 0.1553 | 0.1015 | 0.7456 | A | 0.2289 |
| angle | α=0.5 | 123 | 0.3145 | 0.2451 | 0.6579 | A | 0.2022 |
| angle | α=0.5 | 999 | 0.2874 | 0.4144 | 0.4035 | **B** | 0.0212 |
| angle | α=0.1 | 1 | 0.0781 | 0.1508 | 0.7632 | A | 0.1965 |
| angle | α=0.1 | 7 | 0.2131 | 0.1762 | 0.6316 | **B** | 0.0521 |
| angle | α=0.1 | 42 | 0.0353 | 0.0790 | 0.6316 | **B** | 0.0110 |
| angle | α=0.1 | 123 | 0.0723 | 0.1587 | 0.6316 | **B** | 0.0122 |
| angle | α=0.1 | 999 | 0.0610 | 0.1830 | 0.6316 | **B** | 0.0224 |
| angle | quantity | 1 | 0.1106 | 0.0223 | 0.8333 | A | 0.2999 |
| angle | quantity | 7 | 0.1436 | 0.0424 | 0.8509 | A | 0.2867 |
| angle | quantity | 42 | 0.0298 | 0.0462 | 0.8684 | A | 0.3084 |
| angle | quantity | 123 | 0.0830 | 0.0850 | 0.8772 | A | 0.2844 |
| angle | quantity | 999 | 0.1484 | 0.0898 | 0.8070 | A | 0.2945 |
| amplitude | α=0.5 | 1 | 0.1614 | 0.0341 | 0.9649 | A | 0.3549 |
| amplitude | α=0.5 | 7 | 0.1893 | 0.0325 | 0.9649 | A | 0.3765 |
| amplitude | α=0.5 | 42 | 0.2713 | 0.1667 | 0.9298 | A | 0.4055 |
| amplitude | α=0.5 | 123 | 0.3418§ | 0.2172 | 0.8947 | A | 0.4528 |
| amplitude | α=0.5 | 999 | 0.1280 | 0.1340 | 0.6842 | A | 0.1682 |
| amplitude | α=0.1 | 1 | 0.0988 | 0.1595 | 0.8947 | A | 0.1844 |
| amplitude | α=0.1 | 7 | 0.1238 | 0.2419 | 0.8947 | A | 0.1871 |
| amplitude | α=0.1 | 42 | 0.3804 | 0.4678 | 0.6579 | A | 0.0531 |
| amplitude | α=0.1 | 123 | 0.1775 | 0.0972 | 0.6842 | A | 0.1819 |
| amplitude | α=0.1 | 999 | 0.1844 | 0.2168 | 0.8772 | A | 0.1651 |
| amplitude | quantity | 1 | 0.0989 | 0.0811 | 0.9474 | A | 0.3425 |
| amplitude | quantity | 7 | 0.1114 | 0.0765 | 0.9298 | A | 0.3148 |
| amplitude | quantity | 42 | 0.0818 | 0.0526 | 0.9561 | A | 0.3684 |
| amplitude | quantity | 123 | 0.0933 | 0.0752 | 0.9211 | A | 0.3289 |
| amplitude | quantity | 999 | 0.0565 | 0.0957 | 0.9298 | A | 0.3350 |
| reuploading | α=0.5 | 1 | 0.2256 | 0.1207 | 0.9386 | A | 0.4785 |
| reuploading | α=0.5 | 7 | 0.1856 | 0.1269 | 0.9474 | A | 0.4593 |
| reuploading | α=0.5 | 42 | 0.2419 | 0.1084 | 0.9474 | A | 0.4680 |
| reuploading | α=0.5 | 123 | 0.1932 | 0.1317 | 0.8596 | A | 0.3816 |
| reuploading | α=0.5 | 999 | 0.2674 | 0.4056 | 0.7018 | A | 0.0731‡ |
| reuploading | α=0.1 | 1 | 0.2075 | 0.1185 | 0.8772 | A | 0.5059 |
| reuploading | α=0.1 | 7 | 0.1463 | 0.0735 | 0.9386 | A | 0.5027 |
| reuploading | α=0.1 | 42 | 0.0867† | 0.2431 | 0.7368 | A | 0.2068 |
| reuploading | α=0.1 | 123 | 0.2401 | 0.1389 | 0.8947 | A | 0.4056 |
| reuploading | α=0.1 | 999 | 0.3247 | 0.3770 | 0.6491 | **B** | 0.1176 |
| reuploading | quantity | 1 | 0.1425 | 0.1268 | 0.9561 | A | 0.4229 |
| reuploading | quantity | 7 | 0.1284 | 0.0817 | 0.9298 | A | 0.4717 |
| reuploading | quantity | 42 | 0.1730 | 0.0626 | 0.9474 | A | 0.4668 |
| reuploading | quantity | 123 | 0.2432 | 0.1715 | 0.9298 | A | 0.3462 |
| reuploading | quantity | 999 | 0.2477 | 0.0821 | 0.9474 | A | 0.5128 |
| IQP | α=0.5 | 1 | 0.1178 | 0.1974 | 0.8772 | A | 0.3599 |
| IQP | α=0.5 | 7 | 0.2097 | 0.7990 | 0.6667 | A | 0.1824 |
| IQP | α=0.5 | 42 | 0.2495 | 0.0586 | 0.8947 | A | 0.4437 |
| IQP | α=0.5 | 123 | 0.2985 | 0.3627 | 0.7895 | A | 0.2287 |
| IQP | α=0.5 | 999 | 0.2438 | 0.2481 | 0.7456 | A | 0.2798 |
| IQP | α=0.1 | 1 | 0.1327 | 0.1800 | 0.8947 | A | 0.4102 |
| IQP | α=0.1 | 7 | 0.2975 | 0.6090 | 0.8333 | A | 0.2030 |
| IQP | α=0.1 | 42 | 0.5390 | 0.5850 | 0.6316 | **B** | 0.0031 |
| IQP | α=0.1 | 123 | 0.4226 | 0.4657 | 0.3596 | **B** | 0.0191 |
| IQP | α=0.1 | 999 | 0.3716 | 0.4897 | 0.6316 | **B** | 0.0000 |
| IQP | quantity | 1 | 0.1124 | 0.1048 | 0.8860 | A | 0.3647 |
| IQP | quantity | 7 | 0.5173 | 0.5890 | 0.7982 | A | 0.1232 |
| IQP | quantity | 42 | 0.1306 | 0.1704 | 0.8772 | A | 0.3166 |
| IQP | quantity | 123 | 0.1999 | 0.0966 | 0.8684 | A | 0.3285 |
| IQP | quantity | 999 | 0.1332 | 0.0301 | 0.8860 | A | 0.3946 |

§ amplitude/α=0.5/seed=123: JSD_cent=0.342 (primary) is reproducible — all 3 draws elevated (0.342–0.468); driven by attribution-direction mismatch with centralized model, not by explanation magnitude.  Exclude from condition-level mean or report separately.  See item 4 resolution below.

† reuploading/α=0.1/seed=42: JSD_cent=0.087 was a single-draw positive anomaly (actor-RNG fluke).  Replication draws: 0.220, 0.205 (normal range).  Reliable estimate: ~0.21.  See item 6 resolution below.

‡ reuploading/α=0.5/seed=999: ‖φ‖=0.073 is from the primary draw; 2/3 draws are anomalous (‖φ‖ < 0.15 in draws 1 and 2).  Mean ‖φ‖ across all 3 draws = 0.165 ± 0.159.  This seed is MN-08 boundary-sensitive under α=0.5; exclude from condition-level ‖φ‖ mean or annotate prominently.  See item 3 resolution below.

### Preliminary findings

#### Pattern 1 — Quantity skew causes least explanation divergence

Across all four encodings, quantity skew (IID class distribution, unequal sample counts) produces the lowest JSD vs both baselines (centralized and IID-federated).  JSD vs IID-federated under quantity skew ranges 0.03–0.19 across seeds and encodings; under label skew, it ranges 0.06–0.80.  This is consistent with the accuracy finding (quantity skew preserves accuracy close to IID baseline): when clients see both classes, both the model and its explanation geometry stay close to the balanced-data baseline.

#### Pattern 2 — Severe label skew (α=0.1) collapses angle explanation; artificially depresses JSD

4/5 angle/α=0.1 seeds are Track B with ‖φ‖ < 0.06.  The condition JSD vs centralized (0.092) is the lowest angle value across all three distributions — lower even than quantity skew (0.103).  This is the Track-B artifact: near-zero explanations produce small JSD against any baseline.  For the one Track-A angle/α=0.1 seed (seed=1, acc=0.763, ‖φ‖=0.197), JSD_cent=0.078 — within the range of quantity skew.  The only safe claim for angle/α=0.1 is the collapse rate (4/5 seeds); explanation fidelity is uninterpretable for the collapsed seeds.

IQP/α=0.1 (3/5 Track-B) has the same artifact in the opposite direction: the three collapsed seeds have non-zero but very low ‖φ‖ (0.000–0.019) and moderately high JSD (0.37–0.54), because the near-zero IQP explanation diverges structurally from the IID-federated explanation (which has ‖φ‖ ≈ 0.34).  This inflates the condition mean JSD artificially.

#### Pattern 3 — Amplitude and reuploading show interpretable fidelity across all conditions

Amplitude: 0/15 Track-B across all three non-IID conditions; reuploading: 1/15 Track-B (reuploading/α=0.1/seed=999).  For these two encodings, JSD values are interpretable and show a consistent gradient: quantity < α=0.5 < α=0.1 for JSD vs centralized (0.088, 0.218, 0.193 for amplitude; 0.187, 0.223, 0.201 for reuploading), with the α=0.5 vs α=0.1 ordering slightly reversed in both.  Note: the reuploading condition means exclude seed=999 under α=0.5 and seed=42's primary value under α=0.1, per the replication verdicts in items 3 and 6 below.  With those seeds correctly handled, the condition-level numbers remain stable.

### Replication candidates flagged

LOO outlier detection was applied to each (encoding, distribution) condition: for each seed, compare its value to the LOO mean of the other 4 seeds; flag if deviation > max(2 × LOO-std, 0.10).  Cells explained by documented mechanisms (Track-B collapse, MN-04a bifurcation sensitivity) are noted but do not require new replication runs.

**1. IQP / quantity / seed=7 [RESOLVED — Phase 14]**
- Phase 11 primary: JSD_cent=0.517, JSD_iid=0.589, Track A, acc=0.799, ‖φ‖=0.123 (ANOMALOUS)
- LOO mean for JSD_cent: 0.144 ± 0.033 (dev=3.7σ); for JSD_iid: 0.100 ± 0.050 (dev=4.9σ)
- Phase 14 replications: 3 fresh quantity-skew reps + 3 fresh IID reps; full table in Phase 14 section below.
- **Verdict: initialization-specific instability, not partition-driven.**  Anomaly rate under quantity-skew = 2/4 (50%); anomaly rate under IID = 2/4 (50%).  Since IID has zero non-IID mechanism, the identical rate under IID rules out the quantity partition as the cause.  IQP/seed=7's particular initialization interacts with IQP's entangling structure to produce unstable explanation geometry in approximately half of all training runs, regardless of distribution.
- Correct treatment for RQ3 table: IQP/quantity/seed=7 is an unreliable single value (50% anomaly rate under any distribution).  Exclude from condition-level aggregation; compute IQP/quantity mean and std over seeds 0, 1, 42, 999 only.  Do not attribute the elevated JSD to the quantity partition.

**2. IQP / α=0.5 / seed=7 [RESOLVED — existing Phase 11 data, new MN-08 instance]**
- Phase 11 primary: JSD_cent=0.210, JSD_iid=0.799, Track A, acc=0.667, ‖φ‖=0.182 (ANOMALOUS)
- LOO mean for JSD_iid: 0.217 ± 0.109 (dev=5.3σ); JSD_iid=0.799 is the maximum in the entire Phase 11 dataset.
- Phase 11 also contains 2 MN-08 replication draws for this cell (triggered by G1-geometry flag): draw 2 (‖φ‖=0.119, acc=0.737, Track A) and draw 3 (‖φ‖=0.140, acc=0.465, Track B).  By the strict ‖φ‖ < 0.15 criterion: 2/3 draws are anomalous (draws 2 and 3).  Draw 1 (‖φ‖=0.182) is above the threshold but well below the normal IQP range of 0.28–0.41; all three draws cluster in the 0.12–0.18 band with zero draws reaching normal geometry.  No new runs needed.
- **Verdict: near-persistent anomaly under α=0.5 — new MN-08 instance.**  Unlike the IID/quantity cases (50% stochastic bistability with normal draws clearly present), all 3 α=0.5 draws have below-normal norms (0.12–0.18 vs 0.28–0.41 for other IQP seeds).  The α=0.5 partition tilts IQP/seed=7 strongly toward the degenerate regime; the moderate label skew creates systematic gradient conflicts that make the normal-geometry attractor largely inaccessible for this seed's initialization.
- Structurally unusual combination: JSD_cent=0.210 (within normal range) but JSD_iid=0.799 (maximum in dataset).  The explanation is close to the centralized model but very far from the IID-federated baseline for this seed.  Note: the IID/seed=7 baseline is itself anomalous (Phase 10 norm=0.083, MN-08 single noisy draw), so JSD_iid is not reliable as a reference for any IQP/seed=7 cell.
- Correct treatment: report as new MN-08 instance (3/3 draws anomalous, attractor-boundary-sensitive under α=0.5).  Exclude from IQP/α=0.5 condition mean for JSD_iid.  Use JSD_cent if reporting single-seed data for this cell.

**3. Reuploading / α=0.5 / seed=999 [RESOLVED — Phase 11 inline MN-08 replications]**
- Primary draw: JSD_iid=0.406, acc=0.702, Track A, ‖φ‖=0.073
- LOO mean for JSD_iid: 0.122 ± 0.009 (dev=31.6σ); LOO mean for acc: 0.923 ± 0.037 (dev=5.9σ)
- ‖φ‖=0.073 vs 0.382–0.479 for other reuploading/α=0.5 seeds.

Phase 11 inline MN-08 replication (Trigger S fired: acc=0.702, dev=5.9σ below LOO mean 0.923):

| Draw | Type | acc | Track | ‖φ‖ | JSD_iid | JSD_cent | Anomalous? |
|------|------|-----|-------|-----|---------|---------|-----------|
| 1 (primary) | federated | 0.702 | A | 0.073 | 0.406 | 0.267 | YES (‖φ‖<0.15) |
| 2 | replication | 0.711 | A | 0.072 | 0.371 | 0.261 | YES (‖φ‖<0.15) |
| 3 | replication | 0.798 | A | 0.349 | 0.097 | 0.152 | no |

Anomaly rate: 2/3 draws with ‖φ‖ < 0.15.  Mean ‖φ‖ = 0.165 ± 0.159.

**Verdict: Stochastically persistent near-collapse (new MN instance).** ‖φ‖ < 0.15 in 2/3 draws; the degenerate-explanation attractor is the more probable outcome (~2/3) for this seed under α=0.5.  Unlike IQP/seed=7 (50% bistability, partition-independent), this seed produces the degenerate state more consistently under α=0.5.  The model reaches Track A accuracy (0.70–0.71) in both anomalous draws but develops near-zero explanation geometry — a borderline functional-model/collapsed-explanation state distinct from full Track-B collapse.  Draw 3 (acc=0.798, ‖φ‖=0.349) shows that the normal-geometry attractor is reachable but less probable.

Correct treatment: this seed is MN-08 boundary-sensitive.  Report as 1/5 seed with stochastic explanation instability under α=0.5.  Exclude from reuploading/α=0.5 condition-level ‖φ‖ mean or annotate prominently (including 2/3 anomalous draws would depress the condition mean by ~40%).  JSD_iid for this seed should be reported as the range across draws (0.097–0.406) rather than a single value.

**4. Amplitude / α=0.5 / seed=123 [RESOLVED — experiments/rq3_anomaly_replication.py, 2 fresh draws]**
- Primary draw (phase10): JSD_cent=0.342, JSD_iid=0.217, Track A, acc=0.895, ‖φ‖=0.453
- LOO mean for JSD_cent: 0.188 ± 0.053 (dev=2.9σ); for JSD_iid: 0.092 ± 0.060 (dev=2.1σ)
- Hypothesized mechanism (pre-replication): high ‖φ‖ produces high JSD mechanically.

Fresh replication (2 draws, `rq3_anomaly_replication.py`):

| Draw | Type | acc | Track | ‖φ‖ | JSD_iid | JSD_cent | Elevated (>0.294)? |
|------|------|-----|-------|-----|---------|---------|-------------------|
| 1 (primary) | federated | 0.895 | A | 0.453 | 0.217 | 0.342 | YES |
| 2 | replication | 0.965 | A | 0.172 | 0.392 | 0.468 | YES |
| 3 | replication | 0.965 | A | 0.173 | 0.376 | 0.452 | YES |

Elevated draws: 3/3.  JSD_cent mean=0.421 (4.4σ above LOO mean); range 0.342–0.468.

Mechanical hypothesis FALSIFIED: reps 2 and 3 have ‖φ‖=0.172–0.173, similar to seed=999 (‖φ‖=0.168, JSD_cent=0.128) — yet seed=123 reps have JSD_cent=0.45+.  The elevated JSD_cent is driven by attribution-direction mismatch, not explanation magnitude.

**Verdict: Genuine, reproducible seed-specific partition effect.** Amplitude/α=0.5/seed=123 consistently converges to a local optimum with qualitatively different feature attributions than the centralized model (4.4σ elevation in JSD_cent, stable across actor-RNG draws).  The specific Dirichlet partition at seed=123 creates a distribution of client-local feature correlations that steers federated gradient averaging away from the centralized attribution geometry.  Not a single-draw artifact and not driven by explanation magnitude.

Correct treatment: report seed=123 as an outlier in the amplitude/α=0.5 condition-level JSD_cent analysis.  The condition mean should either exclude this seed or report it separately.  Do NOT attribute the elevation to explanation magnitude — the corrected mechanism is attribution-direction divergence caused by partition geometry.  The finding is consistent with seed=123 being a recurring cross-phase flag (Phase 7 α=0.5 ordering reversal, Phase 13 MIA anomaly — now retracted): this seed's initialization reliably interacts with α=0.5 partition geometry to produce qualitatively distinct model behavior.

**5. Amplitude / α=0.1 / seed=42**
- JSD_cent=0.380, JSD_iid=0.468, Track A, acc=0.658, ‖φ‖=0.053
- LOO mean for JSD_cent: 0.146 ± 0.036 (dev=6.5σ); for acc: 0.838 ± 0.089 (dev=2.0σ)
- Very low ‖φ‖ (0.053 vs 0.165–0.453 for other amplitude/α=0.1 seeds).  This is the MN-04a documented case: amplitude/Wisconsin/α=0.1/seed=42 is explicitly listed as "bifurcation-sensitive" (Phase 10 3 reps: collapsed/functional/mixed outcomes).  The current Phase 11 run represents one draw from that distribution.  **Not a new anomaly; apply the MN-04a bifurcation-sensitive label.  No new replication required beyond what MN-04a already captured.**

**6. Reuploading / α=0.1 / seed=42 (LOW JSD direction) [RESOLVED — Phase 11 inline MN-08 replications]**
- Primary draw: JSD_cent=0.087, LOO mean=0.230 ± 0.064 (dev=2.2σ), Track A, acc=0.737, ‖φ‖=0.207
- Anomalously LOW JSD — federated explanation unusually close to centralized baseline despite severe non-IID.

Phase 11 inline MN-08 replication (Trigger S fired):

| Draw | Type | acc | Track | ‖φ‖ | JSD_iid | JSD_cent | Anomalous (low)? |
|------|------|-----|-------|-----|---------|---------|-----------------|
| 1 (primary) | federated | 0.737 | A | 0.207 | 0.243 | 0.087 | YES (JSD_cent<0.102) |
| 2 | replication | 0.386 | B | 0.068 | 0.192 | 0.220 | no |
| 3 | replication | 0.877 | A | 0.125 | 0.346 | 0.205 | no |

Draws with JSD_cent below anomaly threshold (0.102): 1/3.

**Verdict: Single-draw positive anomaly; not reproducible.** The primary draw's low JSD_cent (0.087) reflects a fortunate actor-RNG training trajectory where the global model converged to near-centralized explanation geometry despite severe label skew.  Draw 2 collapsed to Track B (JSD is meaningless in that state).  Draw 3 recovered Track A but JSD_cent=0.205 — in the normal range for this condition (LOO mean=0.230).  The anomaly is actor-RNG-sensitive and does not reflect a systematic structural property of this partition or seed.  

Correct treatment: do not cite as evidence that α=0.1 preserves explanation fidelity.  The condition's reliable JSD_cent estimate is from the two Track-A draws excluding the primary: mean ≈ 0.21.  No new MN entry required — standard actor-RNG single-draw fluctuation.

---

### Phase 14 — IQP/seed=7 investigation: quantity-skew and IID replications

**Motivation:** Two independent anomaly signals for IQP/seed=7 under quantity-skew — an accuracy-adjacent outlier in earlier sweep, and now JSD_cent=0.517 (3.7σ above LOO mean) in the Phase 11 RQ3 flag.  The α=0.5 case already had 3 Phase 11 draws available; the question there was resolved without new runs.  The quantity case required new runs because the stochastic vs persistent distinction was open.  An IID control arm was added to test whether the anomaly is partition-independent.

**Protocol (experiments/phase14_iqp_seed7_investigation.py):** 3 fresh quantity-skew reps + 3 fresh IID reps.  Data loading: `random_state=42` matching Phase 10/11.  15 rounds FedAvg, 3 local epochs, n_global_explain=10, n_global_bg=20, n_permutations=8.  Anomaly criteria: ‖φ‖ < 0.15 (primary) or JSD_cent > 0.40.  Normal IQP IID seeds show ‖φ‖ 0.32–0.41, JSD_cent 0.11–0.22.

**Full results table:**

| Draw | Split | acc | Track | ‖φ‖ | JSD_cent | Anomalous? |
|------|-------|-----|-------|-----|----------|-----------|
| Phase 11 primary | quantity | 0.799 | A | 0.123 | 0.517 | YES |
| qty rep 1 | quantity | 0.868 | A | 0.406 | 0.144 | no |
| qty rep 2 | quantity | 0.851 | A | 0.074 | 0.326 | YES |
| qty rep 3 | quantity | 0.860 | A | 0.374 | 0.133 | no |
| Phase 10 primary | IID | — | — | 0.083 | — | YES (MN-08 known draw) |
| iid rep 1 | IID | 0.895 | A | 0.387 | 0.158 | no |
| iid rep 2 | IID | 0.868 | A | 0.240 | 0.261 | no |
| iid rep 3 | IID | 0.798 | A | 0.068 | 0.857 | YES |

Quantity draws: 2/4 anomalous (50%).  IID draws: 2/4 anomalous (50%).  Aggregate: ‖φ‖ mean±std = 0.244±0.147 (quantity), 0.232±0.130 (IID) — statistically indistinguishable.

**Verdict — IQP/quantity/seed=7:** Anomaly rate is identical under quantity-skew and IID.  Since IID has zero non-IID mechanism (equal class distribution, equal sample counts, no Dirichlet partition), the quantity partition cannot be the cause.  IQP/seed=7 has approximately a 50% per-draw anomaly rate under any distribution.  This is an initialization-specific stochastic instability: seed=7's random weight initialization places IQP near an optimization saddle where actor-RNG scheduling noise determines which of two regimes — normal convergence or near-degenerate explanation geometry — the training trajectory reaches.

**Contrast with IQP/α=0.5/seed=7:** That case shows 3/3 draws anomalous (persistent), while quantity and IID show 50% stochastic rate.  The α=0.5 partition is not merely amplifying the baseline stochasticity — it qualitatively changes the optimization landscape for this seed, making the degenerate regime the consistent attractor.  Mechanism: the moderate label skew under α=0.5, combined with seed=7's initialization, likely creates a systematic gradient conflict (one or more clients contributing strongly misaligned updates) that the quantity partition (which preserves class balance across clients) does not.

**MN-08 additions from Phase 14:**
- IQP / α=0.5 / seed=7: 2/3 draws anomalous by ‖φ‖ < 0.15; all 3 below normal range → near-persistent MN-08 instance.  Distinct from the IID single-draw case (which resolved to normal after 3 replications) and from the G1-triggered full collapse cases.
- IQP / quantity / seed=7 and IQP / IID / seed=7: ≈50% stochastic anomaly rate, partition-independent → new entry for initialization-sensitive stochastic bistability.  Not captured by original G1/G2 geometry triggers (partition geometry is normal: largest client 45.7% weight, purity 67.8%, well below G1 thresholds).

**IQP/α=0.1/seed=7 — check from existing Phase 11 data:**
Phase 11 contains 3 MN-08 draws for IQP/α=0.1/seed=7 (triggered by Trigger S — statistical outlier in accuracy; G1/G2 geometry flags are NONE for this partition, confirmed by Phase 15):
- Draw 1: norm=0.203, Track A, acc=0.833 → below-normal (above 0.15 threshold; below normal range 0.28–0.41)
- Draw 2: norm=0.091, Track A, acc=0.754 → ANOMALOUS
- Draw 3: norm=0.034, Track B, acc=0.632 → ANOMALOUS (full collapse, not geometry-triggered)

By ‖φ‖ < 0.15: 2/3 anomalous.  Like α=0.5, zero draws reach the normal IQP norm range.  Because G1/G2 = NONE, the Track-B draw is not explained by the geometry-triggered majority-class-lock mechanism that causes collapse in other seeds under α=0.1 — it is seed=7's initialization instability reaching its most severe form under label skew without any unusual partition geometry.  The Track-A anomalous draw (norm=0.091) provides the cleaner seed=7 instability signal.

**Phase 15 — draw 4 each for α=0.5 and α=0.1 (experiments/phase15_iqp_seed7_label_skew.py):**

Both n=3 counts rested on samples too small to distinguish "hard zero escape probability" from "50% Bernoulli rate with 0/3 normal by chance (p=0.125)."  Phase 15 added one fresh draw per condition.  Partition geometry confirmed G1/G2 = NONE for both (consistent with Phase 11 not having been triggered by Trigger G for either condition — Trigger S fired on accuracy outlier).

- α=0.5 draw 4: acc=0.421, Track B, ‖φ‖=0.122, JSD_cent=0.224 → ANOMALOUS (full collapse, second Track-B draw for this condition)
- α=0.1 draw 4: acc=0.798, Track A, ‖φ‖=0.271, JSD_cent=0.326 → below-normal (closest any label-skew draw has reached to the normal ‖φ‖ ≥ 0.28 threshold; 2.5% below)

Both conditions remain at 0/4 normal draws.

**Complete cross-distribution picture for IQP/seed=7 (Phase 15 final):**

| Distribution | Total draws | ‖φ‖ < 0.15 (anomalous) | 0.15 ≤ ‖φ‖ < 0.28 (below-normal) | ‖φ‖ ≥ 0.28 (normal) | Track-B | G1/G2 |
|---|---|---|---|---|---|---|
| IID | 4 | 2/4 (50%) | 1/4 | 1/4 | 0 | NONE |
| quantity | 4 | 2/4 (50%) | 0/4 | 2/4 | 0 | NONE |
| α=0.5 | 4 | 3/4 (75%) | 1/4 | 0/4 | 2 | NONE |
| α=0.1 | 4 | 2/4 (50%) | 2/4 | 0/4 | 1 | NONE |

No other IQP seed shows this pattern.  Normal IQP seeds (1, 42, 123, 999) have ‖φ‖ 0.28–0.52 in nearly all non-Track-B draws.  G1/G2 = NONE across all four conditions confirms the anomaly is not geometry-triggered.

**Implications for RQ3 table:**
- IQP/quantity condition mean and std: compute over seeds 0, 1, 42, 999 only; flag seed=7 exclusion in table footnote.
- IQP/α=0.5/seed=7 single reported value: use JSD_cent (not JSD_iid); the IID/seed=7 baseline is unreliable (MN-08 single noisy draw, norm=0.083).  Better to exclude this seed from the IQP/α=0.5 condition mean and report the near-persistent anomaly as a separate finding.
- IQP/α=0.1/seed=7: exclude from IQP/α=0.1 condition mean for the same reason.  The Track-B draw may be geometry-triggered (not seed=7-specific), but the Track-A anomalous draw is consistent with the seed=7 instability pattern.

See the consolidated IQP/seed=7 section below for the full taxonomic write-up.

---

## RQ2 — Explanation stability across training rounds (IID)

**Source data:** phase10_log.csv, IID federated, global client, all 4 encodings × 5 seeds, rounds 0–15.  
**Metrics:** For each (encoding, seed), compute consecutive round-pair statistics — Kendall's tau on |φ| feature ranks, cosine similarity on φ vectors, L2 distance between consecutive round φ vectors — then average over the 15 pairs (rounds 0→1 through 14→15).  Kendall's tau uses absolute values (|φ|) because the rank of importance magnitude, not sign, is the interpretable quantity.

**Note on metric resolution:** With n=4 features (PCA to 4 qubits), Kendall's tau takes only 7 distinct values (−1, −2/3, −1/3, 0, 1/3, 2/3, 1). Once a single feature dominates the ranking, tau locks at 1.0 for all subsequent pairs regardless of whether the other three features are still shifting. The metric therefore measures "does the top-ranked feature stay the same?" more than "are all four relative importances stable?" This is a resolution limitation, not a stability failure — it should be noted when interpreting values near 1.0.

### Per-seed results

| Encoding | Seed | Track | Acc | ‖φ‖ | mean τ | std τ | mean cos | mean L2 | std L2 |
|----------|------|-------|-----|-----|--------|-------|----------|---------|--------|
| angle | 1 | A | 0.851 | 0.298 | 0.978 | 0.083 | 0.999 | 0.020 | 0.013 |
| angle | 7 | A | 0.816 | 0.320 | 0.911 | 0.191 | 0.882 | 0.022 | 0.011 |
| angle | 42 | A | 0.851 | 0.293 | 0.889 | 0.337 | 0.944 | 0.020 | 0.012 |
| angle | 123 | A | 0.868 | 0.273 | 0.956 | 0.113 | 0.871 | 0.019 | 0.013 |
| angle | 999 | A | 0.868 | 0.324 | 0.911 | 0.147 | 1.000 | 0.021 | 0.014 |
| amplitude | 1 | A | 0.965 | 0.340 | 0.933 | 0.133 | 0.870 | 0.025 | 0.020 |
| amplitude | 7 | A | 0.965 | 0.320 | 0.956 | 0.113 | 1.000 | 0.027 | 0.026 |
| amplitude | 42 | A | 0.930 | 0.339 | 0.889 | 0.199 | 0.972 | 0.025 | 0.021 |
| amplitude | 123 | A | 0.930 | 0.304 | 0.911 | 0.147 | 0.992 | 0.023 | 0.016 |
| amplitude | 999 | A | 0.956 | 0.285 | 0.844 | 0.206 | 0.870 | 0.024 | 0.027 |
| reuploading | 1 | A | 0.939 | 0.445 | 0.911 | 0.147 | 0.870 | 0.035 | 0.037 |
| reuploading | 7 | A | 0.930 | 0.403 | 0.933 | 0.133 | 0.991 | 0.032 | 0.022 |
| reuploading | 42 | A | 0.939 | 0.494 | 0.867 | 0.204 | 0.992 | 0.034 | 0.033 |
| reuploading | 123 | A | 0.947 | 0.399 | 0.822 | 0.401 | 0.975 | 0.030 | 0.026 |
| reuploading | 999 | A | 0.956 | 0.506 | 0.933 | 0.133 | 0.994 | 0.034 | 0.030 |
| IQP | 1 | A | 0.851 | 0.322 | 0.756 | 0.333 | 0.962 | 0.037 | 0.026 |
| IQP | 7 †| A | 0.825 | 0.083 | 0.667 | 0.344 | 0.864 | 0.022 | 0.008 |
| IQP | 42 | A | 0.868 | 0.354 | 0.844 | 0.295 | 0.902 | 0.030 | 0.029 |
| IQP | 123 | A | 0.868 | 0.322 | 0.889 | 0.199 | 0.878 | 0.030 | 0.027 |
| IQP | 999 | A | 0.886 | 0.415 | 0.956 | 0.166 | 0.996 | 0.031 | 0.025 |

† IQP/seed=7: norm=0.083 is from the anomalous Phase 10 primary draw.  MN-09 documents that ~50% of IID draws for this seed are anomalous (norm < 0.15) and ~50% are normal (norm 0.24–0.41).  The τ=0.667 reflects the anomalous trajectory (near-zero explanations have noise-dominated rankings) and is not a representative stability estimate for this seed×encoding.  **Excluded from IQP encoding mean.**

### Summary by encoding

| Encoding | mean τ (5 seeds) | std τ | mean cos | mean L2 | Notes |
|----------|-----------------|-------|----------|---------|-------|
| angle | 0.929 ± 0.033 | — | 0.939 | 0.020 | — |
| amplitude | 0.907 ± 0.038 | — | 0.941 | 0.025 | — |
| reuploading | 0.893 ± 0.043 | — | 0.964 | 0.033 | — |
| IQP | **0.861 ± 0.073** (n=4) | — | 0.921 | 0.030 | seed=7 excluded (MN-09: anomalous draw, norm=0.083); incl. seed=7: 0.822 ± 0.101 |

### Findings

**F1 — All four encodings are stable by round-to-round tau under IID.**  Mean tau (seed-7-excluded IQP) ranges 0.861–0.929.  The lowest per-seed value is IQP/seed=7 (τ=0.667), which is from the Phase 10 anomalous draw (norm=0.083, MN-09); the Phase 10 primary draw is one of two coin-flip outcomes (~50% anomaly rate under IID), not representative of IQP's typical stability.  Excluding seed=7, all 19 remaining (encoding, seed) pairs have mean τ ≥ 0.756 and 17/19 have mean τ ≥ 0.822.

**F2 — Encoding ranking by tau (n=4 for IQP): angle > amplitude > reuploading > IQP.**  Differences are small (0.929 vs 0.861 extremes, ≈7pp spread using corrected IQP mean).  Within-encoding seed variance without seed=7: IQP std=0.073, comparable to angle (std=0.033) and amplitude (std=0.038).  The ranking should not be presented as a firm result; per-seed distributions overlap substantially across all four encodings.

**F3 — Cosine similarity is consistently high across all encodings (0.87–1.00 per seed).**  This confirms that even when the absolute tau drops below 1.0, the explanation vectors are moving in nearly the same direction round-to-round — the explanation geometry is stable, not rotating.

**F4 — Reuploading has higher L2 distance (mean 0.033) than angle and amplitude (0.020–0.025) but also higher norm (0.40–0.51).**  The L2 gap is mechanical: higher-norm vectors produce larger absolute displacements even when relative changes are the same.  Cosine similarity for reuploading (0.964) is the highest of the four encodings, consistent with this interpretation.

**F5 — Within-seed tau variance is higher for IQP than for angle and amplitude.**  Per-seed std_τ for IQP (excluding seed=7): 0.17–0.33 vs angle 0.08–0.34 and amplitude 0.13–0.21.  Lower-norm seeds (IQP/seed=1: norm=0.322, the lowest non-anomalous norm) show higher within-seed variance, consistent with more noise in the ranking signal when the explanation magnitude is smaller.  IQP/seed=7's std_τ=0.344 is computed on the anomalous trajectory and is not included in the IQP characterization.

**Metric limitation caveat:** At n=4 features, tau can only detect rank-order changes at the level of the dominant feature.  Subtle re-orderings among the lower-ranked features are not captured.  All values near τ=1.0 should be interpreted as "the top-ranked feature did not change" rather than "the full explanation geometry was stable."  Cosine similarity provides a complementary check that does not suffer from this discretization issue.

---

## MN-09 — IQP/seed=7: initialization-specific bimodal instability across all distributions

**Summary:** IQP/seed=7 is bimodally unstable in its explanation geometry across all four distribution types (IID, quantity-skew, α=0.5, α=0.1). The anomaly is specific to this seed×encoding combination and is not triggered by partition geometry — G1/G2 flags are absent for all four conditions (confirmed by Phase 15 for the label-skew partitions). It is not a training failure by the accuracy criterion — all anomalous Track-A draws have functional accuracy (0.75–0.85) — but produces near-zero explanation vectors that make the model's attribution output unreliable.

**Evidence:** 16 total draws across all four distributions (Phase 15 closed at n=4 per condition). See Phase 14 section above for the IID and quantity tables; Phase 15 section above for the α=0.5 and α=0.1 draw 4 results. Phase 11 provided draws 1–3 for the label-skew conditions.

**Taxonomy:**

*Class-balanced conditions (IID, quantity-skew):* Stochastic bistability. Approximately 50% of training runs converge to normal explanation geometry (‖φ‖ 0.24–0.41, JSD_cent consistent with peers), the other 50% to near-degenerate geometry (‖φ‖ < 0.15, anomalously high JSD_cent). The outcome is determined by actor-RNG scheduling noise, not the data partition — confirmed by the equal anomaly rate under IID (zero non-IID mechanism) and quantity-skew. Partition geometry is normal in both cases (G1/G2 absent).

*Label-skewed conditions (α=0.5, α=0.1):* The bistability shifts toward the degenerate regime. All 8 draws across both distributions (n=4 each, Phase 15 confirmed) have ‖φ‖ < 0.28 (below the normal IQP range); zero draws reach normal explanation geometry. The closest approach is α=0.1 draw 4 at ‖φ‖=0.271 — 2.5% below the 0.28 threshold, and notably the only label-skew draw that is not also strictly anomalous (‖φ‖ ≥ 0.15). Label skew consistently prevents seed=7's IQP from reaching the normal explanation geometry that roughly half of IID and quantity-skew runs achieve. The 0.271 result suggests the α=0.1 escape probability may be slightly higher than α=0.5 (where the highest norm across 4 draws is 0.182), but both are zero at n=4.

*MIA leakage axis — replication of the balanced-vs-label-skewed taxonomy (RQ4, 2026-08-04):* The pattern established on the accuracy/explanation-norm axis replicates on the membership inference leakage axis, confirming the taxonomy is not axis-specific. Under qty_skew (class-balanced, non-IID): 3/3 replication draws land Track A with AUC 0.844–0.864 (mean=0.856) — fully stable, no bimodality. Under α=0.1 (label-skewed): 3 draws split across Track A (AUC 0.964, 0.866) and Track B (AUC 0.714), reproducing the bistability at a 33% severe-outcome rate. The MIA leakage value is therefore distribution-dependent in exactly the direction MN-09 predicts: high and stable under a class-balanced condition, bimodal and lower on average under the label-skewed condition. This extends the mechanism hypothesis — label skew not only collapses the explanation norm but also destabilizes the accuracy/track classification, placing the draw in a lower-leakage regime when the model collapses. The result is not a new finding but a cross-axis confirmation of MN-09's existing taxonomy.

*Track-B draws under label skew:* Two Track-B draws appear under α=0.5 (acc=0.465 and acc=0.421, Phase 11 draws 3 and 4) and one under α=0.1 (acc=0.632, Phase 11 draw 3). Phase 15 confirms G1/G2 = NONE for both label-skew partitions — these collapses are therefore NOT explained by the geometry-triggered majority-class-lock mechanism that causes Track-B in other seeds under α=0.1. They are seed=7's initialization instability reaching its most severe form (full accuracy collapse) under label skew, without any unusual partition geometry as the proximate trigger. This is distinct from MN-08's geometry-triggered collapse in other seeds, which requires extreme partition structure (G1: ≥60% weight + ≥90% purity).

**Mechanism hypothesis:** Seed=7's random weight initialization places IQP's entangling circuit near an optimization landscape saddle. Under balanced data (IID, quantity), the saddle can be cleared in approximately half of training trajectories depending on which actor-RNG trajectory the Ray scheduler takes. Under label skew (α=0.5, α=0.1), the systematic gradient imbalance from non-IID client updates consistently pushes the trajectory toward the degenerate explanation regime, not because of a catastrophic accuracy failure but because the feature-attribution signal collapses while classification accuracy is maintained.

**Distinction from other documented anomalies:**
- MN-08 IID single-draw (Phase 10 primary): One anomalous draw out of 4 total; 3 normal fresh runs. That case was interpreted as "single noisy draw, not a real finding" — correct for the IID condition in isolation, but underestimated the seed×encoding instability.
- MN-08 α=0.5 persistent (Phase 15 confirmed): 4/4 draws below-normal, 3/4 strictly anomalous, 2/4 Track-B — zero draws reach normal geometry at n=4. Distinctly more severe than IID (~50% normal).
- MN-04a amplitude/α=0.1/seed=42 bifurcation: Track-B collapse specific to G1-flagged partition geometry. IQP/seed=7's Track-B draws under label skew are qualitatively different — G1/G2 = NONE, so the collapse arises from the initialization instability alone, not extreme partition geometry.

**Reporting implications:**
- RQ2 (stability): Exclude IQP/seed=7 from the IQP encoding mean τ. The Phase 10 primary τ=0.667 is from an anomalous draw (norm=0.083) — reporting it inflates IQP's instability score. IQP mean τ over seeds 1, 42, 123, 999 = 0.861 ± 0.073. If reporting seed=7, flag that τ reflects the anomalous trajectory and that ~50% of IID draws for this seed achieve normal norm with expected τ ≈ 0.84–0.96.
- RQ3 (divergence): Exclude IQP/seed=7 from condition means for all four distributions. JSD values from anomalous draws are not informative about distribution-level explanation fidelity.
- RQ1 (fidelity): Same exclusion applies. Do not use IQP/seed=7's JSD as evidence about IQP's general fidelity under any condition.
- What can be reported: IQP/seed=7 as a standalone case study in initialization-specific instability under federated training. The finding is that a specific quantum initialization can produce systematically unreliable explanations without accuracy failure — a failure mode not visible in accuracy metrics alone.
- RQ4 nonIID MIA / IQP/qty_skew/seed=7: Primary draw AUC=0.864 confirmed genuine by 3/3 replication draws Track A, AUC 0.844–0.864, mean=0.856. Use 0.856 for seed=7 in IQP/qty_skew 5-seed mean. Corrected 5-seed mean = 0.722 (negligible shift from 0.723; “IQP most dangerous under qty_skew” finding is unaffected). No annotation needed beyond noting the replication confirms the single draw.
- RQ4 nonIID MIA / IQP/α=0.1/seed=7: Primary draw AUC=0.964 (Track A) is bimodally unstable — 3 draws split Track A (0.964, 0.866) / Track B (0.714), 3-draw mean=0.848. Use 0.848 for seed=7 in IQP/α=0.1 Track A mean. Corrected 4-seed Track A mean (seeds 1, 7, 42, 123; seed=999 is Track B): (0.586 + 0.848 + 0.826 + 0.434)/4 = 0.674. Cite MN-09 for the bimodality; do not re-explain the bistability inline in the write-up.

---

## MN-10: Threshold-Crossing Classification — Within-Seed Reproducibility Before Noise-Floor

**Status:** Established (Phase 16 analysis, 2026-08-04)
**Applies to:** All Track A/B classification decisions where a single draw falls near the threshold.

### Rule

When a single draw crosses the Track A/B boundary by a margin smaller than the
encoding-level 2σ IID noise floor, **check within-seed reproducibility first** —
before invoking the encoding-level noise floor as the classification criterion.

**Procedure:**
1. Identify all draws for the same seed × encoding × condition combination.
2. Count how many other draws (replications) cross the threshold in the same direction.
3. If the marginal crossing does not reproduce in ≥ 1 additional draw of the same seed,
   classify it as **non-reproducible** and do not count it as a confirmed threshold
   crossing for that seed.
4. Only when no other same-seed draws exist (single primary draw, no replications
   triggered) should the encoding-level noise floor alone be used to assess reliability.

### Rationale

The encoding-level noise floor (estimated from IID cross-seed spread) is a
population-level variance estimate. It describes how much accuracy varies between
seeds, not how reliable a given seed's result is across draws. Within-seed
replication is a stricter, more direct test: it measures whether *this seed, in
this condition* consistently lands on the same side of the threshold.

In practice, within-seed reproducibility is the earlier and more decisive filter.
A marginal crossing that fails reproducibility within the same seed's draws is
non-reproducible regardless of whether its margin exceeds the encoding noise floor.

### Originating case

IQP / Pima / α=0.1 / seed=999 (Phase 16, 2026-08-04):

| Draw | Accuracy | Track | Margin to threshold |
|------|----------|-------|-------------------|
| Primary (federated) | 0.6494 | B | −2.00pp |
| Rep 1 | 0.6494 | B | −2.00pp |
| Rep 2 | **0.6753** | **A** | **+0.59pp** |

The Rep 2 crossing (+0.59pp) is well within the IQP encoding noise floor
(2σ = 4.16pp from Phase 16 Phase A). By the noise-floor test alone, classification
is ambiguous. The within-seed test is unambiguous: 0 of 2 other draws for the same
seed cross the threshold. Classified as non-reproducible. The original claim
"IQP does not reliably escape under label-skew conditions" is sustained.

### Interaction with MN-08 / MN-09

This rule complements, not replaces, the encoding noise floor and MN-08 attractor
analysis. The within-seed check answers "is this crossing reproducible for this
seed?"; the noise floor answers "is the spread plausibly random across seeds?";
MN-08/MN-09 explain why certain seeds are structurally boundary-prone. All three
can be relevant for the same marginal crossing, but within-seed reproducibility
should be reported first and is the strongest individual piece of evidence.

---

## MN-11: Angle/Pima α=0.5 Attractor Draws — MN-04c and MN-08c, Not a New Finding

**Status:** Confirmed (Phase 16 analysis, 2026-08-04)
**Triggered by:** angle / Pima / α=0.5 / seeds 123 and 999 showing draws at ~35%
accuracy in Phase 16 Phase B, initially flagged as possible "active class-inversion
under label skew" specific to angle encoding.

### R0 check

For each draw the server-side `evaluate_fn` runs before any training (server_round=0,
initial global parameters). R0 accuracy therefore reflects the untrained model's
predictions on the test set.

| Seed | Draw | R0 | R1 | R15 | Classification |
|------|------|----|----|-----|---------------|
| 123 | Primary (federated) | **0.3506** | 0.3506 | 0.3831 | Init-attractor, training trapped |
| 123 | Rep 1 | **0.3506** | 0.4740 | 0.6494 | Init-attractor, training escaped by R1–R2 |
| 123 | Rep 2 | **0.3506** | 0.6429 | 0.6039 | Init-attractor, training escaped at R1 |
| 999 | Primary (federated) | 0.6494 | 0.6494 | 0.6299 | Standard Track B |
| 999 | Rep 1 | 0.6494 | **0.5584** | **0.3506** | NOT init-attractor; training-driven capture |
| 999 | Rep 2 | 0.6494 | 0.6364 | 0.6234 | Standard Track B |

### Findings

**Seed 123 — MN-04c initialization-attractor (not new).**
R0 = 0.3506 in all three draws. The untrained model already predicts all class-1
(accuracy ≈ class-1 prevalence = 35.1%) before any federated round runs. This is
MN-04c's Pima initialization-attractor mechanism, now confirmed on angle encoding.
The partition geometry (α=0.5 Dirichlet) is not the cause; the same R0 appears
regardless of which draw is made (same `torch.manual_seed(123)` → same initial
weights). Whether training recovers the model is actor-RNG determined: 1 of 3 draws
remained trapped (R15=0.3831), 2 of 3 escaped by R1 (R15=0.6039 and 0.6494).

**Seed 999 — MN-08c within-B bifurcation (not new).**
R0 = 0.6494 in all three draws — the model does NOT start in the class-1 attractor.
One draw (Rep 1) falls into the attractor during training: R1=0.5584 shows
trajectory moving toward class-1 prediction; R15=0.3506 confirms capture. The other
two draws follow standard Track B trajectories (R15=0.6299 and 0.6234). This is
MN-08c's actor-RNG-determined within-B bifurcation: the same initialization produces
opposite constant-output collapsed states in different draws, with the outcome selected
by mini-batch scheduling in early rounds. Previously documented for angle/Wisconsin/α=0.1
(MN-08c); Phase 16 extends the known scope to angle/Pima/α=0.5.

### What this means for Phase 16 reporting

The ~35% draws for angle/Pima/α=0.5 seeds 123 and 999 are not an angle-specific
label-skew finding. They are instances of MN-04c (seed=123, initialization) and
MN-08c (seed=999, training dynamics), both of which are encoding-agnostic and
previously documented. Phase 16 contributes two extensions of known scope:

1. MN-04c initialization-attractor confirmed on angle encoding (previously documented
   for amplitude on Pima; seed=123 angle initialization happens to land in class-1 basin).
2. MN-08c within-B bifurcation confirmed on angle/Pima/α=0.5 (previously documented
   for angle/Wisconsin/α=0.1 and amplitude/Wisconsin/α=0.1).

Neither instance requires new encoding-specific annotation. Both should be cited as
MN-04c / MN-08c in any write-up referencing the angle/Pima attractor draws.

**Implication for the corrected Phase 16 table:** The 0/15 Track A result for
angle across all non-IID conditions is confirmed with these mechanisms understood.
The ~35% draws are collapsed-attractor states (constant class-1 output), not
partial-learning states. They sit below the majority-class baseline and should be
reported alongside the Track B count, not folded into a mean accuracy without
noting the attractor cluster.

---

## MN-12: Angle Encoding — Purity-Driven Collapse; Threshold Location Untested and Dataset-Dependent

**Status:** Established (angle_falsification experiment, 2026-08-04)
**Triggered by:** Angle falsification test using 5 manually constructed G1 partitions across two datasets and 3 draws each (15 total draws), designed to close the "angle uniquely all-B under G1-flagged conditions" claim from Phase 11.

### Data

| Partition | Dataset | Dom. weight | Purity | Track A draws | Track B draws | Anomalous? |
|---|---|---|---|---|---|---|
| G1-W1 | Wisconsin | 60.9% | 100% | 0/3 | 3/3 | Full collapse |
| G1-W2 | Wisconsin | 64.8% | 92.9% | 3/3 | 0/3 | Functional |
| G1-W3 | Wisconsin | 67.9% | 90.9% | 2/3 | 1/3 | Mostly functional |
| G1-P1 | Pima | 61.9% | 100% | 0/3 | 3/3 | Full collapse |
| G1-P2 | Pima | 66.0% | 90.9% | 0/3 | 3/3 | Full collapse |

### Finding

**Collapse rate increases sharply with dominant-client class purity.** At 100% purity, angle collapses in all 6 draws across both datasets (0/6 Track A). At 91–93% purity, angle escapes in 5/6 draws on Wisconsin (G1-W2: 3/3; G1-W3: 2/3). On Pima, angle collapses in all 3 draws even at 90.9% purity.

**The precise threshold location is untested.** Only two distinct purity levels were used (100% and 91–93%). No purity values in the range 93–99% were tested. Do not report this as "the threshold lies between 93% and 100%." The correct statement is: collapse rate is high at 100% purity and low at 91–93% purity on Wisconsin; the boundary is not located and appears to be dataset-dependent (Pima collapses at 90.9%, Wisconsin does not).

### Implication for the "angle uniquely all-B" claim

The Phase 11 claim "angle is uniquely all-B under G1/G2-flagged conditions" is **falsified** by this experiment. Angle reaches Track A in 5/15 draws (G1-W2 and G1-W3, where purity < 100%). The corrected claim is: **angle shows sharply elevated collapse risk under G1/G2 conditions, with 100% dominant-client purity reliably producing collapse across both tested datasets, and sub-100% purity producing variable outcomes that are both dataset-dependent and stochastically sensitive.**

The original Phase 11 claim was an artifact of the Phase 11 seed sample — all 4 geometry-flagged conditions happened to have 100% purity (pure-class dominant client). The falsification test showed that 100% purity is the critical factor, not G1 geometry per se.

### Reporting guidance

- Do not claim a specific purity threshold.
- Report: "angle encoding collapses under 100% dominant-client purity across both tested datasets; collapse rates at 91–93% purity are lower on Wisconsin (1/6) but remain high on Pima (3/3), suggesting the purity sensitivity is dataset-dependent."
- Note the two tested levels, not a boundary estimate.
- The G1 weight criterion (≥60% dominant weight) is a necessary but not sufficient condition: G1-W2 with 64.8% weight and 92.9% purity produced 0 collapses.


---

## MN-13: R0 Initialization Architecture — Reuploading Stays Input-Sensitive; Single-Pass Encodings Default to Majority-Class

**Status:** Established (rq4_reup_seed7_inspection, 2026-08-04)
**Affects:** Mechanism attribution for encoding-specific Track-B collapses under label skew; disambiguation of initialization-driven vs training-driven collapse pathways. Also applies as a general R0 diagnostic for distinguishing MN-04c (initialization-attractor) from MN-08c (training-driven attractor capture) cases across all attractor-sensitive conditions in this project.

### Finding

Under Wisconsin/α=0.1/seed=7, all four encodings start from the same VQC parameter values (`torch.manual_seed(7)`, weights shape (2, 4)) but produce qualitatively different pre-training (R0) prediction distributions:

| Encoding | R0 acc | Prediction type | Final acc (logged) | Track |
|---|---|---|---|---|
| angle | 0.6316 | all-class-1 (114/114) | 0.6754 | A |
| amplitude | 0.6404 | near-all-class-1 (113/114) | 0.8860 | A |
| reuploading | **0.4649** | **mixed (25c0 / 89c1)** | 0.6316 | B (logged) |
| iqp | 0.6404 | near-all-class-1 (113/114) | 0.6930 | A |

Wisconsin majority-class baseline = 0.6316 (class-1 = 72/114 test samples).

Angle locks to the exact majority-class floor before training. Amplitude and IQP land 88 bp above it (113/114 class-1). Reuploading lands 16.7pp *below* it with 25/114 class-0 predictions — genuinely mixed rather than near-uniform.

### Mechanism

The difference is architectural. Angle, amplitude, and IQP are **single-pass** encodings: data is embedded into the quantum state once, and the VQC parameters modulate a fixed gate topology thereafter. With small random initial weights, the circuit output tends toward a near-constant prediction; on an imbalanced dataset (Wisconsin: 62.6% class-1 in train), the "default" output is the majority class.

Reuploading uses **per-layer data re-embedding**: the original features are re-injected at each VQC layer via RY gates interleaved with CNOT entanglement. This repeated coupling keeps the circuit's output input-dependent at initialization — each sample produces different gate expectation values before any training has shaped the weights. The result is genuinely mixed predictions at R0, not a near-constant majority-class output.

### Implications

**Initialization-driven vs training-driven collapse.** A Track-B collapse starting from near-majority-class R0 (angle, amplitude, IQP) may be an initialization-attractor effect: the model was already at the floor before training. A Track-B collapse from a genuinely mixed R0 (reuploading) is necessarily training-driven: training moved the model *to* the majority-class floor from a non-trivial state. The two pathways are observationally distinct — R0 distinguishes them.

**Specific case: reuploading/seed=7/α=0.1.** R0=0.4649 rules out the initialization-attractor interpretation for the logged Track-B outcome (R15=0.6316). The model rose through Track A territory during training (rounds 2–5: 0.667–0.693 in replication) before the actor-RNG trajectory determined the final state — Track B in the logged run, Track A (R15=0.6754) in replication. This is the mid-training bifurcation mechanism from MN-06/MN-09, not MN-04c's initialization-attractor mechanism. See MN-09 for the bifurcation framing; this note provides the R0 evidence that disambiguates the collapse pathway.

**Swept confirmation (all 5 seeds × 4 encodings, zero new compute).** R0 is distribution-independent (pre-training accuracy is unchanged by the federation's client partition), so round=0 global accuracy rows already in the phase10/11 and phase16_pima logs cover a full 5-seed sweep across both datasets.

*Wisconsin (random_state=42 test split, 114 samples; floors: majority=0.6316, minority=0.3684):*

| Encoding | seed=1 | seed=7 | seed=42 | seed=123 | seed=999 | Max preds from nearest floor |
|---|---|---|---|---|---|---|
| angle | 0.3684 | 0.6316 | 0.3684 | 0.6316 | 0.3684 | **0** |
| amplitude | 0.3684 | 0.6404 | 0.3684 | 0.6316 | 0.3684 | **1** |
| IQP | 0.3596 | 0.6316 | 0.3596 | 0.6316 | 0.3684 | **1** |
| reuploading | 0.3509 | 0.6228 | 0.3684 | **0.5614** | 0.3947 | **8** |

Angle: all 5 seeds at exact class floors. Amplitude and IQP: maximum 1 prediction from the nearest floor across all seeds. Reuploading: 3/5 seeds more than 1 prediction from the nearest floor; seed=123 at 0.5614 is 8 predictions from the majority-class floor — genuinely mid-range. The Wisconsin pattern holds across all 5 seeds.

*Pima (random_state=42 test split, 154 samples; floors: majority=0.6494, minority=0.3506):*

| Encoding | seed=1 | seed=7 | seed=42 | seed=123 | seed=999 | Max preds from nearest floor |
|---|---|---|---|---|---|---|
| angle | 0.6494 | 0.3571 | 0.6494 | 0.3506 | 0.6494 | **1** |
| amplitude | 0.6494 | 0.3636 | 0.6494 | 0.3506 | 0.6494 | **2** |
| IQP | 0.6558 | 0.3506 | 0.6364 | 0.3506 | 0.6429 | **2** |
| reuploading | 0.6558 | 0.3442 | 0.6494 | 0.3571 | 0.6494 | **1** |

On Pima, all four encodings stay within 1–2 predictions of a class floor across all 5 seeds. Reuploading shows no distinguishable R0 advantage over the single-pass encodings. The Wisconsin pattern does not replicate on Pima.

**Revised caveat.** The reuploading/single-pass R0 distinction is confirmed across all 5 seeds on Wisconsin, and is absent on Pima. The finding is therefore dataset-specific rather than a universal architectural property. A plausible explanation is that PCA-preprocessed Wisconsin features span a range where reuploading's per-layer re-embedding produces greater cross-sample variation in gate expectation values at initialization, while Pima's PCA-preprocessed features do not produce the same separation. The mechanism claim (per-layer re-embedding preserves input-sensitivity at initialization) is supported by the Wisconsin 5-seed sweep but should not be stated as dataset-agnostic.
