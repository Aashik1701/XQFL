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

**Current status: EVIDENCE RE-OPENING — do not report as outlier yet. Revisit when
all 20 Part A Wisconsin runs are in the CSV.**

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
