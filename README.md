# XQFL

![Python](https://img.shields.io/badge/python-3.11-blue) ![Status](https://img.shields.io/badge/status-active%20development-yellow)

A research codebase for **explainability and privacy analysis in federated quantum machine learning**. It trains variational quantum classifiers across simulated federated clients, generates Shapley-value feature attributions for both local and global models, and measures how those attributions behave under non-IID data partitions and membership-inference attacks.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Running Experiments](#running-experiments)
- [Generating Results](#generating-results)
- [Reproducibility Notes](#reproducibility-notes)
- [Known Limitations](#known-limitations)
- [Testing](#testing)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

XQFL (**X**plainable **Q**uantum **F**ederated **L**earning) combines three normally-separate concerns into one measurable pipeline:

1. **Federated learning** — multiple simulated clients train a shared model without centralizing raw data.
2. **Quantum machine learning** — the shared model is a variational quantum circuit (VQC), evaluated under several distinct data-encoding strategies.
3. **Explainability & privacy** — every round, a from-scratch Shapley-value explainer produces a feature-attribution vector for both the global model and every client's local model; those vectors are then probed for stability across rounds, divergence between local and global models, and susceptibility to membership-inference attacks.

The codebase is organized as a pipeline: **data → federated training → explanation logging → non-IID / privacy stress tests → result aggregation**, with every stage runnable and inspectable independently.

## Key Features

- **4 quantum data encodings** — angle, amplitude, data re-uploading, and IQP — evaluated on the same circuit and training budget
- **Federated simulation** via [Flower](https://flower.ai/) over Ray, with FedAvg and trimmed-mean aggregation strategies
- **Q-SHAP** — a permutation-sampling Shapley explainer implemented from scratch for the quantum feature space (no installable feature-level Q-SHAP package exists)
- **Non-IID partitioning** — IID, quantity skew, and Dirichlet label skew, at configurable severity
- **Privacy attacks** — direct and shadow-model membership-inference attacks on explanation vectors
- **Calibrated noise mechanisms** — configurable perturbations of explanation vectors; empirical scale calibration is not a formal privacy guarantee
- **Deterministic, auditable aggregation** — fixed client ordering and a runtime partition-integrity check so federated results are reproducible and silent data-partition bugs fail loudly instead of silently corrupting a run
- **Result aggregation pipeline** — raw per-seed/per-round logs are turned into analysis-ready tables and figures by a single set of aggregation functions, so downstream tables and plots can never drift apart

## Project Structure

```
xqfl/
├── data/            Dataset loaders and non-IID partitioning strategies
├── models/          Quantum encodings, the VQC classifier, and a classical baseline
├── federation/      Flower client/server, FedAvg + trimmed-mean aggregation strategies
├── explain/         Q-SHAP explainer, gate-level Shapley, and comparison metrics
├── privacy/         Membership-inference attacks and DP noise mechanisms
├── experiments/     Runnable experiment scripts that produce the raw result logs
├── analysis/        Aggregation and plotting pipeline (raw logs -> tables/figures)
├── results/         Raw experiment logs + results/outputs/ (generated tables & figures)
├── tests/           Unit test scaffolding
└── requirements.txt Pinned dependency versions
```

## Installation

```bash
python3.11 -m venv xqfl-env
source xqfl-env/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11. Everything runs on CPU — quantum circuits are simulated, no GPU or QPU access is needed.

## Quick Start

Train a single centralized quantum classifier on the Wisconsin Breast Cancer dataset:

```bash
python experiments/phase1_single_client.py
```

Or as a minimal script:

```python
from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding
from models.quantum_classifier import HybridQuantumClassifier

X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=42)

model = HybridQuantumClassifier(n_qubits=4, n_layers=2, encoding_fn=angle_encoding)
model.fit(X_train, y_train, epochs=30, lr=0.05)

print("test accuracy:", (model.predict(X_test) == y_test).mean())
```

## Core Concepts

### Datasets

| Dataset | Source | Shape | Task |
|---|---|---|---|
| Wisconsin Breast Cancer | `sklearn.datasets.load_breast_cancer` | 569 × 30 | binary malignant/benign |
| Pima Indians Diabetes | `sklearn.datasets.fetch_openml(data_id=37)` | 768 × 8 | binary diabetes diagnosis |

Both are standardized and reduced to 4 PCA components (one per qubit). The projection is fit once on the pooled training set and broadcast unchanged to every client, so all clients embed data into the same feature space and federated weight averaging stays meaningful.

### Quantum encodings (`models/encodings.py`)

| Encoding | Approach |
|---|---|
| Angle | one rotation per feature — a product state |
| Amplitude | feature vector loaded directly into state amplitudes |
| Data re-uploading | re-embeds the input before every variational layer |
| IQP | Hadamard layer + feature-parameterized diagonal and pairwise-product gates |

Each encoding feeds the same 4-qubit, 2-layer entangling variational circuit with a linear + sigmoid classification head (13 trainable parameters), so results are comparable across encodings.

### Federated setup (`federation/`)

Simulated multi-client training with FedAvg or trimmed-mean aggregation, configurable local epochs and rounds. `ExperimentConfig` (`experiments/config.py`) fully specifies one run — dataset, encoding, aggregator, data distribution, client count, and seed — so any logged result row is reproducible from its config alone. Client-to-partition assignment is taken from a stable per-client identifier and verified every round, so a data partition can never be silently dropped or double-counted during aggregation.

### Non-IID partitioning (`data/partition.py`)

- `iid_split` — uniform random assignment across clients
- `dirichlet_split(alpha)` — label skew via Dirichlet-distributed per-client class proportions
- `quantity_skew_split` — exponentially unequal per-client sample counts

### Explainability (`explain/q_shap.py`)

A permutation-sampling Shapley estimator built from scratch, computing feature attributions in the PCA-reduced quantum input space for the global model and every client's local model, every training round. A collapse gate separates functional models from ones that have collapsed to a near-constant prediction, since a collapsed model's attribution vector is trivially near-zero and would otherwise look artificially stable or "faithful" if not filtered out.

### Privacy analysis (`privacy/`)

- `membership_inference.py` — direct and shadow-model membership-inference attacks that try to distinguish training vs. non-training samples from their explanation vectors
- `dp_noise.py` — Laplace and Gaussian perturbation utilities with independent per-record noise
- `release_evaluation.py` — fold-local preprocessing, held-out raw and direction-aware AUC, validation-selected attack thresholds, and matched clean/noisy utility

The current release evaluator uses a shared explanation background for members and non-members. Score direction and thresholds are selected without access to outer test labels. Post-hoc score inversion and low-FPR ROC points are diagnostics, not deployment guarantees. Chance-level performance of one attack does not certify privacy.

## Running Experiments

`experiments/` holds the runnable data-collection scripts:

- **`phaseN_*.py`** — the core sweeps (single-client training, federated baselines, cross-encoding comparisons, non-IID sweeps, privacy sweeps)
- **`rq*.py`** — targeted extensions and robustness checks that add seeds or conditions to an existing sweep
- Two **orchestrator scripts** that run a full multi-stage sweep end-to-end with checkpoint/resume support, so a long-running sweep can be safely interrupted and continued

Logging and checkpoint formats vary by script. The current release evaluation entry point is `experiments/rq4_release_repair.py`; it persists target checkpoints, clean/noisy vectors, labels, held-out predictions and fold records in `results/rq4_release_repair_v1/`. Legacy attack sweeps are retained for provenance and must not be used as the current evaluation path. Running the release evaluator performs model training; aggregation alone does not.

## Generating Results

`analysis/aggregate_results.py` reads the raw per-seed/per-round CSV logs in `results/` and produces analysis-ready tables in `results/outputs/`. `analysis/plots.py` builds the corresponding figures through the same aggregation functions, so a table and its figure can never disagree.

```bash
python analysis/aggregate_results.py   # -> results/outputs/table*.{csv,md}
python analysis/plots.py               # -> results/outputs/figure*.{png,pdf}
python analysis/plot_current_rq4.py    # -> corrected release diagnostics and plots
```

Current release summaries are built by `analysis/current_rq4.py` from the completed release records, with no fallback to legacy attack CSVs. Verify the evaluation helpers without training:

```bash
python -m unittest discover -s tests -p 'test_release_evaluation.py' -v
```

## Reproducibility Notes

- Aggregation sums client updates in a fixed, client-ID-sorted order, so the aggregation step is bit-reproducible given the same client updates.
- The federated strategy raises an error if a round does not train every expected client partition exactly once, rather than silently continuing. This guards against a real defect found during development — partition assignment derived from a backend node identifier instead of a stable per-client index — which is now fixed and covered by a runtime consistency check.
- Residual run-to-run variation comes from actor/mini-batch scheduling non-determinism in the simulation backend, not from aggregation order.

## Known Limitations

- All results use a single circuit architecture (4 qubits, 2 entangling layers) on a noiseless simulator — no shot noise or hardware decoherence is modeled.
- The differential-privacy sensitivity used in the noise sweep is empirically calibrated from observed attribution magnitudes, not a certified bound — treat the noise budget as a comparable relative setting across encodings, not a formal (ε, δ) guarantee.
- The direct membership-inference attack is trained and evaluated on the same target model's labelled explanations; its AUC is best read as a signal-presence proxy, not a real-world attacker's expected success rate.
- Attributions are computed over PCA-component directions, not original domain features.

## Testing

`tests/` contains scaffolding for client, encoding, metric, and partitioning unit tests. These are not yet implemented — contributions welcome.

## Tech Stack

| Purpose | Library |
|---|---|
| Quantum circuit simulation & autodiff | [PennyLane](https://pennylane.ai/) |
| Classical training loop | PyTorch |
| Federated simulation | [Flower](https://flower.ai/) over Ray |
| Baselines & attack classifiers | scikit-learn, SHAP |
| Result aggregation & plotting | pandas, matplotlib |

Exact pinned versions are in `requirements.txt`.

## License

All rights reserved. See [`LICENSE`](LICENSE) for details.
