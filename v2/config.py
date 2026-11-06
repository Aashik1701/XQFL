"""Protocol v2 settings and run definitions. See results/XQFL_PROTOCOL_V2.md."""
import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/protocol_v2"
PROTOCOL = ROOT / "results/XQFL_PROTOCOL_V2.md"

SEEDS = (1, 7, 42, 123, 999, 11, 13, 17, 19, 23, 29, 31, 37, 43, 47,
         53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113)
SWEEP_SEEDS = SEEDS[:15]
DATASETS = ("wisconsin", "pima")
ENCODINGS = ("angle", "amplitude", "reuploading", "iqp")
VARIANT = "repeated_embedding"
CLASSICAL = ("mlp13", "logreg")
REGIMES = ("iid", "qty_skew", "alpha_0.5", "alpha_0.1")
GATE_MARGIN = 0.02
SENSITIVITY_MARGINS = (0.0, 0.01, 0.05, 0.10, None)  # None means ungated
MIN_ESTIMATE_SEEDS = 10
MIN_DESCRIPTIVE_SEEDS = 3
N_GLOBAL_EXPLAINED = 50
N_GLOBAL_BACKGROUND = 50
N_LOCAL_EXPLAINED = 25
N_LOCAL_BACKGROUND = 25
SPARSE_CLIENT = 10
CLIENT_EXPLAIN_EVERY = 3
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.05
BATCH_SIZE = 32
N_MIA_RECORDS = 50

CODE_FILES = (
    "v2/config.py", "v2/data.py", "v2/models.py", "v2/train.py", "v2/explain.py",
    "v2/stats.py", "v2/analysis.py", "v2/mia.py", "v2/estimators.py", "v2/run.py",
    "data/partition.py", "explain/q_shap.py", "explain/metrics.py",
    "models/encodings.py", "models/quantum_classifier.py",
    "privacy/release_evaluation.py",
)


@dataclass(frozen=True)
class RunSpec:
    kind: str  # "fed" or "cent"
    dataset: str
    encoding: str
    seed: int
    regime: str = "iid"
    replicate: int = 1  # centralized only
    n_features: int = 4
    n_layers: int = 2
    n_clients: int = 4
    n_rounds: int = 15
    zero_handling: str = "missing"
    group: str = "primary"
    explain_mode: str = "full"  # "full" or "final"

    @property
    def key(self):
        return (f"{self.kind}_{self.dataset}_{self.encoding}_{self.regime}_s{self.seed}"
                f"_r{self.replicate}_f{self.n_features}_l{self.n_layers}"
                f"_c{self.n_clients}_t{self.n_rounds}_{self.zero_handling}")

    def as_dict(self):
        return asdict(self)


def code_fingerprint():
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def protocol_hash():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def _cent_pair(spec):
    return [replace(spec, kind="cent", regime="iid", replicate=r) for r in (1, 2)]


def primary_specs(seeds=SEEDS):
    """Stage 1 and 2 runs: the full matrix, variant, sensitivity and controls."""
    specs = []
    for seed in seeds:
        for dataset in DATASETS:
            for encoding in ENCODINGS:
                for regime in REGIMES:
                    specs.append(RunSpec("fed", dataset, encoding, seed, regime))
                specs += _cent_pair(RunSpec("fed", dataset, encoding, seed))
            for encoding in (VARIANT,) + CLASSICAL:
                spec = RunSpec("fed", dataset, encoding, seed)
                specs += [spec] + _cent_pair(spec)
        for encoding in ("amplitude", "reuploading"):
            spec = RunSpec("fed", "pima", encoding, seed, zero_handling="as_values")
            specs += [spec] + _cent_pair(spec)
    return specs


def sweep_specs(seeds=SWEEP_SEEDS):
    """Stage 6 runs, in protocol order, so stopping early leaves complete sweeps."""
    out = []

    def add(dataset, regimes=("iid", "alpha_0.5"), **factor):
        for seed in seeds:
            for encoding in ENCODINGS:
                for regime in regimes:
                    spec = RunSpec("fed", dataset, encoding, seed, regime, group="sweep",
                                   explain_mode="final", **factor)
                    out.append(spec)
                out.extend(_cent_pair(RunSpec("fed", dataset, encoding, seed, group="sweep",
                                              explain_mode="final", **factor)))

    for dataset in ("spambase", "phoneme"):
        add(dataset)
    for layers in (4, 6):
        add("wisconsin", n_layers=layers)
        add("pima", regimes=("iid",), n_layers=layers)
    for rounds in (30, 50):
        add("wisconsin", n_rounds=rounds)
    for clients in (8, 10, 20):
        add("wisconsin", n_clients=clients)
    for features in (6, 8):
        add("wisconsin", n_features=features)
        add("pima", regimes=("iid",), n_features=features)
    return list(dict.fromkeys(out))
