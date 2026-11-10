"""Fast v2 acceptance checks. Slow ones are in v2/acceptance.py."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pennylane as qml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explain.metrics import js_divergence
from explain.q_shap import exact_q_shap_values, make_quantum_predict_fn
from v2 import config
from v2.data import PIMA_MISSING_AS_ZERO, explanation_sets, load_raw, make_data, partition_indices
from v2.models import build_model, flatten
from v2.train import fedavg, run_federated

ALL = config.ENCODINGS + (config.VARIANT,)


class DataTests(unittest.TestCase):
    def test_split_differs_by_seed_and_has_expected_shares(self):
        a, b = make_data("wisconsin", 1), make_data("wisconsin", 7)
        self.assertNotEqual(a.meta["test_indices"], b.meta["test_indices"])
        total = len(a.y_train) + len(a.y_val) + len(a.y_test)
        self.assertEqual(total, 569)
        self.assertAlmostEqual(len(a.y_test) / total, 0.2, delta=0.01)
        self.assertAlmostEqual(len(a.y_val) / total, 0.1, delta=0.01)
        used = a.meta["train_indices"] + a.meta["val_indices"] + a.meta["test_indices"]
        self.assertEqual(len(set(used)), total)

    def test_pima_medians_come_from_training_rows_without_zeros(self):
        X, _, columns = load_raw("pima")
        data = make_data("pima", 42)
        for name in PIMA_MISSING_AS_ZERO:
            values = X[data.meta["train_indices"], columns.index(name)]
            self.assertEqual(data.meta["medians"][name], float(np.median(values[values != 0])))
        self.assertGreater(min(data.meta["medians"].values()), 0)

    def test_pima_as_values_keeps_v1_handling(self):
        self.assertEqual(make_data("pima", 42, zero_handling="as_values").meta["medians"], {})

    def test_imputation_changes_the_representation(self):
        a = make_data("pima", 42).X_train
        b = make_data("pima", 42, zero_handling="as_values").X_train
        self.assertFalse(np.allclose(a, b))

    def test_sixteen_features_impossible_on_pima(self):
        with self.assertRaises(ValueError):
            make_data("pima", 1, n_features=16)


class SetTests(unittest.TestCase):
    def setUp(self):
        self.data = make_data("wisconsin", 7)

    def test_global_sets_do_not_depend_on_regime(self):
        results = []
        for regime in ("iid", "alpha_0.1"):
            parts = partition_indices(self.data.y_train, 4, regime, 7)
            results.append(explanation_sets(self.data, parts, 7))
        np.testing.assert_array_equal(results[0].E_global, results[1].E_global)
        np.testing.assert_array_equal(results[0].B_global, results[1].B_global)

    def test_sizes_disjoint_and_stratified(self):
        for regime in config.REGIMES:
            parts = partition_indices(self.data.y_train, 4, regime, 7)
            sets = explanation_sets(self.data, parts, 7)
            self.assertEqual(len(sets.E_global), 50)
            self.assertEqual(len(sets.B_global), 50)
            self.assertFalse(set(sets.half_a) & set(sets.half_b))
            self.assertEqual(len(sets.half_a) + len(sets.half_b), 50)
            self.assertAlmostEqual(self.data.y_test[sets.E_global].mean(),
                                   self.data.y_test.mean(), delta=0.02)
            for k, part in enumerate(parts):
                if sets.E_local[k] is None:
                    self.assertLess(len(part), config.SPARSE_CLIENT)
                    continue
                E_k, B_k = set(sets.E_local[k]), set(sets.B_local[k])
                self.assertFalse(E_k & B_k)
                self.assertTrue((E_k | B_k) <= set(part))
                self.assertLessEqual(len(E_k), 25)
                self.assertLessEqual(len(B_k), 25)
                self.assertAlmostEqual(self.data.y_train[sets.E_local[k]].mean(),
                                       self.data.y_train[part].mean(), delta=2 / len(E_k) + 0.01)


class PartitionTests(unittest.TestCase):
    def test_every_record_assigned_once(self):
        y = make_data("wisconsin", 1).y_train
        for regime in config.REGIMES:
            parts = partition_indices(y, 4, regime, 1)
            self.assertEqual(len(parts), 4)
            np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(y)))


class ModelTests(unittest.TestCase):
    def test_all_models_have_expected_parameters(self):
        for encoding in ALL + config.CLASSICAL[:1]:
            self.assertEqual(len(flatten(build_model(encoding))), 13, encoding)
        self.assertEqual(len(flatten(build_model("logreg"))), 5)

    def test_same_init_across_encodings(self):
        first = flatten(build_model("angle", init_seed=5))
        for encoding in ALL:
            np.testing.assert_array_equal(first, flatten(build_model(encoding, init_seed=5)))

    def test_interleaved_circuit_order(self):
        model = build_model("reuploading")
        x = np.array([0.1, 0.2, 0.3, 0.4])
        tape = qml.workflow.construct_tape(model._qnode)(x, model.vqc_weights)
        names = [op.name for op in tape.operations]
        self.assertEqual(names.count("BasicEntanglerLayers"), 2)
        first = names.index("BasicEntanglerLayers")
        second = names.index("BasicEntanglerLayers", first + 1)
        self.assertEqual(set(names[:first]), {"RY", "CNOT"})
        self.assertEqual(set(names[first + 1:second]), {"RY", "CNOT"})
        self.assertEqual(second, len(names) - 1)

    def test_variant_circuit_order(self):
        model = build_model(config.VARIANT)
        x = np.array([0.1, 0.2, 0.3, 0.4])
        names = [op.name for op in qml.workflow.construct_tape(model._qnode)(
            x, model.vqc_weights).operations]
        self.assertEqual(names[-1], "BasicEntanglerLayers")
        self.assertEqual(names.count("RY"), 8)

    def test_shapley_efficiency_on_each_circuit(self):
        data = make_data("wisconsin", 1)
        for encoding in ALL:
            model = build_model(encoding, init_seed=3)
            predict = make_quantum_predict_fn(model)
            X, background = data.X_test[:4], data.X_train[:20]
            phi = exact_q_shap_values(predict, X, background)
            expected = predict(X) - predict(background.mean(axis=0, keepdims=True).astype(np.float32))
            np.testing.assert_allclose(phi.sum(axis=1), expected, atol=1e-5, err_msg=encoding)


class FederationTests(unittest.TestCase):
    def test_fedavg_weights_by_samples_and_skips_empty(self):
        out = fedavg([(np.array([1.0]), 1), (np.array([5.0]), 3), (np.array([99.0]), 0)])
        np.testing.assert_allclose(out, [4.0])

    def test_same_seed_gives_identical_weights(self):
        spec = config.RunSpec("fed", "wisconsin", "angle", 7, "alpha_0.5", n_rounds=2)
        data = make_data("wisconsin", 7)
        first = run_federated(spec, data)[0]
        second = run_federated(spec, data)[0]
        np.testing.assert_array_equal(first["global_weights"], second["global_weights"])
        np.testing.assert_array_equal(first["client_weights"], second["client_weights"])


class FloorTests(unittest.TestCase):
    def test_identical_inputs_give_zero_distance(self):
        vector = np.array([0.4, 0.1, 0.2, 0.3])
        self.assertAlmostEqual(js_divergence(vector, vector), 0.0)


class SummaryFloorTests(unittest.TestCase):
    def test_summary_reads_zero_when_both_sides_are_the_same(self):
        from types import SimpleNamespace

        from v2.explain import summarize

        rng = np.random.default_rng(0)
        spec = config.RunSpec("fed", "wisconsin", "angle", 1, n_clients=1, n_rounds=2)
        phi = rng.normal(size=(3, 50, 4))
        arrays = dict(phi_global=phi, client_rounds=np.array([1, 2]),
                      phi_matched=phi[1:][:, None], phi_population=phi[1:][:, None],
                      phi_floor=phi[1:][:, None])
        same = np.arange(50)
        sets = SimpleNamespace(half_a=same, half_b=same, E_local=[same], B_local=[same])
        summary = summarize(spec, arrays, sets)
        self.assertAlmostEqual(summary["D_sample_I"], 0.0)
        self.assertAlmostEqual(summary["D_sample_S"], 0.0)
        for name in ("model", "pop", "floor"):
            self.assertAlmostEqual(summary["clients"][0][f"D_{name}_I"], 0.0)
            self.assertAlmostEqual(summary["clients"][0][f"D_{name}_S"], 0.0)

    def test_stability_is_one_for_constant_rankings(self):
        from v2.explain import stability_metrics

        vectors = np.array([[1, 2, 3, 4.0], [4, 3, 2, 1.0], [4.1, 3, 2, 1.0], [4.2, 3, 2, 1.0]])
        result = stability_metrics(vectors)
        self.assertEqual(result["tau"], 1.0)
        self.assertEqual(result["tau_init"], -1.0)
        self.assertEqual(result["top_change_rate"], 0.0)
        self.assertEqual(result["settle_round"], 1)


class ClientRoundTests(unittest.TestCase):
    def test_every_third_round_includes_last(self):
        from v2.explain import client_explanation_rounds

        base = dict(kind="fed", dataset="wisconsin", encoding="angle", seed=1)
        self.assertEqual(client_explanation_rounds(config.RunSpec(**base)), [3, 6, 9, 12, 15])
        self.assertEqual(client_explanation_rounds(config.RunSpec(**base, n_rounds=16)),
                         [3, 6, 9, 12, 15, 16])
        self.assertEqual(client_explanation_rounds(config.RunSpec(**base, explain_mode="final")), [15])


class SpecTests(unittest.TestCase):
    def test_primary_counts(self):
        specs = config.primary_specs(config.SEEDS)
        fed = [s for s in specs if s.kind == "fed"]
        matrix = [s for s in fed if s.encoding in config.ENCODINGS and s.zero_handling == "missing"]
        self.assertEqual(len(matrix), 960)
        self.assertEqual(len([s for s in fed if s.encoding == config.VARIANT]), 60)
        self.assertEqual(len([s for s in fed if s.zero_handling == "as_values"]), 60)
        self.assertEqual(len(specs), len({s.key for s in specs}))

    def test_attacks_run_on_primary_specs_only(self):
        from unittest import mock

        from v2 import run

        def attack_count(specs):
            with mock.patch.object(run, "run_tasks", return_value={}) as tasks:
                run.execute(specs, "unused", 1)
            return next(len(call.args[1]) for call in tasks.call_args_list if call.args[0] == "attacks")

        self.assertEqual(attack_count(config.sweep_specs(seeds=(1,))), 0)
        primary = config.primary_specs((1,))
        expected = sum(s.kind == "fed" and s.encoding in config.ENCODINGS
                       and s.zero_handling == "missing" for s in primary)
        self.assertEqual(attack_count(primary), expected)
        self.assertEqual(expected, 32)

    def test_sweep_keys_unique(self):
        specs = config.sweep_specs()
        self.assertEqual(len(specs), len({(s.group, s.key) for s in specs}))


if __name__ == "__main__":
    unittest.main()
