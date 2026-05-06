"""Tests for deckard/layers/pareto.py — currently 0% coverage."""

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.exceptions import ExperimentalWarning

from deckard.layers.pareto import (
    _coerce_objective_columns_numeric,
    _complete_trials_only,
    _infer_default_optimizers,
    _infer_directions_for_objectives,
    _normalize_direction,
    _objective_to_column,
    _parse_csv_arg,
    _resolve_study,
    pareto_main,
)


class TestParseCsvArg(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(_parse_csv_arg(None), [])

    def test_empty_string_returns_empty(self):
        self.assertEqual(_parse_csv_arg(""), [])

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(_parse_csv_arg("   "), [])

    def test_single_item(self):
        self.assertEqual(_parse_csv_arg("accuracy"), ["accuracy"])

    def test_multi_item(self):
        self.assertEqual(
            _parse_csv_arg("accuracy,precision,f1"),
            ["accuracy", "precision", "f1"],
        )

    def test_strips_whitespace_around_items(self):
        self.assertEqual(
            _parse_csv_arg(" accuracy , precision "),
            ["accuracy", "precision"],
        )

    def test_trailing_comma_ignored(self):
        result = _parse_csv_arg("accuracy,")
        self.assertEqual(result, ["accuracy"])


class TestNormalizeDirection(unittest.TestCase):
    def test_maximize(self):
        self.assertEqual(_normalize_direction("maximize"), "maximize")

    def test_max_alias(self):
        self.assertEqual(_normalize_direction("max"), "maximize")

    def test_minimize(self):
        self.assertEqual(_normalize_direction("minimize"), "minimize")

    def test_min_alias(self):
        self.assertEqual(_normalize_direction("min"), "minimize")

    def test_diff(self):
        self.assertEqual(_normalize_direction("diff"), "diff")

    def test_dotted_prefix_stripped(self):
        self.assertEqual(_normalize_direction("StudyDirection.maximize"), "maximize")
        self.assertEqual(_normalize_direction("StudyDirection.minimize"), "minimize")

    def test_case_insensitive(self):
        self.assertEqual(_normalize_direction("MAXIMIZE"), "maximize")
        self.assertEqual(_normalize_direction("Minimize"), "minimize")

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _normalize_direction("sideways")


class TestCompleteTrialsOnly(unittest.TestCase):
    def _make_df(self):
        return pd.DataFrame(
            {
                "state": ["COMPLETE", "RUNNING", "COMPLETE", "FAIL"],
                "value": [0.9, 0.5, 0.8, 0.1],
            }
        )

    def test_filters_non_complete(self):
        df = self._make_df()
        result = _complete_trials_only(df)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result["state"] == "COMPLETE"))

    def test_no_state_column_returns_all(self):
        df = pd.DataFrame({"value": [0.9, 0.5]})
        result = _complete_trials_only(df)
        self.assertEqual(len(result), 2)


class TestCoerceObjectiveColumnsNumeric(unittest.TestCase):
    def test_already_numeric_unchanged(self):
        df = pd.DataFrame({"accuracy": [0.9, 0.8]})
        result = _coerce_objective_columns_numeric(df, ["accuracy"])
        self.assertTrue(pd.api.types.is_numeric_dtype(result["accuracy"]))

    def test_coerces_string_float_column(self):
        df = pd.DataFrame({"accuracy": ["0.9", "0.8"]})
        result = _coerce_objective_columns_numeric(df, ["accuracy"])
        self.assertAlmostEqual(result["accuracy"].iloc[0], 0.9)

    def test_all_nan_after_coerce_raises(self):
        df = pd.DataFrame({"accuracy": ["bad", "also_bad"]})
        with self.assertRaises(ValueError):
            _coerce_objective_columns_numeric(df, ["accuracy"])


class TestInferDefaultOptimizers(unittest.TestCase):
    def _make_study_with_metric_names(self, names):
        study = MagicMock()
        study.metric_names = names
        return study

    def test_uses_study_metric_names(self):
        study = self._make_study_with_metric_names(["accuracy", "f1"])
        df = pd.DataFrame({"values_0": [0.9], "values_1": [0.8]})
        result = _infer_default_optimizers(study, df)
        self.assertEqual(result, ["accuracy", "f1"])

    def test_falls_back_to_values_columns(self):
        study = self._make_study_with_metric_names([])
        df = pd.DataFrame({"values_0": [0.9], "values_1": [0.8]})
        result = _infer_default_optimizers(study, df)
        self.assertEqual(result, ["values_0", "values_1"])

    def test_falls_back_to_value_column(self):
        study = self._make_study_with_metric_names([])
        df = pd.DataFrame({"value": [0.9]})
        result = _infer_default_optimizers(study, df)
        self.assertEqual(result, ["value"])

    def test_raises_when_no_objective_columns(self):
        study = self._make_study_with_metric_names([])
        df = pd.DataFrame({"state": ["COMPLETE"]})
        with self.assertRaises(ValueError):
            _infer_default_optimizers(study, df)


class TestObjectiveToColumn(unittest.TestCase):
    def test_direct_column_match(self):
        df = pd.DataFrame({"accuracy": [0.9]})
        self.assertEqual(_objective_to_column("accuracy", 0, df, []), "accuracy")

    def test_explicit_values_column_name_match(self):
        df = pd.DataFrame({"values_0": [0.9]})
        self.assertEqual(_objective_to_column("values_0", 0, df, []), "values_0")

    def test_values_prefix_lookup(self):
        df = pd.DataFrame({"values_accuracy": [0.9]})
        self.assertEqual(_objective_to_column("accuracy", 0, df, []), "values_accuracy")

    def test_user_attrs_prefix_lookup(self):
        df = pd.DataFrame({"user_attrs_accuracy": [0.9]})
        self.assertEqual(
            _objective_to_column("accuracy", 0, df, []),
            "user_attrs_accuracy",
        )

    def test_params_prefix_lookup(self):
        df = pd.DataFrame({"params_lr": [0.01]})
        self.assertEqual(_objective_to_column("lr", 0, df, []), "params_lr")

    def test_metric_names_index_fallback(self):
        df = pd.DataFrame({"values_1": [0.8]})
        self.assertEqual(
            _objective_to_column("f1", 1, df, ["accuracy", "f1"]),
            "values_1",
        )

    def test_index_fallback_when_no_metric_names(self):
        df = pd.DataFrame({"values_0": [0.9]})
        self.assertEqual(_objective_to_column("anything", 0, df, []), "values_0")

    def test_raises_when_no_match(self):
        df = pd.DataFrame({"unrelated": [1]})
        with self.assertRaises(ValueError):
            _objective_to_column("accuracy", 0, df, [], allow_index_fallback=False)


class TestInferDirectionsForObjectives(unittest.TestCase):
    def _make_study(self, directions):
        study = MagicMock()
        study.directions = directions
        study.metric_names = []
        return study

    def test_defaults_to_maximize_when_no_study_directions(self):
        study = self._make_study([])
        result = _infer_directions_for_objectives(study, ["acc"], ["values_0"], [])
        self.assertEqual(result, ["maximize"])

    def test_uses_study_directions_when_lengths_match(self):
        study = self._make_study(["minimize", "maximize"])
        study.metric_names = []
        result = _infer_directions_for_objectives(
            study,
            ["loss", "acc"],
            ["values_0", "values_1"],
            [],
        )
        self.assertEqual(result, ["minimize", "maximize"])

    def test_warns_and_defaults_when_objective_cannot_map(self):
        study = self._make_study(["maximize", "minimize"])
        result = _infer_directions_for_objectives(
            study,
            ["unknown_metric"],
            ["values_9"],
            ["known"],
        )
        self.assertEqual(result, ["maximize"])


class TestParetoMain(unittest.TestCase):
    """Integration-level tests for pareto_main using in-memory sqlite Optuna storage."""

    def _make_storage(self):
        return optuna.storages.InMemoryStorage()

    def _create_study_with_trials(self, storage, study_name="test_study"):
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        for val in [0.8, 0.9, 0.75]:
            trial = study.ask()
            study.tell(trial, val)
        return study

    def test_single_objective_maximize_saves_file(self):
        storage = self._make_storage()
        study = self._create_study_with_trials(storage)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "pareto_out.csv")
            pareto_main(
                output_file=output_path,
                optuna_db=storage,
                study_name=study.study_name,
                optimizers="value",
                directions="maximize",
                top_k=1,
            )
            result_df = pd.read_csv(output_path)
            self.assertEqual(len(result_df), 1)
            self.assertIn("_selection_type", result_df.columns)
            self.assertEqual(result_df["_selection_type"].iloc[0], "single_objective")

    def test_single_objective_minimize_selects_lowest(self):
        storage = self._make_storage()
        study = optuna.create_study(
            study_name="min_study",
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )
        for val in [0.5, 0.3, 0.8]:
            t = study.ask()
            study.tell(t, val)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "out.csv")
            pareto_main(
                output_file=output_path,
                optuna_db=storage,
                study_name="min_study",
                optimizers="value",
                directions="minimize",
                top_k=1,
            )
            result_df = pd.read_csv(output_path)
            self.assertAlmostEqual(result_df["value"].iloc[0], 0.3)

    def test_top_k_invalid_raises(self):
        storage = self._make_storage()
        with self.assertRaises(ValueError):
            pareto_main(
                output_file="/tmp/x.csv",
                optuna_db=storage,
                top_k=0,
            )

    def test_no_complete_trials_raises(self):
        storage = self._make_storage()
        optuna.create_study(
            study_name="empty_study",
            storage=storage,
            load_if_exists=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                pareto_main(
                    output_file=str(Path(tmpdir) / "out.csv"),
                    optuna_db=storage,
                    study_name="empty_study",
                    optimizers="accuracy",
                    directions="maximize",
                )

    def test_multi_objective_produces_pareto_front(self):
        storage = self._make_storage()
        study = optuna.create_study(
            study_name="multi_study",
            storage=storage,
            directions=["maximize", "minimize"],
            load_if_exists=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            study.set_metric_names(["accuracy", "loss"])
        pairs = [(0.9, 0.1), (0.8, 0.05), (0.7, 0.3), (0.6, 0.4)]
        for acc, loss in pairs:
            t = study.ask()
            study.tell(t, [acc, loss])
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "pareto.csv")
            pareto_main(
                output_file=output_path,
                optuna_db=storage,
                study_name="multi_study",
                optimizers="accuracy,loss",
                directions="maximize,minimize",
            )
            result_df = pd.read_csv(output_path)
            # All (0.9,0.1) and (0.8,0.05) are Pareto-optimal
            self.assertGreaterEqual(len(result_df), 1)
            self.assertEqual(result_df["_selection_type"].iloc[0], "pareto_front")

    def test_diff_direction_single_raises(self):
        storage = self._make_storage()
        study = self._create_study_with_trials(storage, study_name="diff_study")
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                pareto_main(
                    output_file=str(Path(tmpdir) / "out.csv"),
                    optuna_db=storage,
                    study_name="diff_study",
                    optimizers="accuracy",
                    directions="diff",
                )


class TestResolveStudy(unittest.TestCase):
    def test_no_studies_raises(self):
        storage = optuna.storages.InMemoryStorage()
        with self.assertRaises(ValueError):
            _resolve_study(storage, study_name=None)

    def test_multiple_studies_no_name_raises(self):
        storage = optuna.storages.InMemoryStorage()
        for name in ["study_a", "study_b"]:
            s = optuna.create_study(
                study_name=name,
                storage=storage,
                load_if_exists=True,
            )
            t = s.ask()
            s.tell(t, 0.5)
        with self.assertRaises(ValueError):
            _resolve_study(storage, study_name=None)

    def test_named_study_loads(self):
        storage = optuna.storages.InMemoryStorage()
        optuna.create_study(
            study_name="my_study",
            storage=storage,
            load_if_exists=True,
        )
        study = _resolve_study(storage, study_name="my_study")
        self.assertEqual(study.study_name, "my_study")

    def test_single_summary_without_study_name_uses_name_attr(self):
        storage = optuna.storages.InMemoryStorage()
        summary = MagicMock()
        del summary.study_name
        summary.name = "fallback_name"

        with patch("deckard.layers.pareto.optuna.study.get_all_study_summaries", return_value=[summary]), patch(
            "deckard.layers.pareto.optuna.study.load_study",
        ) as load_mock:
            _resolve_study(storage, study_name=None)
            load_mock.assert_called_once()

    def test_single_summary_without_any_name_raises(self):
        storage = optuna.storages.InMemoryStorage()
        summary = object()
        with patch(
            "deckard.layers.pareto.optuna.study.get_all_study_summaries",
            return_value=[summary],
        ):
            with self.assertRaises(ValueError):
                _resolve_study(storage, study_name=None)


class TestParetoMainDefaultsAndValidation(unittest.TestCase):
    def test_infers_objectives_and_directions_when_omitted(self):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name="auto_defaults",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        for val in [0.2, 0.4, 0.3]:
            t = study.ask()
            study.tell(t, val)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "auto.csv")
            pareto_main(
                output_file=output_path,
                optuna_db=storage,
                study_name="auto_defaults",
                optimizers=None,
                directions=None,
                top_k=1,
            )
            result_df = pd.read_csv(output_path)
            self.assertEqual(len(result_df), 1)

    def test_raises_when_optimizers_and_directions_lengths_mismatch(self):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name="mismatch",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        t = study.ask()
        study.tell(t, 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                pareto_main(
                    output_file=str(Path(tmpdir) / "mismatch.csv"),
                    optuna_db=storage,
                    study_name="mismatch",
                    optimizers="value,values_0",
                    directions="maximize",
                )
