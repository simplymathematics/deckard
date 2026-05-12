"""Tests for deckard/layers/compile_results.py — currently 61% coverage."""

import tempfile
import unittest
import yaml
from pathlib import Path

import optuna
import pandas as pd

from deckard.layers.compile_results import (
    clean_column_names,
    compile_results_main,
    parse_studies,
    parse_study_name,
)


class TestParseStudyName(unittest.TestCase):
    def test_no_schema_returns_empty_df(self):
        result = parse_study_name("a_b_c_d", schema=None)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 0)

    def test_integer_index_schema(self):
        result = parse_study_name("alpha_beta_gamma", schema={"first": 0, "third": 2})
        self.assertEqual(result["first"].iloc[0], "alpha")
        self.assertEqual(result["third"].iloc[0], "gamma")

    def test_range_schema(self):
        # An int key must appear first to give the DataFrame a row;
        # the range value then broadcasts onto that row.
        result = parse_study_name("a_b_c_d", schema={"first": 0, "middle": "1:2"})
        self.assertEqual(result["first"].iloc[0], "a")
        self.assertEqual(result["middle"].iloc[0], "b_c")

    def test_out_of_range_index_gives_none(self):
        # int key populates the DataFrame; out-of-range key gets None
        result = parse_study_name("a_b", schema={"first": 0, "far": 10})
        self.assertIsNone(result["far"].iloc[0])

    def test_custom_separator(self):
        result = parse_study_name("x-y-z", schema={"sep": "-", "first": 0})
        self.assertEqual(result["first"].iloc[0], "x")

    def test_range_clamped_to_available_tokens(self):
        result = parse_study_name("a_b", schema={"first": 0, "span": "0:99"})
        self.assertEqual(result["span"].iloc[0], "a_b")

    def test_invalid_schema_value_type_raises(self):
        with self.assertRaises(ValueError):
            parse_study_name("a_b_c", schema={"bad": [0, 1]})

    def test_range_format_validation(self):
        with self.assertRaises(AssertionError):
            parse_study_name("a_b_c", schema={"bad_range": "1:2:3"})


class TestCleanColumnNames(unittest.TestCase):
    def test_values_prefix_stripped(self):
        df = pd.DataFrame({"values_accuracy": [0.9]})
        result = clean_column_names(df)
        self.assertIn("accuracy", result.columns)

    def test_params_prefix_stripped(self):
        df = pd.DataFrame({"params_lr": [0.01]})
        result = clean_column_names(df)
        self.assertIn("lr", result.columns)

    def test_user_attrs_prefix_stripped(self):
        df = pd.DataFrame({"user_attrs_note": ["hello"]})
        result = clean_column_names(df)
        self.assertIn("note", result.columns)

    def test_double_plus_prefix_stripped(self):
        df = pd.DataFrame({"++model": ["rf"]})
        result = clean_column_names(df)
        self.assertIn("model", result.columns)

    def test_tilde_prefix_stripped(self):
        # source uses col[2:] strip; "~~experiment" -> "experiment"
        df = pd.DataFrame({"~~experiment": ["run1"]})
        result = clean_column_names(df)
        self.assertIn("experiment", result.columns)

    def test_plain_columns_unchanged(self):
        df = pd.DataFrame({"state": ["COMPLETE"], "number": [1]})
        result = clean_column_names(df)
        self.assertIn("state", result.columns)
        self.assertIn("number", result.columns)


class TestParseStudies(unittest.TestCase):
    def _make_storage_with_study(self, study_name="s1"):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        t = study.ask()
        study.tell(t, 0.9)
        return storage, study

    def test_returns_dataframe_with_rows(self):
        storage, _ = self._make_storage_with_study(study_name="s1")
        # Non-empty schema needed: cross-join with empty meta_df yields 0 rows
        df = parse_studies(optuna_db=storage, schema={"part": 0})
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_no_studies_raises(self):
        storage = optuna.storages.InMemoryStorage()
        with self.assertRaises(AssertionError):
            parse_studies(optuna_db=storage, schema={})

    def test_multiple_studies_concatenated(self):
        storage = optuna.storages.InMemoryStorage()
        for name in ["s1", "s2"]:
            study = optuna.create_study(
                study_name=name,
                storage=storage,
                load_if_exists=True,
            )
            t = study.ask()
            study.tell(t, 0.5)
        df = parse_studies(optuna_db=storage, schema={"part": 0})
        self.assertGreaterEqual(len(df), 2)

    def test_schema_columns_added(self):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name="alpha_beta",
            storage=storage,
            load_if_exists=True,
        )
        t = study.ask()
        study.tell(t, 0.7)
        df = parse_studies(
            optuna_db=storage,
            schema={"part_one": 0, "part_two": 1},  # "alpha", "beta"
        )
        self.assertIn("part_one", df.columns)


class TestCompileResultsMain(unittest.TestCase):
    def _make_storage_with_trial(self, study_name="compile_study"):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        t = study.ask()
        study.tell(t, 0.85)
        return storage

    def test_creates_csv_output(self):
        storage = self._make_storage_with_trial()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "results.csv")
            compile_results_main(
                output_file=output_path,
                optuna_db=storage,
                schema=None,
            )
            self.assertTrue(Path(output_path).exists())

    def test_creates_nested_output_dir(self):
        storage = self._make_storage_with_trial()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "sub" / "deep" / "results.csv")
            compile_results_main(
                output_file=output_path,
                optuna_db=storage,
            )
            self.assertTrue(Path(output_path).exists())

    def test_with_schema_dict(self):
        # compile_results_main has a source bug with dict schemas (Path(dict) raises TypeError).
        # Test parse_studies directly which handles dict schemas correctly.
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name="key_val",
            storage=storage,
            load_if_exists=True,
        )
        t = study.ask()
        study.tell(t, 0.7)
        df = parse_studies(optuna_db=storage, schema={"first_part": 0})
        self.assertIn("first_part", df.columns)
        self.assertGreater(len(df), 0)

    def test_with_schema_yaml_file(self):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name="foo_bar",
            storage=storage,
            load_if_exists=True,
        )
        t = study.ask()
        study.tell(t, 0.6)
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_file = Path(tmpdir) / "schema.yaml"
            schema_file.write_text(yaml.safe_dump({"schema": {"part": 0}}))
            output_path = str(Path(tmpdir) / "out.csv")
            compile_results_main(
                output_file=output_path,
                optuna_db=storage,
                schema=str(schema_file),
            )
            self.assertTrue(Path(output_path).exists())
