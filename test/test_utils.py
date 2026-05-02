import argparse
import json
import logging
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd
from omegaconf import OmegaConf

from deckard import utils
from deckard import (
    DataConfig,
    ModelConfig,
    DefenseConfig,
    AttackConfig,
    ExperimentConfig,
    SurvivalExperimentConfig,
    FileConfig,
    ScorerDictConfig,
)
from deckard.utils import (
    ConfigBase,
    coerce_config,
    create_parser_from_function,
    import_class_from_file,
    load_class,
    load_data,
    safe_store,
    save_data,
)


class BaseConfig(ConfigBase):
    def __call__(self):
        return 1


class ParamsConfig(ConfigBase):
    x: int = 10
    y: str = "abc"

    def __call__(self, x, y):
        return x, y


class MissingParamConfig(ConfigBase):
    def __call__(self, required_param):
        return required_param


class FailingConfig(ConfigBase):
    def __call__(self):
        raise RuntimeError("boom")


class TypeAConfig(ConfigBase):
    def __call__(self):
        return "A"


class TypeBConfig(ConfigBase):
    def __call__(self):
        return "B"


class TestUtilsAdditional(unittest.TestCase):
    def test_coerce_config_dictconfig_to_dict(self):
        cfg = OmegaConf.create({"alpha": 1, "beta": {"gamma": 2}})
        out = coerce_config(cfg)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["alpha"], 1)
        self.assertEqual(out["beta"]["gamma"], 2)

    def test_coerce_config_configbase_to_dict(self):
        obj = BaseConfig(score_dict={"x": 1})
        out = coerce_config(obj)
        self.assertIsInstance(out, dict)
        self.assertIn("score_dict", out)

    def test_coerce_config_yaml_path_to_dict(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "scorer.yaml"
            cfg_path.write_text("scorers:\n  acc:\n    score_name: acc\n")
            out = coerce_config(str(cfg_path))
            self.assertIsInstance(out, dict)
            self.assertIn("scorers", out)

    def test_coerce_config_non_yaml_string_passthrough(self):
        class_path = "sklearn.metrics.accuracy_score"
        self.assertEqual(coerce_config(class_path), class_path)

    def test_safe_store_tolerates_duplicate_registration(self):
        group = f"test_safe_store_{uuid4().hex}"
        # First registration
        safe_store(group=group, name="cfg", node={"x": 1})
        # Duplicate registration should not raise
        safe_store(group=group, name="cfg", node={"x": 1})

    def test_hash_stable_after_call_for_core_configbase_objects(self):
        configs = [
            DataConfig(),
            ModelConfig(model_type="sklearn.linear_model.LogisticRegression"),
            DefenseConfig(),
            AttackConfig(),
            FileConfig(),
            ScorerDictConfig(scorers={}),
            ExperimentConfig(data=DataConfig()),
            SurvivalExperimentConfig(data=DataConfig()),
        ]

        for cfg in configs:
            original_hash = hash(cfg)
            cls = cfg.__class__
            original_call = cls.__call__

            def fake_call(self):
                # Simulate runtime-only side effects commonly produced during execution.
                self.training_time = 1.23
                self.prediction_time = 2.34
                self.probabilities = [0.1, 0.9]
                self.predictions = [1, 0]
                self._random_runtime_field = {"seen": True}
                if hasattr(self, "score_dict") and isinstance(self.score_dict, dict):
                    self.score_dict["runtime"] = 1
                return {"ok": 1}

            setattr(cls, "__call__", fake_call)
            try:
                cfg.execute_without_mercy()
            finally:
                setattr(cls, "__call__", original_call)

            self.assertEqual(
                original_hash,
                hash(cfg),
                msg=f"Hash changed after call for {cls.__name__}",
            )

    def test_hash_conf_values_stable_across_dict_order_and_set_order(self):
        left = {
            "b": [3, 2, 1],
            "a": {"y": "yes", "x": {3, 1, 2}},
        }
        right = {
            "a": {"x": {2, 3, 1}, "y": "yes"},
            "b": [3, 2, 1],
        }

        h1 = utils.hash_conf_values(left)
        h2 = utils.hash_conf_values(right)

        self.assertEqual(h1, h2)

    def test_hash_conf_values_stable_for_path_and_bytes(self):
        value = {
            "path": Path("a") / "b" / "c.txt",
            "payload": b"deckard",
        }

        h1 = utils.hash_conf_values(value)
        h2 = utils.hash_conf_values(value)

        self.assertEqual(h1, h2)

    def test_configbase_hash_deterministic_for_equal_content(self):
        cfg1 = BaseConfig(score_dict={"alpha": 1, "beta": 2})
        cfg2 = BaseConfig(score_dict={"beta": 2, "alpha": 1})

        cfg1.custom = {"z": [1, 2], "a": {"m": 9, "n": 8}}
        cfg2.custom = {"a": {"n": 8, "m": 9}, "z": [1, 2]}

        self.assertEqual(hash(cfg1), hash(cfg2))

    def test_get_call_params_success(self):
        cfg = ParamsConfig()
        params = cfg.get_call_params()
        self.assertEqual(params, {"x": 10, "y": "abc"})

    def test_get_call_params_missing_attribute_raises(self):
        cfg = MissingParamConfig()
        with self.assertRaises(AttributeError):
            cfg.get_call_params()

    def test_save_scores_and_load_scores_json(self):
        cfg = BaseConfig(score_dict={"baseline": 1})
        scores = {"acc": 0.9, "files": {"f": "x.csv"}, "params": {"k": 1}}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scores.json"
            cfg.save_scores(scores, p)
            loaded = cfg.load_scores(str(p))
            self.assertIn("acc", loaded)
            self.assertIn("files", loaded)
            self.assertIn("params", loaded)

    def test_save_scores_unsupported_extension_raises(self):
        cfg = BaseConfig()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scores.txt"
            with self.assertRaises(ValueError):
                cfg.save_scores({"acc": 1.0}, p)

    def test_read_scores_from_disk_existing_file_merges(self):
        cfg = BaseConfig(score_dict={"base": 1})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scores.json"
            with open(p, "w") as f:
                json.dump({"new": 2}, f)
            merged = cfg.read_or_initialize_scores(str(p))
            self.assertEqual(merged["base"], 1)
            self.assertEqual(merged["new"], 2)

    def test_read_scores_from_disk_missing_file_creates_directory(self):
        cfg = BaseConfig(score_dict={"base": 1})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "nested" / "scores.json"
            out = cfg.read_or_initialize_scores(str(p))
            self.assertTrue(p.parent.exists())
            self.assertEqual(out, {"base": 1})

    def test_save_data_top_level_and_load_data_roundtrip_pickle(self):
        payload = {"a": [1, 2], "b": [3, 4]}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.pkl"
            save_data(payload, p)
            loaded = load_data(str(p))
            self.assertIsInstance(loaded, pd.DataFrame)
            self.assertEqual(list(loaded.columns), ["a", "b"])

    def test_load_data_none_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_data(None)

    def test_save_overwrite_raises(self):
        cfg = BaseConfig()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obj.pkl"
            cfg.save(str(p))
            with self.assertRaises(ValueError):
                cfg.save(str(p))

    def test_load_type_mismatch_raises(self):
        a = TypeAConfig()
        b = TypeBConfig()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obj.pkl"
            b.save(str(p))
            with self.assertRaises(TypeError):
                a.load(str(p))

    def test_execute_returns_fallback_score_dict_on_exception(self):
        cfg = FailingConfig(score_dict={"fallback": 123})
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "deckard.log"
            handler = logging.FileHandler(log_path)
            utils.logger.addHandler(handler)
            try:
                out = cfg.execute_without_mercy()
                self.assertEqual(out, {"fallback": 123})
                self.assertTrue(log_path.exists())
                self.assertIn("Exception:", log_path.read_text())
            finally:
                utils.logger.removeHandler(handler)
                handler.close()

    def test_import_class_from_file_success(self):
        with tempfile.TemporaryDirectory() as td:
            module_path = Path(td) / "tmp_mod.py"
            module_path.write_text(
                "class MyClass:\n"
                "    def __init__(self, x=0):\n"
                "        self.x = x\n",
            )
            obj = import_class_from_file(str(module_path), "MyClass", 7)
            self.assertEqual(obj.x, 7)

    def test_import_class_from_file_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            import_class_from_file("does_not_exist.py", "Anything")

    def test_load_class_colon_path_success(self):
        with tempfile.TemporaryDirectory() as td:
            module_path = Path(td) / "tmp_mod2.py"
            module_path.write_text(
                "class MyClass:\n"
                "    def __init__(self, name='n'):\n"
                "        self.name = name\n",
            )
            obj = load_class(f"{module_path}:MyClass", "deckard")
            self.assertEqual(obj.name, "deckard")

    def test_create_parser_existing_parser_with_kwargs_raises(self):
        parser = argparse.ArgumentParser()
        with self.assertRaises(ValueError):
            create_parser_from_function(lambda a: a, parser=parser, prog="x")

    def test_create_parser_unannotated_defaults_to_string(self):
        def fn(name, count: int = 1):
            return name, count

        parser = create_parser_from_function(fn)
        args = parser.parse_args(["--name", "alice"])
        self.assertEqual(args.name, "alice")
        self.assertEqual(args.count, 1)

    def test_create_parser_uses_function_docstring_for_description(self):
        def fn(name: str):
            """Create a parser description from the function docstring.

            Parameters
            ----------
            name : str
                Name to echo.
            """

            return name

        parser = create_parser_from_function(fn)

        self.assertEqual(
            parser.description,
            "Create a parser description from the function docstring.",
        )

    def test_create_parser_uses_parameter_docstrings_for_help_text(self):
        def fn(name: str, count: int = 1):
            """Build parser argument help from docstring parameter descriptions.

            Parameters
            ----------
            name : str
                Name to echo in the command output.
            count : int, optional
                Number of iterations to run.
            """

            return name, count

        parser = create_parser_from_function(fn)
        name_action = next(a for a in parser._actions if a.dest == "name")
        count_action = next(a for a in parser._actions if a.dest == "count")

        self.assertEqual(name_action.help, "Name to echo in the command output.")
        self.assertEqual(count_action.help, "Number of iterations to run.")


if __name__ == "__main__":
    unittest.main()
