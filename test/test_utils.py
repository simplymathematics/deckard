import argparse
import json
import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from deckard import utils
from deckard.utils import (
    ConfigBase,
    create_parser_from_function,
    import_class_from_file,
    load_class,
    load_data,
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
            merged = cfg.read_scores_from_disk(str(p))
            self.assertEqual(merged["base"], 1)
            self.assertEqual(merged["new"], 2)

    def test_read_scores_from_disk_missing_file_creates_directory(self):
        cfg = BaseConfig(score_dict={"base": 1})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "nested" / "scores.json"
            out = cfg.read_scores_from_disk(str(p))
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
                out = cfg.execute()
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
                "        self.x = x\n"
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
                "        self.name = name\n"
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


if __name__ == "__main__":
    unittest.main()
