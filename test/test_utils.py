import argparse
import importlib.util
import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
from omegaconf import OmegaConf

from deckard import utils
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
    def test_torch_compiler_backends_handles_missing_or_failing_compiler(self):
        no_compiler = SimpleNamespace()
        self.assertEqual(utils._torch_compiler_backends(no_compiler), [])

        failing_compiler = SimpleNamespace(
            compiler=SimpleNamespace(
                list_backends=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            ),
        )
        self.assertEqual(utils._torch_compiler_backends(failing_compiler), [])

    def test_auto_torch_device_from_backends_prefers_mps_then_cpu(self):
        class FakeTorch:
            def __init__(self, backends):
                self.compiler = SimpleNamespace(list_backends=lambda: backends)
                self.cuda = SimpleNamespace(is_available=lambda: False)
                self.backends = SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: True),
                )

            @staticmethod
            def device(name):
                return name

        self.assertEqual(
            utils._auto_torch_device_from_backends(FakeTorch(["inductor"])),
            "mps",
        )

        cpu_torch = FakeTorch([])
        cpu_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        self.assertEqual(utils._auto_torch_device_from_backends(cpu_torch), "cpu")

    def test_auto_torch_device_from_backends_prefers_cuda_paths(self):
        class FakeTorch:
            def __init__(self, backends, cuda_available=True, mps_available=False):
                self.compiler = SimpleNamespace(list_backends=lambda: backends)
                self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
                self.backends = SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: mps_available),
                )

            @staticmethod
            def device(name):
                return name

        self.assertEqual(
            utils._auto_torch_device_from_backends(FakeTorch(["inductor"])),
            "cuda:0",
        )
        self.assertEqual(
            utils._auto_torch_device_from_backends(FakeTorch([])),
            "cuda:0",
        )

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

    def test_coerce_config_none_passthrough(self):
        self.assertIsNone(coerce_config(None))

    def test_safe_store_tolerates_duplicate_registration(self):
        group = f"test_safe_store_{uuid4().hex}"
        # First registration
        safe_store(group=group, name="cfg", node={"x": 1})
        # Duplicate registration should not raise
        safe_store(group=group, name="cfg", node={"x": 1})

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

    def test_normalize_for_hash_uses_root_lookup_and_object_fallbacks(self):
        cfg = OmegaConf.create({"chosen": {"alpha": 1}})

        class ToDictFails:
            def to_dict(self):
                raise RuntimeError("boom")

            def __str__(self):
                return "fallback-string"

        class PublicAttrs:
            def __init__(self):
                self.visible = 3
                self._hidden = 4

        normalized = utils.normalize_for_hash("chosen", root=cfg)
        self.assertEqual(normalized, {"alpha": 1})
        self.assertEqual(utils.normalize_for_hash(ToDictFails()), "fallback-string")
        self.assertEqual(utils.normalize_for_hash(PublicAttrs()), {"visible": 3})
        self.assertIn("BaseConfig", utils.normalize_for_hash(BaseConfig))

    def test_configbase_hash_deterministic_for_equal_content(self):
        cfg1 = BaseConfig(score_dict={"alpha": 1, "beta": 2})
        cfg2 = BaseConfig(score_dict={"beta": 2, "alpha": 1})

        cfg1.custom = {"z": [1, 2], "a": {"m": 9, "n": 8}}
        cfg2.custom = {"a": {"n": 8, "m": 9}, "z": [1, 2]}

        self.assertEqual(hash(cfg1), hash(cfg2))

    def test_resolve_torch_device_cuda_falls_back_to_best_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch not available")

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=torch.device("mps"),
            ),
        ):
            resolved = utils.resolve_torch_device("cuda")

        self.assertEqual(str(resolved), "mps")

    def test_resolve_torch_device_invalid_cuda_index_falls_back_to_best_available(
        self,
    ):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch not available")

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.device_count",
                return_value=1,
            ),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=torch.device("mps"),
            ),
        ):
            resolved = utils.resolve_torch_device(5)

        self.assertEqual(str(resolved), "mps")

    def test_resolve_torch_device_accepts_existing_device_and_valid_cuda_index(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch not available")

        device = torch.device("cpu")
        self.assertIs(utils.resolve_torch_device(device), device)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            resolved = utils.resolve_torch_device(1)

        self.assertEqual(str(resolved), "cuda:1")

    def test_resolve_torch_device_handles_cuda_string_and_invalid_string(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch not available")

        with patch("torch.cuda.is_available", return_value=True):
            resolved = utils.resolve_torch_device("cuda")
        self.assertEqual(str(resolved), "cuda:0")

        with patch(
            "deckard.utils._auto_torch_device_from_backends",
            return_value=torch.device("cpu"),
        ):
            resolved = utils.resolve_torch_device("not-a-device")
        self.assertEqual(str(resolved), "cpu")

    def test_resolve_torch_device_covers_auto_gpu_cuda_and_mps_paths(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch not available")

        with patch(
            "deckard.utils._auto_torch_device_from_backends",
            return_value=torch.device("cpu"),
        ):
            self.assertEqual(str(utils.resolve_torch_device(None)), "cpu")
            self.assertEqual(str(utils.resolve_torch_device("auto")), "cpu")

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=torch.device("cpu"),
            ),
        ):
            self.assertEqual(str(utils.resolve_torch_device("gpu")), "cpu")
            self.assertEqual(str(utils.resolve_torch_device("cuda:9")), "cpu")

        if hasattr(torch.backends, "mps"):
            with patch("torch.backends.mps.is_available", return_value=True):
                self.assertEqual(str(utils.resolve_torch_device("mps")), "mps")

    def test_coerce_to_list_and_merge_list_of_dicts_guard_paths(self):
        self.assertEqual(utils.coerce_to_list(OmegaConf.create([1, 2])), [1, 2])
        with self.assertRaises(TypeError):
            utils.coerce_to_list("not-a-list")

        merged = utils.merge_list_of_dicts(
            [{"a": 1}, OmegaConf.create({"b": 2}), {"a": 3}],
        )
        self.assertEqual(merged, {"a": 3, "b": 2})
        with self.assertRaises(TypeError):
            utils.merge_list_of_dicts([{"a": 1}, 2])

    def test_resolve_torch_device_mps_unavailable_falls_back_to_best_available(
        self,
    ):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch not available")
        if not hasattr(torch.backends, "mps"):
            self.skipTest("Torch build has no MPS backend")

        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=torch.device("cuda:0"),
            ),
        ):
            resolved = utils.resolve_torch_device("mps")

        self.assertEqual(str(resolved), "cuda:0")

    def test_get_call_params_success(self):
        cfg = ParamsConfig()
        params = cfg.get_call_params()
        self.assertEqual(params, {"x": 10, "y": "abc"})

    def test_get_call_params_missing_attribute_raises(self):
        cfg = MissingParamConfig()
        with self.assertRaises(AttributeError):
            cfg.get_call_params()

    def test_round_scores_handles_non_positive_samples_and_logger(self):
        class CaptureLogger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

        logger_obj = CaptureLogger()
        scores = utils.round_scores(
            {"acc": 0.1234, "label": "ok"},
            n_samples=0,
            logger_obj=logger_obj,
        )
        self.assertEqual(scores["acc"], 0.1)
        self.assertEqual(scores["label"], "ok")
        self.assertTrue(any("Rounding scores" in msg for msg in logger_obj.messages))

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

    def test_save_scores_csv_and_xlsx_paths(self):
        cfg = BaseConfig()
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "scores.csv"
            xlsx_path = Path(td) / "scores.xlsx"

            cfg.save_scores({"acc": 0.9}, csv_path)
            self.assertTrue(csv_path.exists())

            def fake_to_excel(path, index=False):
                _ = index
                Path(path).write_text("xlsx")

            with patch("pandas.core.series.Series.to_excel", side_effect=fake_to_excel) as to_excel:
                cfg.save_scores({"acc": 0.8}, xlsx_path)
            to_excel.assert_called_once()

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

    def test_read_scores_without_score_file_uses_score_dict(self):
        cfg = BaseConfig(score_dict={"base": 1})
        self.assertEqual(cfg.read_or_initialize_scores(None), {"base": 1})

    def test_load_scores_csv_and_xlsx_roundtrip(self):
        cfg = BaseConfig()
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "scores.csv"
            xlsx_path = Path(td) / "scores.xlsx"
            pd.DataFrame([{"acc": 0.9}]).to_csv(csv_path, index=False)
            xlsx_path.write_text("placeholder")

            with patch("pandas.read_excel", return_value=pd.DataFrame([{"acc": 0.8}])):
                csv_scores = cfg.load_scores(str(csv_path))
                xlsx_scores = cfg.load_scores(str(xlsx_path))

        self.assertIsInstance(csv_scores, dict)
        self.assertIsInstance(xlsx_scores, dict)
        self.assertIn("acc", csv_scores)
        self.assertIn("acc", xlsx_scores)

    def test_load_scores_unsupported_extension_raises(self):
        cfg = BaseConfig()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scores.txt"
            path.write_text("x")
            with self.assertRaises(ValueError):
                cfg.load_scores(str(path))

    def test_save_data_top_level_and_load_data_roundtrip_pickle(self):
        payload = {"a": [1, 2], "b": [3, 4]}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.pkl"
            save_data(payload, p)
            loaded = load_data(str(p))
            self.assertIsInstance(loaded, pd.DataFrame)
            self.assertEqual(list(loaded.columns), ["a", "b"])

    def test_save_and_load_data_json_and_unsupported_paths(self):
        frame = pd.DataFrame({"a": [1], "b": [2]})
        with tempfile.TemporaryDirectory() as td:
            json_path = Path(td) / "data.json"
            txt_path = Path(td) / "data.txt"
            save_data(frame, json_path)
            loaded = load_data(str(json_path))
            self.assertEqual(list(loaded.columns), ["a", "b"])
            with self.assertRaises(ValueError):
                save_data(frame, txt_path)
            txt_path.write_text("hello")
            with self.assertRaises(ValueError):
                load_data(str(txt_path))

    def test_save_and_load_data_other_format_paths(self):
        frame = pd.DataFrame({"a": [1], "b": [2]})
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "data.csv"
            parquet_path = Path(td) / "data.parquet"
            html_path = Path(td) / "data.html"
            xlsx_path = Path(td) / "data.xlsx"
            csv_path.write_text("a,b\n1,2\n")
            parquet_path.write_text("placeholder")
            html_path.write_text("<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>")
            xlsx_path.write_text("placeholder")

            with (
                patch("pandas.core.frame.DataFrame.to_parquet") as to_parquet,
                patch("pandas.core.frame.DataFrame.to_html") as to_html,
                patch("pandas.core.frame.DataFrame.to_excel") as to_excel,
                patch("pandas.read_csv", return_value=frame) as read_csv,
                patch("pandas.read_parquet", return_value=frame) as read_parquet,
                patch("pandas.read_html", return_value=[frame]) as read_html,
                patch("pandas.read_excel", return_value=frame) as read_excel,
            ):
                save_data(frame, parquet_path)
                save_data(frame, html_path)
                save_data(frame, xlsx_path)
                self.assertEqual(list(load_data(str(csv_path)).columns), ["a", "b"])
                self.assertEqual(list(load_data(str(parquet_path)).columns), ["a", "b"])
                self.assertEqual(list(load_data(str(html_path)).columns), ["a", "b"])
                self.assertEqual(list(load_data(str(xlsx_path)).columns), ["a", "b"])

            to_parquet.assert_called_once()
            to_html.assert_called_once()
            to_excel.assert_called_once()
            read_csv.assert_called_once()
            read_parquet.assert_called_once()
            read_html.assert_called_once()
            read_excel.assert_called_once()

    def test_load_data_json_explicit_lines_path(self):
        frame = pd.DataFrame({"a": [1]})
        with tempfile.TemporaryDirectory() as td:
            json_path = Path(td) / "data.json"
            json_path.write_text('{"a":1}\n')
            with patch("pandas.read_json", return_value=frame) as read_json:
                loaded = load_data(str(json_path), lines=False)

        self.assertEqual(list(loaded.columns), ["a"])
        read_json.assert_called_once()

    def test_configbase_save_data_and_load_data_wrapper(self):
        cfg = BaseConfig()
        frame = pd.DataFrame({"a": [1]})
        with tempfile.TemporaryDirectory() as td:
            json_path = Path(td) / "nested" / "data.json"
            cfg.save_data(frame, json_path)
            loaded = cfg.load_data(str(json_path))

        self.assertEqual(list(loaded.columns), ["a"])

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

    def test_load_object_ignore_and_delete_corrupt_file(self):
        cfg = BaseConfig()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "corrupt.pkl"
            path.write_bytes(b"not-a-pickle")
            loaded = cfg.load_object(str(path), ignore_corrupt=True, delete_corrupt=True)
            self.assertIsNone(loaded)
            self.assertFalse(path.exists())

    def test_load_missing_file_raises_and_load_updates_instance(self):
        cfg = TypeAConfig()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.pkl"
            cfg.custom = "before"
            cfg.save(str(path))

            restored = TypeAConfig()
            restored.custom = "after"
            result = restored.load(str(path))
            self.assertIs(result, restored)
            self.assertEqual(restored.custom, "before")

            with self.assertRaises(AssertionError):
                restored.load(str(Path(td) / "missing.pkl"))

    def test_from_yaml_from_dict_and_to_yaml_paths(self):
        target_a = f"{__name__}.TypeAConfig"
        target_b = f"{__name__}.TypeBConfig"
        with tempfile.TemporaryDirectory() as td:
            yaml_path = Path(td) / "config.yaml"
            yaml_path.write_text(f"_target_: {target_a}\nscore_dict: {{}}\n")
            loaded = BaseConfig.from_yaml(str(yaml_path))
            self.assertIsInstance(loaded, TypeAConfig)

            bad_yaml_path = Path(td) / "list.yaml"
            bad_yaml_path.write_text("- 1\n- 2\n")
            with self.assertRaises(TypeError):
                BaseConfig.from_yaml(str(bad_yaml_path))

        loaded = BaseConfig.from_dict({"_target_": target_b, "score_dict": {}})
        self.assertIsInstance(loaded, TypeBConfig)
        self.assertIn("score_dict", BaseConfig(score_dict={"x": 1}).to_yaml())

    def test_to_dict_supports_nested_configbase_and_hash_filtering(self):
        parent = BaseConfig(score_dict={"runtime": 1})
        parent.child = BaseConfig(score_dict={"nested": 2})
        parent.runtime_time = 3.5
        parent._private = "hidden"
        parent.extra_cfg = OmegaConf.create({"alpha": 1})

        as_dict = parent.to_dict()
        as_hash = parent.to_dict(for_hash=True)

        self.assertIn("child", as_dict)
        self.assertIn("extra_cfg", as_dict)
        self.assertNotIn("_private", as_dict)
        self.assertNotIn("runtime_time", as_hash)
        self.assertNotIn("score_dict", as_hash)

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

    def test_import_class_from_file_returns_class_when_not_instantiating(self):
        with tempfile.TemporaryDirectory() as td:
            module_path = Path(td) / "tmp_mod3.py"
            module_path.write_text("class MyClass:\n    pass\n")
            cls = import_class_from_file(
                str(module_path),
                "MyClass",
                instantiate_class=False,
            )
            self.assertEqual(cls.__name__, "MyClass")

    def test_import_class_from_file_raises_when_spec_missing(self):
        with tempfile.TemporaryDirectory() as td:
            module_path = Path(td) / "tmp_mod4.py"
            module_path.write_text("class MyClass:\n    pass\n")
            with patch.object(importlib.util, "spec_from_file_location", return_value=None):
                with self.assertRaises(ImportError):
                    import_class_from_file(str(module_path), "MyClass")

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

    def test_resolve_class_and_load_class_guard_paths(self):
        with self.assertRaises(TypeError):
            utils.resolve_class(123)
        with self.assertRaises(TypeError):
            load_class(123)

        with tempfile.TemporaryDirectory() as td:
            module_path = Path(td) / "tmp_mod5.py"
            missing_path = Path(td) / "missing.py"
            module_path.write_text("def helper():\n    return 'ok'\n")
            resolved = utils.resolve_class("json.loads")
            self.assertEqual(resolved.__name__, "loads")
            with self.assertRaises(FileNotFoundError):
                utils.resolve_class(f"{missing_path}:Thing")

        class InlineClass:
            def __init__(self, value):
                self.value = value

        loaded = load_class(InlineClass, "v")
        self.assertEqual(loaded.value, "v")

        with patch("deckard.utils.instantiate", side_effect=lambda cfg: cfg):
            instantiated_cfg = load_class("pkg.mod.Class", 1, label="x")
        self.assertEqual(instantiated_cfg["_args_"], [1])
        self.assertEqual(instantiated_cfg["label"], "x")

    def test_resolve_class_falls_back_to_get_class(self):
        with (
            patch("deckard.utils.importlib.import_module", side_effect=ImportError("boom")),
            patch("deckard.utils.get_class", return_value="fallback-class"),
        ):
            resolved = utils.resolve_class("pkg.mod.Class")
        self.assertEqual(resolved, "fallback-class")

    def test_create_parser_existing_parser_with_kwargs_raises(self):
        parser = argparse.ArgumentParser()
        with self.assertRaises(ValueError):
            create_parser_from_function(lambda a: a, parser=parser, prog="x")

    def test_extract_param_help_and_create_parser_guard_paths(self):
        self.assertEqual(utils._extract_param_help_from_docstring(""), {})

        doc = """Summary.\n\nParameters\n----------\nname : str\n    Name text.\n\nReturns\n-------\nstr\n"""
        self.assertEqual(utils._extract_param_help_from_docstring(doc), {"name": "Name text."})

        with self.assertRaises(ValueError):
            create_parser_from_function("not-callable")

        with self.assertRaises(ValueError):
            create_parser_from_function(lambda x: x, parser="bad-parser")

        existing = argparse.ArgumentParser(description="")

        def fn(alpha: str, beta: int = 1):
            """Parser summary.\n\nParameters\n----------\nalpha : str\n    Alpha value.\nbeta : int\n    Beta value.\n"""

            return alpha, beta

        parser = create_parser_from_function(fn, parser=existing, exclude=["beta"])
        self.assertEqual(parser.description, "Parser summary.")
        parsed = parser.parse_args(["--alpha", "x"])
        self.assertEqual(parsed.alpha, "x")
        self.assertFalse(hasattr(parsed, "beta"))

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

        self.assertEqual(
            name_action.help,
            "Name to echo in the command output.",
        )
        self.assertEqual(count_action.help, "Number of iterations to run.")


if __name__ == "__main__":
    unittest.main()
