import argparse
import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest
from omegaconf import OmegaConf

from deckard import utils
from deckard.utils import (
    ConfigBase,
    _auto_torch_device_from_backends,
    _torch_compiler_backends,
    coerce_config,
    coerce_to_list,
    create_parser_from_function,
    import_class_from_file,
    load_class,
    load_data,
    merge_list_of_dicts,
    merge_scores_with_collision_suffix,
    resolve_class,
    resolve_torch_device,
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

        self.assertEqual(
            name_action.help,
            "Name to echo in the command output.",
        )
        self.assertEqual(count_action.help, "Number of iterations to run.")


# ── Minimal ConfigBase subclass ──────────────────────────────────────────────


class _Cfg(ConfigBase):
    def __call__(self):
        return self.score_dict


class _Fail(ConfigBase):
    def __call__(self):
        raise RuntimeError("deliberate failure")


# ── _torch_compiler_backends ─────────────────────────────────────────────────


class TestTorchCompilerBackends(unittest.TestCase):
    def test_no_compiler_attribute_returns_empty(self):
        mod = SimpleNamespace()  # no .compiler
        self.assertEqual(_torch_compiler_backends(mod), [])

    def test_compiler_no_list_backends_returns_empty(self):
        mod = SimpleNamespace(compiler=SimpleNamespace())
        self.assertEqual(_torch_compiler_backends(mod), [])

    def test_list_backends_exception_returns_empty(self):
        def _raise():
            raise RuntimeError("boom")

        compiler = SimpleNamespace(list_backends=_raise)
        mod = SimpleNamespace(compiler=compiler)
        self.assertEqual(_torch_compiler_backends(mod), [])

    def test_list_backends_returns_normalised_names(self):
        compiler = SimpleNamespace(list_backends=lambda: ["Inductor", " CUDA ", "tvm"])
        mod = SimpleNamespace(compiler=compiler)
        result = _torch_compiler_backends(mod)
        self.assertIn("inductor", result)
        self.assertIn("cuda", result)
        self.assertIn("tvm", result)


# ── _auto_torch_device_from_backends ─────────────────────────────────────────


class TestAutoTorchDevice(unittest.TestCase):
    """Drive all branches of _auto_torch_device_from_backends via mocks."""

    def _make_torch_mock(self, cuda=False, mps=False, backends=None):
        _backends = list(backends or [])

        def device_fn(spec):
            return SimpleNamespace(type=spec.split(":")[0])

        mock = SimpleNamespace(
            device=device_fn,
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: mps),
            ),
            compiler=SimpleNamespace(list_backends=lambda: _backends),
        )
        return mock

    def test_cuda_with_preferred_backend_returns_cuda(self):
        torch_mock = self._make_torch_mock(cuda=True, mps=False, backends=["inductor"])
        dev = _auto_torch_device_from_backends(torch_mock)
        self.assertEqual(dev.type, "cuda")

    def test_mps_with_preferred_backend_returns_mps(self):
        torch_mock = self._make_torch_mock(cuda=False, mps=True, backends=["eager"])
        dev = _auto_torch_device_from_backends(torch_mock)
        self.assertEqual(dev.type, "mps")

    def test_cuda_without_preferred_backend_returns_cuda(self):
        torch_mock = self._make_torch_mock(cuda=True, mps=False, backends=["tvm"])
        dev = _auto_torch_device_from_backends(torch_mock)
        self.assertEqual(dev.type, "cuda")

    def test_mps_without_preferred_backend_returns_mps(self):
        torch_mock = self._make_torch_mock(cuda=False, mps=True, backends=["tvm"])
        dev = _auto_torch_device_from_backends(torch_mock)
        self.assertEqual(dev.type, "mps")

    def test_neither_cuda_nor_mps_returns_cpu(self):
        torch_mock = self._make_torch_mock(cuda=False, mps=False, backends=[])
        dev = _auto_torch_device_from_backends(torch_mock)
        self.assertEqual(dev.type, "cpu")


# ── resolve_torch_device ─────────────────────────────────────────────────────


class TestResolveTorchDevice(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            self.torch = torch
        except ImportError:
            self.skipTest("torch not available")

    def test_torch_device_passthrough(self):
        dev = self.torch.device("cpu")
        result = resolve_torch_device(dev)
        self.assertIs(result, dev)

    def test_none_returns_auto(self):
        # With torch available, None should return some device (auto-selected)
        result = resolve_torch_device(None)
        self.assertIsInstance(result, self.torch.device)

    def test_valid_int_cuda_when_unavailable_falls_back(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=self.torch.device("cpu"),
            ),
        ):
            result = resolve_torch_device(0)
        self.assertEqual(result.type, "cpu")

    def test_gpu_text_cuda_unavailable_falls_back(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=self.torch.device("cpu"),
            ),
        ):
            result = resolve_torch_device("gpu")
        self.assertEqual(result.type, "cpu")

    def test_cuda_text_cuda_unavailable_falls_back(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=self.torch.device("cpu"),
            ),
        ):
            result = resolve_torch_device("cuda:0")
        self.assertEqual(result.type, "cpu")

    def test_null_token_returns_auto(self):
        with patch(
            "deckard.utils._auto_torch_device_from_backends",
            return_value=self.torch.device("cpu"),
        ):
            result = resolve_torch_device("none")
        self.assertEqual(result.type, "cpu")

    def test_auto_token_returns_auto(self):
        with patch(
            "deckard.utils._auto_torch_device_from_backends",
            return_value=self.torch.device("cpu"),
        ):
            result = resolve_torch_device("auto")
        self.assertEqual(result.type, "cpu")

    def test_cpu_string_returns_cpu(self):
        result = resolve_torch_device("cpu")
        self.assertEqual(result.type, "cpu")

    def test_invalid_device_string_falls_back_to_auto(self):
        with patch(
            "deckard.utils._auto_torch_device_from_backends",
            return_value=self.torch.device("cpu"),
        ):
            result = resolve_torch_device("not_a_device_xyz")
        self.assertEqual(result.type, "cpu")

    def test_resolve_torch_device_no_torch(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = resolve_torch_device(None)
        self.assertEqual(result, "cpu")

    def test_resolve_torch_device_no_torch_with_value(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = resolve_torch_device("cpu")
        self.assertEqual(result, "cpu")


# ── coerce_to_list / merge_list_of_dicts ─────────────────────────────────────


class TestCoerceHelpers(unittest.TestCase):
    def test_coerce_to_list_with_plain_list(self):
        self.assertEqual(coerce_to_list([1, 2, 3]), [1, 2, 3])

    def test_coerce_to_list_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            coerce_to_list({"a": 1})

    def test_merge_list_of_dicts_merges(self):
        result = merge_list_of_dicts([{"a": 1}, {"b": 2}])
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_merge_list_of_dicts_later_wins(self):
        result = merge_list_of_dicts([{"a": 1}, {"a": 99}])
        self.assertEqual(result["a"], 99)

    def test_merge_list_of_dicts_invalid_element_raises(self):
        with self.assertRaises(TypeError):
            merge_list_of_dicts(["not_a_dict"])

    def test_coerce_config_none_returns_none(self):
        self.assertIsNone(coerce_config(None))

    def test_coerce_config_yaml_path_returns_dict(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cfg.yaml"
            p.write_text("a: 1\nb: 2\n")
            result = coerce_config(str(p))
            self.assertIsInstance(result, dict)
            self.assertEqual(result["a"], 1)

    def test_merge_scores_with_collision_suffix_keeps_unique_keys(self):
        result = merge_scores_with_collision_suffix(
            {"accuracy": 0.8},
            {"latency": 1.2},
            alias="hsj",
        )
        self.assertEqual(result["accuracy"], 0.8)
        self.assertEqual(result["latency"], 1.2)

    def test_merge_scores_with_collision_suffix_uses_alias_for_collisions(self):
        result = merge_scores_with_collision_suffix(
            {"evasion_accuracy": 0.6},
            {"evasion_accuracy": 0.4, "attack_generation_time": 2.0},
            alias="fgm",
        )
        self.assertEqual(result["evasion_accuracy"], 0.6)
        self.assertEqual(result["evasion_accuracy_fgm"], 0.4)
        self.assertEqual(result["attack_generation_time"], 2.0)

    def test_merge_scores_with_collision_suffix_without_alias_overwrites(self):
        result = merge_scores_with_collision_suffix(
            {"evasion_accuracy": 0.6},
            {"evasion_accuracy": 0.4},
            alias=None,
        )
        self.assertEqual(result["evasion_accuracy"], 0.4)


# ── ConfigBase – save/load scores ────────────────────────────────────────────


class TestConfigBaseScores(unittest.TestCase):
    def test_save_and_load_scores_csv(self):
        cfg = _Cfg()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            cfg.save_scores({"acc": 0.9}, p)
            loaded = cfg.load_scores(str(p))
            self.assertIsNotNone(loaded)

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("openpyxl") is None,
        reason="openpyxl not installed",
    )
    def test_save_and_load_scores_xlsx(self):
        cfg = _Cfg()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.xlsx"
            cfg.save_scores({"acc": 0.9}, p)
            loaded = cfg.load_scores(str(p))
            self.assertIsNotNone(loaded)

    def test_load_scores_unsupported_extension_raises(self):
        cfg = _Cfg()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.txt"
            p.write_text("nothing")
            with self.assertRaises(ValueError):
                cfg.load_scores(str(p))

    def test_save_scores_csv_saves_file(self):
        cfg = _Cfg()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            cfg.save_scores({"x": 1.0, "y": 2.0}, p)
            self.assertTrue(p.exists())


# ── ConfigBase – save_data / load_data ───────────────────────────────────────


class TestConfigBaseSaveLoadData(unittest.TestCase):
    def test_save_data_html(self):
        cfg = _Cfg()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.html"
            cfg.save_data(df, p)
            self.assertTrue(p.exists())

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("openpyxl") is None,
        reason="openpyxl not installed",
    )
    def test_save_data_xlsx(self):
        cfg = _Cfg()
        df = pd.DataFrame({"a": [1, 2]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.xlsx"
            cfg.save_data(df, p)
            self.assertTrue(p.exists())

    def test_save_data_parquet(self):
        cfg = _Cfg()
        df = pd.DataFrame({"a": [1, 2]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.parquet"
            cfg.save_data(df, p)
            self.assertTrue(p.exists())

    def test_save_data_unsupported_raises(self):
        cfg = _Cfg()
        df = pd.DataFrame({"a": [1]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.xyz"
            with self.assertRaises(ValueError):
                cfg.save_data(df, p)

    def test_load_data_delegates_to_top_level(self):
        cfg = _Cfg()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            pd.DataFrame({"a": [1, 2]}).to_csv(p, index=False)
            result = cfg.load_data(str(p))
            self.assertIsInstance(result, pd.DataFrame)


# ── ConfigBase – save_object / load_object ───────────────────────────────────


class TestConfigBaseSaveLoadObject(unittest.TestCase):
    def test_save_object_unsupported_extension_raises(self):
        cfg = _Cfg()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obj.txt"
            with self.assertRaises(ValueError):
                cfg.save_object(cfg, str(p))

    def test_load_object_corrupt_with_ignore_returns_none(self):
        cfg = _Cfg()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "corrupt.pkl"
            p.write_bytes(b"not valid pickle data!!")
            result = cfg.load_object(str(p), ignore_corrupt=True)
            self.assertIsNone(result)

    def test_load_object_corrupt_with_delete_removes_file(self):
        cfg = _Cfg()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "corrupt.pkl"
            p.write_bytes(b"garbage")
            result = cfg.load_object(str(p), ignore_corrupt=True, delete_corrupt=True)
            self.assertIsNone(result)
            self.assertFalse(p.exists())


# ── ConfigBase – from_yaml / to_yaml / to_dict ───────────────────────────────


class TestConfigBaseSerialisation(unittest.TestCase):
    def test_to_yaml_returns_string(self):
        cfg = _Cfg(score_dict={"a": 1})
        yaml_str = cfg.to_yaml()
        self.assertIsInstance(yaml_str, str)
        self.assertIn("score_dict", yaml_str)

    def test_to_dict_returns_dict(self):
        cfg = _Cfg(score_dict={"b": 2})
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("score_dict", d)

    def test_to_dict_for_hash_excludes_score_dict(self):
        cfg = _Cfg(score_dict={"c": 3})
        d = cfg.to_dict(for_hash=True)
        self.assertNotIn("score_dict", d)

    def test_from_dict_round_trip(self):
        data = {
            "_target_": "deckard.utils.ConfigBase",
            "score_dict": {"x": 5},
        }
        obj = ConfigBase.from_dict(data)
        self.assertIsNotNone(obj)


# ── Top-level save_data / load_data ──────────────────────────────────────────


class TestTopLevelSaveLoadData(unittest.TestCase):
    def test_save_and_load_csv(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.csv"
            save_data(df, p)
            result = load_data(str(p))
            self.assertIsInstance(result, pd.DataFrame)

    def test_save_and_load_parquet(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.parquet"
            save_data(df, p)
            result = load_data(str(p))
            self.assertIsInstance(result, pd.DataFrame)

    def test_save_and_load_json(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.json"
            save_data(df, p)
            result = load_data(str(p))
            self.assertIsInstance(result, pd.DataFrame)

    def test_save_and_load_html(self):
        lxml = __import__("importlib").util.find_spec("lxml")
        if lxml is None:
            self.skipTest("lxml not installed")
        df = pd.DataFrame({"x": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.html"
            save_data(df, p)
            result = load_data(str(p))
            self.assertIsInstance(result, pd.DataFrame)

    def test_save_and_load_xlsx(self):
        if __import__("importlib").util.find_spec("openpyxl") is None:
            self.skipTest("openpyxl not installed")
        df = pd.DataFrame({"x": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.xlsx"
            save_data(df, p)
            result = load_data(str(p))
            self.assertIsInstance(result, pd.DataFrame)

    def test_save_and_load_pkl(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.pkl"
            save_data(df, p)
            result = load_data(str(p))
            self.assertIsInstance(result, pd.DataFrame)

    def test_save_data_unsupported_raises(self):
        df = pd.DataFrame({"x": [1]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.xyz"
            with self.assertRaises(ValueError):
                save_data(df, p)

    def test_load_data_unsupported_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.xyz"
            p.write_text("junk")
            with self.assertRaises(ValueError):
                load_data(str(p))

    def test_save_data_converts_non_dataframe(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.csv"
            save_data({"a": [1, 2], "b": [3, 4]}, p)
            self.assertTrue(p.exists())


# ── resolve_class / load_class ───────────────────────────────────────────────


class TestResolveLoadClass(unittest.TestCase):
    def test_resolve_class_non_string_raises(self):
        with self.assertRaises(TypeError):
            resolve_class(123)

    def test_load_class_non_string_non_type_raises(self):
        with self.assertRaises(TypeError):
            load_class(123)

    def test_load_class_with_type_instantiates(self):
        result = load_class(dict)
        self.assertIsInstance(result, dict)

    def test_resolve_class_dotted_path(self):
        cls = resolve_class("sklearn.ensemble.RandomForestClassifier")
        from sklearn.ensemble import RandomForestClassifier

        self.assertIs(cls, RandomForestClassifier)

    def test_resolve_class_file_path_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_class("/nonexistent/path.py:SomeClass")

    def test_load_class_dotted_path(self):
        from sklearn.ensemble import RandomForestClassifier

        obj = load_class(
            "sklearn.ensemble.RandomForestClassifier",
            n_estimators=5,
        )
        self.assertIsInstance(obj, RandomForestClassifier)


# ── execute_without_mercy ─────────────────────────────────────────────────────


class TestExecuteWithoutMercy(unittest.TestCase):
    def test_success_path(self):
        cfg = _Cfg(score_dict={"ok": 1})
        result = cfg.execute_without_mercy()
        self.assertEqual(result, {"ok": 1})

    def test_exception_path_returns_score_dict(self):
        cfg = _Fail(score_dict={"fallback": 99})
        with tempfile.TemporaryDirectory() as td:
            import logging

            log_path = Path(td) / "deckard.log"
            handler = logging.FileHandler(log_path)
            utils.logger.addHandler(handler)
            try:
                result = cfg.execute_without_mercy()
                self.assertEqual(result, {"fallback": 99})
            finally:
                utils.logger.removeHandler(handler)
                handler.close()


if __name__ == "__main__":
    unittest.main()
