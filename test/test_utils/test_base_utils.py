import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
import pytest
from omegaconf import OmegaConf

from deckard import utils
from deckard.utils import (
    BaseConfig,
    _auto_torch_device_from_backends,
    _torch_compiler_backends,
    coerce_config,
    coerce_to_list,
    create_parser_from_function,
    import_class_from_file,
    instantiate_plugin_spec,
    load_class,
    merge_list_of_dicts,
    merge_scores_with_collision_suffix,
    normalize_plugin_specs,
    resolve_class,
    resolve_torch_device,
    safe_store,
)


class BaseConfig(BaseConfig):
    def __call__(self):
        return 1


class ParamsConfig(BaseConfig):
    x: int = 10
    y: str = "abc"

    def __call__(self, x, y):
        return x, y


class MissingParamConfig(BaseConfig):
    def __call__(self, required_param):
        return required_param


class FailingConfig(BaseConfig):
    def __call__(self):
        raise RuntimeError("boom")


class TypeAConfig(BaseConfig):
    def __call__(self):
        return "A"


class TypeBConfig(BaseConfig):
    def __call__(self):
        return "B"


@dataclass(eq=False, kw_only=True)
class ChildComponentConfig(BaseConfig):
    name: str | None = None
    model_type: str | None = None
    defense: Any = None

    def __call__(self):
        return {}


class TestUtilsAdditional:
    def test_coerce_config_dictconfig_to_dict(self):
        cfg = OmegaConf.create({"alpha": 1, "beta": {"gamma": 2}})
        out = coerce_config(cfg)
        assert isinstance(out, dict)
        assert out["alpha"] == 1
        assert out["beta"]["gamma"] == 2

    def test_coerce_config_BaseConfig_to_dict(self):
        obj = BaseConfig(score_dict={"x": 1})
        out = coerce_config(obj)
        assert isinstance(out, dict)
        assert "score_dict" in out

    def test_coerce_config_yaml_path_to_dict(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "scorer.yaml"
            cfg_path.write_text("scorers:\n  acc:\n    score_name: acc\n")
            out = coerce_config(str(cfg_path))
            assert isinstance(out, dict)
            assert "scorers" in out

    def test_coerce_config_non_yaml_string_passthrough(self):
        class_path = "sklearn.metrics.accuracy_score"
        assert coerce_config(class_path) == class_path

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

        assert h1 == h2

    def test_hash_conf_values_stable_for_path_and_bytes(self):
        value = {
            "path": Path("a") / "b" / "c.txt",
            "payload": b"deckard",
        }

        h1 = utils.hash_conf_values(value)
        h2 = utils.hash_conf_values(value)

        assert h1 == h2

    def test_omegaconf_artifact_resolvers_load_and_save(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data_path = td_path / "data.csv"
            saved_data_path = td_path / "saved-data.csv"
            model_path = td_path / "model.pkl"
            saved_model_path = td_path / "saved-model.pkl"

            pd.DataFrame({"x": [1, 2]}).to_csv(data_path, index=False)

            data_cfg = OmegaConf.create(
                {
                    "source": str(data_path),
                    "loaded": "${load_data:${source}}",
                    "saved_path": "${save_data:${loaded},"
                    + str(saved_data_path)
                    + "}",
                },
            )
            resolved_data = OmegaConf.to_container(data_cfg, resolve=True)
            assert saved_data_path.exists()
            assert resolved_data["saved_path"] == str(saved_data_path)
            assert list(resolved_data["loaded"].columns) == ["x"]

            model_cfg = OmegaConf.create(
                {
                    "payload": BaseConfig(score_dict={"metric": 1}),
                    "model_path": str(model_path),
                    "saved_model_path": "${save_model:${payload},"
                    + str(saved_model_path)
                    + "}",
                },
                flags={"allow_objects": True},
            )
            resolved_model = OmegaConf.to_container(model_cfg, resolve=True)
            assert saved_model_path.exists()
            assert resolved_model["saved_model_path"] == str(saved_model_path)

    def test_BaseConfig_hash_deterministic_for_equal_content(self):
        cfg1 = BaseConfig(score_dict={"alpha": 1, "beta": 2})
        cfg2 = BaseConfig(score_dict={"beta": 2, "alpha": 1})

        cfg1.custom = {"z": [1, 2], "a": {"m": 9, "n": 8}}
        cfg2.custom = {"a": {"n": 8, "m": 9}, "z": [1, 2]}

        assert hash(cfg1) == hash(cfg2)

    def test_BaseConfig_fingerprint_is_stable_hex_string(self):
        cfg1 = BaseConfig(score_dict={"alpha": 1, "beta": 2})
        cfg2 = BaseConfig(score_dict={"beta": 2, "alpha": 1})

        assert cfg1.fingerprint == cfg2.fingerprint
        assert isinstance(cfg1.fingerprint, str)
        assert len(cfg1.fingerprint) == 32
        int(cfg1.fingerprint, 16)

    def test_BaseConfig_hash_uses_fingerprint_value(self):
        cfg = BaseConfig(score_dict={"alpha": 1})
        assert cfg.__hash__() == int(cfg.fingerprint, 16)

    def test_BaseConfig_fingerprint_ignores_runtime_mutations(self):
        cfg = BaseConfig(score_dict={"alpha": 1})
        fingerprint = cfg.fingerprint

        cfg.runtime_only = {"ephemeral": True}

        assert cfg.fingerprint == fingerprint
        assert cfg.__hash__() == int(fingerprint, 16)

    def test_BaseConfig_resolve_name_prefers_canonical_name_field(self):
        cfg = BaseConfig(score_dict={"alpha": 1})
        cfg.name = "canonical-name"
        cfg.attack_type = "legacy-attack-type"
        assert cfg.resolve_name() == "canonical-name"

    def test_BaseConfig_resolve_name_falls_back_to_legacy_alias(self):
        cfg = BaseConfig(score_dict={"alpha": 1})
        cfg.name = "legacy-dataset"
        assert cfg.resolve_name() == "legacy-dataset"

    def test_coerce_component_coerces_alias_to_canonical_name(self):
        cfg = BaseConfig(score_dict={"alpha": 1})
        child = cfg.coerce_component(
            {"name": "torch.nn.Linear"},
            ChildComponentConfig,
        )
        assert isinstance(child, ChildComponentConfig)
        assert child.name == "torch.nn.Linear"
        assert child.name == "torch.nn.Linear"

    def test_coerce_component_rejects_blank_name_after_coercion(self):
        cfg = BaseConfig(score_dict={"alpha": 1})
        with pytest.raises(ValueError, match="Invalid name value"):
            cfg.coerce_component({"name": "   "}, ChildComponentConfig)

    def test_coerce_component_applies_defaults_only_when_missing(self):
        cfg = BaseConfig(score_dict={"alpha": 1})
        child = cfg.coerce_component(
            {"name": "child-name"},
            ChildComponentConfig,
            overrides={"defense": "parent-defense"},
        )
        assert child.defense == "parent-defense"

        child_explicit = cfg.coerce_component(
            {"name": "child-name", "defense": "child-defense"},
            ChildComponentConfig,
            overrides={"defense": "parent-defense"},
        )
        assert child_explicit.defense == "child-defense"

    def test_resolve_torch_device_cuda_falls_back_to_best_available(self):
        try:
            import torch
        except ImportError:
            pytest.skip("Torch not available")

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=torch.device("mps"),
            ),
        ):
            resolved = utils.resolve_torch_device("cuda")

        assert str(resolved) == "mps"

    def test_resolve_torch_device_invalid_cuda_index_falls_back_to_best_available(
        self,
    ):
        try:
            import torch
        except ImportError:
            pytest.skip("Torch not available")

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

        assert str(resolved) == "mps"

    def test_resolve_torch_device_mps_unavailable_falls_back_to_best_available(
        self,
    ):
        try:
            import torch
        except ImportError:
            pytest.skip("Torch not available")
        if not hasattr(torch.backends, "mps"):
            pytest.skip("Torch build has no MPS backend")

        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=torch.device("cuda:0"),
            ),
        ):
            resolved = utils.resolve_torch_device("mps")

        assert str(resolved) == "cuda:0"

    def test_get_call_params_success(self):
        cfg = ParamsConfig()
        params = cfg.get_call_params()
        assert params == {"x": 10, "y": "abc"}

    def test_get_call_params_missing_attribute_raises(self):
        cfg = MissingParamConfig()
        with pytest.raises(AttributeError):
            cfg.get_call_params()

    def test_normalize_plugin_specs_requires_list_like(self):
        assert normalize_plugin_specs(None) == []
        with pytest.raises(TypeError):
            normalize_plugin_specs("deckard.plugins.foo.Plugin")

    def test_instantiate_plugin_spec_uses_loader_for_dict_and_string(self):
        calls = []

        def _loader(path, **kwargs):
            calls.append((path, kwargs))
            return {"path": path, **kwargs}

        out_dict = instantiate_plugin_spec(
            {"name": "pkg.Plugin", "alpha": 1},
            loader=_loader,
        )
        out_str = instantiate_plugin_spec("pkg.Plugin2", loader=_loader)

        assert out_dict["path"] == "pkg.Plugin"
        assert out_dict["alpha"] == 1
        assert out_str["path"] == "pkg.Plugin2"
        assert len(calls) == 2

    def test_instantiate_plugin_spec_passthrough_object(self):
        marker = object()
        out = instantiate_plugin_spec(marker, loader=lambda *_a, **_k: None)
        assert out is marker

    def test_execute_returns_fallback_score_dict_on_exception(self):
        cfg = FailingConfig(score_dict={"fallback": 123})
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "deckard.log"
            handler = logging.FileHandler(log_path)
            utils.logger.addHandler(handler)
            try:
                out = cfg.execute_without_mercy()
                assert out == {"fallback": 123}
                assert log_path.exists()
                assert "Exception:" in log_path.read_text()
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
            assert obj.x == 7

    def test_import_class_from_file_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
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
            assert obj.name == "deckard"

    def test_create_parser_existing_parser_with_kwargs_raises(self):
        parser = argparse.ArgumentParser()
        with pytest.raises(ValueError):
            create_parser_from_function(lambda a: a, parser=parser, prog="x")

    def test_create_parser_unannotated_defaults_to_string(self):
        def fn(name, count: int = 1):
            return name, count

        parser = create_parser_from_function(fn)
        args = parser.parse_args(["--name", "alice"])
        assert args.name == "alice"
        assert args.count == 1

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

        assert (
            parser.description
            == "Create a parser description from the function docstring."
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

        assert name_action.help == "Name to echo in the command output."
        assert count_action.help == "Number of iterations to run."


# ── Minimal BaseConfig subclass ──────────────────────────────────────────────


class _Cfg(BaseConfig):
    def __call__(self):
        return self.score_dict


class _Fail(BaseConfig):
    def __call__(self):
        raise RuntimeError("deliberate failure")


# ── _torch_compiler_backends ─────────────────────────────────────────────────


class TestTorchCompilerBackends:
    def test_no_compiler_attribute_returns_empty(self):
        mod = SimpleNamespace()  # no .compiler
        assert _torch_compiler_backends(mod) == []

    def test_compiler_no_list_backends_returns_empty(self):
        mod = SimpleNamespace(compiler=SimpleNamespace())
        assert _torch_compiler_backends(mod) == []

    def test_list_backends_exception_returns_empty(self):
        def _raise():
            raise RuntimeError("boom")

        compiler = SimpleNamespace(list_backends=_raise)
        mod = SimpleNamespace(compiler=compiler)
        assert _torch_compiler_backends(mod) == []

    def test_list_backends_returns_normalised_names(self):
        compiler = SimpleNamespace(list_backends=lambda: ["Inductor", " CUDA ", "tvm"])
        mod = SimpleNamespace(compiler=compiler)
        result = _torch_compiler_backends(mod)
        assert "inductor" in result
        assert "cuda" in result
        assert "tvm" in result


# ── _auto_torch_device_from_backends ─────────────────────────────────────────


class TestAutoTorchDevice:
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
        assert dev.type == "cuda"

    def test_mps_with_preferred_backend_returns_mps(self):
        torch_mock = self._make_torch_mock(cuda=False, mps=True, backends=["eager"])
        dev = _auto_torch_device_from_backends(torch_mock)
        assert dev.type == "mps"

    def test_cuda_without_preferred_backend_returns_cuda(self):
        torch_mock = self._make_torch_mock(cuda=True, mps=False, backends=["tvm"])
        dev = _auto_torch_device_from_backends(torch_mock)
        assert dev.type == "cuda"

    def test_mps_without_preferred_backend_returns_mps(self):
        torch_mock = self._make_torch_mock(cuda=False, mps=True, backends=["tvm"])
        dev = _auto_torch_device_from_backends(torch_mock)
        assert dev.type == "mps"

    def test_neither_cuda_nor_mps_returns_cpu(self):
        torch_mock = self._make_torch_mock(cuda=False, mps=False, backends=[])
        dev = _auto_torch_device_from_backends(torch_mock)
        assert dev.type == "cpu"


# ── resolve_torch_device ─────────────────────────────────────────────────────


class TestResolveTorchDevice:
    def setup_method(self):
        try:
            import torch

            self.torch = torch
        except ImportError:
            pytest.skip("torch not available")

    def test_torch_device_passthrough(self):
        dev = self.torch.device("cpu")
        result = resolve_torch_device(dev)
        assert result is dev

    def test_none_returns_auto(self):
        # With torch available, None should return some device (auto-selected)
        result = resolve_torch_device(None)
        assert isinstance(result, self.torch.device)

    def test_valid_int_cuda_when_unavailable_falls_back(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=self.torch.device("cpu"),
            ),
        ):
            result = resolve_torch_device(0)
        assert result.type == "cpu"

    def test_gpu_text_cuda_unavailable_falls_back(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=self.torch.device("cpu"),
            ),
        ):
            result = resolve_torch_device("gpu")
        assert result.type == "cpu"

    def test_cuda_text_cuda_unavailable_falls_back(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "deckard.utils._auto_torch_device_from_backends",
                return_value=self.torch.device("cpu"),
            ),
        ):
            result = resolve_torch_device("cuda:0")
        assert result.type == "cpu"

    def test_null_token_returns_auto(self):
        with patch(
            "deckard.utils._auto_torch_device_from_backends",
            return_value=self.torch.device("cpu"),
        ):
            result = resolve_torch_device("none")
        assert result.type == "cpu"

    def test_auto_token_returns_auto(self):
        with patch(
            "deckard.utils._auto_torch_device_from_backends",
            return_value=self.torch.device("cpu"),
        ):
            result = resolve_torch_device("auto")
        assert result.type == "cpu"

    def test_cpu_string_returns_cpu(self):
        result = resolve_torch_device("cpu")
        assert result.type == "cpu"

    def test_invalid_device_string_falls_back_to_auto(self):
        with patch(
            "deckard.utils._auto_torch_device_from_backends",
            return_value=self.torch.device("cpu"),
        ):
            result = resolve_torch_device("not_a_device_xyz")
        assert result.type == "cpu"

    def test_resolve_torch_device_no_torch(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = resolve_torch_device(None)
        assert result == "cpu"

    def test_resolve_torch_device_no_torch_with_value(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = resolve_torch_device("cpu")
        assert result == "cpu"


# ── coerce_to_list / merge_list_of_dicts ─────────────────────────────────────


class TestCoerceHelpers:
    def test_coerce_to_list_with_plain_list(self):
        assert coerce_to_list([1, 2, 3]) == [1, 2, 3]

    def test_coerce_to_list_invalid_type_raises(self):
        with pytest.raises(TypeError):
            coerce_to_list({"a": 1})

    def test_merge_list_of_dicts_merges(self):
        result = merge_list_of_dicts([{"a": 1}, {"b": 2}])
        assert result == {"a": 1, "b": 2}

    def test_merge_list_of_dicts_later_wins(self):
        result = merge_list_of_dicts([{"a": 1}, {"a": 99}])
        assert result["a"] == 99

    def test_merge_list_of_dicts_invalid_element_raises(self):
        with pytest.raises(TypeError):
            merge_list_of_dicts(["not_a_dict"])

    def test_coerce_config_none_returns_none(self):
        assert coerce_config(None) is None

    def test_coerce_config_yaml_path_returns_dict(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cfg.yaml"
            p.write_text("a: 1\nb: 2\n")
            result = coerce_config(str(p))
            assert isinstance(result, dict)
            assert result["a"] == 1

    def test_merge_scores_with_collision_suffix_keeps_unique_keys(self):
        result = merge_scores_with_collision_suffix(
            {"accuracy": 0.8},
            {"latency": 1.2},
            alias="hsj",
        )
        assert result["accuracy"] == 0.8
        assert result["latency"] == 1.2

    def test_merge_scores_with_collision_suffix_uses_alias_for_collisions(self):
        result = merge_scores_with_collision_suffix(
            {"evasion_accuracy": 0.6},
            {"evasion_accuracy": 0.4, "attack_generation_time": 2.0},
            alias="fgm",
        )
        assert result["evasion_accuracy"] == 0.6
        assert result["evasion_accuracy_fgm"] == 0.4
        assert result["attack_generation_time"] == 2.0

    def test_merge_scores_with_collision_suffix_without_alias_overwrites(self):
        result = merge_scores_with_collision_suffix(
            {"evasion_accuracy": 0.6},
            {"evasion_accuracy": 0.4},
            alias=None,
        )
        assert result["evasion_accuracy"] == 0.4


# ── BaseConfig – from_yaml / to_yaml / to_dict ───────────────────────────────


class TestBaseConfigSerialisation:
    def test_to_yaml_returns_string(self):
        cfg = _Cfg(score_dict={"a": 1})
        yaml_str = cfg.to_yaml()
        assert isinstance(yaml_str, str)
        assert "score_dict" in yaml_str

    def test_to_dict_returns_dict(self):
        cfg = _Cfg(score_dict={"b": 2})
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "score_dict" in d

    def test_to_dict_for_hash_excludes_score_dict(self):
        cfg = _Cfg(score_dict={"c": 3})
        d = cfg.to_dict(for_hash=True)
        assert "score_dict" not in d

    def test_from_dict_round_trip(self):
        data = {
            "_target_": "deckard.utils.BaseConfig",
            "score_dict": {"x": 5},
        }
        obj = BaseConfig.from_dict(data)
        assert obj is not None


# ── resolve_class / load_class ───────────────────────────────────────────────


class TestResolveLoadClass:
    def test_resolve_class_non_string_raises(self):
        with pytest.raises(TypeError):
            resolve_class(123)

    def test_load_class_non_string_non_type_raises(self):
        with pytest.raises(TypeError):
            load_class(123)

    def test_load_class_with_type_instantiates(self):
        result = load_class(dict)
        assert isinstance(result, dict)

    def test_resolve_class_dotted_path(self):
        cls = resolve_class("sklearn.ensemble.RandomForestClassifier")
        from sklearn.ensemble import RandomForestClassifier

        assert cls is RandomForestClassifier

    def test_resolve_class_file_path_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            resolve_class("/nonexistent/path.py:SomeClass")

    def test_load_class_dotted_path(self):
        from sklearn.ensemble import RandomForestClassifier

        obj = load_class(
            "sklearn.ensemble.RandomForestClassifier",
            n_estimators=5,
        )
        assert isinstance(obj, RandomForestClassifier)


# ── execute_without_mercy ─────────────────────────────────────────────────────


class TestExecuteWithoutMercy:
    def test_success_path(self):
        cfg = _Cfg(score_dict={"ok": 1})
        result = cfg.execute_without_mercy()
        assert result == {"ok": 1}

    def test_exception_path_returns_score_dict(self):
        cfg = _Fail(score_dict={"fallback": 99})
        with tempfile.TemporaryDirectory() as td:
            import logging

            log_path = Path(td) / "deckard.log"
            handler = logging.FileHandler(log_path)
            utils.logger.addHandler(handler)
            try:
                result = cfg.execute_without_mercy()
                assert result == {"fallback": 99}
            finally:
                utils.logger.removeHandler(handler)
                handler.close()
