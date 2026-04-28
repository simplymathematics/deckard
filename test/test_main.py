import importlib
import sys
from argparse import Namespace
from pathlib import Path
import pytest

import logging.config



@pytest.fixture
def main_module():
    import deckard.__main__ as mod

    return importlib.reload(mod)


def test_get_configuration_paths_returns_expected_values(main_module, monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("x: 1\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DECKARD_CONFIG_DIR", "config")
    monkeypatch.delenv("DECKARD_DEFAULT_CONFIG_FILE", raising=False)

    resolved_dir, config_file = main_module.get_configuration_paths()

    assert resolved_dir == "config"
    assert config_file == "default.yaml"


def test_get_configuration_paths_prompts_for_valid_directory(main_module, monkeypatch, tmp_path):
    config_dir = tmp_path / "real_config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("x: 1\n")

    monkeypatch.setenv("DECKARD_CONFIG_DIR", "missing")
    monkeypatch.setattr("builtins.input", lambda _: str(config_dir))

    resolved_dir, config_file = main_module.get_configuration_paths()

    assert resolved_dir == str(config_dir)
    assert config_file == "default.yaml"


def test_get_configuration_paths_raises_for_missing_config_file(main_module, monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    monkeypatch.setenv("DECKARD_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("DECKARD_DEFAULT_CONFIG_FILE", "missing.yaml")

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        main_module.get_configuration_paths()


def test_main_dispatches_to_default_module(main_module, monkeypatch, tmp_path):
    monkeypatch.setenv("DECKARD_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["deckard", "experiment"])

    calls = {"default": 0}

    def fake_default():
        calls["default"] += 1

    monkeypatch.setattr(main_module, "handle_default_module", fake_default)
    monkeypatch.setattr(main_module, "handle_other_layers", lambda layer: None)

    main_module.main()

    assert calls["default"] == 1
    assert Path(main_module.os.environ["DECKARD_CONFIG_DIR"]) == tmp_path.resolve()


def test_main_dispatches_to_supported_layer(main_module, monkeypatch, tmp_path):
    layer = next(
        layer
        for layer in main_module.SUPPORTED_LAYERS
        if layer not in {"experiment", "optimize"}
    )

    monkeypatch.setenv("DECKARD_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["deckard", layer])

    seen = {}

    monkeypatch.setattr(main_module, "handle_default_module", lambda: None)
    monkeypatch.setattr(main_module, "handle_other_layers", lambda value: seen.setdefault("layer", value))

    main_module.main()

    assert seen["layer"] == layer


def test_main_prompts_for_config_directory_when_default_path_missing(main_module, monkeypatch, tmp_path):
    provided_dir = tmp_path / "provided"
    provided_dir.mkdir()

    monkeypatch.delenv("DECKARD_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: str(provided_dir))
    monkeypatch.setattr(sys, "argv", ["deckard", "optimize"])

    calls = {"default": 0}
    monkeypatch.setattr(main_module, "handle_default_module", lambda: calls.__setitem__("default", calls["default"] + 1))
    monkeypatch.setattr(main_module, "handle_other_layers", lambda layer: None)

    main_module.main()

    assert calls["default"] == 1
    assert Path(main_module.os.environ["DECKARD_CONFIG_DIR"]) == provided_dir.resolve()


def test_main_raises_for_unsupported_module(main_module, monkeypatch, tmp_path):
    monkeypatch.setenv("DECKARD_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["deckard", "not-supported"])

    with pytest.raises(ValueError, match="not supported"):
        main_module.main()


def test_handle_default_module_builds_hydra_entrypoint(main_module, monkeypatch, tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()

    seen = {}

    def fake_hydra_main(**kwargs):
        seen["hydra_kwargs"] = kwargs

        def decorator(fn):
            def runner():
                return fn("CFG")

            return runner

        return decorator

    def fake_optimize_main(cfg):
        seen["cfg"] = cfg
        return {"score": 42}

    monkeypatch.setattr(main_module, "get_configuration_paths", lambda: (str(cfg_dir), "default.yaml"))
    monkeypatch.setattr(main_module.hydra, "main", fake_hydra_main)
    monkeypatch.setitem(main_module.layer_dict, "optimize", (object(), fake_optimize_main))

    result = main_module.handle_default_module()

    assert result == {"score": 42}
    assert seen["cfg"] == "CFG"
    assert seen["hydra_kwargs"] == {
        "config_path": str(cfg_dir.resolve()),
        "config_name": "default.yaml",
        "version_base": "1.3",
    }


def test_handle_other_layers_rejects_unknown_layer(main_module):
    with pytest.raises(ValueError):
        main_module.handle_other_layers("unknown-layer")


def test_handle_other_layers_rejects_parser_without_parse_known_args(main_module, monkeypatch):
    monkeypatch.setitem(main_module.layer_dict, "bad", (object(), lambda **kwargs: None))

    with pytest.raises(ValueError, match="parse_known_args"):
        main_module.handle_other_layers("bad")


def test_handle_other_layers_passes_parser_args_and_hydra_overrides(main_module, monkeypatch):
    seen = {}

    class FakeParser:
        def parse_known_args(self, argv):
            seen["argv_to_parser"] = list(argv)
            return Namespace(alpha="cli"), ["alpha=hydra"]

    def fake_main_fn(**kwargs):
        seen["kwargs"] = kwargs
        return "ok"

    def fake_hydra_main(**kwargs):
        seen["hydra_kwargs"] = kwargs

        def decorator(fn):
            def runner():
                return fn({"alpha": "hydra"})

            return runner

        return decorator

    monkeypatch.setitem(main_module.layer_dict, "layer", (FakeParser(), fake_main_fn))
    monkeypatch.setattr(main_module.hydra, "main", fake_hydra_main)
    monkeypatch.setattr(sys, "argv", ["deckard", "--alpha", "cli", "alpha=hydra"])

    result = main_module.handle_other_layers("layer")

    assert result == "ok"
    assert seen["argv_to_parser"] == ["--alpha", "cli", "alpha=hydra"]
    assert sys.argv == ["deckard", "alpha=hydra"]
    assert seen["kwargs"] == {"alpha": "hydra"}
    assert seen["hydra_kwargs"] == {
        "config_path": None,
        "config_name": None,
        "version_base": "1.3",
    }