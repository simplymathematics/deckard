import importlib
import sys
from argparse import Namespace
from pathlib import Path
import pytest


@pytest.fixture
def main_module():
    import deckard.__main__ as mod

    return importlib.reload(mod)


def test_get_configuration_paths_returns_expected_values(
    main_module,
    monkeypatch,
    tmp_path,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("x: 1\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DECKARD_CONFIG_DIR", "config")
    monkeypatch.delenv("DECKARD_DEFAULT_CONFIG_FILE", raising=False)

    resolved_dir, config_file = main_module.get_configuration_paths()

    assert resolved_dir == "config"
    assert config_file == "default.yaml"


def test_get_configuration_paths_prompts_for_valid_directory(
    main_module,
    monkeypatch,
    tmp_path,
):
    config_dir = tmp_path / "real_config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("x: 1\n")

    monkeypatch.setenv("DECKARD_CONFIG_DIR", "missing")
    monkeypatch.setattr("builtins.input", lambda _: str(config_dir))

    resolved_dir, config_file = main_module.get_configuration_paths()

    assert resolved_dir == str(config_dir)
    assert config_file == "default.yaml"


def test_get_configuration_paths_raises_for_missing_config_file(
    main_module,
    monkeypatch,
    tmp_path,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    monkeypatch.setenv("DECKARD_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("DECKARD_DEFAULT_CONFIG_FILE", "missing.yaml")

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        main_module.get_configuration_paths()


def test_main_dispatches_to_default_module(main_module, monkeypatch, tmp_path):
    monkeypatch.setenv("DECKARD_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["deckard"])

    with pytest.raises(SystemExit):
        main_module.main()


def test_main_dispatches_to_supported_layer(main_module, monkeypatch, tmp_path):
    layer = next(
        layer
        for layer in main_module.SUPPORTED_LAYERS
        if layer not in {"experiment", "optimize"}
    )

    monkeypatch.setenv("DECKARD_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["deckard", layer])

    seen = {}

    monkeypatch.setattr(
        main_module,
        "generate_hydra_main",
        lambda layer_name, argv=None: seen.setdefault("layer", layer_name),
    )

    main_module.main()

    assert seen["layer"] == layer


def test_main_raises_for_unsupported_module(main_module, monkeypatch, tmp_path):
    monkeypatch.setenv("DECKARD_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["deckard", "not-supported"])

    with pytest.raises(SystemExit):
        main_module.main()


def test_build_router_includes_supported_layers(main_module):
    parser = main_module._build_router()
    subparser_action = next(
        action
        for action in parser._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict)
    )
    subcommands = set(subparser_action.choices.keys())

    assert set(main_module.SUPPORTED_LAYERS).issubset(subcommands)



def test_generate_hydra_main_rejects_unknown_layer(main_module):
    with pytest.raises(ValueError):
        main_module.generate_hydra_main("unknown-layer")


def test_generate_hydra_main_rejects_parser_without_parse_known_args(
    main_module,
    monkeypatch,
):
    monkeypatch.setitem(
        main_module.layer_dict,
        "bad",
        (object(), lambda **kwargs: None),
    )

    with pytest.raises(ValueError, match="parse_known_args"):
        main_module.generate_hydra_main("bad")


def test_generate_hydra_main_passes_parser_args_and_hydra_overrides(
    main_module,
    monkeypatch,
):
    seen = {}

    class FakeParser:
        def parse_known_args(self, argv):
            seen["argv_to_parser"] = list(argv)
            return Namespace(alpha="cli", overrides=["alpha=hydra"]), []

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
    monkeypatch.setattr(
        main_module,
        "get_configuration_paths",
        lambda: (None, None),
    )
    monkeypatch.setattr(sys, "argv", ["deckard", "--alpha", "cli", "alpha=hydra"])

    result = main_module.generate_hydra_main("layer")

    assert result == "ok"
    assert seen["argv_to_parser"] == ["--alpha", "cli", "alpha=hydra"]
    assert sys.argv == ["deckard", "alpha=hydra"]
    assert seen["kwargs"] == {"alpha": "hydra"}
    assert seen["hydra_kwargs"] == {
        "config_path": None,
        "config_name": None,
        "version_base": "1.3",
    }


def test_generate_hydra_main_forwards_hydra_multirun_flag(
    main_module,
    monkeypatch,
):
    seen = {}

    class FakeParser:
        def parse_known_args(self, argv):
            seen["argv_to_parser"] = list(argv)
            return (
                Namespace(
                    alpha="cli",
                    overrides=["alpha=hydra"],
                    run=False,
                    multirun=True,
                    shell_completion=False,
                    hydra_help=False,
                    help=False,
                    resolve=False,
                    cfg=None,
                    package=None,
                    info=None,
                    experimental_rerun=None,
                    config_path=None,
                    config_name=None,
                    config_dir=None,
                ),
                [],
            )

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
    monkeypatch.setattr(
        main_module,
        "get_configuration_paths",
        lambda: (None, None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["deckard", "--alpha", "cli", "--multirun", "alpha=hydra"],
    )

    result = main_module.generate_hydra_main("layer")

    assert result == "ok"
    assert seen["argv_to_parser"] == ["--alpha", "cli", "--multirun", "alpha=hydra"]
    assert sys.argv == ["deckard", "--multirun", "alpha=hydra"]
    assert seen["kwargs"] == {"alpha": "hydra"}
    assert seen["hydra_kwargs"] == {
        "config_path": None,
        "config_name": None,
        "version_base": "1.3",
    }
