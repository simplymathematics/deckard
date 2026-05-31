import pytest

from deckard.layers import extensions_cli as mod


def test_plugins_list_mode_returns_items_without_install(monkeypatch, capsys):
    monkeypatch.setattr(mod, "_is_entry_installed", lambda entry: True)

    summary = mod.plugins_main(list=True, format="json")
    out = capsys.readouterr().out

    assert summary["mode"] == "list"
    assert len(summary["items"]) == len(mod.PluginRegistry)
    assert '"kind": "plugins"' in out


def test_frameworks_list_mode_and_human_output(monkeypatch, capsys):
    monkeypatch.setattr(mod, "_is_entry_installed", lambda entry: False)

    summary = mod.frameworks_main(list=True, format="human")
    out = capsys.readouterr().out

    assert summary["mode"] == "list"
    assert summary["kind"] == "frameworks"
    assert "kind: frameworks" in out


@pytest.mark.parametrize(
    "fn,arg",
    [
        (mod.plugins_main, {"plugin": "does_not_exist"}),
        (mod.frameworks_main, {"framework": "does_not_exist"}),
    ],
)
def test_unknown_name_hard_fails(fn, arg):
    with pytest.raises(SystemExit) as exc:
        fn(**arg)
    # CLI contract: unknown names fail with exit code 2.
    assert exc.value.code == 2


def test_plugins_anjana_no_install_path(monkeypatch):
    monkeypatch.setattr(mod, "_is_entry_installed", lambda entry: False)

    summary = mod.plugins_main(plugin="anjana", list=False, format="json")
    assert summary["items"][0]["action"] == "noop"
    assert summary["items"][0]["status"] == "optional-dependency-missing"


def test_plugins_non_anjana_is_noop(monkeypatch):
    monkeypatch.setattr(mod, "_is_entry_installed", lambda entry: True)

    summary = mod.plugins_main(plugin="fairlearn", list=False, format="json")
    assert summary["items"][0]["action"] == "noop"
    assert summary["items"][0]["status"] == "available-noop"


def test_anjana_registry_requires_anjana_and_pycanon():
    assert mod.PluginRegistry["anjana"]["required_imports"] == ["anjana", "pycanon"]


def test_registry_extras_come_from_pyproject_groups_except_anjana():
    # anjana is explicitly not mapped to an extra group.
    assert mod.PluginRegistry["anjana"]["extra"] is None

    # plugin extras map to same-name optional-dependency groups when available.
    assert mod.PluginRegistry["fairlearn"]["extra"] == "fairlearn"
    assert mod.PluginRegistry["lifelines"]["extra"] == "lifelines"
    assert mod.PluginRegistry["seaborn"]["extra"] == "seaborn"
    assert mod.PluginRegistry["yellowbrick"]["extra"] == "yellowbrick"

    # framework extras use alias mapping where needed (pytorch -> torch).
    assert mod.FrameworkRegistry["pytorch"]["extra"] == "torch"
    assert mod.FrameworkRegistry["sklearn"]["extra"] is None
    assert mod.FrameworkRegistry["transformers"]["extra"] == [
        "datasets",
        "openattack",
        "textattack",
    ]
