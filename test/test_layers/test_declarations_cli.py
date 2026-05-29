from pathlib import Path

import pytest

from deckard.layers import declarations_cli as mod


@pytest.fixture
def declaration_root(tmp_path: Path) -> Path:
    root = tmp_path / "config"
    (root / "attack" / "evasion").mkdir(parents=True)
    (root / "defense" / "preprocessor").mkdir(parents=True)

    (root / "attack" / "evasion" / "fgm.yaml").write_text(
        "name: fgm\n_target_: attacks.FGMAttack\nepsilon: 0.1\n",
        encoding="utf-8",
    )
    (root / "defense" / "preprocessor" / "gaussian.yaml").write_text(
        "name: gaussian\nsigma: 0.3\n",
        encoding="utf-8",
    )
    return root


def test_list_components(monkeypatch, declaration_root, capsys):
    monkeypatch.setattr(mod.decl, "discover_config_roots", lambda: [declaration_root])

    summary = mod.declarations_main(command="list", format="json")
    out = capsys.readouterr().out

    assert summary["kind"] == "component"
    assert summary["items"] == ["attack", "defense"]
    assert '"kind": "component"' in out


def test_list_subcomponents(monkeypatch, declaration_root):
    monkeypatch.setattr(mod.decl, "discover_config_roots", lambda: [declaration_root])

    summary = mod.declarations_main(
        command="list",
        selector="attack",
        format="json",
    )

    assert summary["kind"] == "subcomponent"
    assert summary["items"] == ["evasion"]


def test_show_validate_and_compose(monkeypatch, declaration_root):
    monkeypatch.setattr(mod.decl, "discover_config_roots", lambda: [declaration_root])

    shown = mod.declarations_main(
        command="show",
        selector="attack/evasion/fgm",
        format="json",
    )
    assert shown["entry"]["name"] == "fgm"
    assert shown["entry"]["payload"]["epsilon"] == 0.1

    validated = mod.declarations_main(
        command="validate",
        selector="defense/preprocessor/gaussian",
        format="json",
    )
    assert validated["result"]["valid"] is True

    composed = mod.declarations_main(
        command="compose",
        selector="attack/evasion/fgm",
        set=["epsilon=0.25"],
        format="json",
    )
    assert composed["result"]["payload"]["epsilon"] == 0.25


def test_show_missing_selector_exits(monkeypatch, declaration_root):
    monkeypatch.setattr(mod.decl, "discover_config_roots", lambda: [declaration_root])

    with pytest.raises(SystemExit) as exc:
        mod.declarations_main(
            command="show",
            selector="attack/evasion/unknown",
            format="json",
        )

    assert exc.value.code == 2
