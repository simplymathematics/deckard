"""Consolidated compose tests for attack configs.

This test suite validates that attack configuration profiles are correctly
included in default composed configurations.
"""

from .shared_compose import compose_pytorch, compose_sklearn


def test_sklearn_default_includes_attack_config():
    """Test that sklearn default config includes attack configuration."""
    cfg = compose_sklearn("default")
    assert cfg is not None
    assert "attack" in cfg
    # Default attack is hsj
    assert cfg.attack.alias == "hsj"


def test_pytorch_torch_default_includes_attack_config():
    """Test that pytorch torch_default config includes attack configuration."""
    cfg = compose_pytorch("torch_default")
    assert cfg is not None
    assert "attack" in cfg
    # Default attack is fgm
    assert cfg.attack.alias == "fgm"


def test_pytorch_attack_group_override_composes():
    """Test that attack=<attack> overrides compose through the public attack group."""
    cfg = compose_pytorch("torch_default", overrides=["attack=fgm"])

    assert cfg is not None
    assert "attack" in cfg
    assert cfg.attack.alias == "fgm"


def test_pytorch_model_defense_group_override_composes():
    """Test that ++defense@model.defense=<defense> composes through the defense group."""
    cfg = compose_pytorch(
        "torch_default",
        overrides=["model=default", "+defense@model.defense=class_labels"],
    )

    assert cfg is not None
    assert "model" in cfg
    assert "defense" in cfg.model
    assert cfg.model.defense.name == "art.defences.postprocessor.ClassLabels"
