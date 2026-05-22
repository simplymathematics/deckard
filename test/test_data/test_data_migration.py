from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from deckard.plugins.base import compose_hook_plugins
from deckard.plugins.anjana.data import (
    ANJANA_PIPELINE_HOOKS,
    ANJANA_SCORING_HOOKS,
)
from deckard.plugins.fairlearn.data import (
    FAIRLEARN_PIPELINE_HOOKS,
    FAIRLEARN_SCORING_HOOKS,
)


def _iter_data_config_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return sorted((repo_root / "examples").glob("*/config/data/*.yaml"))


def _flatten_scalars(node: Any):
    if isinstance(node, dict):
        for value in node.values():
            yield from _flatten_scalars(value)
        return
    if isinstance(node, list):
        for value in node:
            yield from _flatten_scalars(value)
        return
    yield node


def test_pipeline_and_scoring_hook_bundles_are_separated_and_composable():
    assert all(
        not hook.hook_name.startswith("after_score")
        for hook in FAIRLEARN_PIPELINE_HOOKS.hooks
    )
    assert all(
        hook.hook_name.startswith("after_score")
        for hook in FAIRLEARN_SCORING_HOOKS.hooks
    )
    assert all(
        not hook.hook_name.startswith("after_score")
        for hook in ANJANA_PIPELINE_HOOKS.hooks
    )
    assert all(
        hook.hook_name.startswith("after_score")
        for hook in ANJANA_SCORING_HOOKS.hooks
    )

    fairlearn_order = [
        plugin.hook_name
        for plugin in compose_hook_plugins(
            FAIRLEARN_PIPELINE_HOOKS,
            FAIRLEARN_SCORING_HOOKS,
        )
    ]
    assert fairlearn_order == ["before_sample", "after_pipeline", "after_score"]

    anjana_order = [
        plugin.hook_name
        for plugin in compose_hook_plugins(
            ANJANA_PIPELINE_HOOKS,
            ANJANA_SCORING_HOOKS,
        )
    ]
    assert anjana_order == ["before_sample", "after_score_post_pipeline"]


def test_examples_data_configs_have_no_legacy_persistence_keys():
    forbidden_keys = {
        "data_file",
        "score_file",
        "post_sample_data_file",
        "post_pipeline_data_file",
    }
    offenders: list[str] = []

    for yaml_path in _iter_data_config_files():
        cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
        if not isinstance(cfg, dict):
            continue
        bad = sorted(forbidden_keys.intersection(set(cfg.keys())))
        if bad:
            offenders.append(f"{yaml_path}: {bad}")

    assert not offenders, "\n".join(offenders)


def test_examples_data_configs_have_no_legacy_pytorch_targets():
    offenders: list[str] = []

    for yaml_path in _iter_data_config_files():
        cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
        for scalar in _flatten_scalars(cfg):
            if isinstance(scalar, str) and "deckard.data.pytorch" in scalar:
                offenders.append(f"{yaml_path}: {scalar}")

    assert not offenders, "\n".join(offenders)


def test_examples_plugin_data_targets_use_top_level_plugin_apis():
    offenders: list[str] = []

    for yaml_path in _iter_data_config_files():
        cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
        if not isinstance(cfg, dict):
            continue
        target = cfg.get("_target_")
        if not isinstance(target, str):
            continue
        if target.startswith("deckard.plugins.") and ".data." in target:
            offenders.append(f"{yaml_path}: {target}")

    assert not offenders, "\n".join(offenders)


def test_examples_score_mode_values_use_split_scope_only():
    allowed = {"train", "test", "val", "all"}
    offenders: list[str] = []

    for yaml_path in _iter_data_config_files():
        cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
        if not isinstance(cfg, dict):
            continue
        score_mode = cfg.get("score_mode")
        if score_mode is None:
            continue
        token = str(score_mode).strip().lower()
        if token not in allowed:
            offenders.append(f"{yaml_path}: score_mode={score_mode}")

    assert not offenders, "\n".join(offenders)
