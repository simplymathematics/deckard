from types import SimpleNamespace

import pytest

import deckard.experiment.dvc as dvc_module
from deckard.score.dvc import DVCSystemScorerDictConfig, dvc_component_stats_score


def test_dvc_system_scorer_defaults_to_component_score_stages():
    cfg = DVCSystemScorerDictConfig()
    assert set(cfg.scorers.keys()) == {"data", "model", "attack", "defense"}
    assert cfg.scorers["data"].stage == ["data-score"]
    assert cfg.scorers["model"].stage == ["model-score"]
    assert cfg.scorers["attack"].stage == ["attack-score"]
    assert cfg.scorers["defense"].stage == ["detector-score"]


def test_dvc_component_stats_score_returns_available_default_when_disabled():
    exp = SimpleNamespace(dvc_plugin={"enabled": False})
    scores = dvc_component_stats_score(
        [0.0],
        [0.0],
        experiment=exp,
        stage="data-score",
    )

    assert scores["available"] == 0.0


def test_dvc_component_stats_score_includes_system_monitor_metrics(monkeypatch):
    exp = SimpleNamespace(dvc_plugin={"enabled": True})

    monkeypatch.setattr(
        dvc_module,
        "_collect_system_monitor_scores",
        lambda experiment, plugin: {"system_monitor/cpu": 12.5, "system_monitor/ram": 33.0},
    )

    scores = dvc_component_stats_score(
        [0.0],
        [0.0],
        experiment=exp,
        stage="model-score",
    )

    assert scores["cpu"] == pytest.approx(12.5)
    assert scores["memory"] == pytest.approx(33.0)


def test_dvc_component_stats_score_includes_power_namespace_metrics():
    exp = SimpleNamespace(
        dvc_plugin={"enabled": True},
        score_dict={
            "power/detector/gpu_watts": 21.0,
            "power/detector/cpu_watts": 7.0,
        },
    )

    scores = dvc_component_stats_score(
        [0.0],
        [0.0],
        experiment=exp,
        component="defense",
        stage="detector-score",
    )

    assert scores["gpu_power"] == pytest.approx(21.0)
    assert scores["cpu_power"] == pytest.approx(7.0)
