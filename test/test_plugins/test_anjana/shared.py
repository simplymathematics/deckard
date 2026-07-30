"""Shared support helpers for Anjana test suites."""

from __future__ import annotations

from typing import Any, cast

from helpers import load_canonical_data_profile

from deckard.experiment import ExperimentConfig
from deckard.file import FileConfig


def make_anjana_data(
    n: int = 12,
    monkeypatch: Any = None,
    defense: dict[str, Any] | None = None,
    resolve_class_fn: Any = None,
    **overrides: Any,
):
    """Build a minimal AnjanaDataConfig with optional stubbed runtime defense."""
    from deckard.plugins.anjana.data import AnjanaDataConfig

    base = load_canonical_data_profile("classification", framework="sklearn")
    base["data_params"].update(
        {
            "n_samples": n,
            "n_features": 4,
            "n_informative": 2,
            "n_redundant": 0,
            "random_state": 0,
        },
    )
    base.update(
        {
            "classifier": True,
            "sampler": {
                "name": "deckard.data.sample.SplitSampler",
                "train_size": 0.5,
                "test_size": 0.5,
                "random_state": 42,
            },
            "identifiers": None,
            "quasi_identifiers": ["feature_0", "feature_1"],
            "sensitive_attribute": "target",
            "anjana_defense": defense,
            "hierarchy_interval_sizes": {
                "feature_0": [1, 2],
                "feature_1": [1, 2],
            },
        },
    )
    if "sample" in base and "sampler" not in base:
        base["sampler"] = base.pop("sample")
    else:
        base.pop("sample", None)
    base.update(overrides)
    cfg = AnjanaDataConfig(**base)
    if monkeypatch is not None and defense is not None:

        def _fake_k_anon(data, **kwargs):
            _ = kwargs
            return data.copy()

        monkeypatch.setattr(
            "deckard.plugins.anjana.data.resolve_class",
            lambda _: resolve_class_fn or _fake_k_anon,
        )

    return cfg


def stub_drop_half_rows_resolver(monkeypatch: Any) -> None:
    """Patch the Anjana resolver to drop half the rows deterministically."""

    def _drop_half_rows(data, **kwargs):
        _ = kwargs
        return data.iloc[: len(data) // 2].copy()

    monkeypatch.setattr(
        "deckard.plugins.anjana.data.resolve_class",
        lambda _: _drop_half_rows,
    )


def make_art_postprocessor_defense(*, include_model_name: bool = True):
    """Return the canonical ART postprocessor defense pipeline used in tests."""
    from deckard.model import DefenseConfig

    defense = {
        "name": "art.defences.postprocessor.ClassLabels",
        "defense_params": {"apply_fit": False, "apply_predict": True},
        "classifier": True,
    }
    if include_model_name:
        defense["model_name"] = "sklearn.linear_model.LogisticRegression"
    return DefenseConfig(defenses=[defense])


def make_logistic_model(*, max_iter: int = 25, defense: Any = None):
    """Return the canonical logistic regression model config used in tests."""
    from deckard.model import ModelConfig

    return ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": max_iter},
        defense=defense,
    )


def make_hopskipjump_attack(*, attack_size: int = 3):
    """Return the canonical lightweight HopSkipJump attack config used in tests."""
    from deckard.attack import AttackConfig

    return AttackConfig(
        name="art.attacks.evasion.HopSkipJump",
        attack_size=attack_size,
        attack_params={
            "max_iter": 1,
            "init_eval": 1,
            "max_eval": 2,
            "init_size": 5,
            "norm": 2,
            "targeted": False,
        },
    )


def run_experiment(
    *,
    data: Any,
    model: Any,
    attack: Any = None,
    score: Any = None,
) -> ExperimentConfig:
    """Build and execute an ExperimentConfig for shared Anjana test chains."""

    experiment = ExperimentConfig(
        data=data,
        model=model,
        attack=attack,
        files=FileConfig(),
        score=score,
        classifier=True,
    )
    return experiment


def assert_anjana_privacy_scores(scores: dict[str, Any]) -> None:
    """Assert that Anjana privacy metrics are present in runtime or persisted payloads."""
    anjana_metrics = {"k_anonymity", "l_diversity", "t_closeness"}
    metric_source = cast(
        dict[str, Any],
        scores.get("payload") if isinstance(scores.get("payload"), dict) else scores,
    )

    post_pipeline = metric_source.get("post-pipeline")
    if isinstance(post_pipeline, dict):
        assert anjana_metrics.issubset(set(post_pipeline.keys())), (
            f"Expected privacy metrics under 'post-pipeline' key, "
            f"but found keys: {sorted(post_pipeline.keys())}"
        )
        return

    for value in metric_source.values():
        if isinstance(value, dict) and anjana_metrics.issubset(set(value.keys())):
            return

    flat_source = scores.get("flat") if isinstance(scores.get("flat"), dict) else None
    if isinstance(flat_source, dict):
        flattened_keys = {key.rsplit(".", 1)[-1] for key in flat_source.keys()}
        assert anjana_metrics.issubset(flattened_keys), (
            "Expected Anjana-specific privacy metrics in flattened scores "
            "(k_anonymity, l_diversity, t_closeness), "
            f"but found keys: {sorted(flat_source.keys())}"
        )
        return

    flattened_keys = {key for key in metric_source if key in anjana_metrics}
    assert anjana_metrics.issubset(flattened_keys), (
        "Expected Anjana-specific privacy metrics in scores (k_anonymity, "
        "l_diversity, t_closeness), "
        f"but found keys: {sorted(metric_source.keys())}"
    )


def assert_wrapper_reordered_last(
    caplog: Any,
    *,
    warning_substring: str,
    data_defense_name: str,
) -> None:
    """Assert wrapper defenses are reordered to run after data defenses."""
    import logging

    from deckard.model.defense.base import DefenseConfig

    call_order: list[str] = []

    class _StubDataDefense:
        name = data_defense_name
        defense_application_time = 0.0

        def apply_to(self, estimator, data):
            _ = data
            call_order.append("data")
            return estimator

    class _StubArtDefense:
        name = "art.mock.MockArtDefense"
        defense_application_time = 0.0

        def apply_to(self, estimator, data):
            _ = data
            call_order.append("art")
            return estimator

    pipeline = DefenseConfig(defenses=[_StubArtDefense(), _StubDataDefense()])

    with caplog.at_level(logging.WARNING, logger="deckard.model.defense.base"):
        estimator = cast(Any, object())
        data = cast(Any, object())
        pipeline.apply(estimator=estimator, data=data)

    assert call_order == ["data", "art"]
    assert any(warning_substring in rec.message for rec in caplog.records)
