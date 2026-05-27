import pandas as pd
import pytest

from deckard.plugins.anjana.data import AnjanaDataConfig
from deckard.plugins.anjana.score import DefaultAnjanaDataScorerDictConfig


def _fake_anjana_defense(data, **kwargs):
    _ = kwargs
    # Keep first 3 rows to simulate suppression/anonymization filtering.
    return data.iloc[:3].copy()


def test_anjana_data_defense_applies_callable_and_updates_xy(monkeypatch):
    cfg = AnjanaDataConfig(
        dataset_name="make_classification",
        data_params={"n_samples": 10, "n_features": 4, "n_informative": 2},
        classifier=True,
        sampler={
            "name": "deckard.data.sample.SplitSampler",
            "train_size": 6,
            "test_size": 4,
        },
        target="target",
        identifiers=["id_col"],
        quasi_identifiers=["f0", "f1"],
        sensitive_attribute="target",
        anjana_defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    cfg._X = pd.DataFrame(
        {
            "id_col": list(range(10)),
            "f0": list(range(10)),
            "f1": list(range(10, 20)),
            "f2": list(range(20, 30)),
            "f3": list(range(30, 40)),
        },
    )
    cfg._y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    monkeypatch.setattr(
        "deckard.plugins.anjana.data.resolve_class",
        lambda _: _fake_anjana_defense,
    )

    cfg._apply_anjana_defense()

    assert len(cfg._X) == 3
    assert len(cfg._y) == 3


def test_anjana_data_score_uses_auto_default(monkeypatch):
    cfg = AnjanaDataConfig(
        dataset_name="make_classification",
        data_params={"n_samples": 8, "n_features": 3, "n_informative": 2},
        classifier=True,
        sampler={
            "name": "deckard.data.sample.SplitSampler",
            "train_size": 5,
            "test_size": 3,
        },
        scorer=None,
    )
    cfg._X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cfg._y = pd.Series([0, 1])

    class _StubScorer:
        def __call__(self, **kwargs):
            _ = kwargs
            return {"k_anonymity": 2.0}

    monkeypatch.setattr(
        "deckard.plugins.anjana.data.load_class",
        lambda _: _StubScorer(),
    )
    cfg.scorer = "auto"
    cfg.X_train = pd.DataFrame({"a": [1], "b": [3]})
    cfg.y_train = pd.Series([0])
    cfg.X_test = pd.DataFrame({"a": [2], "b": [4]})
    cfg.y_test = pd.Series([1])

    out = cfg.score()
    assert out["k_anonymity"] == 2.0


def test_generate_anjana_hierarchy_dict_builds_interval_levels():
    cfg = AnjanaDataConfig(
        dataset_name="make_classification",
        data_params={"n_samples": 4, "n_features": 2, "n_informative": 2},
        classifier=True,
        sampler={
            "name": "deckard.data.sample.SplitSampler",
            "train_size": 2,
            "test_size": 2,
        },
        quasi_identifiers=["age", "zip"],
        hierarchy_interval_sizes={"age": [10, 20]},
    )
    frame = pd.DataFrame(
        {
            "age": [21, 27, 42, 49],
            "zip": [10001, 10002, 10003, 10004],
        },
    )

    hierarchies = cfg.generate_anjana_hierarchy_dict(frame=frame)

    assert set(hierarchies) == {"age", "zip"}
    assert list(hierarchies["age"]) == [0, 1, 2]
    assert list(hierarchies["age"][0]) == [21, 27, 42, 49]
    assert list(hierarchies["age"][1]) == [
        "[20, 30)",
        "[20, 30)",
        "[40, 50)",
        "[40, 50)",
    ]
    assert list(hierarchies["age"][2]) == [
        "[20, 40)",
        "[20, 40)",
        "[40, 60)",
        "[40, 60)",
    ]
    assert list(hierarchies["zip"]) == [0, 1]
    assert list(hierarchies["zip"][1]) == ["*", "*", "*", "*"]


def test_anjana_data_defense_auto_injects_generated_hierarchies(monkeypatch):
    cfg = AnjanaDataConfig(
        dataset_name="make_classification",
        data_params={"n_samples": 6, "n_features": 3, "n_informative": 2},
        classifier=True,
        sampler={
            "name": "deckard.data.sample.SplitSampler",
            "train_size": 3,
            "test_size": 3,
        },
        target="target",
        identifiers=["id_col"],
        quasi_identifiers=["age"],
        hierarchy_interval_sizes={"age": 10},
        anjana_defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    cfg._X = pd.DataFrame(
        {
            "id_col": list(range(6)),
            "age": [21, 27, 42, 49, 53, 61],
            "feature": [1, 2, 3, 4, 5, 6],
        },
    )
    cfg._y = pd.Series([0, 1, 0, 1, 0, 1])
    seen = {}

    def _capture_hierarchy_kwargs(**kwargs):
        seen.update(kwargs)
        return kwargs["data"].copy()

    monkeypatch.setattr(
        "deckard.plugins.anjana.data.resolve_class",
        lambda _: _capture_hierarchy_kwargs,
    )

    cfg._apply_anjana_defense()

    assert "hierarchies" in seen
    assert list(seen["hierarchies"]["age"][1]) == [
        "[20, 30)",
        "[20, 30)",
        "[40, 50)",
        "[40, 50)",
        "[50, 60)",
        "[60, 70)",
    ]


def test_default_anjana_data_scorer_includes_privacy_metrics():
    scorer = DefaultAnjanaDataScorerDictConfig()

    # DefaultAnjanaDataScoreConfig combines data metrics with privacy metrics
    privacy_metrics = {"k_anonymity", "l_diversity", "t_closeness"}
    assert privacy_metrics.issubset(set(scorer.scorers))
    # Verify privacy metrics are present alongside data metrics
    assert len(scorer.scorers) > 3
