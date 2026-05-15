from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from deckard.score.base import (
    DefaultModelScorerConfig,
    DefaultRegressorConfig,
    ScorerConfig,
    ScorerDictConfig,
    build_scorer,
    build_scorer_dict,
)
from deckard.score.attack import DefaultEvasionAttackScorerConfig
from deckard.score.data import DefaultDataScorerConfig


def test_scorer_config_post_init_dict_and_string_paths(monkeypatch):
    def _loaded_metric(target, *args, **kwargs):
        def _metric(y_true, y_pred, **call_kwargs):
            _ = (y_true, y_pred, call_kwargs)
            return (target, args, kwargs)

        return _metric

    monkeypatch.setattr("deckard.score.base.load_class", _loaded_metric)
    monkeypatch.setattr(
        "deckard.score.base.resolve_class",
        lambda path: (lambda y_true, y_pred, **kwargs: 1.0),
    )

    cfg = ScorerConfig(
        score_name="custom",
        score_function=OmegaConf.create(
            {"_target_": "pkg.metric", "_args_": 3, "flag": True},
        ),
        score_params=None,
    )
    assert cfg.score_function([0], [0])[0] == "pkg.metric"
    assert cfg.score_function([0], [0])[1] == (3,)
    assert cfg.score_params == {}

    cfg_str = ScorerConfig(
        score_name="acc",
        score_function="sklearn.metrics.accuracy_score",
    )
    assert callable(cfg_str.score_function)

    with pytest.raises(ValueError, match="must include '_target_' or 'name'"):
        ScorerConfig(score_name="bad", score_function={"x": 1})

    with pytest.raises(TypeError, match="must be callable"):
        ScorerConfig(score_name="bad", score_function=123)


def test_probability_validation_error_paths():
    scorer = ScorerConfig(
        score_name="roc_auc",
        score_function="sklearn.metrics.roc_auc_score",
        needs_proba=True,
    )

    with pytest.raises(ValueError, match="1D/2D probability input"):
        scorer._validate_probability_input([0, 1], np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="matching sample counts"):
        scorer._validate_probability_input([0, 1], np.array([[0.1], [0.2], [0.3]]))
    with pytest.raises(ValueError, match="numeric probabilities"):
        scorer._validate_probability_input(
            [0, 1],
            np.array([["a"], ["b"]], dtype=object),
        )
    with pytest.raises(ValueError, match=r"values in \[0, 1\]"):
        scorer._validate_probability_input([0, 1], np.array([[1.2], [0.2]]))


def test_normalize_predictions_branches_for_probability_and_labels():
    roc = ScorerConfig(
        score_name="roc_auc",
        score_function="sklearn.metrics.roc_auc_score",
        needs_proba=True,
    )
    assert np.array_equal(
        roc._normalize_predictions_for_metric([0, 1], np.array([[0.2], [0.9]])),
        np.array([0.2, 0.9]),
    )
    assert np.array_equal(
        roc._normalize_predictions_for_metric(
            [0, 1],
            np.array([[0.8, 0.2], [0.1, 0.9]]),
        ),
        np.array([0.2, 0.9]),
    )

    acc = ScorerConfig(
        score_name="accuracy",
        score_function="sklearn.metrics.accuracy_score",
    )
    assert np.array_equal(
        acc._normalize_predictions_for_metric([0, 1], np.array([[0.2], [0.9]])),
        np.array([0, 1]),
    )
    assert np.array_equal(
        acc._normalize_predictions_for_metric([0, 1], np.array([[-2.0], [3.0]])),
        np.array([0, 1]),
    )
    assert np.array_equal(
        acc._normalize_predictions_for_metric(
            [0, 1],
            np.array([[0.8, 0.2], [0.1, 0.9]]),
        ),
        np.array([0, 1]),
    )
    non_numeric = np.array([["a"], ["b"]], dtype=object)
    assert acc._normalize_predictions_for_metric([0, 1], non_numeric) is non_numeric


def test_scorer_call_filters_kwargs_without_var_kwargs():
    def metric(y_true, y_pred, sample_weight=None):
        assert sample_weight == [1, 1]
        return 0.5

    cfg = ScorerConfig(score_name="m", score_function=metric)
    assert cfg([0, 1], [0, 1], sample_weight=[1, 1], ignored=3) == 0.5


def test_scorer_dict_init_iter_getitem_and_builders():
    sd = ScorerDictConfig(
        scorers={
            "acc": {"score_function": "sklearn.metrics.accuracy_score"},
        },
    )
    assert list(iter(sd))[0][0] == "acc"
    assert sd["acc"].score_name == "acc"
    assert build_scorer(sd["acc"]) is sd["acc"]
    assert isinstance(
        build_scorer(
            {"score_name": "acc", "score_function": "sklearn.metrics.accuracy_score"},
        ),
        ScorerConfig,
    )
    assert build_scorer_dict(sd) is sd
    assert isinstance(build_scorer_dict({"scorers": {}}), ScorerDictConfig)

    with pytest.raises(TypeError, match="must be ScorerConfig or dict"):
        ScorerDictConfig(scorers={"bad": 7})


def test_task_aware_model_scorer_normalizes_explicit_classifier_aliases():
    reg = DefaultModelScorerConfig(classifier="regressor")
    assert reg.classifier is False
    assert set(reg.scorers) == {"mse", "mae", "r2"}

    wrapped = DefaultRegressorConfig()
    assert wrapped.classifier is False
    assert set(wrapped.scorers) == {"mse", "mae", "r2"}


def test_task_aware_scorer_resolves_from_model_data_and_attack_context():
    custom = {
        "custom": ScorerConfig(
            score_name="custom",
            score_function="sklearn.metrics.accuracy_score",
        ),
    }

    model_cfg = DefaultModelScorerConfig(scorers=custom, classifier=None)
    assert (
        model_cfg.resolve_classifier(model=SimpleNamespace(classifier=False)) is False
    )

    data_cfg = DefaultDataScorerConfig(scorers=custom, classifier=None)
    assert data_cfg.resolve_classifier(data=SimpleNamespace(classifier=False)) is False

    attack_cfg = DefaultEvasionAttackScorerConfig(scorers=custom, classifier=None)
    assert (
        attack_cfg.resolve_classifier(attack=SimpleNamespace(_is_continuous=True))
        is False
    )


def testresolve_mode_features_and_predict_proba_paths():
    data = SimpleNamespace(X_train="train", X_test="test", X_val="val", _X="full")
    assert ScorerDictConfig.resolve_mode_features("train", data) == "train"
    assert ScorerDictConfig.resolve_mode_features("test", data) == "test"
    assert ScorerDictConfig.resolve_mode_features("val", data) == "val"
    assert ScorerDictConfig.resolve_mode_features("attack-val", data) == "val"
    assert ScorerDictConfig.resolve_mode_features("pre-sample", data) == "full"
    assert ScorerDictConfig.resolve_mode_features("attack", data) is None
    assert ScorerDictConfig.resolve_mode_features("test", None) is None

    import pytest

    with pytest.raises(
        ValueError,
        match="Cannot compute probabilities: model or input X is None",
    ):
        ScorerDictConfig.predict_proba_from_model(None, [1])
    with pytest.raises(
        ValueError,
        match="Cannot compute probabilities: model or input X is None",
    ):
        ScorerDictConfig.predict_proba_from_model(object(), None)

    class Estimator:
        def predict_proba(self, x):
            if not isinstance(x, np.ndarray):
                raise TypeError("need ndarray")
            return np.array([[0.2, 0.8]])

    class ModelWithBrokenGetter:
        def get_model(self):
            raise RuntimeError("boom")

        _model = Estimator()

    import pytest

    with pytest.raises(TypeError, match="need ndarray"):
        ScorerDictConfig.predict_proba_from_model(ModelWithBrokenGetter(), [[1.0]])

    with pytest.raises(
        ValueError,
        match="Model must have a predict or predict_proba function for probability metrics.",
    ):
        ScorerDictConfig.predict_proba_from_model(
            SimpleNamespace(_model=object()),
            [[1.0]],
        )


def test_scorer_dict_call_mode_and_probability_routing(tmp_path, monkeypatch):
    score_file = tmp_path / "scores.csv"
    score_file.write_text("dummy")

    acc = ScorerConfig(
        score_name="accuracy",
        score_function="sklearn.metrics.accuracy_score",
    )
    roc = ScorerConfig(
        score_name="roc_auc",
        score_function="sklearn.metrics.roc_auc_score",
        needs_proba=True,
    )
    scorer_dict = ScorerDictConfig(scorers={"accuracy": acc, "roc_auc": roc})

    loaded = {"accuracy": 0.9}
    saved = {}
    monkeypatch.setattr(scorer_dict, "load_scores", lambda path: dict(loaded))
    monkeypatch.setattr(
        scorer_dict,
        "save_scores",
        lambda scores, path: saved.update(scores),
    )

    data = SimpleNamespace(
        y_test=np.array([0, 1]),
        y_train=np.array([1, 0]),
        y_val=np.array([1, 1]),
        X_test=np.array([[1.0], [2.0]]),
        X_train=np.array([[3.0], [4.0]]),
        X_val=np.array([[5.0], [6.0]]),
    )

    class Estimator:
        def predict_proba(self, x):
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    model = SimpleNamespace(
        predictions=np.array([0, 1]),
        test_predictions=None,
        training_predictions=np.array([1, 0]),
        val_predictions=np.array([1, 1]),
        predict_proba=lambda x: np.array([[0.8, 0.2], [0.1, 0.9]]),
        _model=Estimator(),
        get_model=lambda: Estimator(),
    )
    attack = SimpleNamespace(
        attack_size=1,
        attack_predictions=np.array([0]),
        _attack="atk",
    )
    attack_val = SimpleNamespace(
        attack_size=1,
        attack_predictions=np.array([0, 1]),
        _attack="atk",
    )

    result_test = scorer_dict(
        mode="test",
        data=data,
        model=model,
        attack=None,
        score_file=score_file.as_posix(),
    )
    assert result_test["accuracy"] == 0.9
    assert "roc_auc" in result_test
    assert saved

    result_train = scorer_dict(
        mode="train",
        data=data,
        model=model,
        y_proba=np.array([0.2, 0.8]),
    )
    assert "training_accuracy" in result_train
    assert "training_roc_auc" in result_train

    result_attack = scorer_dict(
        mode="attack",
        data=data,
        model=model,
        attack=attack,
        y_proba=np.array([0.7]),
    )
    assert "attack_accuracy" in result_attack
    assert "attack_roc_auc" in result_attack

    result_val = scorer_dict(
        mode="val",
        data=data,
        model=model,
        y_proba=np.array([0.6, 0.6]),
    )
    assert "accuracy" in result_val
    result_attack_val = scorer_dict(
        mode="attack-val",
        data=data,
        model=model,
        attack=attack_val,
    )
    assert "accuracy" in result_attack_val

    with pytest.raises(AssertionError, match="y_true must also be provided"):
        scorer_dict(y_pred=np.array([0, 1]))

    with pytest.raises(
        AssertionError,
        match="y_true must be provided if mode is None",
    ):
        scorer_dict(mode=None)


def test_scorer_dict_attack_placeholder_and_missing_probability_context():
    scorer = ScorerDictConfig(
        scorers={
            "roc_auc": ScorerConfig(
                score_name="roc_auc",
                score_function="sklearn.metrics.roc_auc_score",
                needs_proba=True,
            ),
        },
    )
    SimpleNamespace(_attack="resolved")

    with pytest.raises(ValueError, match="requires probabilities from predict_proba"):
        scorer(
            mode="attack",
            data=SimpleNamespace(y_test=np.array([0, 1])),
            model=None,
            attack=SimpleNamespace(
                attack_size=1,
                attack_predictions=np.array([0]),
                _attack="resolved",
            ),
            y_true=np.array([0]),
            y_pred=np.array([0]),
            extra="{attack}",
        )


def test_scorer_dict_attack_mode_prefers_attack_attacked_labels():
    scorer_dict = ScorerDictConfig(
        scorers={
            "accuracy": ScorerConfig(
                score_name="accuracy",
                score_function="sklearn.metrics.accuracy_score",
            ),
        },
    )

    data = SimpleNamespace(y_test=np.array([1, 1, 1]))
    attack = SimpleNamespace(
        attack_size=2,
        attack_predictions=np.array([0, 1]),
        attacked_labels=np.array([0, 1]),
        _attack="atk",
    )

    result = scorer_dict(mode="attack", data=data, attack=attack)

    assert result["attack_accuracy"] == 1.0


def test_scorer_dict_pre_sample_mode_uses_full_dataset_vectors():
    scorer_dict = ScorerDictConfig(
        scorers={
            "n": ScorerConfig(
                score_name="n",
                score_function=lambda y_true, y_pred: len(y_true),
            ),
        },
    )

    data = SimpleNamespace(
        _y=np.array([0, 1, 2, 3]),
        _X=np.array([[0.0], [1.0], [2.0], [3.0]]),
    )

    result = scorer_dict(mode="pre-sample", data=data)

    assert result["n"] == 4


def test_scorer_dict_pre_sample_rejects_probability_metrics():
    scorer_dict = ScorerDictConfig(
        scorers={
            "roc_auc": ScorerConfig(
                score_name="roc_auc",
                score_function="sklearn.metrics.roc_auc_score",
                needs_proba=True,
            ),
        },
    )

    data = SimpleNamespace(
        _y=np.array([0, 1, 0, 1]),
        _X=np.array([[0.0], [1.0], [2.0], [3.0]]),
    )

    with pytest.raises(ValueError, match="reserved for full-dataset diagnostics"):
        scorer_dict(mode="pre-sample", data=data)



import numpy as np
import pandas as pd
import pytest

import deckard.score.data as score_data








def test_score_data_coerce_features_dataframe_series_and_vector():
    s = pd.Series([1.0, 2.0], name="named")
    out_series = score_data._coerce_features_dataframe(s)
    assert list(out_series.columns) == ["named"]

    out_vector = score_data._coerce_features_dataframe(np.array([1.0, 2.0]))
    assert list(out_vector.columns) == ["feature_0"]


def test_score_data_reference_resolution_reference_and_missing_column_error():
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    ref, out_X = score_data._resolve_reference_vector([0, 1], X, reference=[9, 8])
    assert np.array_equal(ref, np.array([9, 8]))
    assert list(out_X.columns) == ["a", "b"]

    with pytest.raises(ValueError, match="reference_column 'nope' not found"):
        score_data._resolve_reference_vector([0, 1], X, reference_column="nope")


def test_score_data_is_discrete_reference_empty_and_object():
    assert score_data._is_discrete_reference(np.array([])) is True
    assert (
        score_data._is_discrete_reference(np.array(["x", "y"], dtype=object)) is True
    )


def test_score_data_mutual_information_raises_when_no_features_left():
    y = np.array([0, 1, 0, 1])
    X = pd.DataFrame({"label": y})

    with pytest.raises(ValueError, match="No feature columns available"):
        score_data._feature_mutual_information_vector(
            y_true=y,
            X=X,
            reference_column="label",
        )


def test_score_data_class_imbalance_ratio_empty_and_zero_min_count(monkeypatch):
    assert score_data.data_class_imbalance_ratio_score([], None) == 0.0

    original = score_data.pd.Series.value_counts

    def _fake_value_counts(self, dropna=False):
        _ = dropna
        return pd.Series([3.0, 0.0])

    monkeypatch.setattr(score_data.pd.Series, "value_counts", _fake_value_counts)
    try:
        assert score_data.data_class_imbalance_ratio_score([0, 1], None) == float(
            "inf",
        )
    finally:
        monkeypatch.setattr(score_data.pd.Series, "value_counts", original)


def test_score_data_empirical_cdf_empty_reference_raises():
    y = pd.Series([np.nan, np.nan])
    X = pd.DataFrame({"x": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Reference vector is empty"):
        score_data.data_empirical_cdf_function_score(
            y_true=y,
            X=X,
            reference=[np.nan, np.nan],
        )
