from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import deckard.score as score_mod
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score

from deckard.score import (
    AttackScorerConfig,
    DefaultDataRegressionScorerDictConfig,
    ScorerConfig,
    ScorerDictConfig,
    survival_aic_score,
    survival_bic_score,
    survival_concordance_score,
)
from deckard.score.base import (
    DefaultModelScorerDictConfig,
    _DataScorerMarker,
    coerce_scorer_config,
)
from deckard.score.data import DefaultDataScorerDictConfig
import pytest


@pytest.mark.parametrize(
    ("module_name", "available"),
    [("fairlearn", True), ("missing_optional_dep", False)],
)
def test_optional_score_dependency_detection(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    available: bool,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        score_mod.importlib.util,
        "find_spec",
        lambda name: sentinel if name == module_name and available else None,
    )

    assert score_mod._is_available(module_name) is available


@pytest.mark.parametrize(
    ("wrapper_name", "loader_name", "symbol_name"),
    [
        (
            "fairness_demographic_parity_difference",
            "_load_fairlearn_score_symbol",
            "fairness_demographic_parity_difference",
        ),
        (
            "fairness_equalized_odds_difference",
            "_load_fairlearn_score_symbol",
            "fairness_equalized_odds_difference",
        ),
        (
            "fairness_group_mean_prediction_difference",
            "_load_fairlearn_score_symbol",
            "fairness_group_mean_prediction_difference",
        ),
        (
            "fairness_group_mae_difference",
            "_load_fairlearn_score_symbol",
            "fairness_group_mae_difference",
        ),
        (
            "fairness_group_mse_difference",
            "_load_fairlearn_score_symbol",
            "fairness_group_mse_difference",
        ),
        (
            "anjana_k_anonymity_score",
            "_load_anjana_score_symbol",
            "anjana_k_anonymity_score",
        ),
        (
            "anjana_l_diversity_score",
            "_load_anjana_score_symbol",
            "anjana_l_diversity_score",
        ),
        (
            "anjana_t_closeness_score",
            "_load_anjana_score_symbol",
            "anjana_t_closeness_score",
        ),
        (
            "survival_concordance_score",
            "_load_lifelines_score_symbol",
            "survival_concordance_score",
        ),
        (
            "survival_aic_score",
            "_load_lifelines_score_symbol",
            "survival_aic_score",
        ),
        (
            "survival_bic_score",
            "_load_lifelines_score_symbol",
            "survival_bic_score",
        ),
    ],
)
def test_score_wrapper_functions_delegate_to_lazy_loaders(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_name: str,
    loader_name: str,
    symbol_name: str,
) -> None:
    monkeypatch.setattr(
        score_mod,
        loader_name,
        lambda requested: lambda *args, **kwargs: (requested, args, kwargs),
    )

    wrapper = getattr(score_mod, wrapper_name)
    result = wrapper(1, flag=True)

    assert result == (symbol_name, (1,), {"flag": True})


@pytest.mark.parametrize(
    ("symbol_name", "loader_name"),
    [
        ("DefaultFairlearnScorerDictConfig", "_load_fairlearn_score_symbols"),
        ("DefaultAnjanaScorerDictConfig", "_load_anjana_score_symbols"),
        ("DefaultLifelinesConfig", "_load_lifelines_score_symbols"),
    ],
)
def test_score_module_getattr_returns_lazy_loaded_symbol(
    monkeypatch: pytest.MonkeyPatch,
    symbol_name: str,
    loader_name: str,
) -> None:
    sentinel = object()
    monkeypatch.delattr(score_mod, symbol_name, raising=False)

    def _loader() -> bool:
        score_mod.__dict__[symbol_name] = sentinel
        return True

    monkeypatch.setattr(score_mod, loader_name, _loader)

    assert score_mod.__getattr__(symbol_name) is sentinel


def test_score_module_getattr_raises_for_unknown_or_unloaded_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(score_mod, "_load_fairlearn_score_symbols", lambda: False)

    with pytest.raises(AttributeError):
        score_mod.__getattr__("DefaultFairlearnScorerDictConfig")

    with pytest.raises(AttributeError):
        score_mod.__getattr__("DefinitelyMissingScoreSymbol")


class TestScorerDictConfigMerge:
    """ScorerDictConfig.merge() should union scorer dicts from multiple specs."""

    acc_dict = {
        "accuracy": {"score_function": "sklearn.metrics.accuracy_score"},
    }
    prec_dict = {
        "precision": {
            "score_function": "sklearn.metrics.precision_score",
            "score_params": {"average": "weighted", "zero_division": 0},
        },
    }
    f1_dict = {
        "f1": {
            "score_function": "sklearn.metrics.f1_score",
            "score_params": {"average": "weighted", "zero_division": 0},
        },
    }

    def test_merge_scorer_dict_config_instances(self):
        a = ScorerDictConfig(scorers=self.acc_dict)
        b = ScorerDictConfig(scorers=self.prec_dict)
        result = ScorerDictConfig.merge([a, b])
        assert "accuracy" in result.scorers
        assert "precision" in result.scorers

    def test_merge_dict_with_scorers_key(self):
        wrapped = {"scorers": self.acc_dict}
        result = ScorerDictConfig.merge([wrapped, self.prec_dict])
        assert "accuracy" in result.scorers
        assert "precision" in result.scorers


class TestScorerConfig:
    def test_scorer_config_swap(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        config = ScorerConfig(
            score_name="accuracy",
            score_function=accuracy_score,
            score_params={},
        )
        score_swap = config(y_true=y_true, y_pred=y_pred, swap=True)
        score_normal = config(y_true=y_pred, y_pred=y_true)
        assert score_swap == score_normal

    def test_scorer_config_accepts_torch_tensors_when_available(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch is optional and not installed")
        except RuntimeError as exc:
            if "_has_torch_function' already has a docstring" in str(exc):
                pytest.skip(
                    "torch import is broken under the current coverage/instrumentation environment",
                )
            # TODO audit optional imports to fix bug-- probably related to re-exports
            raise

        y_true = torch.tensor([1, 0, 1, 1])
        y_pred = torch.tensor([1, 0, 0, 1])
        config = ScorerConfig(
            score_name="accuracy",
            score_function="sklearn.metrics.accuracy_score",
            score_params={},
        )
        score = config(y_true=y_true, y_pred=y_pred)
        assert score == 0.75


class TestScorerDictConfig:
    def test_scorer_dict_config_get_callables(self):
        scorer_dict = ScorerDictConfig(
            scorers={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function=accuracy_score,
                    score_params={},
                ),
            },
        )
        callables = scorer_dict.get_callables()
        assert "accuracy" in callables
        assert callable(callables["accuracy"])


class TestDefaultScorerDicts:
    @pytest.mark.parametrize(
        ("profile_name", "kwargs", "expected_keys"),
        [
            (
                "classifier",
                {
                    "y_true": [1, 0, 1, 1],
                    "y_pred": [1, 0, 0, 1],
                    "y_proba": [0.9, 0.1, 0.3, 0.8],
                },
                {"accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"},
            ),
            (
                "regressor",
                {
                    "y_true": [1.0, 2.0, 3.0, 4.0],
                    "y_pred": [1.1, 1.9, 3.2, 3.8],
                },
                {"mse", "mae", "r2"},
            ),
        ],
    )
    def test_canonical_default_score_profiles(
        self,
        profile_name,
        kwargs,
        expected_keys,
    ):
        scores = _load_score_profile(profile_name)(**kwargs)
        assert expected_keys.issubset(scores)

    def test_default_classifier_dict_requires_probabilities_for_probability_metrics(
        self,
    ):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        with pytest.raises(ValueError):
            _load_score_profile("classifier")(y_true=y_true, y_pred=y_pred)

    @pytest.mark.parametrize("profile_name", ["classifier", "regressor"])
    def test_default_profiles_with_empty_predictions_raise(self, profile_name):
        with pytest.raises(ValueError):
            _load_score_profile(profile_name)(y_true=[], y_pred=[])


class TestSurvivalScorers:
    class _MockFitter:
        def __init__(self):
            self.concordance_index_ = 0.71
            self.AIC_ = 123.4
            self.log_likelihood_ = -50.0
            self.params_ = [1.0, 2.0, 3.0]

    def test_survival_aic_score(self):
        fitter = self._MockFitter()
        score = survival_aic_score(y_true=[1, 2, 3], y_pred=fitter)
        assert score == fitter.AIC_

    def test_survival_bic_score_computed(self):
        fitter = self._MockFitter()
        score = survival_bic_score(y_true=[1, 2, 3, 4, 5], y_pred=fitter)
        assert isinstance(score, float)


SCORE_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config" / "score"
)


def _load_score_profile(name: str):
    cfg = OmegaConf.load(SCORE_DIR / f"{name}.yaml")
    return coerce_scorer_config(OmegaConf.to_container(cfg, resolve=True))


def test_model_and_data_default_profiles_infer_task_from_context():
    model_scorer = DefaultModelScorerDictConfig(classifier=None, scorers={})
    data_scorer = DefaultDataScorerDictConfig(classifier=None, scorers={})

    model_scores = model_scorer(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 1, 1],
        y_proba=[0.1, 0.8, 0.7, 0.6],
        mode=None,
    )
    assert "accuracy" in model_scores

    data_scores = data_scorer(
        data=SimpleNamespace(
            _X=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]],
            _y=[0, 1, 1, 0],
            classifier=True,
        ),
        mode="pre-sample",
    )
    assert "pre-sample" in data_scores
    assert "num_classes" in data_scores["pre-sample"]

    def test_survival_concordance_requires_attribute(self):
        with pytest.raises(ValueError):
            survival_concordance_score(y_true=[1, 2], y_pred=object())

    def test_survival_aic_score_supports_partial_aic(self):
        class _Fitter:
            pass

        fitter = _Fitter()
        fitter.partial_AIC_ = 42.5
        score = survival_aic_score(y_true=[1, 2], y_pred=fitter)
        assert score == 42.5

    def test_survival_aic_score_computes_from_log_likelihood_and_params_callable(self):
        class _Fitter:
            pass

        fitter = _Fitter()
        fitter.log_likelihood_ = -12.0
        fitter.params = lambda: [0.1, 0.2, 0.3]
        score = survival_aic_score(y_true=[1, 2], y_pred=fitter)
        assert score == 30.0

    def test_survival_aic_score_raises_when_unavailable(self):
        with pytest.raises(ValueError):
            survival_aic_score(y_true=[1, 2], y_pred=object())

    def test_survival_bic_score_prefers_direct_attribute(self):
        class _Fitter:
            pass

        fitter = _Fitter()
        fitter.BIC_ = 88.0
        score = survival_bic_score(y_true=[1, 2], y_pred=fitter)
        assert score == 88.0

    def test_survival_bic_score_uses_n_samples_kwarg(self):
        class _Fitter:
            pass

        fitter = _Fitter()
        fitter.log_likelihood_ = -10.0
        fitter.params = lambda: [0.1, 0.2]
        score = survival_bic_score(y_true=None, y_pred=fitter, n_samples=10)
        assert isinstance(score, float)

    def test_survival_bic_score_raises_when_unavailable(self):
        with pytest.raises(ValueError):
            survival_bic_score(y_true=None, y_pred=object())


class TestAttackScorers:
    def test_evasion_success_score_requires_benign_labels(self):
        from deckard.score.attack import evasion_success_score

        with pytest.raises(ValueError):
            evasion_success_score(y_true=[0, 1], y_pred=[0, 1], ben_pred_labels=None)

    def test_canonical_attack_evasion_score_profile(self):
        scorer = _load_score_profile("evasion-classification")
        scores = scorer(
            y_true=[0, 1, 0, 1],
            y_pred=[0, 0, 0, 1],
            ben_pred_labels=[0, 1, 0, 1],
        )
        assert "accuracy" in scores
        assert "success" in scores
        assert "precision" in scores

    def test_canonical_attack_evasion_regression_score_profile(self):
        scorer = _load_score_profile("evasion-regression")
        scores = scorer(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred=[1.1, 1.9, 3.2, 3.8],
        )
        assert "mse" in scores
        assert "mae" in scores
        assert "r2" in scores

    def test_attack_scorer_membership(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_membership(
            labels=[1, 1, 0, 0],
            inferred=[1, 0, 0, 0],
            attack_size=4,
        )
        assert "membership_inference_accuracy" in scores
        assert "membership_inference_precision" in scores
        assert "attack_score_time" in scores

    def test_attack_score_with_profile_flattens_mode_and_stage_payloads(self):
        scorer = AttackScorerConfig()

        class _Profile:
            def __call__(self, y_true, y_pred, **kwargs):
                _ = (y_true, y_pred, kwargs)
                return {
                    "attack": {"accuracy": 1.0},
                    "attack-score": {"precision": 0.5},
                    "metadata": 2.0,
                }

        mode_scores = scorer._score_with_profile(
            profile=_Profile(),
            y_true=[0, 1],
            y_pred=[0, 1],
            prefix="evasion",
            n_samples=2,
            mode="attack",
        )
        assert mode_scores["evasion_accuracy"] == 1.0
        assert mode_scores["evasion_metadata"] == 2.0

        stage_scores = scorer._score_with_profile(
            profile=_Profile(),
            y_true=[0, 1],
            y_pred=[0, 1],
            prefix="evasion",
            n_samples=2,
            stage="attack-score",
        )
        assert stage_scores["evasion_precision"] == 0.5

    def test_attack_scorer_attribute_classification(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_attribute(
            target=[1, 0, 1, 0],
            inferred=[1, 1, 1, 0],
            attack_size=4,
            targeted_attribute="age",
            is_classification=True,
            attack_generation_time=0.1,
        )
        assert "inferred_age_accuracy" in scores
        assert "inferred_age_f1" in scores
        assert "attack_generation_time" in scores

    def test_attack_scorer_attribute_regression(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_attribute(
            target=[1.0, 2.0, 3.0, 4.0],
            inferred=[1.1, 1.9, 3.2, 3.8],
            attack_size=4,
            targeted_attribute="income",
            is_classification=False,
        )
        assert "inferred_income_mse" in scores
        assert "inferred_income_r2" in scores
        assert "attack_score_time" in scores

    def test_attack_scorer_attribute_requires_targeted_attribute(self):
        scorer = AttackScorerConfig()
        with pytest.raises(ValueError):
            scorer._score(
                attack_kind="attribute",
                y_true=[0, 1],
                y_pred=[0, 1],
                attack_size=2,
                targeted_attribute=None,
                is_classification=True,
            )

    def test_attack_scorer_attribute_requires_is_classification(self):
        scorer = AttackScorerConfig()
        with pytest.raises(ValueError):
            scorer._score(
                attack_kind="attribute",
                y_true=[0, 1],
                y_pred=[0, 1],
                attack_size=2,
                targeted_attribute="age",
                is_classification=None,
            )

    def test_attack_scorer_rejects_unknown_kind(self):
        scorer = AttackScorerConfig()
        with pytest.raises(ValueError):
            scorer._score(
                attack_kind="unknown",
                y_true=[0, 1],
                y_pred=[0, 1],
                attack_size=2,
            )

    def test_attack_scorer_coerce_profile_from_dict(self):
        scorer = AttackScorerConfig(
            evasion={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function="sklearn.metrics.accuracy_score",
                ),
            },
        )
        assert isinstance(scorer.evasion, ScorerDictConfig)

    def test_attack_scorer_coerce_profile_invalid_type_raises(self):
        with pytest.raises(TypeError):
            AttackScorerConfig(evasion=123)


class TestDataInspectionScorers:
    def test_canonical_data_classification_score_profile(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        X = pd.DataFrame(
            {
                "feature_0": [0, 1, 0, 1, 1, 0, 1, 0],
                "feature_1": [1, 2, 1, 2, 2, 1, 2, 1],
            },
        )
        scores = _load_score_profile("data-classification")(
            y_true=y_true,
            y_pred=X,
            mode=None,
        )
        assert "num_classes" in scores
        assert "class_count_min" in scores
        assert "class_count_max" in scores
        assert "class_imbalance_ratio" in scores
        assert "mutual_information_mean" in scores
        assert "mutual_information_max" in scores
        assert scores["num_classes"] == 2

    def test_data_regression_default_scores_include_ecdf(self):
        y_true = np.array([2.1, 2.5, 3.0, 3.4, 3.6, 4.0])
        X = pd.DataFrame(
            {
                "feature_0": [0.1, 0.2, 0.4, 0.7, 0.8, 1.0],
                "feature_1": [10, 12, 14, 18, 19, 22],
            },
        )
        scores = DefaultDataRegressionScorerDictConfig()(
            y_true=y_true,
            y_pred=X,
            mode=None,
        )
        assert "mutual_information_mean" in scores
        assert "mutual_information_max" in scores
        assert "empirical_cdf" in scores
        assert callable(scores["empirical_cdf"])

        ecdf = scores["empirical_cdf"]
        values = ecdf(np.array([2.1, 3.0, 4.0]))
        assert np.all(values[:-1] <= values[1:])
