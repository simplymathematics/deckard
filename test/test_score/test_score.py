from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score

from deckard.score import (
    AttackScorerConfig,
    DefaultClassifierScorerDictConfig,
    DefaultDataClassificationScorerDictConfig,
    DefaultDataRegressionScorerDictConfig,
    DefaultRegressorScorerDictConfig,
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

    def test_merge_two_bare_dicts(self):
        result = ScorerDictConfig.merge([self.acc_dict, self.prec_dict])
        assert isinstance(result, ScorerDictConfig)
        assert "accuracy" in result.scorers
        assert "precision" in result.scorers

    def test_merge_three_dicts(self):
        result = ScorerDictConfig.merge(
            [self.acc_dict, self.prec_dict, self.f1_dict],
        )
        assert set(result.scorers.keys()) == {"accuracy", "precision", "f1"}

    def test_merge_scorer_dict_config_instances(self):
        a = ScorerDictConfig(scorers=self.acc_dict)
        b = ScorerDictConfig(scorers=self.prec_dict)
        result = ScorerDictConfig.merge([a, b])
        assert "accuracy" in result.scorers
        assert "precision" in result.scorers

    def test_merge_later_wins_on_key_conflict(self):
        override = {
            "accuracy": {
                "score_function": "sklearn.metrics.balanced_accuracy_score",
            },
        }
        result = ScorerDictConfig.merge([self.acc_dict, override])
        assert (
            result.scorers["accuracy"].score_function.__name__
            == "balanced_accuracy_score"
        )

    def test_merge_dict_with_scorers_key(self):
        wrapped = {"scorers": self.acc_dict}
        result = ScorerDictConfig.merge([wrapped, self.prec_dict])
        assert "accuracy" in result.scorers
        assert "precision" in result.scorers

    def test_merge_single_element_list(self):
        result = ScorerDictConfig.merge([self.acc_dict])
        assert "accuracy" in result.scorers


class TestScorerConfig:
    def test_scorer_config_initialization(self):
        config = ScorerConfig(
            score_name="accuracy",
            score_function=accuracy_score,
            score_params={"normalize": True},
        )
        assert config.score_name == "accuracy"
        assert callable(config.score_function)

    def test_scorer_config_callable(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        config = ScorerConfig(
            score_name="accuracy",
            score_function=accuracy_score,
            score_params={},
        )
        score = config(y_true=y_true, y_pred=y_pred)
        assert score == accuracy_score(y_true, y_pred)

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

    def test_scorer_config_with_additional_params(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        config = ScorerConfig(
            score_name="precision",
            score_function=precision_score,
            score_params={"average": "binary", "zero_division": 0},
        )
        score = config(y_true=y_true, y_pred=y_pred)
        assert score == precision_score(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

    def test_scorer_config_accepts_torch_tensors_when_available(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch is optional and not installed")

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
    def test_scorer_dict_config_initialization_and_call(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        scorer_dict = ScorerDictConfig(
            scorers={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function=accuracy_score,
                    score_params={},
                ),
                "mse": ScorerConfig(
                    score_name="mse",
                    score_function=mean_squared_error,
                    score_params={},
                ),
            },
        )
        scores = scorer_dict(y_true=y_true, y_pred=y_pred)
        assert "accuracy" in scores
        assert "mse" in scores
        assert scores["accuracy"] == accuracy_score(y_true, y_pred)
        assert scores["mse"] == mean_squared_error(y_true, y_pred)

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
    def test_default_classifier_dict(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        y_proba = [0.9, 0.1, 0.3, 0.8]
        scores = DefaultClassifierScorerDictConfig()(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )
        assert "accuracy" in scores
        assert "precision" in scores
        assert "recall" in scores
        assert "f1" in scores
        assert "roc_auc" in scores
        assert "log_loss" in scores

    def test_default_classifier_dict_requires_probabilities_for_probability_metrics(
        self,
    ):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        with pytest.raises(ValueError):
            DefaultClassifierScorerDictConfig()(y_true=y_true, y_pred=y_pred)

    def test_default_regressor_dict(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.2, 3.8]
        scores = DefaultRegressorScorerDictConfig()(y_true=y_true, y_pred=y_pred)
        assert "mse" in scores
        assert "mae" in scores
        assert "r2" in scores

    def test_default_classifier_dict_with_empty_predictions(self):
        y_true = []
        y_pred = []
        with pytest.raises(ValueError):
            DefaultClassifierScorerDictConfig()(y_true=y_true, y_pred=y_pred)

    def test_default_regressor_dict_with_empty_predictions(self):
        y_true = []
        y_pred = []
        with pytest.raises(ValueError):
            DefaultRegressorScorerDictConfig()(y_true=y_true, y_pred=y_pred)


class TestSurvivalScorers:
    class _MockFitter:
        def __init__(self):
            self.concordance_index_ = 0.71
            self.AIC_ = 123.4
            self.log_likelihood_ = -50.0
            self.params_ = [1.0, 2.0, 3.0]

    def test_survival_concordance_score(self):
        fitter = self._MockFitter()
        score = survival_concordance_score(y_true=[1, 2, 3], y_pred=fitter)
        assert score == fitter.concordance_index_

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


def test_model_default_score_profile_executes_from_yaml_defaults():
    scorer = _load_score_profile("classification")

    scores = scorer(
        y_true=[1, 0, 1, 1],
        y_pred=[1, 0, 0, 1],
        y_proba=[0.9, 0.1, 0.3, 0.8],
        mode=None,
    )

    assert "accuracy" in scores
    assert "precision" in scores
    assert "recall" in scores
    assert "f1" in scores
    assert "log_loss" in scores


def test_data_default_score_profile_executes_from_yaml_defaults():
    scorer = _load_score_profile("data-classification")
    assert isinstance(scorer, _DataScorerMarker)

    data = SimpleNamespace(
        _X=[[0.1, 1.0], [0.2, 1.0], [0.9, 0.0], [0.8, 0.0]],
        _y=[0, 0, 1, 1],
        classifier=True,
    )

    scores = scorer(data=data, mode="pre-sample")

    assert "pre-sample" in scores
    assert "num_classes" in scores["pre-sample"]
    assert "class_count_min" in scores["pre-sample"]
    assert "class_count_max" in scores["pre-sample"]
    assert "class_imbalance_ratio" in scores["pre-sample"]
    assert "mutual_information_mean" in scores["pre-sample"]
    assert "mutual_information_max" in scores["pre-sample"]


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

    def test_attack_scorer_evasion(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_evasion(
            ben_pred_labels=[0, 1, 0, 1],
            adv_pred_labels=[0, 0, 0, 1],
            y_true=[0, 1, 0, 1],
            attack_size=4,
        )
        assert "evasion_accuracy" in scores
        assert "evasion_success" in scores
        assert "attack_score_time" in scores

    def test_attack_scorer_evasion_regression(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_evasion(
            ben_pred_labels=[1.0, 2.0, 3.0, 4.0],
            adv_pred_labels=[1.1, 1.9, 3.2, 3.8],
            y_true=[1.0, 2.0, 3.0, 4.0],
            attack_size=4,
            is_classification=False,
        )
        assert "evasion_mse" in scores
        assert "evasion_mae" in scores
        assert "evasion_r2" in scores
        assert "evasion_success" not in scores
        assert "attack_score_time" in scores

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

    def test_attack_score_configstores_registered(self):
        scorer = AttackScorerConfig()
        cs = ConfigStore.instance()
        assert cs is not None
        assert scorer.evasion is not None
        assert scorer.membership_inference is not None
        assert scorer.attribute_inference is not None

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
    def test_data_classification_default_scores(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        X = pd.DataFrame(
            {
                "feature_0": [0, 1, 0, 1, 1, 0, 1, 0],
                "feature_1": [1, 2, 1, 2, 2, 1, 2, 1],
            },
        )
        scores = DefaultDataClassificationScorerDictConfig()(
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

    def test_data_classification_reference_column_override(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        X = pd.DataFrame(
            {
                "age": [20, 24, 41, 45, 44, 23, 43, 21],
                "income_proxy": [1.2, 1.5, 2.8, 3.0, 2.9, 1.4, 3.1, 1.3],
            },
        )
        scores = DefaultDataClassificationScorerDictConfig()(
            y_true=y_true,
            y_pred=X,
            mode=None,
            reference_column="age",
        )
        assert "mutual_information_mean" in scores
        assert scores["mutual_information_mean"] >= 0.0

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

    def test_data_scorer_configstores_registered(self):
        cs = ConfigStore.instance()
        assert cs is not None
        assert isinstance(
            DefaultDataClassificationScorerDictConfig(),
            DefaultDataClassificationScorerDictConfig,
        )
        assert isinstance(
            DefaultDataRegressionScorerDictConfig(),
            DefaultDataRegressionScorerDictConfig,
        )
