import math
from types import SimpleNamespace

import pytest

from deckard.plugins.lifelines.score import (
    DefaultLifelinesConfig,
    survival_aic_score,
    survival_bic_score,
    survival_concordance_score,
)


def test_survival_concordance_score_success_and_error():
    model = SimpleNamespace(concordance_index_=0.731)
    assert survival_concordance_score(y_true=None, y_pred=model) == pytest.approx(0.731)

    with pytest.raises(ValueError, match="concordance_index_"):
        survival_concordance_score(y_true=None, y_pred=SimpleNamespace())


def test_survival_aic_score_prefers_aic_then_partial_aic():
    model_aic = SimpleNamespace(AIC_=12.5)
    assert survival_aic_score(y_true=None, y_pred=model_aic) == pytest.approx(12.5)

    model_partial = SimpleNamespace(partial_AIC_=19.2)
    assert survival_aic_score(y_true=None, y_pred=model_partial) == pytest.approx(19.2)


def test_survival_aic_score_from_log_likelihood_with_params_variants():
    model_params_attr = SimpleNamespace(log_likelihood_=-42.0, params_=[1.0, 2.0, 3.0])
    # AIC = -2*ll + 2*k = -2*(-42) + 2*3 = 90
    assert survival_aic_score(y_true=None, y_pred=model_params_attr) == pytest.approx(90.0)

    model_params_method = SimpleNamespace(
        log_likelihood_=-10.0,
        params=lambda: [1.0, 2.0],
    )
    # AIC = -2*(-10) + 2*2 = 24
    assert survival_aic_score(y_true=None, y_pred=model_params_method) == pytest.approx(24.0)


@pytest.mark.parametrize(
    "bad_model",
    [
        SimpleNamespace(log_likelihood_=-10.0),
        SimpleNamespace(params_=[1.0, 2.0]),
        SimpleNamespace(),
    ],
)
def test_survival_aic_score_error_when_insufficient_model_info(bad_model):
    with pytest.raises(ValueError, match="compute AIC"):
        survival_aic_score(y_true=None, y_pred=bad_model)


def test_survival_bic_score_prefers_explicit_bic():
    model = SimpleNamespace(BIC_=33.7)
    assert survival_bic_score(y_true=[1, 2, 3], y_pred=model) == pytest.approx(33.7)


def test_survival_bic_score_from_log_likelihood_with_inferred_n_and_params():
    model_params_attr = SimpleNamespace(log_likelihood_=-5.0, params_=[1.0, 2.0])
    y_true = [0, 1, 1, 0, 1]
    expected = -2.0 * (-5.0) + 2.0 * math.log(len(y_true))
    assert survival_bic_score(y_true=y_true, y_pred=model_params_attr) == pytest.approx(expected)



def test_survival_bic_score_from_log_likelihood_with_n_samples_kw_and_params_method():
    model_params_method = SimpleNamespace(
        log_likelihood_=-8.0,
        params=lambda: [1.0, 2.0, 3.0],
    )
    n_samples = 10
    expected = -2.0 * (-8.0) + 3.0 * math.log(n_samples)
    assert survival_bic_score(y_true=None, y_pred=model_params_method, n_samples=n_samples) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs,bad_model",
    [
        ({}, SimpleNamespace(log_likelihood_=-1.0, params_=[1.0])),
        ({"n_samples": 0}, SimpleNamespace(log_likelihood_=-1.0, params_=[1.0])),
        ({"n_samples": 5}, SimpleNamespace(log_likelihood_=-1.0)),
        ({"n_samples": 5}, SimpleNamespace()),
    ],
)
def test_survival_bic_score_error_when_insufficient_model_info(kwargs, bad_model):
    with pytest.raises(ValueError, match="compute BIC"):
        survival_bic_score(y_true=None, y_pred=bad_model, **kwargs)


def test_default_lifelines_config_contains_expected_scorers():
    cfg = DefaultLifelinesConfig()
    assert set(cfg.scorers.keys()) == {"concordance", "aic", "bic"}
    assert cfg.scorers["aic"].greater_is_better is False
    assert cfg.scorers["bic"].greater_is_better is False
