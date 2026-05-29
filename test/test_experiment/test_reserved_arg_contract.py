from types import SimpleNamespace
from typing import Any, Callable, cast

import numpy as np
import pytest

from deckard.attack import AttackConfig
from deckard.data import DataConfig
from deckard.detector import DetectorConfig
from deckard.experiment import ExperimentConfig
from deckard.model import ModelConfig
from deckard.model.defense.base import DefenseConfig, DefensePipelineConfig
from deckard.score.base import ScorerDictConfig


class _SpyScorer(ScorerDictConfig):
    def __post_init__(self):
        super().__post_init__()
        self.calls = []

    def __call__(self, *args, **kwargs) -> Any:
        self.calls.append((args, dict(kwargs)))
        mode = kwargs.get("mode", "test")
        return {mode: {"ok": 1.0}}


def _call_with_unknown_kwargs(factory: Callable[..., Any], **kwargs: Any) -> Any:
    return factory(**kwargs)


def _make_runtime_experiment() -> ExperimentConfig:
    exp = cast(Any, ExperimentConfig.__new__(ExperimentConfig))
    exp.data = cast(
        Any,
        SimpleNamespace(
            pipeline=SimpleNamespace(name="pipeline"),
            sampler=SimpleNamespace(name="sampler"),
            _sensitive_test=np.array([9, 8]),
        ),
    )
    exp.model = cast(
        Any,
        SimpleNamespace(
            defense=SimpleNamespace(name="defense"),
            trainer=SimpleNamespace(name="trainer"),
        ),
    )
    exp.attack = cast(Any, SimpleNamespace())
    exp.detector = cast(Any, SimpleNamespace())
    exp.files = cast(Any, SimpleNamespace())
    exp.evaluation_mode = "standard"
    exp.score_mode = "test"
    exp._has_explicit_score_mode = lambda: True
    exp._resolve_score_modes = lambda: ["test"]
    exp._ensure_mode_predictions = lambda mode: None
    exp._resolve_mode_model_outputs = (
        lambda mode: (
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0.1, 0.9]),
        )
    )
    return cast(ExperimentConfig, exp)


def _sensitive_test_values(exp: ExperimentConfig) -> np.ndarray:
    return cast(Any, exp.data)._sensitive_test


def test_post_init_rejects_unknown_kwargs_data_model_attack_detector_score_and_defense():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call_with_unknown_kwargs(
            DataConfig,
            name="adult",
            __phase4_unknown__=1,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call_with_unknown_kwargs(
            ModelConfig,
            model_params={"_target_": "sklearn.linear_model.LogisticRegression"},
            __phase4_unknown__=1,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call_with_unknown_kwargs(
            AttackConfig,
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"_target_": "builtins.object"},
            __phase4_unknown__=1,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call_with_unknown_kwargs(DetectorConfig, __phase4_unknown__=1)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call_with_unknown_kwargs(ScorerDictConfig, scorers={}, __phase4_unknown__=1)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call_with_unknown_kwargs(DefensePipelineConfig, __phase4_unknown__=1)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call_with_unknown_kwargs(DefenseConfig, __phase4_unknown__=1)


def test_post_init_rejects_unknown_kwargs_experiment():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call_with_unknown_kwargs(
            ExperimentConfig,
            data=SimpleNamespace(),
            __phase4_unknown__=1,
        )


def test_experiment_reserved_runtime_kwargs_are_emitted_for_scorer_dict_paths():
    exp = _make_runtime_experiment()

    scorer = _SpyScorer(scorers={})
    exp.score = scorer

    out = ExperimentConfig._run_experiment_scorer_modes(exp)
    assert out == {"ok": 1.0}
    assert len(scorer.calls) == 1
    _, kwargs = scorer.calls[0]

    assert kwargs["__deckard__labels__"] is not None
    assert kwargs["__deckard__labels__test__"] is not None
    assert kwargs["__deckard__predictions__"] is not None
    assert kwargs["__deckard__predictions__test__"] is not None
    assert kwargs["__deckard__probabilities__"] is not None
    assert kwargs["__deckard__probabilities__test__"] is not None
    assert kwargs["__deckard__mode__"] == "test"
    assert kwargs["__deckard__mode__test__"] == "test"

    assert kwargs["__deckard__data__"] is exp.data
    assert kwargs["__deckard__data__test__"] is exp.data
    assert kwargs["__deckard__model__"] is exp.model
    assert kwargs["__deckard__model__test__"] is exp.model
    assert kwargs["__deckard__attack__"] is exp.attack
    assert kwargs["__deckard__detector__"] is exp.detector
    assert kwargs["__deckard__experiment__"] is exp
    assert kwargs["__deckard__files__"] is exp.files
    assert kwargs["__deckard__score__"] is exp.score
    assert kwargs["__deckard__scorer__"] is exp.score

    assert kwargs["__deckard__defense__"] is exp.model.defense
    assert kwargs["__deckard__defense__test__"] is exp.model.defense
    assert kwargs["__deckard__trainer__"] is exp.model.trainer
    assert kwargs["__deckard__trainer__test__"] is exp.model.trainer
    assert kwargs["__deckard__pipeline__"] is exp.data.pipeline
    assert kwargs["__deckard__pipeline__test__"] is exp.data.pipeline
    assert kwargs["__deckard__sampler__"] is exp.data.sampler
    assert kwargs["__deckard__sampler__test__"] is exp.data.sampler
    assert np.array_equal(kwargs["__deckard__sensitive__"], _sensitive_test_values(exp))
    assert np.array_equal(
        kwargs["__deckard__sensitive__test__"],
        _sensitive_test_values(exp),
    )

    assert "y_true" not in kwargs
    assert "y_pred" not in kwargs
    assert "y_proba" not in kwargs


def test_experiment_reserved_runtime_collision_merge_hard_fails():
    with pytest.raises(ValueError, match="Reserved runtime key collision"):
        ExperimentConfig._merge_reserved_runtime_kwargs(
            {"__deckard__labels__": np.array([1])},
            {"__deckard__labels__": np.array([2])},
        )
