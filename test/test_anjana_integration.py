"""Integration and runtime tests for Anjana chains in examples/sklearn."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from deckard.attack import AttackConfig
from deckard.experiment import ExperimentConfig
from deckard.file import FileConfig
from deckard.model import DefensePipelineConfig, ModelConfig


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_SKLEARN_DIR = ROOT / "examples" / "sklearn"
DECKARD_RC_PATH = EXAMPLES_SKLEARN_DIR / ".deckard_rc"


def _load_env_from_deckard_rc(path: Path) -> dict[str, str]:
    env_overrides: dict[str, str] = {}
    if not path.exists():
        return env_overrides
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        key_value = line[len("export ") :]
        if "=" not in key_value:
            continue
        key, value = key_value.split("=", 1)
        env_overrides[key.strip()] = value.strip().strip('"').strip("'")
    return env_overrides


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(_load_env_from_deckard_rc(DECKARD_RC_PATH))
    env["DECKARD_TEST_MAX_SAMPLES"] = "200"
    env.setdefault("MPLBACKEND", "Agg")
    return env


def _make_anjana_data(n=40, monkeypatch=None, defense=None):
    from deckard.data.anjana import AnjanaDataConfig

    cfg = AnjanaDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": n,
            "n_features": 6,
            "n_informative": 3,
            "n_redundant": 0,
            "random_state": 0,
            "n_clusters_per_class": 1,
        },
        train_size=0.7,
        test_size=0.3,
        classifier=True,
        random_state=42,
        quasi_identifiers=["feature_0", "feature_1"],
        sensitive_attribute="target",
        sensitive_columns=["feature_0"],
        anjana_defense=defense,
        hierarchy_interval_sizes={"feature_0": [1, 2], "feature_1": [1, 2]},
    )

    if monkeypatch is not None and defense is not None:

        def _stub_k_anon(data, **kwargs):
            _ = kwargs
            return data.copy()

        monkeypatch.setattr(
            "deckard.data.anjana.resolve_class",
            lambda _: _stub_k_anon,
        )
    return cfg


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
def test_deckard_optimize_help_in_examples_sklearn():
    result = subprocess.run(
        [sys.executable, "-m", "deckard", "optimize", "--help"],
        cwd=str(EXAMPLES_SKLEARN_DIR),
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
def test_deckard_optimize_smoke_matrix_sklearn():
    cmd = [
        sys.executable,
        "-m",
        "deckard",
        "optimize",
        "data=classification",
        "model=logistic",
        "attack=boundary",
        "defense=class-labels",
        "score=classification",
        "experiment_name=sklearn_smoke_chain",
        "model.model_params.max_iter=20",
        "data.data_params.n_samples=80",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(EXAMPLES_SKLEARN_DIR),
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_anjana_attack_chain_type_and_scores(monkeypatch):
    def _stub_k_anon(data, **kwargs):
        _ = kwargs
        return data.copy()

    monkeypatch.setattr(
        "deckard.data.anjana.resolve_class",
        lambda _: _stub_k_anon,
    )

    data_cfg = _make_anjana_data(
        n=40,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    model_cfg = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 30},
    )
    attack_cfg = AttackConfig(
        attack_type="art.attacks.evasion.HopSkipJump",
        attack_size=3,
        attack_params={
            "max_iter": 1,
            "init_eval": 1,
            "max_eval": 2,
            "init_size": 5,
            "norm": 2,
            "targeted": False,
        },
    )

    exp = ExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        attack=attack_cfg,
        files=FileConfig(),
        classifier=True,
    )
    scores = exp()

    assert isinstance(exp.data.X_train, pd.DataFrame)
    assert "accuracy" in scores
    assert "evasion_accuracy" in scores


def test_anjana_fairness_and_art_chain_type_and_transform(monkeypatch):
    pytest.importorskip("fairlearn")
    from art.estimators.classification.scikitlearn import (
        ScikitlearnLogisticRegression,
    )

    def _drop_half_rows(data, **kwargs):
        _ = kwargs
        return data.iloc[: len(data) // 2].copy()

    monkeypatch.setattr(
        "deckard.data.anjana.resolve_class",
        lambda _: _drop_half_rows,
    )

    data_cfg = _make_anjana_data(
        n=60,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    data_cfg.fairness_defense = {
        "name": "fairlearn.preprocessing.CorrelationRemover",
        "step_name": "fairness_correlation_remover",
    }

    model_cfg = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 30},
        defense=DefensePipelineConfig(
            defenses=[
                {
                    "defense_name": "art.defences.postprocessor.ClassLabels",
                    "defense_params": {
                        "apply_fit": False,
                        "apply_predict": True,
                    },
                    "classifier": True,
                },
            ],
        ),
    )

    exp = ExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        attack=None,
        files=FileConfig(),
        classifier=True,
    )
    scores = exp()

    assert isinstance(exp.model._model, ScikitlearnLogisticRegression)
    assert len(exp.data._X) == 30
    assert len(exp.data.X_train) + len(exp.data.X_test) == 30
    assert hasattr(exp.data, "_sensitive_train")
    assert "accuracy" in scores


def test_wrapper_defenses_reordered_last_with_warning(caplog):
    import logging

    call_order = []

    class _StubDataDefense:
        defense_name = "custom.mock.DataDefense"
        defense_application_time = 0.0

        def apply_to(self, estimator, data):
            _ = data
            call_order.append("data")
            return estimator

    class _StubArtDefense:
        defense_name = "art.mock.MockArtDefense"
        defense_application_time = 0.0

        def apply_to(self, estimator, data):
            _ = data
            call_order.append("art")
            return estimator

    pipeline = DefensePipelineConfig(
        defenses=[_StubArtDefense(), _StubDataDefense()],
    )

    with caplog.at_level(logging.WARNING, logger="deckard.model.defend"):
        pipeline.apply(estimator=object(), data=object())

    assert call_order == ["data", "art"]
    assert any(
        "automatically reordered" in rec.message for rec in caplog.records
    )
