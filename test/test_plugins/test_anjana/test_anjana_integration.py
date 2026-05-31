"""Integration and runtime tests for Anjana chains in examples/sklearn."""

import importlib.util
import json
import subprocess
import sys
import uuid
from pathlib import Path

import optuna
import pandas as pd
import pytest
from helpers import make_runtime_env

from deckard.attack import AttackConfig
from deckard.experiment import ExperimentConfig
from deckard.file import FileConfig
from deckard.model import DefensePipelineConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_SKLEARN_DIR = ROOT / "examples" / "sklearn"
DECKARD_RC_PATH = EXAMPLES_SKLEARN_DIR / ".deckard_rc"
DVCLIVE_AVAILABLE = importlib.util.find_spec("dvclive") is not None


def _runtime_env() -> dict[str, str]:
    env = make_runtime_env(DECKARD_RC_PATH)
    env["DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION"] = "1"
    env.setdefault("DECKARD_TEST_MAX_SAMPLES", "200")
    return env


def _run_optimize_and_load_scores(
    overrides: list[str],
    *,
    timeout: int = 300,
) -> tuple[dict, subprocess.CompletedProcess]:
    experiment_override = [
        item
        for item in overrides
        if isinstance(item, str) and item.startswith("experiment_name=")
    ]
    assert len(experiment_override) == 1
    experiment_name = experiment_override[0].split("=", 1)[1]

    score_override = (
        f"+files={{score_file:outputs/logs/{experiment_name}/scores.json}}"
    )
    has_score_override = any(
        isinstance(item, str)
        and "score_file" in item
        and item.startswith(("files", "+files"))
        for item in overrides
    )
    final_overrides = list(overrides)

    if any(item == "score=classification" for item in final_overrides):
        final_overrides.append(
            "++score._target_=deckard.score.base.DefaultClassifierScorerDictConfig",
        )
    if any(
        isinstance(item, str) and item.startswith("+score@score.model=classification")
        for item in final_overrides
    ):
        final_overrides.append(
            "++score.model._target_=deckard.score.base.DefaultClassifierScorerDictConfig",
        )
    if any(
        isinstance(item, str) and item.startswith("+score@score.data=anjana")
        for item in final_overrides
    ):
        final_overrides.append(
            "++score.data._target_=deckard.plugins.anjana.score.DefaultAnjanaDataScorerDictConfig",
        )
    if any(
        isinstance(item, str)
        and item.startswith("+score@score.attack=evasion-classification")
        for item in final_overrides
    ):
        final_overrides.append(
            "++score.attack._target_=deckard.score.attack.DefaultEvasionAttackScorerDictConfig",
        )
    for alias_override in (
        "~data_alias",
        "~model_alias",
        "~attack_alias",
        "~defense_alias",
    ):
        if alias_override not in final_overrides:
            final_overrides.append(alias_override)
    if not has_score_override:
        final_overrides.append(score_override)
    if not any(
        isinstance(item, str)
        and item.startswith(("hydra.job.num=", "+hydra.job.num="))
        for item in final_overrides
    ):
        final_overrides.append("hydra.job.num=0")

    cmd = [sys.executable, "-m", "deckard", "optimize", *final_overrides]
    result = subprocess.run(
        cmd,
        cwd=str(EXAMPLES_SKLEARN_DIR),
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    score_file = (
        EXAMPLES_SKLEARN_DIR / "outputs" / "logs" / experiment_name / "scores.json"
    )
    if not score_file.exists():
        candidates = sorted(
            EXAMPLES_SKLEARN_DIR.glob("outputs/logs/**/scores.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        assert len(candidates) > 0, "Expected at least one scores.json artifact"
        matched = None
        for candidate in candidates:
            try:
                with candidate.open("r") as handle:
                    payload = json.load(handle)
                if payload.get("experiment_name") == experiment_name:
                    matched = candidate
                    break
            except Exception:
                continue
        score_file = matched if matched is not None else candidates[0]
    with score_file.open("r") as handle:
        return json.load(handle), result


def _contains_metric(scores: dict, metric: str) -> bool:
    if isinstance(scores.get("payload"), dict):
        payload = scores["payload"]
        if metric in payload:
            return True
        flat = scores.get("flat")
        if isinstance(flat, dict) and metric in flat:
            return True
        return any(
            isinstance(value, dict) and metric in value for value in payload.values()
        )
    if metric in scores:
        return True
    return any(
        isinstance(value, dict) and metric in value for value in scores.values()
    )


def _make_anjana_data(n=40, monkeypatch=None, defense=None):
    from deckard.plugins.anjana.data import AnjanaDataConfig

    cfg = AnjanaDataConfig(
        name="make_classification",
        data_params={
            "n_samples": n,
            "n_features": 6,
            "n_informative": 3,
            "n_redundant": 0,
            "random_state": 0,
            "n_clusters_per_class": 1,
        },
        sampler={
            "name": "deckard.data.sample.SplitSampler",
            "train_size": 0.7,
            "test_size": 0.3,
        },
        classifier=True,
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
            "deckard.plugins.anjana.data.resolve_class",
            lambda _: _stub_k_anon,
        )
    return cfg


def _assert_anjana_privacy_scores(scores: dict) -> None:
    anjana_metrics = {
        "k_anonymity",
        "l_diversity",
        "t_closeness",
    }

    # Persisted score artifacts are envelope-based; runtime scores are plain mappings.
    metric_source = (
        scores.get("payload") if isinstance(scores.get("payload"), dict) else scores
    )

    # Check for metrics under "post-pipeline" key (expected default location)
    post_pipeline = metric_source.get("post-pipeline")
    if isinstance(post_pipeline, dict):
        assert anjana_metrics.issubset(set(post_pipeline.keys())), (
            f"Expected privacy metrics under 'post-pipeline' key, "
            f"but found keys: {sorted(post_pipeline.keys())}"
        )
        return

    # Direct scorer calls may return mode-scoped payloads, e.g. {"test": {...}}.
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

    # Check for flat top-level keys (legacy fallback)
    flattened_keys = {key for key in metric_source if key in anjana_metrics}
    assert anjana_metrics.issubset(flattened_keys), (
        "Expected Anjana-specific privacy metrics in scores (k_anonymity, "
        "l_diversity, t_closeness), "
        f"but found keys: {sorted(metric_source.keys())}"
    )


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
@pytest.mark.skipif(
    not DVCLIVE_AVAILABLE,
    reason="dvclive is required for optimize runtime hooks in this CLI smoke test",
)
def test_deckard_optimize_smoke_matrix_sklearn():
    cmd = [
        sys.executable,
        "-m",
        "deckard",
        "optimize",
        "data=test-classification",
        "model=test-logistic",
        "attack=boundary",
        "defense=class-labels",
        "score=classification",
        "experiment_name=sklearn_smoke_chain",
        "~data_alias",
        "~model_alias",
        "~attack_alias",
        "~defense_alias",
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


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
def test_joblib_launcher_syncs_scores_and_attrs_sklearn(tmp_path):
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")

    study_name = f"joblib_sync_{uuid.uuid4().hex[:8]}"
    db_path = tmp_path / "joblib_sync.db"
    storage = f"sqlite:///{db_path}"

    cmd = [
        sys.executable,
        "-m",
        "deckard",
        "optimize",
        "-m",
        "data=test-classification",
        "model=test-logistic",
        "attack=boundary",
        "defense=class-labels",
        "score=classification",
        "hydra/launcher=joblib",
        "hydra.launcher.n_jobs=1",
        "hydra.launcher.backend=null",
        "hydra.launcher.prefer=processes",
        "hydra.launcher.require=null",
        "hydra.launcher.verbose=0",
        "hydra.launcher.timeout=null",
        "hydra.launcher.pre_dispatch=2*n_jobs",
        "hydra.launcher.batch_size=auto",
        "hydra.launcher.temp_folder=null",
        "hydra.launcher.max_nbytes=null",
        "hydra.launcher.mmap_mode=r",
        "hydra.sweeper.n_trials=1",
        "hydra.sweeper.n_jobs=1",
        f"hydra.sweeper.study_name={study_name}",
        f"hydra.sweeper.storage={storage}",
        "~data_alias",
        "~model_alias",
        "~attack_alias",
        "~defense_alias",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(EXAMPLES_SKLEARN_DIR),
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=360,
        check=False,
    )

    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    score_file = (
        EXAMPLES_SKLEARN_DIR / "outputs" / "logs" / study_name / "0" / "scores.json"
    )
    if not score_file.exists():
        candidates = sorted(
            EXAMPLES_SKLEARN_DIR.glob(f"outputs/logs/{study_name}/**/scores.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            candidates = sorted(
                EXAMPLES_SKLEARN_DIR.glob("outputs/logs/**/scores.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        assert len(candidates) > 0, "Expected at least one scores.json artifact"
        score_file = candidates[0]

    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.get_trials(deepcopy=False)
    assert len(trials) >= 1
    attrs = trials[0].user_attrs
    assert "experiment_name" in attrs
    assert len(attrs) > 1
    score_attr_present = any(
        key.startswith("benign_")
        or key.startswith("evasion_")
        or key.endswith("_time")
        or key in {"accuracy", "evasion_accuracy", "attack_generation_time"}
        for key in attrs
    )
    metadata_attr_present = any(
        key in attrs
        for key in (
            "data",
            "model",
            "attack",
            "defense",
            "++defense.defense_name",
        )
    )
    assert score_attr_present or metadata_attr_present


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
def test_deckard_optimize_hydra_multirun_syncs_optuna_trial_attrs_sklearn(tmp_path):
    study_name = f"callback_cov_{uuid.uuid4().hex[:8]}"
    db_path = tmp_path / "callback_cov.db"
    storage = f"sqlite:///{db_path}"

    cmd = [
        sys.executable,
        "-m",
        "deckard",
        "optimize",
        "-m",
        "data=test-classification",
        "model=test-logistic",
        "attack=boundary",
        "defense=class-labels",
        "score=classification",
        "hydra.sweeper.n_trials=1",
        "hydra.sweeper.n_jobs=1",
        f"hydra.sweeper.study_name={study_name}",
        f"hydra.sweeper.storage={storage}",
        "~data_alias",
        "~model_alias",
        "~attack_alias",
        "~defense_alias",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(EXAMPLES_SKLEARN_DIR),
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.get_trials(deepcopy=False)
    assert len(trials) >= 1

    attrs = trials[0].user_attrs
    assert "experiment_name" in attrs
    assert len(attrs) > 1

    score_attr_present = any(
        key.startswith("benign_")
        or key.startswith("evasion_")
        or key.endswith("_time")
        or key in {"accuracy", "evasion_accuracy", "attack_generation_time"}
        for key in attrs
    )
    metadata_attr_present = any(
        key in attrs
        for key in (
            "data",
            "model",
            "attack",
            "defense",
            "++defense.defense_name",
        )
    )
    assert score_attr_present or metadata_attr_present


def test_anjana_attack_chain_type_and_scores(monkeypatch):
    def _stub_k_anon(data, **kwargs):
        _ = kwargs
        return data.copy()

    monkeypatch.setattr(
        "deckard.plugins.anjana.data.resolve_class",
        lambda _: _stub_k_anon,
    )

    data_cfg = _make_anjana_data(
        n=40,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    model_cfg = ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 30},
    )
    attack_cfg = AttackConfig(
        name="art.attacks.evasion.HopSkipJump",
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
    _assert_anjana_privacy_scores(scores)
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
        "deckard.plugins.anjana.data.resolve_class",
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
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 30},
        defense=DefensePipelineConfig(
            defenses=[
                {
                    "name": "art.defences.postprocessor.ClassLabels",
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
    assert not hasattr(exp.data, "_sensitive_train")
    _assert_anjana_privacy_scores(scores)
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

    with caplog.at_level(logging.WARNING, logger="deckard.model.defense.base"):
        pipeline.apply(estimator=object(), data=object())

    assert call_order == ["data", "art"]
    assert any("automatically reordered" in rec.message for rec in caplog.records)


@pytest.mark.slow
@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
@pytest.mark.skipif(
    importlib.util.find_spec("pycanon") is None,
    reason="pycanon is required for Anjana scorer integration",
)
def test_cli_score_chain_without_model_or_attack_sklearn():
    scorer = pytest.importorskip(
        "deckard.plugins.anjana.score",
        reason="Anjana scorer module is required for privacy-only scoring",
    )

    data_cfg = _make_anjana_data(
        n=40,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    data_cfg()

    score_cfg = scorer.DefaultAnjanaScorerDictConfig()
    scores = score_cfg(
        data=data_cfg,
        X=data_cfg._X,
        y=data_cfg._y,
        mode="test",
    )

    _assert_anjana_privacy_scores(scores)
    assert not _contains_metric(scores, "accuracy")
    assert not _contains_metric(scores, "evasion_accuracy")


@pytest.mark.slow
@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
@pytest.mark.skipif(
    importlib.util.find_spec("pycanon") is None,
    reason="pycanon is required for Anjana scorer integration",
)
def test_cli_score_chain_with_model_without_attack_sklearn():
    experiment_name = f"anjana_chain_model_{uuid.uuid4().hex[:8]}"
    scores, _ = _run_optimize_and_load_scores(
        [
            "data=test-classification",
            "model=test-logistic",
            "~attack",
            "~search/attacks",
            "attack_alias=no_attack",
            "+data.quasi_identifiers=[feature_0,feature_1]",
            "+data.sensitive_attribute=target",
            "+data.sensitive_columns=[feature_0]",
            "~score",
            "+score@score.model=classification",
            "+score@score.data=anjana",
            f"experiment_name={experiment_name}",
        ],
    )

    assert _contains_metric(scores, "accuracy")
    _assert_anjana_privacy_scores(scores)


@pytest.mark.slow
@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
@pytest.mark.skipif(
    importlib.util.find_spec("pycanon") is None,
    reason="pycanon is required for Anjana scorer integration",
)
def test_cli_score_chain_data_anjana_evasion_sklearn():
    experiment_name = f"anjana_chain_attack_{uuid.uuid4().hex[:8]}"
    scores, _ = _run_optimize_and_load_scores(
        [
            "data=test-classification",
            "model=test-logistic",
            "attack=boundary",
            "defense=class-labels",
            "+data.quasi_identifiers=[feature_0,feature_1]",
            "+data.sensitive_attribute=target",
            "+data.sensitive_columns=[feature_0]",
            "~score",
            "+score@score.model=classification",
            "+score@score.data=anjana",
            "+score@score.attack=evasion-classification",
            f"experiment_name={experiment_name}",
        ],
        timeout=420,
    )

    assert _contains_metric(scores, "accuracy")
    assert _contains_metric(scores, "evasion_accuracy")
    _assert_anjana_privacy_scores(scores)


@pytest.mark.slow
@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
@pytest.mark.skipif(
    importlib.util.find_spec("pycanon") is None,
    reason="pycanon is required for Anjana scorer integration",
)
@pytest.mark.skipif(
    importlib.util.find_spec("fairlearn") is None,
    reason="fairlearn is required for fairness scorer chain integration",
)
def test_cli_score_chain_with_fairness_sklearn():
    experiment_name = f"anjana_chain_fair_{uuid.uuid4().hex[:8]}"
    scores, _ = _run_optimize_and_load_scores(
        [
            "data=test-classification",
            "model=test-logistic",
            "attack=boundary",
            "defense=class-labels",
            "+data.quasi_identifiers=[feature_0,feature_1]",
            "+data.sensitive_attribute=target",
            "+data.sensitive_columns=[feature_0]",
            "~score",
            "+score@score.model=fairness-classification",
            "+score@score.data=anjana",
            "+score@score.attack=evasion-classification",
            f"experiment_name={experiment_name}",
        ],
        timeout=420,
    )

    assert _contains_metric(scores, "accuracy")
    assert _contains_metric(scores, "evasion_accuracy")
    assert _contains_metric(scores, "demographic_parity_difference")
    assert _contains_metric(scores, "equalized_odds_difference")
    _assert_anjana_privacy_scores(scores)
