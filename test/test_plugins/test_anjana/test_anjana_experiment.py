"""Comprehensive tests for Anjana-based experiments.

Working directory context: examples/sklearn (uses default.yaml via .deckard_rc).
Covers: unit, integration, hash stability, persistence, chain (anjana+art, anjana+fairness,
anjana+attack) and subcommand tests.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from helpers import load_canonical_data_profile, make_runtime_env

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_SKLEARN_DIR = ROOT / "examples" / "sklearn"
DECKARD_RC_PATH = EXAMPLES_SKLEARN_DIR / ".deckard_rc"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anjana_data(
    n=20,
    monkeypatch=None,
    defense=None,
    resolve_class_fn=None,
    **overrides,
):
    """Build a minimal AnjanaDataConfig with synthetic tabular data.

    Parameters
    ----------
    defense:
        anjana_defense dict to pass to AnjanaDataConfig.
    monkeypatch:
        If provided, patches resolve_class so no real anjana package is needed.
    """
    from deckard.plugins.anjana.data import AnjanaDataConfig

    base = load_canonical_data_profile("anjana", framework="sklearn")
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
            "train_size": int(n * 0.7),
            "test_size": int(n * 0.3),
            "classifier": True,
            "random_state": 42,
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
        # Stub the actual anjana call to avoid dependency on real anjana library
        def _fake_k_anon(data, **kwargs):
            _ = kwargs
            # Return the full frame unchanged (no suppression for simplicity)
            return data.copy()

        monkeypatch.setattr(
            "deckard.plugins.anjana.data.resolve_class",
            lambda _: resolve_class_fn or _fake_k_anon,
        )

    return cfg


def _run_anjana_data(cfg):
    """Call the data config to load and split data."""
    cfg()
    return cfg


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_anjana_data_config_initialises():
    cfg = _make_anjana_data(
        n=20,
        quasi_identifiers=["feature_0"],
        anjana_defense=None,
    )
    assert cfg.quasi_identifiers == ["feature_0"]
    assert cfg.anjana_defense is None


def test_anjana_data_config_hash_stable_before_loading():
    cfg = _make_anjana_data(n=20)
    h1 = hash(cfg)
    h2 = hash(cfg)
    assert h1 == h2


def test_anjana_hierarchy_dict_generates_levels():
    cfg = _make_anjana_data(
        n=10,
        quasi_identifiers=["age", "zip"],
        hierarchy_interval_sizes={"age": [10]},
    )
    frame = pd.DataFrame({"age": [21, 27, 42, 49], "zip": [101, 102, 103, 104]})
    hierarchies = cfg.generate_anjana_hierarchy_dict(frame=frame)

    assert set(hierarchies) == {"age", "zip"}
    assert 0 in hierarchies["age"]
    assert 1 in hierarchies["age"]  # interval level
    assert list(hierarchies["zip"][1]) == ["*", "*", "*", "*"]


def test_anjana_defense_applied_before_split(monkeypatch):
    """Anjana defense must modify _X/_y BEFORE the train/test split occurs."""
    apply_calls = []

    def _tracking_defense(data, **kwargs):
        _ = kwargs
        apply_calls.append(len(data))
        return data.copy()

    cfg = _make_anjana_data(
        n=20,
        monkeypatch=monkeypatch,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
        resolve_class_fn=_tracking_defense,
    )

    cfg()

    # The defense should have been called on the FULL dataset (before split), not just test/train split
    assert (
        len(apply_calls) == 1
    ), f"Expected exactly 1 defense call, got {len(apply_calls)}"
    # Defense was called on the full dataset (20 rows + target col)
    assert (
        apply_calls[0] == 20
    ), f"Expected defense called on all 20 rows, got {apply_calls[0]}"


def test_anjana_defense_transforms_x_and_y_before_split(monkeypatch):
    """After anjana defense, X and y should reflect the transformed frame."""

    def _fake_defense_drop_half(data, **kwargs):
        _ = kwargs
        return data.iloc[: len(data) // 2].copy()

    cfg = _make_anjana_data(
        n=20,
        monkeypatch=monkeypatch,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
        resolve_class_fn=_fake_defense_drop_half,
    )
    cfg()

    # After defense, only 10 rows should remain (half of 20)
    assert len(cfg._X) == 10
    assert len(cfg._y) == 10


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_anjana_experiment_end_to_end(monkeypatch):
    """Full ExperimentConfig run with AnjanaDataConfig and no defense."""
    from deckard.experiment import ExperimentConfig
    from deckard.file import FileConfig
    from deckard.model import ModelConfig
    from deckard.score.base import DefaultClassifierScorerDictConfig

    cfg = _make_anjana_data(n=30)

    exp = ExperimentConfig(
        data=cfg,
        model=ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 50},
        ),
        attack=None,
        files=FileConfig(),
        score=DefaultClassifierScorerDictConfig(),
        classifier=True,
    )

    scores = exp()
    assert isinstance(scores, dict)
    assert "class_count_max" in scores or "accuracy" in scores


def test_anjana_experiment_with_defense(monkeypatch):
    """Full ExperimentConfig run with AnjanaDataConfig and a stubbed defense."""
    from deckard.experiment import ExperimentConfig
    from deckard.file import FileConfig
    from deckard.model import ModelConfig
    from deckard.score.base import DefaultClassifierScorerDictConfig

    def _stub_k_anon(data, **kwargs):
        _ = kwargs
        return data.copy()

    monkeypatch.setattr(
        "deckard.plugins.anjana.data.resolve_class",
        lambda _: _stub_k_anon,
    )

    cfg = _make_anjana_data(
        n=30,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )

    exp = ExperimentConfig(
        data=cfg,
        model=ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 50},
        ),
        attack=None,
        files=FileConfig(),
        score=DefaultClassifierScorerDictConfig(),
        classifier=True,
    )

    scores = exp()
    assert isinstance(scores, dict)
    assert "class_count_max" in scores or "accuracy" in scores


# ---------------------------------------------------------------------------
# Chain tests: anjana + ART
# ---------------------------------------------------------------------------


def test_anjana_data_with_art_model_defense_chain(monkeypatch):
    """ART defense wraps model AFTER anjana data anonymization.

    Data type: AnjanaDataConfig (tabular, sklearn).
    Defense chain: [ART postprocessor only - no Anjana in model defense anymore].
    Verification: type check (model is ART wrapper) + data transform check.
    """
    from deckard.experiment import ExperimentConfig
    from deckard.file import FileConfig
    from deckard.model import DefensePipelineConfig, ModelConfig

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

    defense_cfg = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "art.defences.postprocessor.ClassLabels",
                "defense_params": {"apply_fit": False, "apply_predict": True},
                "classifier": True,
                "model_type": "sklearn.linear_model.LogisticRegression",
            },
        ],
    )
    model_cfg = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 50},
        defense=defense_cfg,
    )

    exp = ExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        attack=None,
        files=FileConfig(),
        classifier=True,
    )

    scores = exp()

    # Type check: model's estimator should be an ART wrapper
    from art.estimators.classification.scikitlearn import (
        ScikitlearnLogisticRegression,
    )

    art_estimator = exp.model._model
    assert isinstance(
        art_estimator,
        ScikitlearnLogisticRegression,
    ), f"Expected ART wrapper, got {type(art_estimator)}"

    # Data transform check: AnjanaDataConfig must have applied defense before split
    assert isinstance(
        data_cfg.X_train,
        pd.DataFrame,
    ), "Expected tabular (DataFrame) X_train from AnjanaDataConfig"
    assert "class_count_max" in scores or "accuracy" in scores


def test_anjana_art_defense_is_applied_last(monkeypatch):
    """When a model is defended via pipeline with only ART, ART runs last.

    This is trivially satisfied when there's only one defense. For mixed chains,
    see test_art_last_ordering tests in test_torch_experiment.py.
    """
    from deckard.model import DefensePipelineConfig

    pipeline = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "art.defences.postprocessor.ClassLabels",
                "defense_params": {"apply_fit": False, "apply_predict": True},
                "classifier": True,
                "model_type": "sklearn.linear_model.LogisticRegression",
            },
        ],
    )
    # Single ART defense: nothing to reorder
    art_defenses = [d for d in pipeline.defenses if pipeline._is_art_defense(d)]
    non_art_defenses = [
        d for d in pipeline.defenses if not pipeline._is_art_defense(d)
    ]
    assert len(art_defenses) == 1
    assert len(non_art_defenses) == 0


def test_anjana_fairness_data_and_art_model_chain(monkeypatch):
    """Combined chain: Anjana pre-split data transform + ART model wrapper.

    Anjana data configs are decoupled from fairlearn runtime caches; this test
    validates Anjana + ART interaction only.
    """
    pytest.importorskip("fairlearn")

    from art.estimators.classification.scikitlearn import (
        ScikitlearnLogisticRegression,
    )

    from deckard.experiment import ExperimentConfig
    from deckard.file import FileConfig
    from deckard.model import DefensePipelineConfig, ModelConfig
    from deckard.plugins.anjana.data import AnjanaDataConfig

    def _drop_half_rows(data, **kwargs):
        _ = kwargs
        return data.iloc[: len(data) // 2].copy()

    monkeypatch.setattr(
        "deckard.plugins.anjana.data.resolve_class",
        lambda _: _drop_half_rows,
    )

    data_cfg = AnjanaDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 60,
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
        anjana_defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
        fairness_defense={
            "name": "fairlearn.preprocessing.CorrelationRemover",
            "step_name": "fairness_correlation_remover",
        },
        pipeline={},
    )

    model_cfg = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 50},
        defense=DefensePipelineConfig(
            defenses=[
                {
                    "defense_name": "art.defences.postprocessor.ClassLabels",
                    "defense_params": {
                        "apply_fit": False,
                        "apply_predict": True,
                    },
                    "classifier": True,
                    "model_type": "sklearn.linear_model.LogisticRegression",
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

    # Type checks
    assert isinstance(exp.data, AnjanaDataConfig)
    assert isinstance(exp.model._model, ScikitlearnLogisticRegression)

    # Data transform checks: Anjana reduced rows before splitting.
    assert len(exp.data._X) == 30
    assert len(exp.data.X_train) + len(exp.data.X_test) == 30
    assert not hasattr(exp.data, "_sensitive_train")
    assert not hasattr(exp.data, "_sensitive_test")
    assert "accuracy" in scores


def test_art_defense_reordered_last_with_warning(caplog):
    """If ART is placed before non-ART defenses, pipeline reorders and logs a warning."""
    import logging

    from deckard.model.defense.base import DefensePipelineConfig

    call_order = []

    class _StubFairnessDefense:
        defense_name = "fairlearn.mock.MockFairnessDefense"
        defense_application_time = 0.0

        def apply_to(self, estimator, data):
            _ = data
            call_order.append("fairness")
            return estimator

    class _StubArtDefense:
        defense_name = "art.mock.MockArtDefense"
        defense_application_time = 0.0

        def apply_to(self, estimator, data):
            _ = data
            call_order.append("art")
            return estimator

    pipeline = DefensePipelineConfig(
        defenses=[_StubArtDefense(), _StubFairnessDefense()],
    )

    with caplog.at_level(logging.WARNING, logger="deckard.model.defense.base"):
        pipeline.apply(estimator=object(), data=object())

    assert call_order == ["fairness", "art"]
    assert any("automatically reordered" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Chain tests: anjana + fairness
# ---------------------------------------------------------------------------


def test_anjana_data_with_fairlearn_model_chain(monkeypatch):
    """Fairlearn model defense + AnjanaDataConfig data anonymization.

    Verification: AnjanaDataConfig present, FairlearnModelConfig used,
    fairlearn defense is not treated as ART.
    """
    pytest.importorskip("fairlearn")

    from deckard.model import DefensePipelineConfig
    from deckard.plugins.anjana.data import AnjanaDataConfig

    def _stub_k_anon(data, **kwargs):
        _ = kwargs
        return data.copy()

    monkeypatch.setattr(
        "deckard.plugins.anjana.data.resolve_class",
        lambda _: _stub_k_anon,
    )

    data_cfg = _make_anjana_data(
        n=60,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    # Data type check
    assert isinstance(data_cfg, AnjanaDataConfig)

    pipeline = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "fairlearn.reductions.ExponentiatedGradient",
                "defense_params": {"constraints": "DemographicParity"},
                "classifier": True,
                "model_type": "sklearn.linear_model.LogisticRegression",
            },
        ],
    )

    # Fairlearn defense must NOT be classified as an ART defense
    from deckard.plugins.fairlearn.model import FairlearnDefenseConfig as FLDef

    assert not pipeline._is_art_defense(
        pipeline.defenses[0],
    ), "Fairlearn defense must not be treated as ART defense"
    assert isinstance(
        pipeline.defenses[0],
        FLDef,
    ), "Defense should have been coerced to FairlearnDefenseConfig"


# ---------------------------------------------------------------------------
# Chain tests: anjana + attack
# ---------------------------------------------------------------------------


def test_anjana_data_with_attack_scoring(monkeypatch):
    """Anonymized data should still allow the attack pipeline to run and score."""
    from deckard.attack import AttackConfig
    from deckard.experiment import ExperimentConfig
    from deckard.file import FileConfig
    from deckard.model import ModelConfig

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
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 50},
    )
    attack_cfg = AttackConfig(
        attack_type="art.attacks.evasion.HopSkipJump",
        attack_params={
            "max_iter": 1,
            "init_eval": 1,
            "max_eval": 2,
            "init_size": 5,
            "norm": 2,
            "targeted": False,
        },
        attack_size=3,
    )

    exp = ExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        attack=attack_cfg,
        files=FileConfig(),
        classifier=True,
    )

    scores = exp()
    assert isinstance(scores, dict)
    assert "accuracy" in scores
    assert "evasion_accuracy" in scores


# ---------------------------------------------------------------------------
# Hash stability tests
# ---------------------------------------------------------------------------


def test_anjana_data_config_hash_stable_after_load(monkeypatch):
    def _stub(data, **kwargs):
        return data.copy()

    monkeypatch.setattr("deckard.plugins.anjana.data.resolve_class", lambda _: _stub)

    cfg = _make_anjana_data(
        n=20,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    h_before = hash(cfg)
    cfg()
    h_after = hash(cfg)
    assert h_before == h_after, "AnjanaDataConfig hash must not change after loading"


def test_anjana_experiment_hash_stable_after_execution(monkeypatch):
    from deckard.experiment import ExperimentConfig
    from deckard.file import FileConfig
    from deckard.model import ModelConfig

    def _stub(data, **kwargs):
        return data.copy()

    monkeypatch.setattr("deckard.plugins.anjana.data.resolve_class", lambda _: _stub)

    cfg = _make_anjana_data(
        n=20,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    model_cfg = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 50},
    )
    exp = ExperimentConfig(
        data=cfg,
        model=model_cfg,
        attack=None,
        files=FileConfig(),
        classifier=True,
    )

    h_before = hash(exp)
    exp()
    h_after = hash(exp)
    assert h_before == h_after, "ExperimentConfig hash must not change after execution"


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


def test_anjana_experiment_scores_persist_to_json(monkeypatch, tmp_path):
    from deckard.experiment import ExperimentConfig
    from deckard.file import FileConfig
    from deckard.model import ModelConfig

    def _stub(data, **kwargs):
        return data.copy()

    monkeypatch.setattr("deckard.plugins.anjana.data.resolve_class", lambda _: _stub)

    score_file = str(tmp_path / "scores.json")
    cfg = _make_anjana_data(
        n=20,
        defense={"name": "anjana.anonymity.k_anonymity", "k": 2},
    )
    model_cfg = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 50},
    )
    file_cfg = FileConfig(score_file=score_file)
    exp = ExperimentConfig(
        data=cfg,
        model=model_cfg,
        attack=None,
        files=file_cfg,
        classifier=True,
    )
    exp()

    assert Path(score_file).exists()
    with open(score_file) as f:
        loaded = json.load(f)
    assert "accuracy" in loaded


# ---------------------------------------------------------------------------
# Subcommand test: deckard optimize in examples/sklearn context
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
def test_deckard_optimize_subcommand_help_in_sklearn_dir():
    """Verify deckard optimize --help runs cleanly from examples/sklearn context."""

    env = make_runtime_env(DECKARD_RC_PATH)
    env["DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "deckard", "optimize", "--help"],
        cwd=str(EXAMPLES_SKLEARN_DIR),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, f"deckard optimize --help failed:\n{result.stderr}"
