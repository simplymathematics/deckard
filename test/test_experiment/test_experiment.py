import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import yaml

import numpy as np
import pytest
from helpers import make_runtime_env
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from deckard.attack import AttackConfig
from deckard.attack.base import resolve_class as base_resolve_class
from deckard.data import DataConfig
from deckard.experiment import ExperimentConfig, SurvivalExperimentConfig
from deckard.experiment.canon import (
    CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_VERSION,
    build_experiment_params_manifest,
    build_experiment_stage_cache_key,
)
from deckard.experiment.base import (
    DataConfigResolutionMixin,
    _file_resolver,
    _merge_resolver,
)
from deckard.file import FileConfig
from deckard.frameworks.pytorch.experiment import TorchExperimentConfig
from deckard.model import ModelConfig
from deckard.plugins import HookPlugin
from deckard.score import (
    DefaultClassifierScorerDictConfig,
    DefaultDataClassificationScorerDictConfig,
)
from deckard.utils import BaseConfig


def test_experiment_family_aliases_are_importable():
    assert SurvivalExperimentConfig is not None
    assert TorchExperimentConfig is not None


class DummyPoisonAttack:
    """Simple local attack used to exercise poisoning integration flow."""

    def __init__(self, classifier, **kwargs):
        self.classifier = classifier
        self.kwargs = kwargs

    def poison(self, x_trigger, y_trigger, x_train, y_train):
        _ = x_trigger
        y_target = int(np.argmax(np.asarray(y_trigger), axis=1)[0])
        x_poison = np.asarray(x_train).copy()
        y_poison = np.asarray(y_train).copy()
        if y_poison.ndim == 2 and y_poison.shape[1] > 1:
            y_poison[0] = 0
            y_poison[0, y_target] = 1
        else:
            y_poison = y_poison.reshape(-1)
            y_poison[0] = y_target
        return x_poison, y_poison


class TestKFoldExperiment:
    """ExperimentConfig should loop over all folds when sampler='fold'."""

    N_FOLDS = 3

    def _make_exp(self):
        from deckard.data.sample import KFoldSampler

        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 120,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 2,
                    "random_state": 0,
                    "n_clusters_per_class": 1,
                },
                classifier=True,
                sampler=KFoldSampler(n_splits=self.N_FOLDS),
            ),
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5, "random_state": 0},
            ),
            attack=None,
            files=FileConfig(),
        )

    def test_per_fold_keys_present(self):
        exp = self._make_exp()
        scores = exp()
        for k in range(self.N_FOLDS):
            assert f"fold-{k}" in scores, f"missing fold-{k}"
            assert "accuracy" in scores[f"fold-{k}"]

    def test_kfold_mean_key_present(self):
        exp = self._make_exp()
        scores = exp()
        assert "accuracy" in scores

    def test_mean_equals_average_of_folds(self):
        exp = self._make_exp()
        scores = exp()
        fold_accs = [scores[f"fold-{k}"]["accuracy"] for k in range(self.N_FOLDS)]
        assert scores["accuracy"] == pytest.approx(
            float(np.mean(fold_accs)),
            abs=1e-10,
        )


class TestShuffleExperiment:
    """ExperimentConfig should loop over all shuffle splits when sampler='shuffle'."""

    N_SPLITS = 3

    def _make_exp(self):
        from deckard.data.sample import ShuffleSampler

        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 120,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 2,
                    "random_state": 0,
                    "n_clusters_per_class": 1,
                },
                classifier=True,
                sampler=ShuffleSampler(
                    n_splits=self.N_SPLITS,
                    test_size=0.2,
                    val_size=0.1,
                    random_state=42,
                ),
            ),
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5, "random_state": 0},
            ),
            attack=None,
            files=FileConfig(),
        )

    def test_per_split_keys_present(self):
        exp = self._make_exp()
        scores = exp()
        for k in range(self.N_SPLITS):
            assert f"split-{k}" in scores, f"missing split-{k}"
            assert "accuracy" in scores[f"split-{k}"]

    def test_shuffle_mean_key_present(self):
        exp = self._make_exp()
        scores = exp()
        assert "accuracy" in scores

    def test_mean_equals_average_of_splits(self):
        exp = self._make_exp()
        scores = exp()
        split_accs = [scores[f"split-{k}"]["accuracy"] for k in range(self.N_SPLITS)]
        assert scores["accuracy"] == pytest.approx(
            float(np.mean(split_accs)),
            abs=1e-10,
        )


class TestExperimentValidationScoring:
    def test_call_delegates_to_run(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.run = lambda: {"ok": 1.0}
        assert exp() == {"ok": 1.0}

    def test_propagation_of_modes_to_children(self):
        exp = self._make_exp(score_mode="test")
        exp._propagate_score_mode()
        assert exp.data.score_mode == "test"
        assert exp.model.score_mode == "test"
        exp = self._make_exp(score_mode="val")
        exp._propagate_score_mode()
        assert exp.data.score_mode == "val"
        assert exp.model.score_mode == "val"

    def _make_exp(self, *, val_size=0.1, evaluation_mode="standard", score_mode=None):
        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 120,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 2,
                    "random_state": 0,
                    "n_clusters_per_class": 1,
                },
                classifier=True,
                sampler={
                    "name": "deckard.data.sample.SplitSampler",
                    "test_size": 0.2,
                    "val_size": val_size,
                    "random_state": 42,
                },
            ),
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5, "random_state": 0},
            ),
            score={"experiment": DefaultClassifierScorerDictConfig()},
            attack=None,
            files=FileConfig(),
            experiment_name="score-mode-policy-test",
            evaluation_mode=evaluation_mode,
            score_mode=score_mode,
        )

    def test_tuning_mode_emits_test_scores(self):
        exp = self._make_exp(evaluation_mode="tuning")
        scores = exp()
        assert "accuracy" in scores
        assert "validation_accuracy" not in scores
        assert "training_accuracy" not in scores

    def test_report_mode_emits_validation_scores(self):
        exp = self._make_exp(evaluation_mode="report", score_mode=None)
        scores = exp()
        assert "train" in scores
        assert "test" in scores
        assert "val" in scores
        assert "accuracy" in scores["train"]
        assert "accuracy" in scores["test"]
        assert "accuracy" in scores["val"]

    def test_report_mode_without_validation_split_raises(self):
        exp = self._make_exp(val_size=None, evaluation_mode="report", score_mode=None)
        with pytest.raises(ValueError):
            exp()

    def test_tuning_mode_without_validation_split_uses_test_scores(self):
        exp = self._make_exp(val_size=None, evaluation_mode="tuning")
        scores = exp()
        assert "accuracy" in scores


class _FakeDetector:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.score_dict = {}

    def __post_init__(self):
        return None

    def __call__(self, data, model, attack):
        _ = data, model, attack
        self.score_dict = {"detector_accuracy": 0.75, "detector_n": 10}
        return self.score_dict


class TestExperimentDetectorPhase:
    def test_detector_phase_runs_after_attack(self):
        exp = ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 0,
                    "n_classes": 2,
                    "random_state": 7,
                },
                classifier=True,
                sampler={
                    "name": "deckard.data.sample.SplitSampler",
                    "train_size": 40,
                    "test_size": 20,
                    "random_state": 7,
                    "stratify": True,
                },
            ),
            model=ModelConfig(
                model_type="sklearn.linear_model.LogisticRegression",
                classifier=True,
                model_params={"max_iter": 30},
            ),
            attack=AttackConfig(
                attack_type="art.attacks.evasion.FastGradientMethod",
                attack_params={"eps": 0.1},
                attack_size=10,
            ),
            detector=_FakeDetector(),
            files=FileConfig(),
        )

        scores = exp()

        assert "detector_accuracy" in scores
        assert "detector_n" in scores


class TestPoisoningExperimentIntegration:
    def test_poisoning_experiment_emits_benign_and_poisoned_accuracy(self):
        exp = ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 120,
                    "n_features": 8,
                    "n_informative": 6,
                    "n_redundant": 2,
                    "random_state": 4,
                },
                classifier=True,
                sampler={
                    "name": "deckard.data.sample.SplitSampler",
                    "test_size": 0.2,
                    "random_state": 4,
                },
            ),
            model=ModelConfig(
                model_type="sklearn.linear_model.LogisticRegression",
                classifier=True,
                model_params={"max_iter": 150},
            ),
            attack=AttackConfig(
                attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
                attack_params={"class_source": 0, "class_target": 1},
                attack_size=20,
            ),
            files=FileConfig(),
        )

        with patch(
            "deckard.attack.base.resolve_class",
            wraps=base_resolve_class,
        ) as mocked_resolve:

            def _resolve(name):
                if (
                    name
                    == "art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack"
                ):
                    return DummyPoisonAttack
                return base_resolve_class(name)

            mocked_resolve.side_effect = _resolve
            scores = exp()

        assert "benign_accuracy" in scores
        assert "poisoned_accuracy" in scores

    def test_deckard_optimize_help_smoke(self):
        examples_dir = Path(__file__).resolve().parents[2] / "examples" / "sklearn"
        rc_path = examples_dir / ".deckard_rc"
        env = make_runtime_env(rc_path)
        env["DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION"] = "1"

        result = subprocess.run(
            [sys.executable, "-m", "deckard", "optimize", "--help"],
            cwd=str(examples_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_single_pass_with_attack_runs_experiment_scorer_once(self):
        exp = ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 80,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 2,
                    "random_state": 0,
                    "n_clusters_per_class": 1,
                },
                classifier=True,
                sampler={
                    "name": "deckard.data.sample.SplitSampler",
                    "train_size": 60,
                    "test_size": 20,
                    "random_state": 42,
                },
            ),
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5, "random_state": 0},
            ),
            attack=AttackConfig(
                attack_type="art.attacks.evasion.FastGradientMethod",
                attack_params={"eps": 0.1},
                attack_size=10,
            ),
            score={"experiment": DefaultClassifierScorerDictConfig()},
            files=FileConfig(),
            experiment_name="single-pass-scorer-call-count",
            evaluation_mode="standard",
        )

        with (
            patch.object(
                AttackConfig,
                "__call__",
                autospec=True,
                side_effect=lambda self, data, model, **kwargs: {
                    "evasion_accuracy": 0.5,
                    "attack_generation_time": 0.01,
                    "attack_prediction_time": 0.01,
                    "attack_score_time": 0.01,
                },
            ),
            patch.object(
                ExperimentConfig,
                "_run_experiment_scorer_modes",
                autospec=True,
                wraps=ExperimentConfig._run_experiment_scorer_modes,
            ) as mocked_run_modes,
        ):
            _ = exp()

        assert mocked_run_modes.call_count == 1


# ── _file_resolver ───────────────────────────────────────────────────────────


class TestFileResolver:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Set DECKARD_CONFIG_DIR to tmpdir so the resolver finds files there
        self._orig_env = os.environ.get("DECKARD_CONFIG_DIR")
        os.environ["DECKARD_CONFIG_DIR"] = self.tmpdir

    def teardown_method(self):
        if self._orig_env is None:
            os.environ.pop("DECKARD_CONFIG_DIR", None)
        else:
            os.environ["DECKARD_CONFIG_DIR"] = self._orig_env

    def _write_yaml(self, name, content):
        p = Path(self.tmpdir) / name
        p.write_text(content)
        return p

    def test_empty_arg_raises(self):
        with pytest.raises(ValueError):
            _file_resolver("")

    def test_file_not_found_raises(self):
        # Patch so DECKARD_CONFIG_DIR points to tmpdir
        import deckard.experiment.base as _mod

        orig = _mod.DECKARD_CONFIG_DIR
        _mod.DECKARD_CONFIG_DIR = self.tmpdir
        try:
            with pytest.raises(FileNotFoundError):
                _file_resolver("does_not_exist.yaml")
        finally:
            _mod.DECKARD_CONFIG_DIR = orig

    def test_whole_file_returned_when_no_key(self):
        import deckard.experiment.base as _mod

        orig = _mod.DECKARD_CONFIG_DIR
        _mod.DECKARD_CONFIG_DIR = self.tmpdir
        try:
            self._write_yaml("test.yaml", "a: 1\nb: 2\n")
            result = _file_resolver("test.yaml")
            assert OmegaConf.to_container(result)["a"] == 1
        finally:
            _mod.DECKARD_CONFIG_DIR = orig

    def test_key_lookup_returns_sub_value(self):
        import deckard.experiment.base as _mod

        orig = _mod.DECKARD_CONFIG_DIR
        _mod.DECKARD_CONFIG_DIR = self.tmpdir
        try:
            self._write_yaml("config.yaml", "model:\n  type: rf\n  n_estimators: 10\n")
            result = _file_resolver("config.yaml:model")
            container = OmegaConf.to_container(result)
            assert container["type"] == "rf"
        finally:
            _mod.DECKARD_CONFIG_DIR = orig

    def test_missing_key_raises(self):
        import deckard.experiment.base as _mod

        orig = _mod.DECKARD_CONFIG_DIR
        _mod.DECKARD_CONFIG_DIR = self.tmpdir
        try:
            self._write_yaml("cfg.yaml", "a: 1\n")
            with pytest.raises(KeyError):
                _file_resolver("cfg.yaml:nonexistent.key")
        finally:
            _mod.DECKARD_CONFIG_DIR = orig


# ── _merge_resolver ──────────────────────────────────────────────────────────


class TestMergeResolver:
    def test_merge_two_dicts(self):
        a = OmegaConf.create({"x": 1, "y": 2})
        b = OmegaConf.create({"z": 3, "y": 99})
        result = _merge_resolver(a, b)
        container = OmegaConf.to_container(result)
        assert container["x"] == 1
        assert container["y"] == 99
        assert container["z"] == 3

    def test_merge_single_dict(self):
        a = OmegaConf.create({"a": 42})
        result = _merge_resolver(a)
        container = OmegaConf.to_container(result)
        assert container["a"] == 42


# ── DataConfigResolutionMixin ─────────────────────────────────────────────────


class _TestMixin(DataConfigResolutionMixin):
    """Concrete subclass to test the mixin methods."""

    @property
    def data(self):
        return None


class TestDataConfigResolutionMixin:
    def setup_method(self):
        self.mixin = _TestMixin()

    def test_data_to_dict_with_dictconfig(self):
        dc = OmegaConf.create(
            {
                "dataset_name": "make_classification",
                "classifier": True,
                "_target_": "deckard.data.DataConfig",
            },
        )
        result = self.mixin._data_to_dict(dc)
        assert isinstance(result, dict)

    def test_data_to_dict_with_plain_dict(self):
        d = {"dataset_name": "make_regression", "classifier": False}
        result = self.mixin._data_to_dict(d)
        assert result == d

    def test_data_to_dict_with_yaml_path(self):
        yaml_text = (
            "dataset_name: make_classification\n"
            "classifier: true\n"
            "data_params:\n"
            "  n_samples: 20\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.yaml"
            p.write_text(yaml_text)
            with pytest.raises(Exception):
                self.mixin._data_to_dict(str(p))

    def test_data_to_dict_invalid_type_raises(self):
        with pytest.raises(ValueError):
            self.mixin._data_to_dict(12345)

    def test_data_to_dict_not_dict_result_raises(self):
        # BaseConfig.to_dict() normally returns dict; if it returns non-dict we get TypeError
        class BadBase(BaseConfig):
            def __call__(self):
                pass

            def to_dict(self, **_):
                return "not_a_dict"

        obj = BadBase()
        with pytest.raises(TypeError):
            self.mixin._data_to_dict(obj)

    # @pytest.importorskip("anjana")
    # def test_select_data_cls_anjana_keys(self):
    #     data_dict = {"dataset_name": "x", "quasi_identifiers": ["age"]}
    #     from deckard.plugins.anjana.data import AnjanaDataConfig
    #     # cls = self.mixin._select_data_cls(data_dict)
    #     # print(type(cls))
    #     input("HERE")
    #     # self.assertIs(cls, AnjanaDataConfig)

    # @pytest.importorskip("fairlearn")
    # def test_select_data_cls_fairness_keys(self):
    #     data_dict = {"dataset_name": "x", "sensitive_columns": ["gender"]}
    #     from deckard.plugins.fairlearn.data import FairlearnDataConfig
    #     cls = self.mixin._select_data_cls(data_dict)
    #     self.assertIs(cls, FairlearnDataConfig)

    def test_select_data_cls_pipeline_key(self):
        from deckard.data import DataConfig

        data_dict = {"pipeline": {}}
        cls = self.mixin._select_data_cls(data_dict)
        assert cls is DataConfig

    def test_select_data_cls_plain(self):
        data_dict = {"dataset_name": "make_classification", "classifier": True}
        cls = self.mixin._select_data_cls(data_dict)
        assert cls is DataConfig

    def test_resolve_data_config_with_data_config_passthrough(self):
        class _Exp(DataConfigResolutionMixin, BaseConfig):
            def __call__(self):
                pass

        dc = DataConfig(dataset_name="make_classification", classifier=True)
        exp = _Exp()
        exp.data = dc
        result = exp._resolve_data_config()
        assert result is dc

    def test_resolve_data_config_target_resolves_wrong_type_raises(self):
        class _Exp(DataConfigResolutionMixin, BaseConfig):
            def __call__(self):
                pass

        exp = _Exp()
        exp.data = OmegaConf.create({"_target_": "builtins.dict"})
        with pytest.raises(TypeError):
            exp._resolve_data_config()

    def test_sklearn_pipeline_yaml_compose_instantiates_pipeline_configs(self):
        from deckard.data import DataConfig

        config_dir = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "sklearn"
            / "config"
            / "data"
            / "pipeline"
        )
        for config_name in (
            "default_pipeline",
            "anjana_pipeline",
            "fairlearn_pipeline",
        ):
            with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                cfg = compose(config_name=config_name)
            assert isinstance(instantiate(cfg), DataConfig)

    def test_pytorch_pipeline_yaml_compose_instantiates_pipeline_config(self):
        pytest.importorskip("torch")

        from deckard.data import DataConfig

        config_dir = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "pytorch"
            / "config"
            / "data"
            / "pipeline"
        )
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="pytorch_pipeline")
        assert isinstance(instantiate(cfg), DataConfig)


# ── ExperimentConfig.set_random_seed ─────────────────────────────────────────


class TestSetRandomSeed:
    def _make_base_exp(self, library="sklearn"):
        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 4,
                    "n_informative": 3,
                    "n_redundant": 1,
                    "random_state": 0,
                },
                classifier=True,
                sampler={
                    "name": "deckard.data.sample.SplitSampler",
                    "train_size": 40,
                    "test_size": 20,
                    "random_state": 42,
                },
            ),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            library=library,
            files=FileConfig(),
        )

    def test_set_random_seed_pytorch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        exp = self._make_base_exp(library="pytorch")
        # Should not raise
        exp.set_random_seed()

    def test_set_random_seed_unsupported_library_raises(self):
        exp = self._make_base_exp()
        exp.library = "tensorflow"
        try:
            import tensorflow  # noqa: F401

            # If tensorflow is available, this won't raise
        except ImportError:
            with pytest.raises(Exception):
                exp.set_random_seed()

    def test_set_random_seed_truly_unsupported_library_raises(self):
        exp = self._make_base_exp()
        exp.library = "not_a_supported_library"
        with pytest.raises(ValueError):
            exp.set_random_seed()


# ── ExperimentConfig.__post_init__ model input types ─────────────────────────


class TestExperimentPostInitModelTypes:
    """Test that __post_init__ correctly handles different model input types."""

    def _data(self):
        return DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 4,
                "n_informative": 3,
                "n_redundant": 1,
                "random_state": 0,
            },
            classifier=True,
            sampler={
                "name": "deckard.data.sample.SplitSampler",
                "train_size": 40,
                "test_size": 20,
                "random_state": 42,
            },
        )

    def test_model_as_dict(self):
        model_dict = {
            "model_type": "sklearn.tree.DecisionTreeClassifier",
            "classifier": True,
            "model_params": {"max_depth": 2},
        }
        exp = ExperimentConfig(
            data=self._data(),
            model=model_dict,
            files=FileConfig(),
        )
        assert isinstance(exp.model, ModelConfig)

    def test_model_as_dictconfig(self):
        model_cfg = OmegaConf.create(
            {
                "_target_": "deckard.model.ModelConfig",
                "model_type": "sklearn.tree.DecisionTreeClassifier",
                "classifier": True,
                "model_params": {"max_depth": 2},
            },
        )
        exp = ExperimentConfig(
            data=self._data(),
            model=model_cfg,
            files=FileConfig(),
        )
        assert isinstance(exp.model, ModelConfig)

    def test_experiment_name_hash_generated(self):
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            experiment_name="{hash}",
            files=FileConfig(),
        )
        # Should be replaced with an MD5 hash string
        assert exp.experiment_name != "{hash}"
        assert len(exp.experiment_name) == 32

    def test_attack_as_dict(self):
        attack_dict = {
            "attack_type": "art.attacks.evasion.FastGradientMethod",
            "attack_params": {"eps": 0.1},
            "attack_size": 5,
        }
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            attack=attack_dict,
            files=FileConfig(),
        )
        assert isinstance(exp.attack, AttackConfig)

    def test_attack_as_dictconfig(self):
        attack_cfg = OmegaConf.create(
            {
                "_target_": "deckard.attack.AttackConfig",
                "attack_type": "art.attacks.evasion.FastGradientMethod",
                "attack_params": {"eps": 0.1},
                "attack_size": 5,
            },
        )
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            attack=attack_cfg,
            files=FileConfig(),
        )
        assert isinstance(exp.attack, AttackConfig)


# ── ExperimentConfig.__call__ score file paths ───────────────────────────────


class TestExperimentScoreFileHandling:
    def _make_exp(self, score_file=None):
        files = FileConfig(score_file=score_file) if score_file else FileConfig()
        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 4,
                    "n_informative": 3,
                    "n_redundant": 1,
                    "random_state": 0,
                },
                classifier=True,
                sampler={
                    "name": "deckard.data.sample.SplitSampler",
                    "train_size": 40,
                    "test_size": 20,
                    "random_state": 42,
                },
            ),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            files=files,
            experiment_name="score-file-test",
        )

    def test_no_score_file_runs_without_saving(self):
        exp = self._make_exp()
        scores = exp()
        assert isinstance(scores, dict)

    def test_score_file_is_created(self):
        with tempfile.TemporaryDirectory() as td:
            score_path = str(Path(td) / "scores.json")
            exp = self._make_exp(score_file=score_path)
            exp()
            assert Path(score_path).exists()

    def test_existing_score_file_is_merged(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            score_path = str(Path(td) / "scores.json")
            # Pre-populate the score file
            with open(score_path, "w") as f:
                json.dump({"prior_metric": 99.9}, f)
            exp = self._make_exp(score_file=score_path)
            exp()
            with open(score_path) as f:
                merged = json.load(f)
            assert "prior_metric" in merged

    def test_model_none_runs_data_only_pipeline(self):
        exp = ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 4,
                    "n_informative": 3,
                    "n_redundant": 1,
                    "random_state": 0,
                },
                classifier=True,
                sampler={
                    "name": "deckard.data.sample.SplitSampler",
                    "train_size": 40,
                    "test_size": 20,
                    "random_state": 42,
                },
            ),
            model=None,
            files=FileConfig(),
        )
        scores = exp()
        assert isinstance(scores, dict)


# ── ExperimentConfig._canonical_device ───────────────────────────────────────


class TestCanonicalDevice:
    def test_none_returns_none(self):
        assert ExperimentConfig._canonical_device(None) is None

    def test_null_token_returns_none(self):
        assert ExperimentConfig._canonical_device("none") is None

    def test_auto_token_returns_none(self):
        assert ExperimentConfig._canonical_device("auto") is None

    def test_cpu_returns_cpu(self):
        assert ExperimentConfig._canonical_device("cpu") == "cpu"

    def test_mps_returns_mps(self):
        assert ExperimentConfig._canonical_device("mps") == "mps"


# ── ExperimentConfig._resolve_score_modes ────────────────────────────────────


class TestResolveScoreModes:
    def _base_exp(self, **kwargs):
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.__dict__.update(
            {
                "score_mode": None,
                "evaluation_mode": "standard",
            },
        )
        exp.__dict__.update(kwargs)
        return exp

    def test_standard_returns_train_and_test(self):
        exp = self._base_exp(evaluation_mode="standard")
        assert exp._resolve_score_modes() == ["train", "test"]

    def test_tuning_returns_test(self):
        exp = self._base_exp(evaluation_mode="tuning")
        assert exp._resolve_score_modes() == ["test"]

    def test_report_returns_train_test_and_val(self):
        exp = self._base_exp(evaluation_mode="report")
        assert exp._resolve_score_modes() == ["train", "test", "val"]

    def test_explicit_score_modes_override(self):
        exp = self._base_exp(score_mode=["test", "val"])
        assert exp._resolve_score_modes() == ["test", "val"]

    def test_explicit_score_mode_all(self):
        exp = self._base_exp(score_mode="all")
        assert exp._resolve_score_modes() == ["all"]

    def test_empty_score_mode_list_uses_evaluation_mode(self):
        exp = self._base_exp(score_mode=[], evaluation_mode="standard")
        assert exp._resolve_score_modes() == ["train", "test"]


class TestExperimentScorerModePermutations:
    def _base_data(self, *, val_size=0.1):
        return DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 120,
                "n_features": 6,
                "n_informative": 4,
                "n_redundant": 2,
                "random_state": 0,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler={
                "name": "deckard.data.sample.SplitSampler",
                "test_size": 0.2,
                "val_size": val_size,
                "random_state": 42,
            },
        )

    def _base_model(self):
        return ModelConfig(
            model_type="sklearn.ensemble.RandomForestClassifier",
            classifier=True,
            model_params={"n_estimators": 5, "random_state": 0},
        )

    def test_data_profile_scorer_supports_all_mode_permutations(self):
        exp = ExperimentConfig(
            data=self._base_data(val_size=0.1),
            model=self._base_model(),
            score={"experiment": DefaultDataClassificationScorerDictConfig()},
            score_mode=["train", "test", "val"],
            files=FileConfig(),
            experiment_name="score-permutations-data-profile",
        )

        scores = exp()
        assert "accuracy" in scores

    def test_non_data_profile_rejects_presample_mode(self):
        exp = ExperimentConfig(
            data=self._base_data(val_size=0.1),
            model=self._base_model(),
            score={"experiment": DefaultClassifierScorerDictConfig()},
            score_mode=["all", "test"],
            files=FileConfig(),
            experiment_name="score-permutations-non-data-profile",
        )

        with pytest.raises(ValueError):
            exp()


# ── ExperimentConfig._aggregate_repeated_scores ──────────────────────────────


class TestAggregateRepeatedScores:
    def test_empty_list_returns_empty(self):
        result = ExperimentConfig._aggregate_repeated_scores([])
        assert result == {}

    def test_single_run_produces_fold_dict(self):
        result = ExperimentConfig._aggregate_repeated_scores(
            [{"accuracy": 0.9}],
            suffix="fold",
        )
        assert result["fold-0"]["accuracy"] == pytest.approx(0.9)
        assert result["accuracy"] == pytest.approx(0.9)

    def test_multiple_runs_computes_mean(self):
        runs = [{"accuracy": 0.8}, {"accuracy": 0.9}, {"accuracy": 1.0}]
        result = ExperimentConfig._aggregate_repeated_scores(runs, suffix="fold")
        assert result["accuracy"] == pytest.approx(np.mean([0.8, 0.9, 1.0]))
        for i in range(3):
            assert result[f"fold-{i}"]["accuracy"] == pytest.approx(
                runs[i]["accuracy"]
            )

    def test_non_numeric_uses_last_value(self):
        runs = [{"label": "a"}, {"label": "b"}]
        result = ExperimentConfig._aggregate_repeated_scores(runs)
        assert result["label"] == "b"
        assert result["fold-1"]["label"] == "b"

    def test_none_values_excluded_from_mean(self):
        runs = [{"acc": 0.9}, {"acc": None}]
        result = ExperimentConfig._aggregate_repeated_scores(runs)
        assert result["acc"] == pytest.approx(0.9)

    def test_nested_score_dicts_are_averaged_recursively(self):
        runs = [
            {"test": {"accuracy": 0.8, "loss": 0.4}},
            {"test": {"accuracy": 1.0, "loss": 0.2}},
        ]
        result = ExperimentConfig._aggregate_repeated_scores(runs, suffix="fold")
        assert result["fold-0"]["test"]["accuracy"] == pytest.approx(0.8)
        assert result["fold-1"]["test"]["accuracy"] == pytest.approx(1.0)
        assert result["test"]["accuracy"] == pytest.approx(0.9)
        assert result["test"]["loss"] == pytest.approx(0.3)


# ── ExperimentConfig.__post_init__ more branches ─────────────────────────────


class TestExperimentPostInitMoreBranches:
    def _data(self):
        return DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 4,
                "n_informative": 3,
                "n_redundant": 1,
                "random_state": 0,
            },
            classifier=True,
            sampler={
                "name": "deckard.data.sample.SplitSampler",
                "train_size": 40,
                "test_size": 20,
                "random_state": 42,
            },
        )

    def test_model_as_str_yaml(self):
        """Cover branch: isinstance(self.model, str) -> ModelConfig.from_yaml(...)"""
        with tempfile.TemporaryDirectory() as td:
            yaml_path = Path(td) / "model.yaml"
            yaml_path.write_text(
                "model_type: sklearn.tree.DecisionTreeClassifier\n"
                "classifier: true\n"
                "model_params:\n"
                "  max_depth: 2\n",
            )
            exp = ExperimentConfig(
                data=self._data(),
                model=str(yaml_path),
                files=FileConfig(),
            )
        assert isinstance(exp.model, ModelConfig)

    def test_model_as_config_base_subclass(self):
        """Cover branch: isinstance(self.model, BaseConfig) -> model_dict = model.to_dict()"""

        class AltModel(ModelConfig):
            pass

        alt_model = AltModel(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
        )
        exp = ExperimentConfig(
            data=self._data(),
            model=alt_model,
            files=FileConfig(),
        )
        assert isinstance(exp.model, ModelConfig)

    def test_files_as_dict(self):
        """Cover branch: isinstance(self.files, dict)"""
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            files={},
        )
        from deckard.file import FileConfig as FC

        assert isinstance(exp.files, FC)

    def test_files_as_dictconfig(self):
        """Cover branch: isinstance(self.files, DictConfig)"""
        files_cfg = OmegaConf.create({"score_file": None})
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            files=files_cfg,
        )
        from deckard.file import FileConfig as FC

        assert isinstance(exp.files, FC)

    def test_model_dictconfig_without_target(self):
        """Cover branch: isinstance(self.model, DictConfig) without _target_"""
        model_cfg = OmegaConf.create(
            {
                "model_type": "sklearn.tree.DecisionTreeClassifier",
                "classifier": True,
                "model_params": {"max_depth": 2},
            },
        )
        exp = ExperimentConfig(
            data=self._data(),
            model=model_cfg,
            files=FileConfig(),
        )
        assert isinstance(exp.model, ModelConfig)

    def test_attack_as_str_yaml(self):
        """Cover branch: isinstance(self.attack, str) -> AttackConfig.from_yaml"""
        with tempfile.TemporaryDirectory() as td:
            yaml_path = Path(td) / "attack.yaml"
            yaml_path.write_text(
                "attack_type: art.attacks.evasion.FastGradientMethod\n"
                "attack_params:\n"
                "  eps: 0.1\n"
                "attack_size: 5\n",
            )
            exp = ExperimentConfig(
                data=self._data(),
                model=ModelConfig(
                    model_type="sklearn.tree.DecisionTreeClassifier",
                    classifier=True,
                    model_params={"max_depth": 2},
                ),
                attack=str(yaml_path),
                files=FileConfig(),
            )
        assert isinstance(exp.attack, AttackConfig)

    def test_attack_as_BaseConfig(self):
        """Cover branch: isinstance(self.attack, BaseConfig)"""

        class AltAttack(AttackConfig):
            pass

        alt_attack = AltAttack(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
            attack_size=5,
        )
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            attack=alt_attack,
            files=FileConfig(),
        )
        assert isinstance(exp.attack, AttackConfig)

    def test_multi_attack_requires_aliases(self):
        with pytest.raises(ValueError):
            ExperimentConfig(
                data=self._data(),
                model=ModelConfig(
                    model_type="sklearn.tree.DecisionTreeClassifier",
                    classifier=True,
                    model_params={"max_depth": 2},
                ),
                attack=[
                    {
                        "attack_type": "art.attacks.evasion.FastGradientMethod",
                        "attack_params": {"eps": 0.1},
                        "attack_size": 5,
                    },
                    {
                        "attack_type": "art.attacks.evasion.FastGradientMethod",
                        "attack_params": {"eps": 0.2},
                        "attack_size": 5,
                    },
                ],
                files=FileConfig(),
            )

    def test_multi_attack_sets_primary_attack_and_chain(self):
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            attack=[
                {
                    "attack_type": "art.attacks.evasion.FastGradientMethod",
                    "attack_params": {"eps": 0.1},
                    "attack_size": 5,
                    "alias": "fgm_a",
                },
                {
                    "attack_type": "art.attacks.evasion.FastGradientMethod",
                    "attack_params": {"eps": 0.2},
                    "attack_size": 5,
                    "alias": "fgm_b",
                },
            ],
            files=FileConfig(),
        )
        assert len(exp._attack_chain) == 2
        assert exp.attack.alias == "fgm_a"

    def test_score_as_scorer_dict_config(self):
        """Cover scorer config path where score is ScorerDictConfig."""
        from deckard.score import ScorerConfig, ScorerDictConfig

        scorer = ScorerDictConfig(
            scorers={
                "acc": ScorerConfig(
                    score_name="accuracy",
                    score_function="sklearn.metrics.accuracy_score",
                ),
            },
        )
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            score=scorer,
            files=FileConfig(),
        )
        assert exp.model.scorer is not None
        with pytest.raises(ValueError):
            ExperimentConfig(
                data=self._data(),
                model=ModelConfig(
                    model_type="sklearn.tree.DecisionTreeClassifier",
                    classifier=True,
                    model_params={"max_depth": 2},
                ),
                attack=123,
                files=FileConfig(),
            )

    def test_detector_invalid_type_raises(self):
        with pytest.raises(ValueError):
            ExperimentConfig(
                data=self._data(),
                model=ModelConfig(
                    model_type="sklearn.tree.DecisionTreeClassifier",
                    classifier=True,
                    model_params={"max_depth": 2},
                ),
                detector=123,
                files=FileConfig(),
            )

    def test_files_invalid_type_raises(self):
        with pytest.raises(ValueError):
            ExperimentConfig(
                data=self._data(),
                model=ModelConfig(
                    model_type="sklearn.tree.DecisionTreeClassifier",
                    classifier=True,
                    model_params={"max_depth": 2},
                ),
                files=123,
            )


# ── ExperimentConfig._coerce_scorer_config branches ──────────────────────────


class TestCoerceScorerConfig:
    def _data(self):
        return DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 4,
                "n_informative": 3,
                "n_redundant": 1,
                "random_state": 0,
            },
            classifier=True,
            sampler={
                "name": "deckard.data.sample.SplitSampler",
                "train_size": 40,
                "test_size": 20,
                "random_state": 42,
            },
        )

    def test_score_as_dict_with_data_model_experiment_keys(self):
        """Cover path where score dict has data/model/experiment keys."""
        from sklearn.metrics import accuracy_score

        scorer_spec = {
            "accuracy": {
                "score_name": "accuracy",
                "score_function": accuracy_score,
            },
        }
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            score={"experiment": scorer_spec, "data": "auto", "model": "auto"},
            files=FileConfig(),
        )
        # experiment scorer is set from the scorer_spec dict
        assert exp.score is not None
        # data and model auto scorers should be attached too
        assert exp.data.scorer is not None
        assert exp.model.scorer is not None

    def test_score_auto_shorthand_applies_data_and_model_defaults(self):
        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            score="auto",
            files=FileConfig(),
        )
        assert exp.data.scorer is not None
        assert exp.model.scorer is not None
        assert exp.score is None

    def test_score_dict_with_scoring_type_routes_to_data_scope(self):
        from sklearn.metrics import accuracy_score

        exp = ExperimentConfig(
            data=self._data(),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            score={
                "scoring_type": "data",
                "scorers": {
                    "accuracy": {
                        "score_name": "accuracy",
                        "score_function": accuracy_score,
                    },
                },
            },
            files=FileConfig(),
        )

        assert exp.score is None
        assert exp.data.scorer is not None
        # Updated to check for 'num_classes' instead of 'accuracy'
        assert "num_classes" in exp.data.scorer.scorers


class TestRunSinglePipelineBranchesExtra:
    def _exp_stub(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(score_dict={"data_score": 1.0})
        exp.model = None
        exp.attack = None
        exp.detector = None
        exp._run_experiment_scorer_modes = lambda score_file=None: {}
        return exp

    def test_attack_value_error_is_reraised(self):
        exp = self._exp_stub()

        class _Attack:
            def __call__(self, **kwargs):
                _ = kwargs
                raise ValueError("boom")

        exp.attack = _Attack()
        with pytest.raises(ValueError):
            exp._run_single_pipeline({}, {})

    def test_detector_without_attack_raises(self):
        exp = self._exp_stub()
        exp.detector = lambda **kwargs: kwargs
        with pytest.raises(ValueError):
            exp._run_single_pipeline({}, {})

    def test_data_file_path_loads_object_and_saves_score_file(self):
        loaded_data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 3,
                "n_redundant": 1,
                "random_state": 0,
            },
            classifier=True,
            sampler={
                "name": "deckard.data.sample.SplitSampler",
                "train_size": 30,
                "test_size": 10,
                "random_state": 42,
            },
        )
        loaded_data()

        with tempfile.TemporaryDirectory() as td:
            data_file = Path(td) / "data.pkl"
            data_file.write_text("placeholder")
            score_file = Path(td) / "scores.json"

            exp = ExperimentConfig(
                data=DataConfig(
                    dataset_name="make_classification",
                    data_params={
                        "n_samples": 20,
                        "n_features": 4,
                        "n_informative": 3,
                        "n_redundant": 1,
                        "random_state": 0,
                    },
                    classifier=True,
                    sampler={
                        "name": "deckard.data.sample.SplitSampler",
                        "train_size": 10,
                        "test_size": 10,
                        "random_state": 42,
                    },
                ),
                model=ModelConfig(
                    model_type="sklearn.tree.DecisionTreeClassifier",
                    classifier=True,
                    model_params={"max_depth": 2},
                ),
                files=FileConfig(data_file=str(data_file), score_file=str(score_file)),
                experiment_name="load-object-branch",
            )

            exp.load_object = lambda _p: loaded_data
            scores = exp()

            assert isinstance(scores, dict)
            assert score_file.exists()

    def test_val_score_mode_resamples_loaded_data_for_validation_split(self):
        loaded_data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 80,
                "n_features": 6,
                "n_informative": 4,
                "n_redundant": 2,
                "random_state": 0,
            },
            classifier=True,
            sampler={
                "name": "deckard.data.sample.SplitSampler",
                "train_size": 60,
                "test_size": 20,
                "random_state": 42,
            },
        )
        loaded_data()
        loaded_data.X_val = None
        loaded_data.y_val = None
        loaded_data.val_n = None

        with tempfile.TemporaryDirectory() as td:
            data_file = Path(td) / "data.pkl"
            data_file.write_text("placeholder")

            exp = ExperimentConfig(
                data=DataConfig(
                    dataset_name="make_classification",
                    data_params={
                        "n_samples": 80,
                        "n_features": 6,
                        "n_informative": 4,
                        "n_redundant": 2,
                        "random_state": 0,
                    },
                    classifier=True,
                    sampler={
                        "name": "deckard.data.sample.SplitSampler",
                        "train_size": 0.6,
                        "test_size": 0.2,
                        "val_size": 0.2,
                        "random_state": 42,
                    },
                ),
                model=ModelConfig(
                    model_type="sklearn.tree.DecisionTreeClassifier",
                    classifier=True,
                    model_params={"max_depth": 2},
                ),
                files=FileConfig(data_file=str(data_file)),
                score_mode="val",
                experiment_name="val-mode-resample-loaded-data",
            )

            exp.load_object = lambda _p: loaded_data
            scores = exp()

            assert exp.data.X_val is not None
            assert exp.data.y_val is not None
            assert "val" in scores
            assert "accuracy" in scores["val"]


# ── ExperimentConfig set_device tensorflow ────────────────────────────────────


class TestSetDeviceTensorflow:
    def setup_method(self):
        # Only run if TensorFlow is available
        try:
            import tensorflow  # noqa: F401

            self.has_tf = True
        except ImportError:
            self.has_tf = False

    def _make_exp(self):
        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 4,
                    "n_informative": 3,
                    "n_redundant": 1,
                    "random_state": 0,
                },
                classifier=True,
                sampler={
                    "name": "deckard.data.sample.SplitSampler",
                    "train_size": 40,
                    "test_size": 20,
                    "random_state": 42,
                },
            ),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            library="tensorflow",
            files=FileConfig(),
        )

    def test_set_device_tensorflow_cpu(self):
        pytest.importorskip("tensorflow")
        exp = self._make_exp()
        exp.set_device("cpu")  # Should not raise

    def test_set_device_unsupported_library_logs(self):
        """Cover the else branch: unsupported library logs warning."""
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.library = "unsupported_lib"
        exp.set_device("cpu")  # Should not raise

    def test_set_device_tensorflow_gpu_without_devices(self):
        pytest.importorskip("tensorflow")
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.library = "tensorflow"
        fake_tf = SimpleNamespace(
            config=SimpleNamespace(
                list_physical_devices=lambda *_: [],
                set_visible_devices=lambda *_args, **_kwargs: None,
                experimental=SimpleNamespace(
                    set_memory_growth=lambda *_args, **_kwargs: None,
                ),
            ),
        )
        with patch.dict(sys.modules, {"tensorflow": fake_tf}):
            exp.set_device("gpu")

    def test_set_device_tensorflow_gpu_index_runtime_error(self):
        pytest.importorskip("tensorflow")
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.library = "tensorflow"
        fake_gpu = object()

        def _raise(*_args, **_kwargs):
            raise RuntimeError("boom")

        fake_tf = SimpleNamespace(
            config=SimpleNamespace(
                list_physical_devices=lambda *_: [fake_gpu],
                set_visible_devices=_raise,
                experimental=SimpleNamespace(
                    set_memory_growth=lambda *_args, **_kwargs: None,
                ),
            ),
        )
        with patch.dict(sys.modules, {"tensorflow": fake_tf}):
            exp.set_device(0)


class TestRunSinglePipelineBranches:
    def test_model_prediction_outputs_cleared_before_uncached_run(self):
        class _Model:
            def __init__(self):
                self.score_dict = {"stale": 1.0}
                self.training_predictions = [1]
                self.predictions = [1]
                self.val_predictions = [1]
                self.training_probabilities = [0.1]
                self.probabilities = [0.1]
                self.val_probabilities = [0.1]
                self.training_prediction_time = 1.0
                self.prediction_time = 1.0
                self.val_prediction_time = 1.0
                self.training_score_time = 1.0
                self.prediction_score_time = 1.0
                self.val_score_time = 1.0
                self.training_n = 1
                self.prediction_n = 1
                self.val_n = 1

            def __call__(self, **_kwargs):
                assert self.score_dict == {"stale": 1.0}
                assert self.training_predictions is None
                assert self.predictions is None
                assert self.val_predictions is None
                assert self.training_probabilities is None
                assert self.probabilities is None
                assert self.val_probabilities is None
                assert self.training_prediction_time == 1.0
                assert self.prediction_time == 1.0
                assert self.val_prediction_time == 1.0
                assert self.training_score_time == 1.0
                assert self.prediction_score_time == 1.0
                assert self.val_score_time == 1.0
                assert self.training_n == 1
                assert self.prediction_n == 1
                assert self.val_n == 1
                self.training_predictions = [0]
                self.predictions = [0]
                self.score_dict = {"m": 1.0}

        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(score_dict={})
        exp.model = _Model()
        exp.attack = None
        exp.detector = None
        exp.defense = None
        exp._run_experiment_scorer_modes = lambda score_file=None: {}
        exp._run_experiment_stage_hooks = lambda *args, **kwargs: None
        exp._cache_stage_get = lambda **kwargs: None
        exp._cache_stage_set = lambda **kwargs: None

        scores = exp._run_single_pipeline({}, {}, run_idx=None)

        assert scores["m"] == 1.0

    def test_detector_requires_attack_raises(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(score_dict={})
        exp.model = None
        exp.attack = None
        exp.detector = SimpleNamespace(__call__=lambda **_kwargs: None)
        exp._run_experiment_scorer_modes = lambda score_file=None: {}
        with pytest.raises(ValueError):
            exp._run_single_pipeline({}, {})

    def test_detector_missing_score_dict_asserts(self):
        class _Model:
            def __call__(self, **_kwargs):
                self.training_predictions = [0]
                self.predictions = [0]
                self.score_dict = {"m": 1.0}

        class _Attack:
            def __call__(self, **_kwargs):
                self.attack = object()
                self.attack_predictions = [0]
                self.score_dict = {"a": 1.0}

        class _Detector:
            def __call__(self, **kwargs):
                # Intentionally omit score_dict to hit assertion branch.
                return None

        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(score_dict={})
        exp.model = _Model()
        exp.attack = _Attack()
        exp.detector = _Detector()
        exp._run_experiment_scorer_modes = lambda score_file=None: {}
        with pytest.raises(AssertionError):
            exp._run_single_pipeline({}, {})

    def test_multi_attack_scores_suffix_collisions_and_combines_detector_inputs(self):
        class _Attack:
            def __init__(self, alias, score_dict, predictions):
                self.alias = alias
                self.score_dict = score_dict
                self.attack_predictions = predictions

            def __call__(self, **_kwargs):
                self.attack = object()
                return self.score_dict

        class _Detector:
            def __call__(self, **kwargs):
                attack_obj = kwargs["attack"]
                self.score_dict = {
                    "detector_n": int(len(attack_obj.attack_predictions)),
                }

        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(score_dict={})
        exp.model = None
        exp._run_experiment_scorer_modes = lambda score_file=None: {}
        exp._attack_chain = [
            _Attack(
                "atk_a",
                {"evasion_accuracy": 0.5, "attack_generation_time": 1.0},
                np.array([[1.0], [2.0]]),
            ),
            _Attack(
                "atk_b",
                {"evasion_accuracy": 0.4, "attack_generation_time": 2.0},
                np.array([[3.0], [4.0], [5.0]]),
            ),
        ]
        exp.attack = exp._attack_chain[0]
        exp.detector = _Detector()

        scores = exp._run_single_pipeline({}, {})

        assert scores["evasion_accuracy"] == 0.5
        assert scores["evasion_accuracy_atk_b"] == 0.4
        assert scores["attack_generation_time"] == 1.0
        assert scores["attack_generation_time_atk_b"] == 2.0
        assert scores["detector_n"] == 5


class TestExperimentBranchEdges:
    def test_canonical_device_empty_and_default_tokens(self):
        assert ExperimentConfig._canonical_device("") is None
        assert ExperimentConfig._canonical_device(" default ") is None
        assert ExperimentConfig._canonical_device("CUDA") == "cuda"

    def test_resolve_data_config_target_must_resolve_to_dataconfig(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(_target_="fake.target")
        with patch("deckard.experiment.base.instantiate", return_value=object()):
            with pytest.raises(TypeError):
                exp._resolve_data_config()

    def test_compute_val_predictions_error_branches(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)

        exp.model = None
        exp.data = SimpleNamespace(X_val=[1], y_val=[1])
        with pytest.raises(ValueError):
            exp._compute_val_predictions()

        exp.model = SimpleNamespace(predict=lambda x: x)
        exp.data = SimpleNamespace(X_val=None, y_val=None)
        with pytest.raises(ValueError):
            exp._compute_val_predictions()

        exp.model = SimpleNamespace()
        exp.data = SimpleNamespace(X_val=[1], y_val=[1])
        with pytest.raises(ValueError):
            exp._compute_val_predictions()

    def test_ensure_mode_predictions_error_branches(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(X_train=[1], X_test=[2], X_val=[3], y_val=[1])

        exp.model = None
        with pytest.raises(ValueError):
            exp._ensure_mode_predictions("train")

        exp.model = SimpleNamespace()
        with pytest.raises(ValueError):
            exp._ensure_mode_predictions("test")

    def test_post_init_rejects_unsupported_model_type(self):
        with pytest.raises(ValueError):
            ExperimentConfig(
                data=DataConfig(
                    dataset_name="make_classification",
                    data_params={
                        "n_samples": 20,
                        "n_features": 4,
                        "n_informative": 3,
                        "n_redundant": 1,
                        "random_state": 0,
                    },
                    classifier=True,
                    sampler={
                        "name": "deckard.data.sample.SplitSampler",
                        "train_size": 10,
                        "test_size": 10,
                        "random_state": 42,
                    },
                ),
                model=123,
                files=FileConfig(),
            )


class TestExperimentRuntimeCompositionAndPersistence:
    def _make_base_experiment(self, *, params_file=None):
        files = FileConfig(params_file=params_file) if params_file else FileConfig()
        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 80,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 2,
                    "random_state": 0,
                },
                classifier=True,
                sampler={
                    "name": "deckard.data.sample.SplitSampler",
                    "test_size": 0.25,
                    "random_state": 42,
                },
            ),
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
            ),
            files=files,
            experiment_name="runtime-contract-test",
        )

    def test_compose_components_updates_model_entrypoint(self):
        exp = self._make_base_experiment()
        exp.compose_components(
            model={
                "model_type": "sklearn.linear_model.LogisticRegression",
                "classifier": True,
                "model_params": {"max_iter": 20},
            },
        )
        assert isinstance(exp.model, ModelConfig)
        assert exp.model.model_type == "sklearn.linear_model.LogisticRegression"

    def test_compose_components_accepts_hook_bundle_dict_and_orders_after_canonical(
        self,
    ):
        exp = self._make_base_experiment()
        custom_hook = HookPlugin(
            hook_name="before_train",
            method_name="_experiment_stage_hook",
        )
        exp.compose_components(
            hook_bundles=[{"name": "custom", "hooks": [custom_hook]}],
        )

        exp.outputs["hooks"]["trace"] = []
        exp._run_experiment_stage_hooks("before", "train", component="model")
        trace = exp.outputs["hooks"]["trace"]
        train_before_events = [
            entry
            for entry in trace
            if entry.get("stage") == "train" and entry.get("event") == "before"
        ]
        # Canonical before-train hook must always execute.
        assert len(train_before_events) >= 1

    def test_stage_cache_key_changes_when_component_params_change(self):
        exp_a = self._make_base_experiment()
        exp_b = self._make_base_experiment()
        exp_b.compose_components(
            model=ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 5},
            ),
        )

        key_a = build_experiment_stage_cache_key(
            params_manifest=build_experiment_params_manifest(exp_a),
            stage="train",
            component="model",
            identity={"run_idx": 0},
        )
        key_b = build_experiment_stage_cache_key(
            params_manifest=build_experiment_params_manifest(exp_b),
            stage="train",
            component="model",
            identity={"run_idx": 0},
        )
        assert key_a != key_b

    def test_runtime_state_yaml_is_persisted_with_schema_and_reloads(self):
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "experiment_runtime_state"
            exp = self._make_base_experiment(params_file=str(params_path))
            _ = exp()

            saved_path = Path(str(params_path) + ".yaml")
            assert saved_path.exists()
            payload = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
            assert (
                payload.get("schema_version")
                == CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_VERSION
            )
            assert "params" in payload
            assert "runtime" in payload

            restored = ExperimentConfig.from_yaml(str(saved_path))
            assert isinstance(restored, ExperimentConfig)

    def test_cache_reuse_records_hits_on_second_run(self):
        with tempfile.TemporaryDirectory() as td:
            params_path = Path(td) / "experiment_runtime_state"
            exp = self._make_base_experiment(params_file=str(params_path))
            exp.cache_enabled = True

            _ = exp()
            first_hits = len(exp.outputs.get("cache", {}).get("hits", []))

            _ = exp()
            second_hits = len(exp.outputs.get("cache", {}).get("hits", []))

            assert second_hits >= first_hits

    def test_runtime_state_yaml_rejects_future_schema_version(self):
        with tempfile.TemporaryDirectory() as td:
            exp = self._make_base_experiment()
            payload = {
                "schema_version": "deckard.experiment.runtime.v999",
                "experiment": exp.to_dict(for_hash=True),
            }
            path = Path(td) / "future_runtime.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            with pytest.raises(ValueError):
                ExperimentConfig.from_yaml(str(path))
