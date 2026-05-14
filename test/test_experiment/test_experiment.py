import unittest
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from deckard.attack import AttackConfig
from deckard.data import DataConfig
from deckard.experiment import ExperimentConfig, SurvivalExperimentConfig
from deckard.frameworks.pytorch.experiment import TorchExperimentConfig
from deckard.experiment.base import (
    DataConfigResolutionMixin,
    _file_resolver,
    _merge_resolver,
)
from deckard.file import FileConfig
from deckard.model import ModelConfig
from deckard.score import DefaultClassifierConfig, DefaultDataClassificationConfig
from deckard.utils import ConfigBase
from helpers import make_runtime_env
from deckard.attack.base import resolve_class as base_resolve_class


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


class TestKFoldExperiment(unittest.TestCase):
    """ExperimentConfig should loop over all folds when sample='fold'."""

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
                test_size=0.2,
                random_state=42,
                classifier=True,
                sample=KFoldSampler(n_splits=self.N_FOLDS),
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
            self.assertIn(
                f"accuracy_fold_{k}",
                scores,
                f"missing accuracy_fold_{k}",
            )

    def test_kfold_mean_key_present(self):
        exp = self._make_exp()
        scores = exp()
        self.assertIn("accuracy", scores)

    def test_mean_equals_average_of_folds(self):
        exp = self._make_exp()
        scores = exp()
        fold_accs = [scores[f"accuracy_fold_{k}"] for k in range(self.N_FOLDS)]
        self.assertAlmostEqual(
            scores["accuracy"],
            float(np.mean(fold_accs)),
            places=10,
        )


class TestShuffleExperiment(unittest.TestCase):
    """ExperimentConfig should loop over all shuffle splits when sample='shuffle'."""

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
                test_size=0.2,
                val_size=0.1,
                random_state=42,
                classifier=True,
                sample=ShuffleSampler(n_splits=self.N_SPLITS),
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
            self.assertIn(
                f"accuracy_split_{k}",
                scores,
                f"missing accuracy_split_{k}",
            )

    def test_shuffle_mean_key_present(self):
        exp = self._make_exp()
        scores = exp()
        self.assertIn("accuracy", scores)

    def test_mean_equals_average_of_splits(self):
        exp = self._make_exp()
        scores = exp()
        split_accs = [scores[f"accuracy_split_{k}"] for k in range(self.N_SPLITS)]
        self.assertAlmostEqual(
            scores["accuracy"],
            float(np.mean(split_accs)),
            places=10,
        )


class TestExperimentValidationScoring(unittest.TestCase):
    def test_propagation_of_modes_to_children(self):
        exp = self._make_exp(score_mode="test")
        exp._propagate_score_mode()
        self.assertEqual(exp.data.score_mode, "test")
        self.assertEqual(exp.model.score_mode, "test")
        exp = self._make_exp(score_mode="val")
        exp._propagate_score_mode()
        self.assertEqual(exp.data.score_mode, "val")
        self.assertEqual(exp.model.score_mode, "val")

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
                test_size=0.2,
                val_size=val_size,
                random_state=42,
                classifier=True,
                sample="split",
            ),
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5, "random_state": 0},
            ),
            score={"experiment": DefaultClassifierConfig()},
            attack=None,
            files=FileConfig(),
            experiment_name="score-mode-policy-test",
            evaluation_mode=evaluation_mode,
            score_mode=score_mode,
        )

    def test_tuning_mode_emits_test_scores(self):
        exp = self._make_exp(evaluation_mode="tuning")
        scores = exp()
        self.assertIn("accuracy", scores)
        self.assertNotIn("validation_accuracy", scores)
        self.assertNotIn("training_accuracy", scores)

    def test_report_mode_emits_validation_scores(self):
        exp = self._make_exp(evaluation_mode="report", score_mode=None)
        scores = exp()
        self.assertIn("validation_accuracy", scores)
        self.assertIn("accuracy", scores)
        self.assertIn("training_accuracy", scores)

    def test_report_mode_without_validation_split_raises(self):
        exp = self._make_exp(val_size=None, evaluation_mode="report", score_mode=None)
        with self.assertRaises(ValueError):
            exp()

    def test_tuning_mode_without_validation_split_uses_test_scores(self):
        exp = self._make_exp(val_size=None, evaluation_mode="tuning")
        scores = exp()
        self.assertIn("accuracy", scores)


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


class TestExperimentDetectorPhase(unittest.TestCase):
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
                train_size=40,
                test_size=20,
                random_state=7,
                stratify=True,
                classifier=True,
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

        self.assertIn("detector_accuracy", scores)
        self.assertIn("detector_n", scores)


class TestSurvivalExperimentConfig(unittest.TestCase):
    def test_allows_survival_only_config_without_attack(self):
        config = SurvivalExperimentConfig(
            data=DataConfig(dataset_name="make_regression", classifier=False),
            model="cox",
            target="E",
            classifier=False,
            duration_col="T",
            event_col="E",
        )
        self.assertIsInstance(config, SurvivalExperimentConfig)
        self.assertEqual(config.model, "cox")

    def test_requires_attack_when_aux_model_present(self):
        with self.assertRaises(ValueError):
            SurvivalExperimentConfig(
                data=DataConfig(
                    dataset_name="make_regression",
                    classifier=False,
                ),
                model="cox",
                target="E",
                classifier=False,
                aux_model=ModelConfig(
                    model_type="sklearn.linear_model.LogisticRegression",
                    classifier=True,
                    model_params={"max_iter": 10},
                ),
                duration_col="T",
                event_col="E",
            )

    def test_requires_data_config(self):
        with self.assertRaises(ValueError):
            SurvivalExperimentConfig(
                data=None,
                model="cox",
                target="E",
                duration_col="T",
                event_col="E",
                classifier=False,
            )

    def test_survival_config_initializes(self):
        config = SurvivalExperimentConfig(
            data=DataConfig(
                dataset_name="make_regression",
                classifier=False,
                target=None,
            ),
            model="cox",
            target="E",
            classifier=False,
            duration_col="T",
            event_col="E",
        )
        self.assertIsInstance(config, SurvivalExperimentConfig)
        self.assertEqual(config.model, "cox")


class TestPoisoningExperimentIntegration(unittest.TestCase):
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
                test_size=0.2,
                random_state=4,
                classifier=True,
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

        self.assertIn("benign_accuracy", scores)
        self.assertIn("poisoned_accuracy", scores)

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
        self.assertEqual(result.returncode, 0, msg=result.stderr)

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
                train_size=60,
                test_size=20,
                random_state=42,
                classifier=True,
                sample="split",
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
            score={"experiment": DefaultClassifierConfig()},
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


class TestFileResolver(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Set DECKARD_CONFIG_DIR to tmpdir so the resolver finds files there
        self._orig_env = os.environ.get("DECKARD_CONFIG_DIR")
        os.environ["DECKARD_CONFIG_DIR"] = self.tmpdir

    def tearDown(self):
        if self._orig_env is None:
            os.environ.pop("DECKARD_CONFIG_DIR", None)
        else:
            os.environ["DECKARD_CONFIG_DIR"] = self._orig_env

    def _write_yaml(self, name, content):
        p = Path(self.tmpdir) / name
        p.write_text(content)
        return p

    def test_empty_arg_raises(self):
        with self.assertRaises(ValueError):
            _file_resolver("")

    def test_file_not_found_raises(self):
        # Patch so DECKARD_CONFIG_DIR points to tmpdir
        import deckard.experiment.base as _mod

        orig = _mod.DECKARD_CONFIG_DIR
        _mod.DECKARD_CONFIG_DIR = self.tmpdir
        try:
            with self.assertRaises(FileNotFoundError):
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
            self.assertEqual(OmegaConf.to_container(result)["a"], 1)
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
            self.assertEqual(container["type"], "rf")
        finally:
            _mod.DECKARD_CONFIG_DIR = orig

    def test_missing_key_raises(self):
        import deckard.experiment.base as _mod

        orig = _mod.DECKARD_CONFIG_DIR
        _mod.DECKARD_CONFIG_DIR = self.tmpdir
        try:
            self._write_yaml("cfg.yaml", "a: 1\n")
            with self.assertRaises(KeyError):
                _file_resolver("cfg.yaml:nonexistent.key")
        finally:
            _mod.DECKARD_CONFIG_DIR = orig


# ── _merge_resolver ──────────────────────────────────────────────────────────


class TestMergeResolver(unittest.TestCase):
    def test_merge_two_dicts(self):
        a = OmegaConf.create({"x": 1, "y": 2})
        b = OmegaConf.create({"z": 3, "y": 99})
        result = _merge_resolver(a, b)
        container = OmegaConf.to_container(result)
        self.assertEqual(container["x"], 1)
        self.assertEqual(container["y"], 99)
        self.assertEqual(container["z"], 3)

    def test_merge_single_dict(self):
        a = OmegaConf.create({"a": 42})
        result = _merge_resolver(a)
        container = OmegaConf.to_container(result)
        self.assertEqual(container["a"], 42)


# ── DataConfigResolutionMixin ─────────────────────────────────────────────────


class _TestMixin(DataConfigResolutionMixin):
    """Concrete subclass to test the mixin methods."""

    @property
    def data(self):
        return None


class TestDataConfigResolutionMixin(unittest.TestCase):
    def setUp(self):
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
        self.assertIsInstance(result, dict)

    def test_data_to_dict_with_plain_dict(self):
        d = {"dataset_name": "make_regression", "classifier": False}
        result = self.mixin._data_to_dict(d)
        self.assertEqual(result, d)

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
            with self.assertRaises(Exception):
                self.mixin._data_to_dict(str(p))

    def test_data_to_dict_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            self.mixin._data_to_dict(12345)

    def test_data_to_dict_not_dict_result_raises(self):
        # ConfigBase.to_dict() normally returns dict; if it returns non-dict we get TypeError
        class BadBase(ConfigBase):
            def __call__(self):
                pass

            def to_dict(self, **_):
                return "not_a_dict"

        obj = BadBase()
        with self.assertRaises(TypeError):
            self.mixin._data_to_dict(obj)

    def test_select_data_cls_anjana_keys(self):
        data_dict = {"dataset_name": "x", "quasi_identifiers": ["age"]}
        from deckard.data import AnjanaDataConfig

        if AnjanaDataConfig is None:
            self.skipTest("AnjanaDataConfig not available")
        cls = self.mixin._select_data_cls(data_dict)
        self.assertIs(cls, AnjanaDataConfig)

    def test_select_data_cls_fairness_keys(self):
        data_dict = {"dataset_name": "x", "sensitive_columns": ["gender"]}
        try:
            from deckard.data import FairlearnDataConfig

            if FairlearnDataConfig is None:
                self.skipTest("FairlearnDataConfig not available")
        except ImportError:
            self.skipTest("FairlearnDataConfig not available")
        cls = self.mixin._select_data_cls(data_dict)
        self.assertIs(cls, FairlearnDataConfig)

    def test_select_data_cls_pipeline_key(self):
        from deckard.data import DataPipelineConfig

        data_dict = {"pipeline": {}}
        cls = self.mixin._select_data_cls(data_dict)
        self.assertIs(cls, DataPipelineConfig)

    def test_select_data_cls_plain(self):
        data_dict = {"dataset_name": "make_classification", "classifier": True}
        cls = self.mixin._select_data_cls(data_dict)
        self.assertIs(cls, DataConfig)

    def test_resolve_data_config_with_data_config_passthrough(self):
        class _Exp(DataConfigResolutionMixin, ConfigBase):
            def __call__(self):
                pass

        dc = DataConfig(dataset_name="make_classification", classifier=True)
        exp = _Exp()
        exp.data = dc
        result = exp._resolve_data_config()
        self.assertIs(result, dc)

    def test_select_data_cls_anjana_missing_dependency_raises(self):
        data_dict = {"quasi_identifiers": ["age"]}
        with patch("deckard.experiment.base.AnjanaDataConfig", None):
            with self.assertRaises(ImportError):
                self.mixin._select_data_cls(data_dict)

    def test_select_data_cls_fairness_missing_dependency_raises(self):
        data_dict = {"sensitive_columns": ["gender"]}
        with patch("deckard.experiment.base.FairlearnDataConfig", None):
            with self.assertRaises(ImportError):
                self.mixin._select_data_cls(data_dict)

    def test_resolve_data_config_target_resolves_wrong_type_raises(self):
        class _Exp(DataConfigResolutionMixin, ConfigBase):
            def __call__(self):
                pass

        exp = _Exp()
        exp.data = OmegaConf.create({"_target_": "builtins.dict"})
        with self.assertRaises(TypeError):
            exp._resolve_data_config()

    def test_sklearn_pipeline_yaml_compose_instantiates_pipeline_configs(self):
        from deckard.data import DataPipelineConfig

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
            self.assertIsInstance(instantiate(cfg), DataPipelineConfig)

    def test_pytorch_pipeline_yaml_compose_instantiates_pipeline_config(self):
        pytest.importorskip("torch")

        from deckard.data import DataPipelineConfig

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
        self.assertIsInstance(instantiate(cfg), DataPipelineConfig)


# ── ExperimentConfig.set_random_seed ─────────────────────────────────────────


class TestSetRandomSeed(unittest.TestCase):
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
                train_size=40,
                test_size=20,
                random_state=42,
                classifier=True,
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
            self.skipTest("torch not available")
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
            with self.assertRaises(Exception):
                exp.set_random_seed()

    def test_set_random_seed_truly_unsupported_library_raises(self):
        exp = self._make_base_exp()
        exp.library = "not_a_supported_library"
        with self.assertRaises(ValueError):
            exp.set_random_seed()


# ── ExperimentConfig.__post_init__ model input types ─────────────────────────


class TestExperimentPostInitModelTypes(unittest.TestCase):
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
            train_size=40,
            test_size=20,
            random_state=42,
            classifier=True,
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
        self.assertIsInstance(exp.model, ModelConfig)

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
        self.assertIsInstance(exp.model, ModelConfig)

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
        self.assertNotEqual(exp.experiment_name, "{hash}")
        self.assertEqual(len(exp.experiment_name), 32)

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
        self.assertIsInstance(exp.attack, AttackConfig)

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
        self.assertIsInstance(exp.attack, AttackConfig)


# ── ExperimentConfig.__call__ score file paths ───────────────────────────────


class TestExperimentScoreFileHandling(unittest.TestCase):
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
                train_size=40,
                test_size=20,
                random_state=42,
                classifier=True,
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
        self.assertIsInstance(scores, dict)

    def test_score_file_is_created(self):
        with tempfile.TemporaryDirectory() as td:
            score_path = str(Path(td) / "scores.json")
            exp = self._make_exp(score_file=score_path)
            exp()
            self.assertTrue(Path(score_path).exists())

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
            self.assertIn("prior_metric", merged)

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
                train_size=40,
                test_size=20,
                random_state=42,
                classifier=True,
            ),
            model=None,
            files=FileConfig(),
        )
        scores = exp()
        self.assertIsInstance(scores, dict)


# ── ExperimentConfig._canonical_device ───────────────────────────────────────


class TestCanonicalDevice(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(ExperimentConfig._canonical_device(None))

    def test_null_token_returns_none(self):
        self.assertIsNone(ExperimentConfig._canonical_device("none"))

    def test_auto_token_returns_none(self):
        self.assertIsNone(ExperimentConfig._canonical_device("auto"))

    def test_cpu_returns_cpu(self):
        self.assertEqual(ExperimentConfig._canonical_device("cpu"), "cpu")

    def test_mps_returns_mps(self):
        self.assertEqual(ExperimentConfig._canonical_device("mps"), "mps")


# ── ExperimentConfig._resolve_score_modes ────────────────────────────────────


class TestResolveScoreModes(unittest.TestCase):
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
        self.assertEqual(exp._resolve_score_modes(), ["train", "test"])

    def test_tuning_returns_test(self):
        exp = self._base_exp(evaluation_mode="tuning")
        self.assertEqual(exp._resolve_score_modes(), ["test"])

    def test_report_returns_train_test_and_val(self):
        exp = self._base_exp(evaluation_mode="report")
        self.assertEqual(exp._resolve_score_modes(), ["train", "test", "val"])

    def test_explicit_score_modes_override(self):
        exp = self._base_exp(score_mode=["test", "val"])
        self.assertEqual(exp._resolve_score_modes(), ["test", "val"])

    def test_explicit_score_mode_presample(self):
        exp = self._base_exp(score_mode="pre-sample")
        self.assertEqual(exp._resolve_score_modes(), ["pre-sample"])

    def test_normalize_mode_score_keys_presample(self):
        out = ExperimentConfig._normalize_mode_score_keys(
            "pre-sample",
            {"num_classes": 3},
        )
        self.assertEqual(out, {"presample_num_classes": 3})


class TestExperimentScorerModePermutations(unittest.TestCase):
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
            test_size=0.2,
            val_size=val_size,
            random_state=42,
            classifier=True,
            sample="split",
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
            score={"experiment": DefaultDataClassificationConfig()},
            score_mode=["pre-sample", "train", "test", "val"],
            files=FileConfig(),
            experiment_name="score-permutations-data-profile",
        )

        scores = exp()

        self.assertIn("presample_num_classes", scores)
        self.assertIn("training_num_classes", scores)
        self.assertIn("num_classes", scores)
        self.assertIn("validation_num_classes", scores)

    def test_non_data_profile_rejects_presample_mode(self):
        exp = ExperimentConfig(
            data=self._base_data(val_size=0.1),
            model=self._base_model(),
            score={"experiment": DefaultClassifierConfig()},
            score_mode=["pre-sample", "test"],
            files=FileConfig(),
            experiment_name="score-permutations-non-data-profile",
        )

        with self.assertRaises(ValueError):
            exp()


# ── ExperimentConfig._aggregate_repeated_scores ──────────────────────────────


class TestAggregateRepeatedScores(unittest.TestCase):
    def test_empty_list_returns_empty(self):
        result = ExperimentConfig._aggregate_repeated_scores([])
        self.assertEqual(result, {})

    def test_single_run_produces_fold_keys(self):
        result = ExperimentConfig._aggregate_repeated_scores(
            [{"accuracy": 0.9}],
            suffix="fold",
        )
        self.assertIn("accuracy_fold_0", result)
        self.assertAlmostEqual(result["accuracy"], 0.9)

    def test_multiple_runs_computes_mean(self):
        runs = [{"accuracy": 0.8}, {"accuracy": 0.9}, {"accuracy": 1.0}]
        result = ExperimentConfig._aggregate_repeated_scores(runs, suffix="fold")
        self.assertAlmostEqual(result["accuracy"], np.mean([0.8, 0.9, 1.0]))
        for i in range(3):
            self.assertIn(f"accuracy_fold_{i}", result)

    def test_non_numeric_uses_last_value(self):
        runs = [{"label": "a"}, {"label": "b"}]
        result = ExperimentConfig._aggregate_repeated_scores(runs)
        self.assertEqual(result["label"], "b")

    def test_none_values_excluded_from_mean(self):
        runs = [{"acc": 0.9}, {"acc": None}]
        result = ExperimentConfig._aggregate_repeated_scores(runs)
        self.assertAlmostEqual(result["acc"], 0.9)


# ── ExperimentConfig.__post_init__ more branches ─────────────────────────────


class TestExperimentPostInitMoreBranches(unittest.TestCase):
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
            train_size=40,
            test_size=20,
            random_state=42,
            classifier=True,
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
        self.assertIsInstance(exp.model, ModelConfig)

    def test_model_as_config_base_subclass(self):
        """Cover branch: isinstance(self.model, ConfigBase) -> model_dict = model.to_dict()"""

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
        self.assertIsInstance(exp.model, ModelConfig)

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

        self.assertIsInstance(exp.files, FC)

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

        self.assertIsInstance(exp.files, FC)

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
        self.assertIsInstance(exp.model, ModelConfig)

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
        self.assertIsInstance(exp.attack, AttackConfig)

    def test_attack_as_configbase(self):
        """Cover branch: isinstance(self.attack, ConfigBase)"""

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
        self.assertIsInstance(exp.attack, AttackConfig)

    def test_multi_attack_requires_aliases(self):
        with self.assertRaises(ValueError):
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
        self.assertEqual(len(exp._attack_chain), 2)
        self.assertEqual(exp.attack.alias, "fgm_a")

    def test_score_as_scorer_dict_config(self):
        """Cover scorer config path where score is ScorerDictConfig."""
        from deckard.score import ScorerDictConfig, ScorerConfig

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
        self.assertIsNotNone(exp.model.scorer)
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
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


class TestCoerceScorerConfig(unittest.TestCase):
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
            train_size=40,
            test_size=20,
            random_state=42,
            classifier=True,
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
        self.assertIsNotNone(exp.score)
        # data and model auto scorers should be attached too
        self.assertIsNotNone(exp.data.scorer)
        self.assertIsNotNone(exp.model.scorer)

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
        self.assertIsNotNone(exp.data.scorer)
        self.assertIsNotNone(exp.model.scorer)
        self.assertIsNone(exp.score)


class TestRunSinglePipelineBranchesExtra(unittest.TestCase):
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
        with self.assertRaises(ValueError):
            exp._run_single_pipeline({}, {})

    def test_detector_without_attack_raises(self):
        exp = self._exp_stub()
        exp.detector = lambda **kwargs: kwargs
        with self.assertRaises(ValueError):
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
            train_size=30,
            test_size=10,
            random_state=42,
            classifier=True,
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
                    train_size=10,
                    test_size=10,
                    random_state=42,
                    classifier=True,
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

            self.assertIsInstance(scores, dict)
            self.assertTrue(score_file.exists())

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
            train_size=60,
            test_size=20,
            random_state=42,
            classifier=True,
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
                    train_size=0.6,
                    test_size=0.2,
                    val_size=0.2,
                    sample="split",
                    random_state=42,
                    classifier=True,
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

            self.assertIsNotNone(exp.data.X_val)
            self.assertIsNotNone(exp.data.y_val)
            self.assertIn("validation_accuracy", scores)


# ── ExperimentConfig set_device tensorflow ────────────────────────────────────


class TestSetDeviceTensorflow(unittest.TestCase):
    def setUp(self):
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
                train_size=40,
                test_size=20,
                random_state=42,
                classifier=True,
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


class TestRunSinglePipelineBranches(unittest.TestCase):
    def test_detector_requires_attack_raises(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(score_dict={})
        exp.model = None
        exp.attack = None
        exp.detector = SimpleNamespace(__call__=lambda **_kwargs: None)
        exp._run_experiment_scorer_modes = lambda score_file=None: {}
        with self.assertRaises(ValueError):
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
            def __call__(self, **_kwargs):
                # Intentionally omit score_dict to hit assertion branch.
                return None

        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(score_dict={})
        exp.model = _Model()
        exp.attack = _Attack()
        exp.detector = _Detector()
        exp._run_experiment_scorer_modes = lambda score_file=None: {}
        with self.assertRaises(AssertionError):
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

        self.assertEqual(scores["evasion_accuracy"], 0.5)
        self.assertEqual(scores["evasion_accuracy_atk_b"], 0.4)
        self.assertEqual(scores["attack_generation_time"], 1.0)
        self.assertEqual(scores["attack_generation_time_atk_b"], 2.0)
        self.assertEqual(scores["detector_n"], 5)


class TestExperimentBranchEdges(unittest.TestCase):
    def test_canonical_device_empty_and_default_tokens(self):
        self.assertIsNone(ExperimentConfig._canonical_device(""))
        self.assertIsNone(ExperimentConfig._canonical_device(" default "))
        self.assertEqual(ExperimentConfig._canonical_device("CUDA"), "cuda")

    def test_resolve_data_config_target_must_resolve_to_dataconfig(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(_target_="fake.target")
        with patch("deckard.experiment.base.instantiate", return_value=object()):
            with self.assertRaises(TypeError):
                exp._resolve_data_config()

    def test_compute_val_predictions_error_branches(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)

        exp.model = None
        exp.data = SimpleNamespace(X_val=[1], y_val=[1])
        with self.assertRaises(ValueError):
            exp._compute_val_predictions()

        exp.model = SimpleNamespace(_predict=lambda x: x)
        exp.data = SimpleNamespace(X_val=None, y_val=None)
        with self.assertRaises(ValueError):
            exp._compute_val_predictions()

        exp.model = SimpleNamespace()
        exp.data = SimpleNamespace(X_val=[1], y_val=[1])
        with self.assertRaises(ValueError):
            exp._compute_val_predictions()

    def test_ensure_mode_predictions_error_branches(self):
        exp = ExperimentConfig.__new__(ExperimentConfig)
        exp.data = SimpleNamespace(X_train=[1], X_test=[2], X_val=[3], y_val=[1])

        exp.model = None
        with self.assertRaises(ValueError):
            exp._ensure_mode_predictions("train")

        exp.model = SimpleNamespace()
        with self.assertRaises(ValueError):
            exp._ensure_mode_predictions("test")

    def test_post_init_rejects_unsupported_model_type(self):
        with self.assertRaises(ValueError):
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
                    train_size=10,
                    test_size=10,
                    random_state=42,
                    classifier=True,
                ),
                model=123,
                files=FileConfig(),
            )


if __name__ == "__main__":
    unittest.main()
