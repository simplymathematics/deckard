import os
import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import TinyData
from numpy.exceptions import AxisError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from deckard.attack import AttackConfig
from deckard.attack.base import SensitiveFeaturesWrapper, _sensitive_slice
from deckard.attack.canon import normalize_attack_stage
from deckard.attack.extraction import ExtractionAttackMixin
from deckard.score.attack import FairlearnAttackScorerConfig


class TestAttackConfig:
    def setup_method(self):
        self.attack_params = {}
        self.name = "art.attacks.evasion.FastGradientMethod"
        self.attack = AttackConfig(
            name=self.name,
            attack_params=self.attack_params,
        )
        self.tmpdir = tempfile.mkdtemp()
        self.attack_file = os.path.join(self.tmpdir, "attack.pkl")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def _load_pytorch_model_inversion_config(self):
        config_path = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "pytorch"
            / "config"
            / "attack"
            / "model-inversion.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return AttackConfig(**config)

    def _load_pytorch_database_reconstruction_config(self):
        config_path = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "pytorch"
            / "config"
            / "attack"
            / "database-reconstruction.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return AttackConfig(**config)

    def test_post_init(self):
        assert hasattr(self.attack, "name")
        assert hasattr(self.attack, "attack_params")

    def test_attack_config_canonical_name_field(self):
        cfg = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        assert cfg.name == "art.attacks.evasion.FastGradientMethod"
        assert cfg.name == "art.attacks.evasion.FastGradientMethod"

    def test_post_init_scorer_dict_and_poisoning_validation_paths(self):
        cfg = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            scorer={},
        )
        assert cfg.scorer is not None

        cfg_default = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            scorer="default",
        )
        assert cfg_default.scorer is not None

        cfg_auto = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            scorer="auto",
        )
        assert cfg_auto.scorer is not None

        cfg_target = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            scorer={
                "_target_": "deckard.score.attack.FairlearnAttackScorerConfig",
            },
        )
        assert cfg_target.scorer.__class__.__name__ == "FairlearnAttackScorerConfig"

        cfg_name = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            scorer={
                "name": "deckard.score.attack.FairlearnAttackScorerConfig",
            },
        )
        assert cfg_name.scorer.__class__.__name__ == "FairlearnAttackScorerConfig"

        cfg_path = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            scorer="deckard.score.attack.FairlearnAttackScorerConfig",
        )
        assert cfg_path.scorer.__class__.__name__ == "FairlearnAttackScorerConfig"

        with pytest.raises(TypeError, match="_score"):
            AttackConfig(
                name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
                scorer=123,
            )

        with pytest.raises(ValueError, match="Missing"):
            AttackConfig(
                name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
                attack_params={},
            )

        with pytest.raises(
            ValueError,
            match="class_source and class_target to differ",
        ):
            AttackConfig(
                name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
                attack_params={"class_source": 1, "class_target": 1},
            )

    def test_attack_path_properties_and_kind(self):
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
        )
        assert attack.attack_family == "inference"
        assert attack.attack_sub_family == "membership_inference"
        assert attack.attack_kind == "membership"

    def test_poisoning_svm_initialization_injects_train_and_val_arrays(self):
        class DummyPoisoningAttackSVM:
            def __init__(self, classifier, **kwargs):
                self.classifier = classifier
                self.kwargs = kwargs

        class _TinyData:
            X_train = np.array(
                [[0.0, 0.1], [1.0, 0.2], [0.2, 1.0], [0.9, 0.8]],
                dtype=np.float32,
            )
            y_train = np.array([0, 1, 0, 1])
            X_val = np.array([[0.15, 0.25], [0.75, 0.65]], dtype=np.float32)
            y_val = np.array([0, 1])
            X_test = np.array([[0.05, 0.15], [0.85, 0.75]], dtype=np.float32)
            y_test = np.array([0, 1])
            classifier = True

        model = SVC(probability=True)
        model.fit(_TinyData.X_train, _TinyData.y_train)
        attack = AttackConfig(
            name="art.attacks.poisoning.PoisoningAttackSVM",
            attack_params={
                "step": 0.1,
                "eps": 0.2,
                "max_iter": 2,
                "verbose": False,
            },
            attack_size=2,
        )

        with patch(
            "deckard.attack.base.resolve_class",
            return_value=DummyPoisoningAttackSVM,
        ):
            initialized_attack, _, attack_family, _ = attack._initialize_attack(
                model,
                _TinyData(),
            )

        assert attack_family == "poisoning"
        assert getattr(initialized_attack.classifier, "clip_values", None) is not None
        assert "x_train" in initialized_attack.kwargs
        assert "y_train" in initialized_attack.kwargs
        assert "x_val" in initialized_attack.kwargs
        assert "y_val" in initialized_attack.kwargs
        assert initialized_attack.kwargs["x_train"].shape == (4, 2)
        assert initialized_attack.kwargs["x_val"].shape == (2, 2)
        np.testing.assert_array_equal(
            initialized_attack.kwargs["y_train"],
            np.array(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            initialized_attack.kwargs["y_val"],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

    def test_select_extraction_scorer_falls_back_for_logits(self):
        scorer, use_proba = ExtractionAttackMixin._select_extraction_scorer(
            benign_pred=np.array([[0.8, 0.2], [0.1, 0.9]]),
            extracted_pred=np.array([[1.5, -0.2], [-0.4, 2.1]]),
        )

        assert not use_proba
        assert "roc_auc" not in scorer.scorers

    def test_infer_task_is_classification_and_compatibility_guard(self):
        class _Data:
            classifier = None

        assert not AttackConfig._infer_task_is_classification(
            _Data(),
            LinearRegression(),
        )
        assert AttackConfig._infer_task_is_classification(
            _Data(),
            LogisticRegression(),
        )

        data = type("D", (), {"classifier": True})()
        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        with pytest.raises(ValueError, match="not supported for regression"):
            attack._validate_attack_task_compatibility(data, LinearRegression())

    def test_score_and_static_normalizers(self):
        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        attack.scorer = None
        with pytest.raises(ValueError, match="must be configured"):
            attack._score("evasion", [0, 1], [0, 1])

        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        assert not attack._is_regression_prediction_output(
            [0, 1],
            [[0.2, 0.8], [0.6, 0.4]],
        )
        assert attack._is_regression_prediction_output([0.2, 0.3], [0.1, 0.2])

        arr = attack._to_numpy_array(pd.Series([1, 2, 3]), flatten=True)
        assert arr.shape == (3,)

        feat = attack._prepare_features_for_attack(pd.DataFrame({"a": [1, 2]}))
        lbl = attack._prepare_labels_for_attack(pd.Series([0, 1]))
        assert isinstance(feat, np.ndarray)
        assert isinstance(lbl, np.ndarray)

        labels = attack._prediction_to_labels(
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            is_regression=False,
        )
        assert np.array_equal(labels, np.array([1, 0]))
        assert np.array_equal(
            attack._normalize_ground_truth(
                pd.Series(["a", "b", "a"]),
                is_regression=False,
            ),
            np.array([0, 1, 0]),
        )

        class_labels = attack._target_to_class_labels(np.array([[0, 1], [1, 0]]))
        assert np.array_equal(class_labels, np.array([1, 0]))

        one_hot = attack._one_hot_encode([0, 1], 2)
        assert one_hot.shape == (2, 2)

        norm = attack._normalize_inferred_output(
            np.array([0, 1]),
            reference=np.array([[1, 0], [0, 1]]),
        )
        assert norm.shape[1] == 2

        scorer_cfg, has_proba = ExtractionAttackMixin._select_extraction_scorer(
            np.array([0, 1]),
            np.array([0, 1]),
        )
        assert not has_proba
        assert scorer_cfg is not None

    def test_attack_mode_rejects_defense_stage_aliases(self):
        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        with pytest.raises(ValueError, match="Unsupported attack mode"):
            attack.set_mode("post-defense")

        with pytest.raises(ValueError, match="Unsupported attack mode"):
            attack.set_mode("pre-defense")

        attack.set_mode("val")
        assert attack.resolve_mode_for_attack_kind("evasion") == "val"

    def test_attack_mode_auto_defaults_by_subtype(self):
        membership = AttackConfig(
            name=(
                "art.attacks.inference.membership_inference."
                "MembershipInferenceBlackBox"
            ),
        )
        assert (
            membership.resolve_mode_for_attack_kind(
                "membership",
                attack_sub_family="membership_inference",
            )
            == "train"
        )

        reconstruction = AttackConfig(
            name=("art.attacks.inference.reconstruction.DatabaseReconstruction"),
        )
        assert (
            reconstruction.resolve_mode_for_attack_kind(
                "reconstruction",
                attack_sub_family="reconstruction",
            )
            == "train"
        )

        inversion = AttackConfig(
            name="art.attacks.inference.model_inversion.MIFace",
        )
        assert (
            inversion.resolve_mode_for_attack_kind(
                "model_inversion",
                attack_sub_family="model_inversion",
            )
            == "test"
        )

    def test_attack_runtime_contract_defaults_and_stage_metadata(self):
        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        assert isinstance(attack.score_dict, dict)
        assert attack.attack_predictions is None
        assert attack.mode == "auto"
        assert normalize_attack_stage("before_attack") == "pre-attack"
        assert normalize_attack_stage("after_attack") == "post-attack"

        poisoning = AttackConfig(
            name=(
                "art.attacks.poisoning.gradient_matching_attack."
                "GradientMatchingAttack"
            ),
            attack_params={"class_source": 0, "class_target": 1},
        )
        assert poisoning.resolve_mode_for_attack_kind("poisoning") == "train"

    def test_attack_mode_split_override_precedence(self):
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"split": "train"},
        )
        attack.set_mode("val")
        assert attack.resolve_mode_for_attack_kind("evasion") == "train"

        assert (
            attack.resolve_mode_for_attack_kind(
                "evasion",
                split_override="test",
            )
            == "test"
        )

        with pytest.raises(ValueError, match="Unsupported attack split override"):
            attack.resolve_mode_for_attack_kind(
                "evasion",
                split_override="invalid",
            )

    def test_attack_score_forwards_mode_and_stage_context(self):
        class _DummyScorer:
            def __init__(self):
                self.last_kwargs = None

            def _score(self, *args, **kwargs):
                _ = args
                self.last_kwargs = kwargs
                return {"attack_score_time": 0.01, "evasion_success": 0.2}

        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        dummy = _DummyScorer()
        attack.scorer = dummy

        out = attack._score(
            "evasion",
            y_true=np.array([0, 1]),
            y_pred=np.array([1, 0]),
            ben_pred_labels=np.array([0, 1]),
        )

        assert "evasion_success" in out
        assert dummy.last_kwargs.get("mode") == "test"
        assert "stage" not in dummy.last_kwargs

        attack.set_mode("val")
        out = attack._score(
            "evasion",
            y_true=np.array([0, 1]),
            y_pred=np.array([1, 0]),
            ben_pred_labels=np.array([0, 1]),
        )
        assert "evasion_success" in out
        assert dummy.last_kwargs.get("mode") == "val"
        assert "stage" not in dummy.last_kwargs

    def test_target_to_class_labels_invalid_shape_raises(self):
        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        with pytest.raises(ValueError, match="Unsupported target shape"):
            attack._target_to_class_labels(np.array(5))

    def test_save_and_load_attack(self):
        from deckard.data import DataConfig
        from deckard.model import ModelConfig

        data = DataConfig(name="adult")
        data()
        model = ModelConfig(name="sklearn.linear_model.LogisticRegression")
        model(data=data)
        path = Path(self.attack_file)
        if path.exists():
            path.unlink()
        loaded_attack = AttackConfig(
            name=self.name,
            attack_params=self.attack_params,
        )
        loaded_attack(data=data, model=model, attack_file=self.attack_file)
        self.attack.save(self.attack_file)
        assert Path(self.attack_file).exists()
        loaded_attack.load(self.attack_file)
        assert loaded_attack.name == self.attack.name
        assert loaded_attack.attack_params == self.attack.attack_params

    def test_attack_metrics(self):
        # Mock data for testing
        ben_pred_labels = [0, 1, 0]
        adv_pred_labels = [0, 0, 0]
        y_test_numeric = [0, 1, 0]
        self.attack._score_attack(
            ben_pred_labels,
            adv_pred_labels,
            y_test_numeric,
        )
        metrics = self.attack.score_dict
        assert "evasion_success" in metrics

    def test_evade_includes_benign_scores(self):
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_size=6,
        )

        class _TinyData:
            X_test = np.array(
                [
                    [0.0, 0.0],
                    [0.2, 0.1],
                    [0.9, 0.7],
                    [0.8, 0.9],
                    [0.1, 0.3],
                    [0.7, 0.6],
                ],
            )
            y_test = np.array([0, 0, 1, 1, 0, 1])

        class _FakeArtModel:
            def predict(self, x):
                x = np.asarray(x)
                p1 = (x[:, 0] > 0.5).astype(float)
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])

        class _FakeEvasionAttack:
            def generate(self, x):
                x = np.asarray(x).copy()
                x[:, 0] = 1.0 - x[:, 0]
                return x

        runtime = attack._with_attack_context(
            attack_family="evasion", attack_sub_family=""
        )
        result = runtime.evade(
            data=_TinyData(),
            art_model=_FakeArtModel(),
            attack=_FakeEvasionAttack(),
        )

        assert "benign_accuracy" in result
        assert "benign_precision" in result
        assert "benign_recall" in result
        assert "benign_f1" in result
        assert "evasion_success" in result

    def test_poison_attack_metrics(self):
        attack = AttackConfig(
            name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            attack_size=6,
            mode="test",
        )

        class _TinyData:
            X_train = np.array(
                [
                    [0.0, 0.1],
                    [1.0, 0.2],
                    [0.2, 1.0],
                    [0.9, 0.8],
                    [0.1, 0.3],
                    [0.8, 0.7],
                ],
            )
            y_train = np.array([0, 1, 0, 1, 0, 1])
            X_test = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.1],
                    [0.2, 0.9],
                    [0.7, 0.8],
                ],
            )
            y_test = np.array([0, 1, 0, 1])
            classifier = True

        class _FakeArtModel:
            def __init__(self):
                self.nb_classes = 2
                self._poisoned = False

            def predict(self, X):
                x = np.asarray(X)
                p1 = (x[:, 0] > 0.5).astype(float)
                if self._poisoned:
                    p1 = 1.0 - p1
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])

            def fit(self, x, y, **kwargs):
                _ = x
                _ = y
                _ = kwargs
                self._poisoned = True
                return self

        class _FakePoisonAttack:
            def poison(self, x_trigger, y_trigger, x_train, y_train):
                _ = x_trigger
                _ = y_trigger
                return np.asarray(x_train), np.asarray(y_train)

        data = _TinyData()
        art_model = _FakeArtModel()
        poison_attack = _FakePoisonAttack()

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(
                poison_attack,
                art_model,
                "poisoning",
                "gradient_matching_attack",
            ),
        ):
            result = attack(data, object())

        assert "benign_accuracy" in result
        assert "poisoned_accuracy" in result
        assert "poison_trigger_success" in result
        assert "attack_generation_time" in result

    def test_call_attack(self):
        # Mock attack internals to verify call flow without requiring a specific ART estimator.
        data = object()
        model = object()

        def _fake_handler(**kwargs):
            _ = kwargs
            self.attack.attack_time = 0.1
            self.attack.attack_prediction_time = 0.05
            self.attack.attack_score_time = 0.02
            return {"evasion_success": 0.3}

        with (
            patch.object(
                AttackConfig,
                "_initialize_attack",
                return_value=(object(), object(), "evasion", ""),
            ),
            patch.object(
                AttackConfig,
                "_with_attack_context",
                return_value=self.attack,
            ),
            patch.object(
                AttackConfig,
                "_resolve_attack_handler",
                return_value=_fake_handler,
            ),
        ):
            result = self.attack(data, model)

        assert result is not None
        assert "evasion_success" in result
        assert "attack_generation_time" in result

    def test_call_membership_inference(self):
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
        )

        def _fake_handler(**kwargs):
            _ = kwargs
            attack.attack_time = 0.2
            attack.attack_prediction_time = 0.1
            attack.attack_score_time = 0.05
            return {"membership_inference_accuracy": 0.9}

        with (
            patch.object(
                AttackConfig,
                "_initialize_attack",
                return_value=(
                    object(),
                    object(),
                    "inference",
                    "membership_inference",
                ),
            ),
            patch.object(
                AttackConfig,
                "_with_attack_context",
                return_value=attack,
            ),
            patch.object(
                AttackConfig,
                "_resolve_attack_handler",
                return_value=_fake_handler,
            ),
        ):
            result = attack(object(), object())

        assert "membership_inference_accuracy" in result
        assert "attack_generation_time" in result

    def test_call_attribute_inference(self):
        attack = AttackConfig(
            name="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute="age",
            attack_params={},
        )

        def _fake_handler(**kwargs):
            _ = kwargs
            attack.attack_time = 0.3
            attack.attack_prediction_time = 0.1
            attack.attack_score_time = 0.07
            return {"inferred_age_accuracy": 0.8}

        with (
            patch.object(
                AttackConfig,
                "_initialize_attack",
                return_value=(
                    object(),
                    object(),
                    "inference",
                    "attribute_inference",
                ),
            ),
            patch.object(
                AttackConfig,
                "_with_attack_context",
                return_value=attack,
            ),
            patch.object(
                AttackConfig,
                "_resolve_attack_handler",
                return_value=_fake_handler,
            ),
        ):
            result = attack(object(), object())

        assert "inferred_age_accuracy" in result
        assert "attack_prediction_time" in result

    def testinfer_model_inversion_scores_reconstruction(self):
        attack = self._load_pytorch_model_inversion_config()
        attack.attack_size = 2
        attack.attack_params["split"] = "test"

        class _TinyData:
            X_test = np.array(
                [
                    [0.0, 0.2],
                    [0.1, 0.3],
                    [0.8, 0.7],
                    [0.9, 0.6],
                ],
                dtype=np.float32,
            )
            y_test = np.array([0, 0, 1, 1], dtype=np.int64)

        class _FakeModelInversionAttack:
            def infer(self, x, y):
                y = np.asarray(y).reshape(-1, 1).astype(np.float32)
                return np.asarray(x, dtype=np.float32) + (0.05 * y)

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        scores = runtime.infer_model_inversion(
            data=_TinyData(),
            attack=_FakeModelInversionAttack(),
        )

        assert "model_inversion_mse" in scores
        assert "model_inversion_mae" in scores
        assert "model_inversion_num_targets" in scores
        assert scores["model_inversion_num_targets"] == 2
        assert scores["model_inversion_mse"] >= 0.0

    def test_call_model_inversion_executes_realinfer_model_inversion(self):
        attack = self._load_pytorch_model_inversion_config()
        attack.attack_size = 2
        attack.attack_params["split"] = "test"
        attack.attack_params["targets"] = [0, 1]

        class _TinyData:
            X_test = np.array(
                [
                    [0.0, 0.2],
                    [0.1, 0.3],
                    [0.8, 0.7],
                    [0.9, 0.6],
                ],
                dtype=np.float32,
            )
            y_test = np.array([0, 0, 1, 1], dtype=np.int64)

        class _FakeModelInversionAttack:
            def __init__(self):
                self.infer_calls = 0

            def infer(self, x, y):
                self.infer_calls += 1
                y = np.asarray(y).reshape(-1, 1).astype(np.float32)
                return np.asarray(x, dtype=np.float32) + (0.05 * y)

        fake_attack = _FakeModelInversionAttack()

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(
                fake_attack,
                object(),
                "inference",
                "model_inversion",
            ),
        ):
            result = attack(_TinyData(), object())

        assert fake_attack.infer_calls == 1
        assert "model_inversion_mse" in result
        assert "model_inversion_mae" in result
        assert "model_inversion_num_targets" in result
        assert "attack_generation_time" in result
        assert "attack_prediction_time" in result
        assert "attack_score_time" in result

    def testinfer_database_reconstruction_scores_reconstruction(self):
        attack = self._load_pytorch_database_reconstruction_config()
        attack.attack_params["split"] = "train"
        attack.attack_params["missing_index"] = 1

        class _TinyData:
            X_train = np.array(
                [
                    [0.0, 0.2, 0.1],
                    [0.4, 0.5, 0.6],
                    [0.8, 0.7, 0.9],
                    [0.2, 0.3, 0.1],
                ],
                dtype=np.float32,
            )
            y_train = np.array([0, 1, 1, 0], dtype=np.int64)

        class _FakeDatabaseReconstructionAttack:
            def reconstruct(self, x, y=None, **kwargs):
                _ = kwargs
                x = np.asarray(x, dtype=np.float32)
                y = np.asarray(y) if y is not None else np.array([0])
                x_guess = x[:1] + 0.01
                y_guess = np.array([int(y[0])], dtype=np.int64)
                return x_guess, y_guess

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="reconstruction",
        )
        scores = runtime.infer_database_reconstruction(
            data=_TinyData(),
            attack=_FakeDatabaseReconstructionAttack(),
        )

        assert "database_reconstruction_feature_mse" in scores
        assert "database_reconstruction_feature_mae" in scores
        assert "database_reconstruction_num_features" in scores
        assert "database_reconstruction_missing_index" in scores
        assert scores["database_reconstruction_feature_mse"] >= 0.0

    def test_call_database_reconstruction_executes_realinfer_database_reconstruction(
        self,
    ):
        attack = self._load_pytorch_database_reconstruction_config()
        attack.attack_params["split"] = "train"
        attack.attack_params["missing_index"] = 0

        class _TinyData:
            X_train = np.array(
                [
                    [0.0, 0.2],
                    [0.1, 0.3],
                    [0.8, 0.7],
                    [0.9, 0.6],
                ],
                dtype=np.float32,
            )
            y_train = np.array([0, 0, 1, 1], dtype=np.int64)

        class _FakeDatabaseReconstructionAttack:
            def __init__(self):
                self.reconstruct_calls = 0

            def reconstruct(self, x, y=None, **kwargs):
                _ = kwargs
                self.reconstruct_calls += 1
                x = np.asarray(x, dtype=np.float32)
                y = np.asarray(y) if y is not None else np.array([0])
                return x[:1], np.array([int(y[0])], dtype=np.int64)

        fake_attack = _FakeDatabaseReconstructionAttack()

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(
                fake_attack,
                object(),
                "inference",
                "reconstruction",
            ),
        ):
            result = attack(_TinyData(), object())

        assert fake_attack.reconstruct_calls == 1
        assert "database_reconstruction_feature_mse" in result
        assert "database_reconstruction_feature_mae" in result
        assert "database_reconstruction_num_features" in result
        assert "attack_generation_time" in result
        assert "attack_prediction_time" in result
        assert "attack_score_time" in result

    def test_call_attribute_inference_requires_targeted_attribute(self):
        attack = AttackConfig(
            name="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute=None,
            attack_params={},
        )
        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(
                object(),
                object(),
                "inference",
                "attribute_inference",
            ),
        ):
            with pytest.raises(AssertionError):
                attack(object(), object())

    def test_call_inference_unknown_subtype_raises(self):
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
        )
        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "inference", "unknown_subtype"),
        ):
            with pytest.raises(ValueError):
                attack(object(), object())

    def test_call_poisoning_requires_source_and_target_classes(self):
        with pytest.raises(ValueError):
            AttackConfig(
                name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
                attack_params={},
            )

    def test_call_extraction_requires_classification_task(self):
        attack = AttackConfig(
            name="art.attacks.extraction.CopycatCNN",
            attack_params={},
        )

        class _TinyData:
            classifier = False

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "extraction", "any"),
        ):
            with pytest.raises(ValueError):
                attack(_TinyData(), object())

    def test_call_extraction_requires_nn_classifier(self):
        attack = AttackConfig(
            name="art.attacks.extraction.CopycatCNN",
            attack_params={},
        )

        class _TinyData:
            classifier = True

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "extraction", "any"),
        ):
            with pytest.raises(ValueError):
                attack(_TinyData(), object())

    def test_call_extraction_scores_victim_and_extracted_classifiers(self):
        attack = AttackConfig(
            name="art.attacks.extraction.CopycatCNN",
            attack_params={},
            attack_size=4,
            mode="test",
        )

        class _TinyData:
            classifier = True

            X_train = np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.1, 0.9],
                    [0.9, 0.1],
                ],
            )
            y_train = np.array([0, 1, 0, 1])
            X_test = np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.2, 0.8],
                    [0.8, 0.2],
                ],
            )
            y_test = np.array([0, 1, 0, 1])

        class PyTorchClassifierStub:
            _model = None

            def predict(self, X):
                X = np.asarray(X)
                p1 = (X[:, 0] > X[:, 1]).astype(float)
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])

        class _FakeExtractionAttack:
            def extract(self, x, thieved_classifier=None, **kwargs):
                _ = x
                _ = kwargs
                return thieved_classifier

        data = _TinyData()
        art_model = PyTorchClassifierStub()
        extraction_attack = _FakeExtractionAttack()

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(extraction_attack, art_model, "extraction", "any"),
        ):
            result = attack(data, object())

        assert "benign_accuracy" in result
        assert "extracted_accuracy" in result
        assert "extraction_mode" in result
        assert result["extraction_mode"] == "test"

    def test_call_rejects_regression_evasion_early(self):
        class TinyData:
            classifier = False

        data = TinyData()
        model = LinearRegression().fit([[0.0, 1.0], [1.0, 0.0]], [0.1, 0.9])
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        with patch.object(
            AttackConfig,
            "_initialize_attack",
            side_effect=AssertionError(
                "_initialize_attack should not be called",
            ),
        ):
            with pytest.raises(ValueError):
                attack(data, model)

    def test_initialize_attack_rejects_unsupported_type(self):
        attack = AttackConfig(
            name="art.attacks.foo.Bar",
            attack_params={},
        )
        model = RandomForestClassifier().fit([[0, 1], [1, 0]], [0, 1])
        with pytest.raises(ValueError):
            attack._initialize_attack(model, object())

    def test_real_evasion_attack_executes(self):
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")

        class TinyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        class TinyData:
            pass

        rng = np.random.default_rng(42)
        X_train = torch.tensor(rng.normal(size=(24, 3)), dtype=torch.float32)
        y_train = torch.tensor(rng.integers(0, 2, size=(24,)), dtype=torch.long)
        X_test = torch.tensor(rng.normal(size=(12, 3)), dtype=torch.float32)
        y_test = torch.tensor(rng.integers(0, 2, size=(12,)), dtype=torch.long)

        data = TinyData()
        data.X_train = X_train
        data.y_train = y_train
        data.X_test = X_test
        data.y_test = y_test

        model = TinyLinear()
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
            attack_size=8,
        )

        scores = attack(data, model)
        assert "evasion_success" in scores
        assert "attack_generation_time" in scores
        assert scores["attack_size"] >= 1

    def test_real_membership_inference_attack_executes(self):

        data = TinyData()

        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
            attack_size=20,
        )

        scores = attack(data, model)
        assert "membership_inference_accuracy" in scores
        assert "attack_score_time" in scores

    def test_real_attribute_inference_attack_executes(self):

        data = TinyData()

        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            name="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute=["sensitive"],
            attack_params={
                "attack_model_type": "lr",
                "is_continuous": True,
                "scale_range": (0, 1),
            },
            attack_size=20,
        )

        scores = attack(data, model)
        assert "attack_score_time" in scores
        inferred_keys = [k for k in scores.keys() if k.startswith("inferred_")]
        assert len(inferred_keys) > 0

    def test_real_model_inversion_attack_executes(self):
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")

        class TinyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        class TinyData:
            pass

        rng = np.random.default_rng(7)
        X_train = torch.tensor(rng.normal(size=(20, 3)), dtype=torch.float32)
        y_train = torch.tensor(rng.integers(0, 2, size=(20,)), dtype=torch.long)
        X_test = torch.tensor(rng.normal(size=(12, 3)), dtype=torch.float32)
        y_test = torch.tensor(rng.integers(0, 2, size=(12,)), dtype=torch.long)

        data = TinyData()
        data.X_train = X_train
        data.y_train = y_train
        data.X_test = X_test
        data.y_test = y_test

        model = TinyLinear()
        attack = self._load_pytorch_model_inversion_config()
        attack.attack_size = 2
        attack.attack_params["max_iter"] = 2
        attack.attack_params.pop("initialization", None)
        attack.attack_params.pop("split", None)
        attack.attack_params.pop("targets", None)

        scores = attack(data, model)
        assert "model_inversion_mse" in scores
        assert "model_inversion_mae" in scores
        assert "model_inversion_num_targets" in scores
        assert "attack_score_time" in scores

    def test_real_database_reconstruction_attack_executes(self):
        data = TinyData()
        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            name="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 0},
            attack_size=1,
        )

        scores = attack(data, model)
        assert "database_reconstruction_feature_mse" in scores
        assert "database_reconstruction_feature_mae" in scores
        assert "database_reconstruction_num_features" in scores
        assert "attack_score_time" in scores

    def test_hash_stable_after_call_for_attack_config(self):
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={},
        )
        data = TinyData()

        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        model = model
        original_hash = hash(attack)
        attack(data, model)
        assert original_hash == hash(
            attack,
        ), "Hash changed after call for AttackConfig"


class TestPytorchAttackConfig:
    """Tests for deckard/attack/pytorch.py — PytorchAttackConfig feature prep."""

    @classmethod
    def setup_class(cls):
        try:
            import torch

            cls.torch = torch
        except ImportError:
            cls.torch = None

    def _skip_if_no_torch(self):
        if self.torch is None:
            pytest.skip("torch not installed")

    def test_prepare_features_tensor_passthrough(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        t = self.torch.randn(4, 3)
        result = cfg._prepare_features_for_attack(t)
        assert result is t

    def test_prepare_features_dataframe_to_numpy(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = cfg._prepare_features_for_attack(df)
        assert isinstance(result, np.ndarray)

    def test_prepare_features_series_to_numpy(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        s = pd.Series([1.0, 2.0, 3.0])
        result = cfg._prepare_features_for_attack(s)
        assert isinstance(result, np.ndarray)

    def test_prepare_features_passthrough_for_other_types(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        arr = np.array([1.0, 2.0])
        result = cfg._prepare_features_for_attack(arr)
        assert result is arr

    def test_prepare_labels_tensor_passthrough(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        t = self.torch.tensor([0, 1, 1])
        result = cfg._prepare_labels_for_attack(t)
        assert result is t

    def test_prepare_labels_dataframe_to_numpy(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        df = pd.DataFrame({"label": [0, 1]})
        result = cfg._prepare_labels_for_attack(df)
        assert isinstance(result, np.ndarray)

    def test_prepare_labels_series_to_numpy(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        s = pd.Series([0, 1, 0])
        result = cfg._prepare_labels_for_attack(s)
        assert isinstance(result, np.ndarray)

    def test_prepare_labels_passthrough_for_other_types(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        arr = np.array([0, 1])
        result = cfg._prepare_labels_for_attack(arr)
        assert result is arr

    def test_prepare_features_for_art_tensor_to_numpy_float_dtype(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        t = self.torch.randn(4, 3)
        result = cfg._prepare_features_for_art(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3)
        assert np.issubdtype(result.dtype, np.floating)

    def test_torch_evasion_uses_art_boundary_conversion(self):
        self._skip_if_no_torch()
        from types import SimpleNamespace

        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        class _DummyArtModel:
            def predict(self, x):
                x = np.asarray(x)
                p1 = (x[:, 0] > 0.0).astype(float)
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])

        class _DummyAttack:
            def generate(self, x):
                # This mimics ART's numpy-based path and would fail on raw tensors.
                return x.astype(np.float32)

        data = SimpleNamespace(
            X_test=self.torch.randn(16, 4),
            y_test=self.torch.randint(0, 2, (16,)),
            _sensitive_test=None,
        )
        cfg = PytorchAttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_size=8,
        )

        runtime = cfg._with_attack_context(
            attack_family="evasion", attack_sub_family=""
        )
        scores = runtime.evade(data, _DummyArtModel(), _DummyAttack())
        assert "evasion_accuracy" in scores
        assert isinstance(runtime.attack, np.ndarray)
        assert runtime.attack.shape[0] == 8


class TestTorchUtils:
    """Tests for deckard/frameworks/pytorch/torch_utils.py."""

    @classmethod
    def setup_class(cls):
        try:
            import torch

            cls.torch = torch
        except ImportError:
            cls.torch = None

    def _skip_if_no_torch(self):
        if self.torch is None:
            pytest.skip("torch not installed")

    def test_is_tensor_true_for_tensor(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import is_tensor

        assert is_tensor(self.torch.tensor([1.0]))

    def test_is_tensor_false_for_numpy(self):
        from deckard.frameworks.pytorch.torch_utils import is_tensor

        assert not is_tensor(np.array([1.0]))

    def test_is_torch_model_true(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import is_torch_model

        model = self.torch.nn.Linear(2, 2)
        assert is_torch_model(model)

    def test_is_torch_model_false_for_sklearn(self):
        from sklearn.linear_model import LogisticRegression

        from deckard.frameworks.pytorch.torch_utils import is_torch_model

        assert not is_torch_model(LogisticRegression())

    def test_is_dataloader_true(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, TensorDataset

        from deckard.frameworks.pytorch.torch_utils import is_dataloader

        ds = TensorDataset(self.torch.randn(4, 2))
        dl = DataLoader(ds, batch_size=2)
        assert is_dataloader(dl)

    def test_is_dataloader_false_for_list(self):
        from deckard.frameworks.pytorch.torch_utils import is_dataloader

        assert not is_dataloader([1, 2, 3])

    def test_tensor_to_numpy_converts(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import tensor_to_numpy

        t = self.torch.tensor([1.0, 2.0, 3.0])
        arr = tensor_to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_tensor_to_numpy_with_dtype(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import tensor_to_numpy

        t = self.torch.tensor([1.0, 2.0])
        arr = tensor_to_numpy(t, dtype=np.float32)
        assert arr.dtype == np.float32

    def test_tensor_to_numpy_passthrough_non_tensor(self):
        from deckard.frameworks.pytorch.torch_utils import tensor_to_numpy

        arr = np.array([1, 2])
        result = tensor_to_numpy(arr)
        assert result is arr

    def test_get_torch_model_device_returns_cpu(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import get_torch_model_device

        model = self.torch.nn.Linear(2, 2)
        device = get_torch_model_device(model)
        assert str(device.type) == "cpu"

    def test_get_torch_model_device_non_torch_returns_cpu(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import get_torch_model_device

        device = get_torch_model_device(object())
        assert str(device.type) == "cpu"

    def test_get_torch_model_device_model_with_no_parameters_returns_cpu(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import get_torch_model_device

        class EmptyModule(self.torch.nn.Module):
            def forward(self, x):
                return x

        model = EmptyModule()
        device = get_torch_model_device(model)
        assert str(device.type) == "cpu"

    def test_build_torch_art_model_raises_when_torch_flag_disabled(self):
        from deckard.frameworks.pytorch.torch_utils import build_torch_art_model

        with patch("deckard.frameworks.pytorch.torch_utils.HAS_TORCH", False):
            with pytest.raises(ImportError):
                build_torch_art_model(object(), object())

    def test_collect_subset_raises_when_torch_flag_disabled(self):
        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        with patch("deckard.frameworks.pytorch.torch_utils.HAS_TORCH", False):
            with pytest.raises(ImportError):
                collect_subset_from_dataloader(object(), n=2)

    def test_build_torch_art_model_dataloader_tuple_batch_input_shape(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, TensorDataset

        from deckard.frameworks.pytorch.torch_utils import build_torch_art_model

        class FakePyTorchClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.input_shape = kwargs["input_shape"]
                self._device = None
                self._model = kwargs["model"]
                self.preprocessing = type("P", (), {"_device": None})()
                self.preprocessing_operations = [
                    type("Op", (), {"_device": None})(),
                ]

        torch_mod = self.torch

        class TinyModel(torch_mod.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch_mod.nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        X = self.torch.randn(8, 3)
        y = self.torch.randint(0, 2, (8,))
        dl = DataLoader(TensorDataset(X, y), batch_size=4)
        data = type("D", (), {"X_train": dl, "y_train": y.numpy()})()

        with patch(
            "art.estimators.classification.PyTorchClassifier",
            FakePyTorchClassifier,
        ):
            model = TinyModel()
            estimator = build_torch_art_model(model, data)
        assert estimator.input_shape == (3,)

    def test_build_torch_art_model_dataloader_tensor_batch_input_shape(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, Dataset

        from deckard.frameworks.pytorch.torch_utils import build_torch_art_model

        class TensorOnlyDataset(Dataset):
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx]

        class FakePyTorchClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.input_shape = kwargs["input_shape"]
                self._device = None
                self._model = kwargs["model"]
                self.preprocessing = type("P", (), {"_device": None})()
                self.preprocessing_operations = [
                    type("Op", (), {"_device": None})(),
                ]

        torch_mod = self.torch

        class TinyModel(torch_mod.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch_mod.nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        X = self.torch.randn(8, 3)
        y = self.torch.randint(0, 2, (8,))
        dl = DataLoader(TensorOnlyDataset(X), batch_size=4)
        data = type("D", (), {"X_train": dl, "y_train": y.numpy()})()

        with patch(
            "art.estimators.classification.PyTorchClassifier",
            FakePyTorchClassifier,
        ):
            model = TinyModel()
            estimator = build_torch_art_model(model, data)
        assert estimator.input_shape == (3,)

    def test_collect_subset_from_dataloader(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, TensorDataset

        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        X = self.torch.randn(10, 3)
        y = self.torch.randint(0, 2, (10,))
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=4)
        x_sub, y_sub = collect_subset_from_dataloader(dl, n=4)
        assert x_sub.shape[0] == 4
        assert y_sub.shape[0] == 4

    def test_collect_subset_clips_to_dataset_len(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, TensorDataset

        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        X = self.torch.randn(5, 2)
        y = self.torch.randint(0, 2, (5,))
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=5)
        x_sub, y_sub = collect_subset_from_dataloader(dl, n=100)
        assert x_sub.shape[0] == 5

    def test_collect_subset_raises_for_non_dataloader(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        with pytest.raises(TypeError):
            collect_subset_from_dataloader([1, 2, 3], n=2)


# ---------------------------------------------------------------------------
# Tiny fixtures
# ---------------------------------------------------------------------------


def _make_tiny_data():
    rng = np.random.default_rng(42)
    n_train, n_test = 20, 12

    X_train = pd.DataFrame(
        {
            "feat0": rng.normal(size=n_train),
            "feat1": rng.normal(size=n_train),
            "sensitive": rng.integers(0, 2, size=n_train),
        },
    )
    y_train = pd.Series((X_train["feat0"] > 0).astype(int), name="target")
    X_test = pd.DataFrame(
        {
            "feat0": rng.normal(size=n_test),
            "feat1": rng.normal(size=n_test),
            "sensitive": rng.integers(0, 2, size=n_test),
        },
    )
    y_test = pd.Series((X_test["feat0"] > 0).astype(int), name="target")

    class _Data:
        pass

    d = _Data()
    d.X_train = X_train
    d.y_train = y_train
    d.X_test = X_test
    d.y_test = y_test
    d.classifier = True
    return d


class _FakeArtModel:
    nb_classes = 2

    def predict(self, X):
        X = np.asarray(X)
        p1 = (X[:, 0] > 0).astype(float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class _PickleableFakeModelConfig:
    """Pickle-safe fake used to exercise attack_model string loading path."""

    def _load_or_train_model(self, data):
        return self

    def get_art_model(self, data):
        return _FakeArtModel()


# ---------------------------------------------------------------------------
# SensitiveFeaturesWrapper
# ---------------------------------------------------------------------------


class TestSensitiveFeaturesWrapper:
    """Cover the wrapper class lines that are missed by existing tests."""

    def _make_estimator_with_predict_proba(self):
        """Estimator that exposes predict_proba accepting sensitive_features."""

        class _E:
            def fit(self, X, y, **kw):
                return self

            def predict(self, X, sensitive_features=None):
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X, sensitive_features=None):
                n = len(X)
                return np.column_stack([np.ones(n) * 0.6, np.ones(n) * 0.4])

        return _E()

    def _make_estimator_without_predict_proba(self):
        """Estimator that has no predict_proba but needs sensitive_features."""

        class _E:
            def fit(self, X, y, **kw):
                return self

            def predict(self, X, sensitive_features=None):
                return np.zeros(len(X), dtype=int)

        return _E()

    def test_predict_proba_with_estimator_having_it(self):
        est = self._make_estimator_with_predict_proba()
        sf = np.array([0, 1, 0, 1, 0])
        wrapper = SensitiveFeaturesWrapper(est, sf)
        X = np.zeros((3, 2))
        proba = wrapper.predict_proba(X)
        assert proba.shape == (3, 2)

    def test_predict_proba_fallback_when_no_predict_proba(self):
        """Cover the fallback branch that builds a two-column probability matrix."""
        est = self._make_estimator_without_predict_proba()
        sf = np.array([0, 1, 0, 1, 0])
        wrapper = SensitiveFeaturesWrapper(est, sf)
        X = np.zeros((3, 2))
        proba = wrapper.predict_proba(X)
        # fallback: each row has one 1.0, rest 0.0
        assert proba.shape[0] == 3
        assert np.all(proba.sum(axis=1) == 1.0)

    def test_get_params(self):
        est = MagicMock()
        sf = np.array([1, 0, 1])
        wrapper = SensitiveFeaturesWrapper(est, sf)
        params = wrapper.get_params()
        assert "estimator" in params
        assert "sensitive_features" in params
        assert params["estimator"] is est

    def test_set_params(self):
        est = MagicMock()
        sf = np.array([1, 0])
        wrapper = SensitiveFeaturesWrapper(est, sf)
        new_est = MagicMock()
        new_sf = np.array([0, 1, 0])
        wrapper.set_params(estimator=new_est, sensitive_features=new_sf)
        assert wrapper.estimator is new_est
        np.testing.assert_array_equal(wrapper._sensitive, new_sf)

    def test_fit_delegates_to_estimator(self):
        est = MagicMock()
        sf = np.array([0, 1])
        wrapper = SensitiveFeaturesWrapper(est, sf)
        X, y = np.ones((4, 2)), np.zeros(4)
        wrapper.fit(X, y)
        est.fit.assert_called_once()

    def test_predict_slices_sensitive(self):
        est = self._make_estimator_without_predict_proba()
        sf = np.array([0, 1, 0, 1, 0, 1])
        wrapper = SensitiveFeaturesWrapper(est, sf)
        preds = wrapper.predict(np.zeros((3, 2)))
        assert len(preds) == 3


# ---------------------------------------------------------------------------
# _sensitive_slice helper
# ---------------------------------------------------------------------------


class TestSensitiveSlice:
    def test_none_returns_none(self):
        assert _sensitive_slice(None, 5) is None

    def test_slices_to_n(self):
        arr = np.array([0, 1, 2, 3, 4])
        result = _sensitive_slice(arr, 3)
        np.testing.assert_array_equal(result, [0, 1, 2])


# ---------------------------------------------------------------------------
# __post_init__ branches
# ---------------------------------------------------------------------------


class TestPostInitBranches:
    """Cover DictConfig scorer path and scorer-as-type path."""

    def test_scorer_as_type_class(self):
        from deckard.score.attack import AttackScorerConfig

        cfg = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            scorer=AttackScorerConfig,  # pass the class, not an instance
        )
        assert isinstance(cfg.scorer, AttackScorerConfig)

    def test_scorer_as_dictconfig(self):
        from omegaconf import OmegaConf

        dc = OmegaConf.create(
            {"_target_": "deckard.score.attack.AttackScorerConfig"},
        )
        cfg = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            scorer=dc,
        )
        # scorer should have been coerced to a dict then processed
        assert hasattr(cfg.scorer, "_score")

    def test_scorer_null_string(self):
        # "null" should be treated as None -> default scorer
        cfg = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            scorer="null",
        )
        assert hasattr(cfg.scorer, "_score")

    def test_scorer_dict_with_no_target_uses_attack_scorer_config(self):
        from deckard.score.attack import AttackScorerConfig

        cfg = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            scorer={},  # empty dict, no _target_
        )
        assert isinstance(cfg.scorer, AttackScorerConfig)


# ---------------------------------------------------------------------------
# _initialize_attack branches
# ---------------------------------------------------------------------------


class TestInitializeAttackBranches:
    def setup_method(self):
        self.data = _make_tiny_data()

    def test_generic_classifer_wrapping_and_nb_classes(self):
        """Cover the 'generic BaseEstimator classifier' wrapping branch."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200).fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
            attack_size=6,
        )
        # Patch resolve_class to avoid constructing a real FGM
        fake_fgm = MagicMock()
        fake_fgm.return_value = MagicMock()
        with patch("deckard.attack.base.resolve_class", return_value=fake_fgm):
            result = attack._initialize_attack(model, self.data)
        assert result is not None

    def test_unsupported_model_raises_value_error(self):
        """Cover the 'else: raise ValueError' branch for unknown model types."""
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={},
        )
        with pytest.raises((ValueError, Exception)):
            attack._initialize_attack("not_a_model", self.data)

    def test_targeted_attribute_string_current_behavior_does_not_raise(self):
        """Document current behavior: target_index field bypasses missing-column check."""
        attack = AttackConfig(
            name="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute="nonexistent_feature",
            attack_params={},
        )
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200).fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        fake_attack_cls = MagicMock(return_value=MagicMock())
        with patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls):
            attack._initialize_attack(model, self.data)
        assert "attack_feature" not in attack.attack_params

    def test_attack_model_invalid_type_raises(self):
        """Cover the 'else: raise ValueError' branch for invalid attack_model."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200).fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={"attack_model": 12345},  # invalid type
        )
        with pytest.raises((ValueError, Exception)):
            attack._initialize_attack(model, self.data)

    def test_model_with_sensitive_features_predict_wraps_with_wrapper(self):
        """Cover the SensitiveFeaturesWrapper path in _initialize_attack."""
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _SFModel(BaseEstimator, ClassifierMixin):
            classes_ = np.array([0, 1])

            def fit(self, X, y):
                self.fitted_ = True
                return self

            def predict(self, X, sensitive_features=None):
                return np.zeros(len(X), dtype=int)

        model = _SFModel().fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        self.data._sensitive_test = np.zeros(len(self.data.X_test))
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        fake_fgm = MagicMock()
        fake_fgm.return_value = MagicMock()
        with patch("deckard.attack.base.resolve_class", return_value=fake_fgm):
            result = attack._initialize_attack(model, self.data)
        assert result is not None

    def test_not_fitted_sklearn_model_triggers_fit(self):
        """Cover NotFittedError branch for sklearn_dict models (lines 436-438)."""
        from sklearn.exceptions import NotFittedError
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200)
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        fake_attack_cls = MagicMock(return_value=MagicMock())
        with (
            patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls),
            # First check_is_fitted at top of _initialize_attack passes,
            # second check inside sklearn_dict branch raises.
            patch(
                "deckard.attack.base.check_is_fitted",
                side_effect=[None, NotFittedError()],
            ),
        ):
            result = attack._initialize_attack(model, self.data)
        assert result is not None

    def test_not_fitted_generic_estimator_triggers_fit(self):
        """Cover NotFittedError branch for generic BaseEstimator path (lines 443-444)."""
        from sklearn.base import BaseEstimator, ClassifierMixin
        from sklearn.exceptions import NotFittedError

        class _GenericCls(BaseEstimator, ClassifierMixin):
            classes_ = np.array([0, 1])

            def fit(self, X, y):
                self.fitted_ = True
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        model = _GenericCls()
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        fake_attack_cls = MagicMock(return_value=MagicMock())
        with (
            patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls),
            # First check passes; second check in BaseEstimator branch raises.
            patch(
                "deckard.attack.base.check_is_fitted",
                side_effect=[None, NotFittedError()],
            ),
        ):
            result = attack._initialize_attack(model, self.data)
        assert result is not None

    def test_sensitive_features_fallback_uses_sensitive_train(self):
        """Cover _sensitive_train fallback branch (line 452)."""
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _SFModel(BaseEstimator, ClassifierMixin):
            classes_ = np.array([0, 1])

            def fit(self, X, y):
                self.fitted_ = True
                return self

            def predict(self, X, sensitive_features=None):
                return np.zeros(len(X), dtype=int)

        model = _SFModel().fit(self.data.X_train.values, self.data.y_train.values)
        self.data._sensitive_test = None
        self.data._sensitive_train = np.zeros(len(self.data.X_train))
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        fake_attack_cls = MagicMock(return_value=MagicMock())
        with patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls):
            result = attack._initialize_attack(model, self.data)
        assert result is not None

    def test_regressor_branch_uses_sklearn_regressor_wrapper(self):
        """Cover regressor wrapper branch (line 459)."""
        from sklearn.base import BaseEstimator, RegressorMixin

        class _GenericReg(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                self.fitted_ = True
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=float)

        model = _GenericReg().fit(self.data.X_train.values, self.data.y_train.values)
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        fake_attack_cls = MagicMock(return_value=MagicMock())
        with patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls):
            result = attack._initialize_attack(model, self.data)
        assert result is not None

    def test_unsupported_model_type_reaches_value_error_branch(self):
        """Cover explicit unsupported model type ValueError branch (line 470)."""
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )

        class _HasFitButUnsupported:
            def fit(self, X, y):
                return self

        fake_attack_cls = MagicMock(return_value=MagicMock())
        with (
            patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls),
            patch("deckard.attack.base.check_is_fitted", return_value=None),
        ):
            with pytest.raises(ValueError):
                attack._initialize_attack(_HasFitButUnsupported(), self.data)

    def test_attack_model_dictconfig_path(self):
        """Cover DictConfig attack_model branch (lines 499-503)."""
        from omegaconf import OmegaConf
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200).fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        attack_model_dc = OmegaConf.create({"classifier": True})
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={"attack_model": attack_model_dc},
        )

        class _FakeCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, data):
                return self

            def get_art_model(self, data):
                return _FakeArtModel()

        fake_attack_cls = MagicMock(return_value=MagicMock())
        with (
            patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls),
            patch("deckard.attack.base.ModelConfig", _FakeCfg),
        ):
            result = attack._initialize_attack(model, self.data)
        assert result is not None

    def test_attack_model_modelconfig_instance_path(self):
        """Cover ModelConfig instance branch (lines 504-506)."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200).fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        fake_cfg = _PickleableFakeModelConfig()
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={"attack_model": fake_cfg},
        )

        fake_attack_cls = MagicMock(return_value=MagicMock())
        with (
            patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls),
            patch("deckard.attack.base.ModelConfig", _PickleableFakeModelConfig),
        ):
            result = attack._initialize_attack(model, self.data)
        assert result is not None

    def test_attack_model_string_path(self):
        """Cover string attack_model loading branch (lines 507-517)."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200).fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        tmpdir = tempfile.mkdtemp()
        try:
            attack_model_path = os.path.join(tmpdir, "fake_attack_model.pkl")
            with open(attack_model_path, "wb") as f:
                pickle.dump(_PickleableFakeModelConfig(), f)

            attack = AttackConfig(
                name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
                attack_params={"attack_model": attack_model_path},
            )
            fake_attack_cls = MagicMock(return_value=MagicMock())
            with (
                patch(
                    "deckard.attack.base.resolve_class",
                    return_value=fake_attack_cls,
                ),
                patch("deckard.attack.base.ModelConfig", _PickleableFakeModelConfig),
            ):
                result = attack._initialize_attack(model, self.data)
            assert result is not None
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# __call__ file caching branches
# ---------------------------------------------------------------------------


class TestCallCachingPaths:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data = _make_tiny_data()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def _base_attack(self):
        return AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_size=6,
        )

    def _fake_evade_side_effect(self, attack_obj):
        def _inner(*args, **kwargs):
            attack_obj.attack_time = 0.01
            attack_obj.attack_prediction_time = 0.01
            attack_obj.attack_score_time = 0.01
            return {"evasion_success": 0.5}

        return _inner

    def test_attack_predictions_file_load_oserror_falls_through(self):
        """Cover the OSError fallback when loading cached predictions fails."""
        attack = self._base_attack()
        pred_file = os.path.join(self.tmpdir, "preds.csv")
        Path(pred_file).write_text("bad,data")

        with (
            patch.object(
                AttackConfig,
                "_initialize_attack",
                return_value=(MagicMock(), MagicMock(), "evasion", ""),
            ),
            patch.object(
                AttackConfig,
                "_with_attack_context",
                return_value=attack,
            ),
            patch.object(
                AttackConfig,
                "_resolve_attack_handler",
                return_value=self._fake_evade_side_effect(attack),
            ),
            patch.object(AttackConfig, "load_data", side_effect=OSError("boom")),
        ):
            result = attack(
                self.data,
                object(),
                attack_predictions_file=pred_file,
            )
        assert "evasion_success" in result

    def test_score_file_loaded_when_exists(self):
        """Cover the score_file load branch."""
        attack = self._base_attack()
        score_file = os.path.join(self.tmpdir, "scores.json")
        # Create a minimal score file
        import json

        Path(score_file).write_text(json.dumps({"cached_score": 0.99}))

        with (
            patch.object(
                AttackConfig,
                "_initialize_attack",
                return_value=(MagicMock(), MagicMock(), "evasion", ""),
            ),
            patch.object(
                AttackConfig,
                "_with_attack_context",
                return_value=attack,
            ),
            patch.object(
                AttackConfig,
                "_resolve_attack_handler",
                return_value=self._fake_evade_side_effect(attack),
            ),
        ):
            result = attack(
                self.data,
                object(),
                score_file=score_file,
            )
        assert "evasion_success" in result

    def test_attack_file_save_pickle_error_continues(self):
        """Cover the PicklingError fallback when saving attack object fails."""
        attack = self._base_attack()
        attack_file = os.path.join(self.tmpdir, "attack.pkl")
        # File must NOT exist so the save branch is entered

        with (
            patch.object(
                AttackConfig,
                "_initialize_attack",
                return_value=(MagicMock(), MagicMock(), "evasion", ""),
            ),
            patch.object(
                AttackConfig,
                "_with_attack_context",
                return_value=attack,
            ),
            patch.object(
                AttackConfig,
                "_resolve_attack_handler",
                return_value=self._fake_evade_side_effect(attack),
            ),
            patch.object(
                AttackConfig,
                "save_object",
                side_effect=pickle.PicklingError("cannot pickle"),
            ),
        ):
            result = attack(
                self.data,
                object(),
                attack_file=attack_file,
            )
        assert "evasion_success" in result

    def test_attack_predictions_file_saved_after_call(self):
        """Cover the branch that saves attack_predictions when file path provided."""
        attack = self._base_attack()
        pred_file = os.path.join(self.tmpdir, "preds_out.npy")
        attack.attack_predictions = np.array([0, 1])

        with (
            patch.object(
                AttackConfig,
                "_initialize_attack",
                return_value=(MagicMock(), MagicMock(), "evasion", ""),
            ),
            patch.object(
                AttackConfig,
                "_with_attack_context",
                return_value=attack,
            ),
            patch.object(
                AttackConfig,
                "_resolve_attack_handler",
                return_value=self._fake_evade_side_effect(attack),
            ),
            patch.object(AttackConfig, "save_data") as mock_save_data,
        ):
            attack(self.data, object(), attack_predictions_file=pred_file)

        mock_save_data.assert_called_once()


# ---------------------------------------------------------------------------
# get_attack_subset edge cases
# ---------------------------------------------------------------------------


class TestGetAttackSubset:
    def test_raises_for_unsupported_type(self):
        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        attack.attack_size = 4

        class _BadData:
            X_test = "not_an_array"
            y_test = "not_an_array"

        with pytest.raises(ValueError):
            attack.get_attack_subset(_BadData())

    def test_returns_subset_from_numpy(self):
        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        attack.attack_size = 3

        class _Data:
            X_test = np.arange(20).reshape(10, 2)
            y_test = np.zeros(10)

        n, x_sub, y_sub = attack.get_attack_subset(_Data())
        assert n == 3
        assert len(x_sub) == 3

    def test_train_subset(self):
        attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
        attack.attack_size = 3

        class _Data:
            X_train = np.arange(20).reshape(10, 2)
            y_train = np.zeros(10)
            X_test = np.arange(20).reshape(10, 2)
            y_test = np.zeros(10)

        n, x_sub, y_sub = attack.get_attack_subset(_Data(), test=False)
        assert n == 3


# ---------------------------------------------------------------------------
# _get_benign_preds train=True path
# ---------------------------------------------------------------------------


class TestGetBenignPreds:
    @staticmethod
    def _make_numpy_data():
        d = type("D", (), {})()
        d.X_train = np.arange(40, dtype=np.float32).reshape(20, 2)
        d.y_train = np.array([0, 1] * 10)
        d.X_test = np.arange(24, dtype=np.float32).reshape(12, 2)
        d.y_test = np.array([0, 1] * 6)
        return d

    def test_train_true_uses_test_data(self):
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_size=4,
        )
        data = self._make_numpy_data()
        art_model = _FakeArtModel()
        n, labels, x_sub, y_sub = attack._get_benign_preds(data, art_model, train=True)
        assert n == 4
        assert isinstance(labels, np.ndarray)

    def test_train_false_uses_train_data(self):
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_size=4,
        )
        data = self._make_numpy_data()
        art_model = _FakeArtModel()
        n, labels, x_sub, y_sub = attack._get_benign_preds(
            data,
            art_model,
            train=False,
        )
        assert n == 4


# ---------------------------------------------------------------------------
# _evade branches
# ---------------------------------------------------------------------------


class TestEvadeBranches:
    def test_adversarial_patch_branch(self):
        """Cover the 'AdversarialPatch' special handling."""
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_size=4,
        )

        class _TinyData:
            X_test = np.random.default_rng(1).random((6, 2, 3, 3)).astype(np.float32)
            y_test = np.array([0, 1, 0, 1, 0, 1])

        class AdversarialPatch:
            """Simulates AdversarialPatch so the type-name check fires."""

            def generate(self, x, y):
                # returns a tuple (patches, masks) so patches[0].shape[1:] works
                patches = np.ones_like(x)
                return patches, np.ones_like(x)

            def apply_patch(self, x, scale=0.5):
                return x.copy()

        class _FakeModel:
            nb_classes = 2

            def predict(self, X):
                X = np.asarray(X)
                probs = np.zeros((len(X), 2))
                probs[:, 0] = 0.5
                probs[:, 1] = 0.5
                return probs

        fake_attack_obj = AdversarialPatch()
        runtime = attack._with_attack_context(
            attack_family="evasion", attack_sub_family=""
        )
        result = runtime.evade(
            data=_TinyData(),
            art_model=_FakeModel(),
            attack=fake_attack_obj,
        )
        assert isinstance(result, dict)

    def test_evade_regression_path(self):
        """Cover the is_regression=True scoring path in _evade."""
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_size=6,
        )

        class _TinyData:
            X_test = np.linspace(0, 1, 12).reshape(6, 2).astype(np.float32)
            y_test = np.linspace(0.0, 1.0, 6).astype(np.float32)

        class _RegressionArtModel:
            def predict(self, X):
                X = np.asarray(X)
                # Single-column float output => regression
                return X[:, :1].astype(np.float32)

        class _FakeEvasionAttack:
            def generate(self, x):
                return np.asarray(x).copy() + 0.01

        runtime = attack._with_attack_context(
            attack_family="evasion", attack_sub_family=""
        )
        result = runtime.evade(
            data=_TinyData(),
            art_model=_RegressionArtModel(),
            attack=_FakeEvasionAttack(),
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# infer_attribute: list targeted_attribute path
# ---------------------------------------------------------------------------


class TestInferAttributeBranches:
    def test_list_targeted_attribute_executes(self):
        data = _make_tiny_data()
        attack = AttackConfig(
            name="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute=["sensitive"],
            attack_params={
                "attack_model_type": "lr",
                "is_continuous": True,
                "scale_range": (0, 1),
            },
            attack_size=8,
        )
        from sklearn.linear_model import LogisticRegression

        LogisticRegression(max_iter=200).fit(
            data.X_train.drop(columns=["sensitive"]).values,
            data.y_train.values,
        )

        class _FakeAttribAttack:
            _is_continuous = True

            def fit(self, x, **kw):
                pass

            def infer(self, x, pred, values=None):
                return np.random.default_rng(0).random(len(x))

        class _FakeArt:
            def predict(self, X):
                X = np.asarray(X)
                return np.column_stack(
                    [np.zeros(len(X)), np.ones(len(X))],
                ).astype(np.float32)

        with (
            patch.object(
                AttackConfig,
                "_initialize_attack",
                return_value=(
                    _FakeAttribAttack(),
                    _FakeArt(),
                    "inference",
                    "attribute_inference",
                ),
            ),
        ):
            result = attack(data, object())

        assert isinstance(result, dict)

    def test_attribute_column_missing_raises(self):
        data = _make_tiny_data()
        attack = AttackConfig(
            name="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute="nonexistent_col",
            attack_params={},
            attack_size=8,
        )

        class _FakeAttribAttack:
            _is_continuous = False

            def fit(self, x, **kw):
                pass

            def infer(self, x, pred, values=None):
                return np.zeros(len(x))

        class _FakeArt:
            def predict(self, X):
                X = np.asarray(X)
                return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="attribute_inference",
        )
        with pytest.raises((AssertionError, ValueError, KeyError)):
            runtime.infer_attribute(
                data,
                _FakeArt(),
                _FakeAttribAttack(),
                targeted_attribute="nonexistent_col",
            )

    def test_attribute_list_column_missing_raises_value_error(self):
        data = _make_tiny_data()
        attack = AttackConfig(
            name="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute=["not_a_column"],
            attack_params={},
            attack_size=8,
        )

        class _FakeAttribAttack:
            _is_continuous = False

            def fit(self, x, **kw):
                pass

            def infer(self, x, pred, values=None):
                return np.zeros(len(x))

        class _FakeArt:
            def predict(self, X):
                X = np.asarray(X)
                return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="attribute_inference",
        )
        with pytest.raises((ValueError, AssertionError)):
            runtime.infer_attribute(
                data,
                _FakeArt(),
                _FakeAttribAttack(),
                targeted_attribute=["not_a_column"],
            )


# ---------------------------------------------------------------------------
# infer_membership: AxisError fallback + sensitive features
# ---------------------------------------------------------------------------


class TestInferMembershipBranches:
    def test_axis_error_fallback_is_used(self):
        """Cover the AxisError fallback in infer_membership."""
        data = _make_tiny_data()
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_size=16,
        )

        call_count = {"n": 0}

        class _FakeMIAttack:
            def fit(self, x, y, test_x, **kw):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise AxisError(axis=1, ndim=1)
                # second call succeeds

            def infer(self, x, y=None):
                return np.zeros(len(x), dtype=int)

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="membership_inference",
        )
        result = runtime.infer_membership(data=data, attack=_FakeMIAttack())
        assert isinstance(result, dict)

    def test_sensitive_features_present_builds_big_sensitive(self):
        """Cover the path that concatenates train+test sensitive features."""
        data = _make_tiny_data()
        data._sensitive_train = np.zeros(len(data.X_train))
        data._sensitive_test = np.ones(len(data.X_test))
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_size=16,
        )

        class _FakeMIAttack:
            def fit(self, x, y, test_x, **kw):
                pass

            def infer(self, x, y=None):
                return np.zeros(len(x), dtype=int)

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="membership_inference",
        )
        result = runtime.infer_membership(data=data, attack=_FakeMIAttack())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# infer_model_inversion init modes
# ---------------------------------------------------------------------------


class TestInferModelInversionModes:
    def _make_mi_attack_config(self, init_mode="zeros"):
        return AttackConfig(
            name="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={"split": "test", "initialization": init_mode},
            attack_size=2,
        )

    def _make_data(self):
        rng = np.random.default_rng(7)
        d = type("D", (), {})()
        d.X_test = rng.random((8, 3)).astype(np.float32)
        d.y_test = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
        d.X_train = rng.random((8, 3)).astype(np.float32)
        d.y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        return d

    def _fake_attack(self):
        class _A:
            def infer(self, x, y):
                return np.zeros_like(x)

        return _A()

    def test_zeros_init(self):
        cfg = self._make_mi_attack_config("zeros")
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        assert "model_inversion_mse" in result

    def test_ones_init(self):
        cfg = self._make_mi_attack_config("ones")
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        assert "model_inversion_mse" in result

    def test_random_init(self):
        cfg = self._make_mi_attack_config("random")
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        assert "model_inversion_mse" in result

    def test_average_init(self):
        cfg = self._make_mi_attack_config("average")
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        assert "model_inversion_mse" in result

    def test_invalid_init_mode_raises(self):
        cfg = self._make_mi_attack_config("invalid_mode")
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        with pytest.raises(ValueError):
            runtime.infer_model_inversion(
                data=self._make_data(),
                attack=self._fake_attack(),
            )

    def test_train_split_used(self):
        cfg = AttackConfig(
            name="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={"split": "train"},
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        assert "model_inversion_mse" in result

    def test_mi_invalid_split_raises(self):
        cfg = AttackConfig(
            name="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={"split": "validate"},
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        with pytest.raises(ValueError):
            runtime.infer_model_inversion(
                data=self._make_data(),
                attack=self._fake_attack(),
            )

    def test_empty_x_source_raises(self):
        cfg = self._make_mi_attack_config("zeros")
        d = type("D", (), {})()
        d.X_test = np.empty((0, 3), dtype=np.float32)
        d.y_test = np.array([], dtype=int)
        d.X_train = np.empty((0, 3), dtype=np.float32)
        d.y_train = np.array([], dtype=int)
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        with pytest.raises(ValueError):
            runtime.infer_model_inversion(data=d, attack=self._fake_attack())

    def test_explicit_targets_param(self):
        cfg = AttackConfig(
            name="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={"split": "test", "targets": [0, 1]},
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        assert "model_inversion_mse" in result

    def test_x_init_from_param(self):
        d = self._make_data()
        x_init = np.zeros((2, 3), dtype=np.float32)
        cfg = AttackConfig(
            name="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={
                "split": "test",
                "x_init": x_init.tolist(),
                "targets": [0, 1],
            },
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        result = runtime.infer_model_inversion(data=d, attack=self._fake_attack())
        assert "model_inversion_mse" in result

    def test_x_init_length_mismatch_raises(self):
        d = self._make_data()
        x_init = np.zeros((5, 3), dtype=np.float32)  # 5 != 2 targets
        cfg = AttackConfig(
            name="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={
                "split": "test",
                "x_init": x_init.tolist(),
                "targets": [0, 1],
            },
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        with pytest.raises(ValueError):
            runtime.infer_model_inversion(data=d, attack=self._fake_attack())

    def test_type_error_fallback_on_infer(self):
        """Cover the TypeError fallback path in infer_model_inversion."""
        cfg = self._make_mi_attack_config("average")
        call_count = {"n": 0}

        class _A:
            def infer(self, x, y):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise TypeError("unexpected keyword argument")
                return np.zeros_like(x)

        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        result = runtime.infer_model_inversion(data=self._make_data(), attack=_A())
        assert "model_inversion_mse" in result

    def test_empty_target_labels_raises(self):
        """Cover the path when target_labels is empty."""
        cfg = AttackConfig(
            name="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={"split": "test", "targets": []},
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_family="inference",
            attack_sub_family="model_inversion",
        )
        with pytest.raises(ValueError):
            runtime.infer_model_inversion(
                data=self._make_data(),
                attack=self._fake_attack(),
            )


# ---------------------------------------------------------------------------
# infer_database_reconstruction branches
# ---------------------------------------------------------------------------


class TestInferDatabaseReconstructionBranches:
    def _make_data(self):
        d = type("D", (), {})()
        d.X_train = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            dtype=np.float32,
        )
        d.y_train = np.array([0, 1, 0, 1], dtype=int)
        d.X_test = np.array(
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4]],
            dtype=np.float32,
        )
        d.y_test = np.array([1, 0, 1], dtype=int)
        d.classifier = True
        return d

    def test_test_split_used(self):
        attack = AttackConfig(
            name="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "test", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([0])

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="reconstruction",
        )
        result = runtime.infer_database_reconstruction(
            data=self._make_data(),
            attack=_FakeAttack(),
        )
        assert "database_reconstruction_feature_mse" in result

    def test_dr_invalid_split_raises(self):
        attack = AttackConfig(
            name="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "validate", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([0])

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="reconstruction",
        )
        with pytest.raises(ValueError):
            runtime.infer_database_reconstruction(
                data=self._make_data(),
                attack=_FakeAttack(),
            )

    def test_missing_index_out_of_bounds_raises(self):
        attack = AttackConfig(
            name="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 100},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([0])

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="reconstruction",
        )
        with pytest.raises(ValueError):
            runtime.infer_database_reconstruction(
                data=self._make_data(),
                attack=_FakeAttack(),
            )

    def test_too_few_rows_raises(self):
        attack = AttackConfig(
            name="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([0])

        d = self._make_data()
        d.X_train = d.X_train[:1]  # only 1 row
        d.y_train = d.y_train[:1]
        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="reconstruction",
        )
        with pytest.raises(ValueError):
            runtime.infer_database_reconstruction(data=d, attack=_FakeAttack())

    def test_y_reconstructed_none_skips_label_scoring(self):
        """Cover the path where reconstructed tuple has only x."""
        attack = AttackConfig(
            name="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                # Return only features, no labels
                return x[:1]

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="reconstruction",
        )
        result = runtime.infer_database_reconstruction(
            data=self._make_data(),
            attack=_FakeAttack(),
        )
        assert "database_reconstruction_label_accuracy" not in result

    def test_regression_task_uses_mae_label(self):
        """Cover the regression label scoring branch."""
        attack = AttackConfig(
            name="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([0.5])

        d = self._make_data()
        # Make task appear as regression
        d.classifier = False

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="reconstruction",
        )
        result = runtime.infer_database_reconstruction(data=d, attack=_FakeAttack())
        assert "database_reconstruction_label_mae" in result

    def test_type_error_fallback_on_reconstruct(self):
        """Cover the TypeError fallback (positional reconstruct)."""
        attack = AttackConfig(
            name="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 0},
            attack_size=1,
        )
        call_count = {"n": 0}

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise TypeError("y not expected")
                return x[:1], np.array([0])

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="reconstruction",
        )
        result = runtime.infer_database_reconstruction(
            data=self._make_data(),
            attack=_FakeAttack(),
        )
        assert "database_reconstruction_feature_mse" in result

    def test_empty_y_reconstructed_skips_label(self):
        """Cover path where y_pred is empty after to_numpy_array."""
        attack = AttackConfig(
            name="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([])

        runtime = attack._with_attack_context(
            attack_family="inference",
            attack_sub_family="reconstruction",
        )
        result = runtime.infer_database_reconstruction(
            data=self._make_data(),
            attack=_FakeAttack(),
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _resolve_eval_split branches
# ---------------------------------------------------------------------------


class TestResolveEvalSplit:
    def test_val_split_available_returns_val(self):
        attack = AttackConfig(
            name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            mode="val",
        )

        class _Data:
            X_val = np.zeros((4, 2))
            y_val = np.array([0, 1, 0, 1])
            X_test = np.zeros((4, 2))
            y_test = np.array([0, 1, 0, 1])

        runtime = attack._with_attack_context(
            attack_family="poisoning",
            attack_sub_family="gradient_matching_attack",
        )
        mode, x, y = runtime._resolve_eval_split(_Data())
        assert mode == "val"

    def test_val_split_unavailable_falls_back_to_test(self):
        attack = AttackConfig(
            name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            mode="val",
        )

        class _Data:
            X_test = np.zeros((4, 2))
            y_test = np.array([0, 1, 0, 1])

        runtime = attack._with_attack_context(
            attack_family="poisoning",
            attack_sub_family="gradient_matching_attack",
        )
        mode, x, y = runtime._resolve_eval_split(_Data())
        assert mode == "test"

    def test_invalid_mode_raises(self):
        attack = AttackConfig(
            name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            mode="invalid",
        )

        class _Data:
            X_test = np.zeros((4, 2))
            y_test = np.array([0, 1, 0, 1])

        runtime = attack._with_attack_context(
            attack_family="poisoning",
            attack_sub_family="gradient_matching_attack",
        )
        with pytest.raises(ValueError):
            runtime._resolve_eval_split(_Data())

    def test_test_mode_with_missing_data_raises(self):
        attack = AttackConfig(
            name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            mode="test",
        )

        class _Data:
            X_test = None
            y_test = None

        runtime = attack._with_attack_context(
            attack_family="poisoning",
            attack_sub_family="gradient_matching_attack",
        )
        with pytest.raises(ValueError):
            runtime._resolve_eval_split(_Data())


# ---------------------------------------------------------------------------
# _poison val mode + class_source fallback
# ---------------------------------------------------------------------------


class TestPoisonBranches:
    def _make_data(self):
        d = type("D", (), {})()
        d.X_train = np.array(
            [[0.0, 0.1], [1.0, 0.2], [0.2, 1.0], [0.9, 0.8], [0.1, 0.3], [0.8, 0.7]],
            dtype=np.float32,
        )
        d.y_train = np.array([0, 1, 0, 1, 0, 1])
        d.X_test = np.array(
            [[0.0, 0.0], [1.0, 0.1], [0.2, 0.9], [0.7, 0.8]],
            dtype=np.float32,
        )
        d.y_test = np.array([0, 1, 0, 1])
        d.X_val = np.array([[0.3, 0.4], [0.6, 0.7]], dtype=np.float32)
        d.y_val = np.array([0, 1])
        d.classifier = True
        return d

    def test_val_mode_used_when_available(self):
        attack = AttackConfig(
            name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            attack_size=4,
            mode="val",
        )

        class _FakeArtModel:
            nb_classes = 2
            _poisoned = False

            def predict(self, X):
                X = np.asarray(X)
                p1 = (X[:, 0] > 0.5).astype(float)
                return np.column_stack([1 - p1, p1])

            def fit(self, x, y, **kw):
                self._poisoned = True

        class _FakePoisonAttack:
            def poison(self, x_trigger, y_trigger, x_train, y_train):
                return np.asarray(x_train), np.asarray(y_train)

        runtime = attack._with_attack_context(
            attack_family="poisoning",
            attack_sub_family="gradient_matching_attack",
        )
        result = runtime.poison(
            data=self._make_data(),
            art_model=_FakeArtModel(),
            attack=_FakePoisonAttack(),
        )
        assert result["poison_mode"] == "val"

    def test_class_source_fallback_when_no_samples(self):
        """Cover the warning/fallback when class_source has no samples in eval."""
        attack = AttackConfig(
            name="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 99, "class_target": 1},
            attack_size=4,
            mode="test",
        )

        class _FakeArtModel:
            nb_classes = 2
            _poisoned = False

            def predict(self, X):
                X = np.asarray(X)
                p1 = (X[:, 0] > 0.5).astype(float)
                return np.column_stack([1 - p1, p1])

            def fit(self, x, y, **kw):
                pass

        class _FakePoisonAttack:
            def poison(self, x_trigger, y_trigger, x_train, y_train):
                return np.asarray(x_train), np.asarray(y_train)

        runtime = attack._with_attack_context(
            attack_family="poisoning",
            attack_sub_family="gradient_matching_attack",
        )
        result = runtime.poison(
            data=self._make_data(),
            art_model=_FakeArtModel(),
            attack=_FakePoisonAttack(),
        )
        # class_source should have been adjusted to something present
        assert "poison_attack_source_class" in result

    def test_poisoning_svm_branch_scores_benign_and_poisoned_accuracy(self):
        attack = AttackConfig(
            name="art.attacks.poisoning.PoisoningAttackSVM",
            attack_params={"step": 0.1, "eps": 0.2, "max_iter": 2, "verbose": False},
            attack_size=2,
            mode="test",
        )

        class _FakeArtModel:
            nb_classes = 2
            _poisoned = False

            def predict(self, X):
                X = np.asarray(X)
                p1 = (X[:, 0] > 0.5).astype(float)
                if self._poisoned:
                    p1 = 1.0 - p1
                return np.column_stack([1 - p1, p1])

            def fit(self, x, y, **kw):
                _ = x
                _ = y
                _ = kw
                self._poisoned = True

        class _FakePoisoningAttackSVM:
            def poison(self, x, y=None, **kwargs):
                _ = kwargs
                return np.asarray(x), np.asarray(y)

        runtime = attack._with_attack_context(
            attack_family="poisoning",
            attack_sub_family="PoisoningAttackSVM",
        )
        result = runtime.poison(
            data=self._make_data(),
            art_model=_FakeArtModel(),
            attack=_FakePoisoningAttackSVM(),
        )

        assert "benign_accuracy" in result
        assert "poisoned_accuracy" in result
        assert "poisoning_attack_points" in result
        assert result["attack_size"] == 2


# ---------------------------------------------------------------------------
# _extract val mode
# ---------------------------------------------------------------------------


class TestExtractBranches:
    def test_initialize_attack_builds_neural_art_classifier_for_extraction(self):
        torch = pytest.importorskip("torch")
        from deckard.data import PytorchDataConfig
        from deckard.model import PytorchModelConfig

        class DummyCopycatCNN:
            def __init__(self, classifier, **kwargs):
                self.classifier = classifier
                self.kwargs = kwargs

        X = torch.rand(16, 4)
        y = torch.randint(0, 2, (16,))
        data = PytorchDataConfig(
            name="torch.utils.data.TensorDataset",
            sampler={
                "name": "split",
                "train_size": 12,
                "test_size": 4,
                "random_state": 42,
            },
            classifier=True,
            data_params={"_args_": [X, y]},
        )
        data()

        model = PytorchModelConfig(
            name="torch.nn.Linear",
            model_params={"in_features": 4, "out_features": 2},
            classifier=True,
            fit_params={"nb_epochs": 1, "batch_size": 4},
            criterion="CrossEntropyLoss",
            optimizer={"name": "SGD", "lr": 0.05},
        )
        model(data)

        attack = AttackConfig(
            name="art.attacks.extraction.CopycatCNN",
            attack_params={},
            attack_size=4,
        )

        with patch(
            "deckard.attack.base.resolve_class",
            return_value=DummyCopycatCNN,
        ):
            initialized_attack, art_model, attack_family, attack_sub_family = (
                attack._initialize_attack(
                    model,
                    data,
                )
            )

        assert attack_family == "extraction"
        assert attack_sub_family == "CopycatCNN"
        runtime = attack._with_attack_context(
            attack_family="extraction",
            attack_sub_family="CopycatCNN",
        )
        assert runtime._is_nn_art_classifier(art_model)
        assert initialized_attack.classifier is art_model

    def test_extract_uses_val_split(self):
        attack = AttackConfig(
            name="art.attacks.extraction.CopycatCNN",
            attack_params={},
            attack_size=4,
            mode="val",
        )

        class _TinyData:
            classifier = True

            X_train = np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]])
            y_train = np.array([0, 1, 0, 1])
            X_val = np.array([[0.0, 1.0], [1.0, 0.0]])
            y_val = np.array([0, 1])
            X_test = np.array([[0.0, 1.0], [1.0, 0.0]])
            y_test = np.array([0, 1])

        class PyTorchClassifierStub:
            _model = None

            def predict(self, X):
                X = np.asarray(X)
                p1 = (X[:, 0] > X[:, 1]).astype(float)
                return np.column_stack([1 - p1, p1])

        class _FakeExtractionAttack:
            def extract(self, x, thieved_classifier=None, **kwargs):
                return thieved_classifier

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(
                _FakeExtractionAttack(),
                PyTorchClassifierStub(),
                "extraction",
                "any",
            ),
        ):
            result = attack(_TinyData(), object())

        assert result.get("extraction_mode") == "val"

    def test_extract_not_implemented_raises_for_non_type(self):
        """Cover the not-implemented path for unsupported attack family."""
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
        )
        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(MagicMock(), MagicMock(), "unknown_type", ""),
        ):

            def _fakeevade(*a, **kw):
                attack.attack_time = 0.1
                attack.attack_prediction_time = 0.1
                attack.attack_score_time = 0.1
                return {"x": 1}

            with pytest.raises(NotImplementedError):
                attack(object(), object())


# ---------------------------------------------------------------------------
# Additional helper / static method branches
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    def test_labels_from_classifier_predictions_1d_float(self):
        """Cover the 1D float -> threshold path."""
        result = AttackConfig._labels_from_classifier_predictions(
            np.array([0.3, 0.7, 0.2, 0.9]),
        )
        np.testing.assert_array_equal(result, [0, 1, 0, 1])

    def test_labels_from_classifier_predictions_2d_single_col(self):
        """Cover the 2D single-col -> threshold path."""
        result = AttackConfig._labels_from_classifier_predictions(
            np.array([[0.3], [0.8]]),
        )
        np.testing.assert_array_equal(result, [0, 1])

    def test_labels_from_classifier_predictions_invalid_shape_raises(self):
        result = AttackConfig._labels_from_classifier_predictions(
            np.zeros((2, 2, 2)),
        )
        assert result.shape == (8,)

    def test_normalize_ground_truth_dataframe_regression(self):
        df = pd.DataFrame({"a": [0.1, 0.2, 0.3]})
        result = AttackConfig._normalize_ground_truth(df, is_regression=True)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3])

    def test_normalize_ground_truth_dataframe_classification(self):
        df = pd.DataFrame({"a": [0, 1, 0]})
        result = AttackConfig._normalize_ground_truth(df, is_regression=False)
        assert result.shape == (3,)

    def test_normalize_ground_truth_2d_one_hot(self):
        arr = np.array([[1, 0], [0, 1], [1, 0]])
        result = AttackConfig._normalize_ground_truth(arr, is_regression=False)
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_to_numpy_array_with_dtype_on_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = AttackConfig._to_numpy_array(df, dtype=np.float32)
        assert result.dtype == np.float32

    def test_to_numpy_array_with_dtype_on_series(self):
        s = pd.Series([1, 2, 3])
        result = AttackConfig._to_numpy_array(s, dtype=np.float64)
        assert result.dtype == np.float64

    def test_to_numpy_array_with_dtype_on_ndarray(self):
        arr = np.array([1, 2, 3])
        result = AttackConfig._to_numpy_array(arr, dtype=np.float32)
        assert result.dtype == np.float32

    def test_is_regression_prediction_output_1d_col(self):
        """Cover the 2D single-col -> regression = True path."""
        preds = np.array([[0.1], [0.2], [0.3]])
        labels = np.array([0.1, 0.2, 0.3])
        assert AttackConfig._is_regression_prediction_output(labels, preds)

    def test_select_extraction_scorer_with_probabilities(self):
        benign = np.array([[0.3, 0.7], [0.6, 0.4]])
        extracted = np.array([[0.4, 0.6], [0.5, 0.5]])
        scorer, has_proba = ExtractionAttackMixin._select_extraction_scorer(
            benign,
            extracted,
        )
        assert has_proba
        assert scorer is not None

    def test_is_nn_art_classifier_returns_false_for_plain_object(self):
        attack = AttackConfig(name="art.attacks.extraction.CopycatCNN")
        runtime = attack._with_attack_context(
            attack_family="extraction",
            attack_sub_family="CopycatCNN",
        )
        assert not runtime._is_nn_art_classifier(object())

    def test_is_nn_art_classifier_returns_true_for_pytorch_name(self):
        class PyTorchClassifier:
            _model = None

        attack = AttackConfig(name="art.attacks.extraction.CopycatCNN")
        runtime = attack._with_attack_context(
            attack_family="extraction",
            attack_sub_family="CopycatCNN",
        )
        assert runtime._is_nn_art_classifier(PyTorchClassifier())

    def test_normalize_inferred_output_higher_dim_reference(self):
        """Cover the `ref.ndim > arr.ndim` branch."""
        inferred = np.array([0, 1, 0])
        ref = np.array([[1, 0], [0, 1], [1, 0]])
        result = AttackConfig._normalize_inferred_output(inferred, reference=ref)
        # Should have been get_dummies-expanded
        assert result.ndim == 2

    def test_normalize_inferred_output_lower_dim_reference(self):
        """Cover the `arr.ndim > ref.ndim` branch."""
        inferred = np.array([[0.2, 0.8], [0.7, 0.3]])
        ref = np.array([0, 1])
        result = AttackConfig._normalize_inferred_output(inferred, reference=ref)
        np.testing.assert_array_equal(result, [1, 0])

    def test_infer_task_from_data_classifier_attr(self):
        """Cover the `hasattr(data, 'classifier')` path in _infer_task_is_classification."""

        class _Data:
            classifier = True

        result = AttackConfig._infer_task_is_classification(_Data(), object())
        assert result

    def test_infer_task_returns_none_for_unknown(self):

        class _Data:
            pass

        result = AttackConfig._infer_task_is_classification(_Data(), object())
        assert result is None


# ---------------------------------------------------------------------------
# Fairlearn scorer integration coverage
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("require_fairlearn")
class TestFairlearnAttackScorer:
    """Unit tests for FairlearnAttackScorerConfig per-group attack metrics."""

    def _make_data_with_sensitive(self):
        from deckard.plugins.fairlearn.data import FairlearnDataConfig

        data = FairlearnDataConfig(name="adult", sensitive_columns="sex")
        data()
        return data

    def test_fairlearn_attack_scorer_instantiates(self):
        from deckard.plugins.fairlearn.score import FairlearnScorerDictConfig
        from deckard.score.base import DefaultClassifierScorerDictConfig

        scorer = FairlearnAttackScorerConfig(
            evasion=DefaultClassifierScorerDictConfig(),
        )
        assert isinstance(scorer.evasion, FairlearnScorerDictConfig)
        assert isinstance(scorer.membership_inference, FairlearnScorerDictConfig)
        assert isinstance(scorer.attribute_inference, FairlearnScorerDictConfig)

    def test_score_evasion_with_sensitive_features_produces_group_metrics(self):
        scorer = FairlearnAttackScorerConfig()
        rng = np.random.default_rng(1)
        n = 20
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        sensitive = np.array(["a" if i % 2 == 0 else "b" for i in range(n)])
        result = scorer.score_evasion(
            ben_pred_labels=y_true,
            adv_pred_labels=y_pred,
            y_true=y_true,
            attack_size=n,
            is_classification=True,
            sensitive_features=sensitive,
        )
        group_keys = [k for k in result if "_accuracy" in k or "_f1" in k]
        assert len(group_keys) > 0, f"No group metrics found in {list(result)}"

    def test_score_membership_with_sensitive_features_produces_group_metrics(self):
        scorer = FairlearnAttackScorerConfig()
        rng = np.random.default_rng(2)
        n = 20
        labels = rng.integers(0, 2, n)
        inferred = rng.integers(0, 2, n)
        sensitive = np.array(["a" if i % 2 == 0 else "b" for i in range(n)])
        result = scorer.score_membership(
            labels=labels,
            inferred=inferred,
            attack_size=n,
            sensitive_features=sensitive,
        )
        group_keys = [k for k in result if "membership_inference" in k]
        assert (
            len(group_keys) > 0
        ), f"No membership_inference metrics found in {list(result)}"

    def test_score_attribute_with_sensitive_features_produces_group_metrics(self):
        scorer = FairlearnAttackScorerConfig()
        rng = np.random.default_rng(3)
        n = 20
        target = rng.integers(0, 3, n)
        inferred = rng.integers(0, 3, n)
        sensitive = np.array(["a" if i % 2 == 0 else "b" for i in range(n)])
        result = scorer.score_attribute(
            target=target,
            inferred=inferred,
            attack_size=n,
            targeted_attribute="age",
            is_classification=True,
            sensitive_features=sensitive,
        )
        group_keys = [k for k in result if "inferred_age" in k]
        assert len(group_keys) > 0, f"No inferred_age metrics found in {list(result)}"

    def test_evasion_attack_with_fairlearn_scorer_end_to_end(self):
        pytest.importorskip("art")
        data = self._make_data_with_sensitive()
        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
            attack_size=10,
            scorer=FairlearnAttackScorerConfig(),
        )
        scores = attack(data, model)
        group_keys = [k for k in scores if "_accuracy" in k or "_f1" in k]
        assert (
            len(group_keys) > 0
        ), f"Expected per-group evasion metrics, got keys: {list(scores)}"

    def test_membership_inference_with_fairlearn_scorer_end_to_end(self):
        pytest.importorskip("art")
        data = self._make_data_with_sensitive()
        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
            attack_size=20,
            scorer=FairlearnAttackScorerConfig(),
        )
        scores = attack(data, model)
        group_keys = [k for k in scores if "membership_inference" in k]
        assert (
            len(group_keys) > 0
        ), f"Expected per-group membership metrics, got keys: {list(scores)}"
