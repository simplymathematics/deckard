import unittest
from pathlib import Path
import os
import tempfile
import shutil
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from deckard.attack import AttackConfig


class TinyData:
    rng = np.random.default_rng(11)
    X_train = pd.DataFrame(
        {
            "feature": rng.normal(size=40),
            "sensitive": rng.integers(0, 2, size=40),
            "other": rng.normal(size=40),
        },
    )
    y_train = pd.Series(
        (X_train["feature"] + X_train["other"] > 0).astype(int),
        name="target",
    )
    X_test = pd.DataFrame(
        {
            "feature": rng.normal(size=24),
            "sensitive": rng.integers(0, 2, size=24),
            "other": rng.normal(size=24),
        },
    )
    y_test = pd.Series(
        (X_test["feature"] + X_test["other"] > 0).astype(int),
        name="target",
    )


class TestAttackConfig(unittest.TestCase):
    def setUp(self):
        self.attack_params = {}
        self.attack_type = "art.attacks.evasion.FastGradientMethod"
        self.attack = AttackConfig(
            attack_type=self.attack_type,
            attack_params=self.attack_params,
        )
        self.tmpdir = tempfile.mkdtemp()
        self.attack_file = os.path.join(self.tmpdir, "attack.pkl")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_post_init(self):
        self.assertTrue(hasattr(self.attack, "attack_type"))
        self.assertTrue(hasattr(self.attack, "attack_params"))

    def test_save_and_load_attack(self):
        self.attack.save(self.attack_file)
        self.assertTrue(Path(self.attack_file).exists())
        loaded_attack = AttackConfig()
        loaded_attack.load(self.attack_file)
        self.assertEqual(loaded_attack.attack_type, self.attack.attack_type)
        self.assertEqual(loaded_attack.attack_params, self.attack.attack_params)

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
        self.assertIn("evasion_success", metrics)

    def test_evade_includes_benign_scores(self):
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
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

        result = attack._evade(
            data=_TinyData(),
            art_model=_FakeArtModel(),
            attack=_FakeEvasionAttack(),
        )

        self.assertIn("benign_accuracy", result)
        self.assertIn("benign_precision", result)
        self.assertIn("benign_recall", result)
        self.assertIn("benign_f1", result)
        self.assertIn("evasion_success", result)

    def test_poison_attack_metrics(self):
        attack = AttackConfig(
            attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            attack_size=6,
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

        self.assertIn("benign_accuracy", result)
        self.assertIn("poisoned_accuracy", result)
        self.assertIn("poison_trigger_success", result)
        self.assertIn("attack_generation_time", result)

    def test_call_attack(self):
        # Mock attack internals to verify call flow without requiring a specific ART estimator.
        data = object()
        model = object()

        def _fake_evade(*args, **kwargs):
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
            patch.object(AttackConfig, "_evade", side_effect=_fake_evade),
        ):
            result = self.attack(data, model)

        self.assertIsNotNone(result)
        self.assertIn("evasion_success", result)
        self.assertIn("attack_generation_time", result)

    def test_call_membership_inference(self):
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
        )

        def _fake_membership(*args, **kwargs):
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
                "_infer_membership",
                side_effect=_fake_membership,
            ),
        ):
            result = attack(object(), object())

        self.assertIn("membership_inference_accuracy", result)
        self.assertIn("attack_generation_time", result)

    def test_call_attribute_inference(self):
        attack = AttackConfig(
            attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute="age",
            attack_params={},
        )

        def _fake_attribute(*args, **kwargs):
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
                "_infer_attribute",
                side_effect=_fake_attribute,
            ),
        ):
            result = attack(object(), object())

        self.assertIn("inferred_age_accuracy", result)
        self.assertIn("attack_prediction_time", result)

    def test_call_attribute_inference_requires_targeted_attribute(self):
        attack = AttackConfig(
            attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
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
            with self.assertRaises(AssertionError):
                attack(object(), object())

    def test_call_inference_unknown_subtype_raises(self):
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
        )
        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "inference", "unknown_subtype"),
        ):
            with self.assertRaises(ValueError):
                attack(object(), object())

    def test_call_poisoning_requires_source_and_target_classes(self):
        with self.assertRaises(ValueError):
            AttackConfig(
                attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
                attack_params={},
            )

    def test_call_extraction_requires_classification_task(self):
        attack = AttackConfig(
            attack_type="art.attacks.extraction.CopycatCNN",
            attack_params={},
        )

        class _TinyData:
            classifier = False

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "extraction", "any"),
        ):
            with self.assertRaises(ValueError):
                attack(_TinyData(), object())

    def test_call_extraction_requires_nn_classifier(self):
        attack = AttackConfig(
            attack_type="art.attacks.extraction.CopycatCNN",
            attack_params={},
        )

        class _TinyData:
            classifier = True

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "extraction", "any"),
        ):
            with self.assertRaises(ValueError):
                attack(_TinyData(), object())

    def test_call_extraction_scores_victim_and_extracted_classifiers(self):
        attack = AttackConfig(
            attack_type="art.attacks.extraction.CopycatCNN",
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

        self.assertIn("benign_accuracy", result)
        self.assertIn("extracted_accuracy", result)
        self.assertIn("extraction_mode", result)
        self.assertEqual(result["extraction_mode"], "test")

    def test_call_rejects_regression_evasion_early(self):
        class TinyData:
            classifier = False

        data = TinyData()
        model = LinearRegression().fit([[0.0, 1.0], [1.0, 0.0]], [0.1, 0.9])
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        with patch.object(
            AttackConfig,
            "_initialize_attack",
            side_effect=AssertionError(
                "_initialize_attack should not be called",
            ),
        ):
            with self.assertRaises(ValueError):
                attack(data, model)

    def test_initialize_attack_rejects_unsupported_type(self):
        attack = AttackConfig(
            attack_type="art.attacks.foo.Bar",
            attack_params={},
        )
        model = RandomForestClassifier().fit([[0, 1], [1, 0]], [0, 1])
        with self.assertRaises(ValueError):
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
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
            attack_size=8,
        )

        scores = attack(data, model)
        self.assertIn("evasion_success", scores)
        self.assertIn("attack_generation_time", scores)
        self.assertGreaterEqual(scores["attack_size"], 1)

    def test_real_membership_inference_attack_executes(self):

        data = TinyData()

        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
            attack_size=20,
        )

        scores = attack(data, model)
        self.assertIn("membership_inference_accuracy", scores)
        self.assertIn("attack_score_time", scores)

    def test_real_attribute_inference_attack_executes(self):

        data = TinyData()

        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute=["sensitive"],
            attack_params={
                "attack_model_type": "lr",
                "is_continuous": True,
                "scale_range": (0, 1),
            },
            attack_size=20,
        )

        scores = attack(data, model)
        self.assertIn("attack_score_time", scores)
        inferred_keys = [k for k in scores.keys() if k.startswith("inferred_")]
        self.assertTrue(len(inferred_keys) > 0)

    def test_hash_stable_after_call_for_attack_config(self):
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
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
        self.assertEqual(
            original_hash,
            hash(attack),
            msg="Hash changed after call for AttackConfig",
        )


class TestFairlearnAttackScorer(unittest.TestCase):
    """Unit tests for FairlearnAttackScorerConfig per-group attack metrics."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_fairlearn(self):
        pytest.importorskip("fairlearn")

    def _make_data_with_sensitive(self):
        data = TinyData()
        data._sensitive_train = pd.Series(
            ["group_a" if i % 2 == 0 else "group_b" for i in range(len(data.X_train))],
            name="sensitive",
        )
        data._sensitive_test = pd.Series(
            ["group_a" if i % 2 == 0 else "group_b" for i in range(len(data.X_test))],
            name="sensitive",
        )
        return data

    def test_fairlearn_attack_scorer_instantiates(self):
        from deckard.score.attack import FairlearnAttackScorerConfig
        from deckard.score.fairness import FairlearnScoreDictConfig

        scorer = FairlearnAttackScorerConfig()
        self.assertIsInstance(scorer.evasion, FairlearnScoreDictConfig)
        self.assertIsInstance(
            scorer.membership_inference,
            FairlearnScoreDictConfig,
        )
        self.assertIsInstance(
            scorer.attribute_inference,
            FairlearnScoreDictConfig,
        )

    def test_score_evasion_with_sensitive_features_produces_group_metrics(self):
        from deckard.score.attack import FairlearnAttackScorerConfig

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
        self.assertTrue(
            len(group_keys) > 0,
            f"No group metrics found in {list(result)}",
        )

    def test_score_membership_with_sensitive_features_produces_group_metrics(
        self,
    ):
        from deckard.score.attack import FairlearnAttackScorerConfig

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
        self.assertTrue(
            len(group_keys) > 0,
            f"No membership_inference metrics found in {list(result)}",
        )

    def test_score_attribute_with_sensitive_features_produces_group_metrics(
        self,
    ):
        from deckard.score.attack import FairlearnAttackScorerConfig

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
        self.assertTrue(
            len(group_keys) > 0,
            f"No inferred_age metrics found in {list(result)}",
        )

    def test_evasion_attack_with_fairlearn_scorer_end_to_end(self):
        from deckard.score.attack import FairlearnAttackScorerConfig

        pytest.importorskip("art")
        data = self._make_data_with_sensitive()
        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
            attack_size=10,
            scorer=FairlearnAttackScorerConfig(),
        )
        scores = attack(data, model)
        group_keys = [k for k in scores if "_accuracy" in k or "_f1" in k]
        self.assertTrue(
            len(group_keys) > 0,
            f"Expected per-group evasion metrics, got keys: {list(scores)}",
        )

    def test_membership_inference_with_fairlearn_scorer_end_to_end(self):
        from deckard.score.attack import FairlearnAttackScorerConfig

        pytest.importorskip("art")
        data = self._make_data_with_sensitive()
        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
            attack_size=20,
            scorer=FairlearnAttackScorerConfig(),
        )
        scores = attack(data, model)
        group_keys = [k for k in scores if "membership_inference" in k]
        self.assertTrue(
            len(group_keys) > 0,
            f"Expected per-group membership metrics, got keys: {list(scores)}",
        )


if __name__ == "__main__":
    unittest.main()
