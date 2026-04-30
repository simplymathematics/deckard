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
from sklearn.linear_model import LogisticRegression
from deckard.attack import AttackConfig


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
        self.attack._score_attack(ben_pred_labels, adv_pred_labels, y_test_numeric)
        metrics = self.attack.score_dict
        self.assertIn("evasion_success", metrics)

    def test_call_attack(self):
        # Mock attack internals to verify call flow without requiring a specific ART estimator.
        data = object()
        model = object()

        def _fake_evade(*args, **kwargs):
            self.attack.attack_time = 0.1
            self.attack.attack_prediction_time = 0.05
            self.attack.attack_score_time = 0.02
            return {"evasion_success": 0.3}

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "evasion", ""),
        ), patch.object(AttackConfig, "_evade", side_effect=_fake_evade):
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

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "inference", "membership_inference"),
        ), patch.object(AttackConfig, "_infer_membership", side_effect=_fake_membership):
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

        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "inference", "attribute_inference"),
        ), patch.object(AttackConfig, "_infer_attribute", side_effect=_fake_attribute):
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
            return_value=(object(), object(), "inference", "attribute_inference"),
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

    def test_call_poisoning_not_implemented(self):
        attack = AttackConfig(
            attack_type="art.attacks.poisoning.PoisoningAttackBackdoor",
            attack_params={},
        )
        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "poisoning", "any"),
        ):
            with self.assertRaises(NotImplementedError):
                attack(object(), object())

    def test_call_extraction_not_implemented(self):
        attack = AttackConfig(
            attack_type="art.attacks.extraction.CopycatCNN",
            attack_params={},
        )
        with patch.object(
            AttackConfig,
            "_initialize_attack",
            return_value=(object(), object(), "extraction", "any"),
        ):
            with self.assertRaises(NotImplementedError):
                attack(object(), object())

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
        class TinyData:
            pass

        rng = np.random.default_rng(7)
        X_train = pd.DataFrame(rng.normal(size=(30, 4)), columns=["f1", "f2", "f3", "f4"])
        y_train = pd.Series(rng.integers(0, 2, size=(30,)), name="target")
        X_test = pd.DataFrame(rng.normal(size=(20, 4)), columns=["f1", "f2", "f3", "f4"])
        y_test = pd.Series(rng.integers(0, 2, size=(20,)), name="target")

        data = TinyData()
        data.X_train = X_train
        data.y_train = y_train
        data.X_test = X_test
        data.y_test = y_test

        model = LogisticRegression(max_iter=200).fit(X_train.values, y_train.values)
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
            attack_size=20,
        )

        scores = attack(data, model)
        self.assertIn("membership_inference_accuracy", scores)
        self.assertIn("attack_score_time", scores)

    def test_real_attribute_inference_attack_executes(self):
        class TinyData:
            pass

        rng = np.random.default_rng(11)
        X_train = pd.DataFrame(
            {
                "feature": rng.normal(size=40),
                "sensitive": rng.integers(0, 2, size=40),
                "other": rng.normal(size=40),
            }
        )
        y_train = pd.Series((X_train["feature"] + X_train["other"] > 0).astype(int), name="target")
        X_test = pd.DataFrame(
            {
                "feature": rng.normal(size=24),
                "sensitive": rng.integers(0, 2, size=24),
                "other": rng.normal(size=24),
            }
        )
        y_test = pd.Series((X_test["feature"] + X_test["other"] > 0).astype(int), name="target")

        data = TinyData()
        data.X_train = X_train
        data.y_train = y_train
        data.X_test = X_test
        data.y_test = y_test

        model = LogisticRegression(max_iter=200).fit(X_train.values, y_train.values)
        attack = AttackConfig(
            attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
            targeted_attribute=["sensitive"],
            attack_params={"attack_model_type": "lr", "is_continuous": True, "scale_range": (0, 1)},
            attack_size=20,
        )

        scores = attack(data, model)
        self.assertIn("attack_score_time", scores)
        inferred_keys = [k for k in scores.keys() if k.startswith("inferred_")]
        self.assertTrue(len(inferred_keys) > 0)


if __name__ == "__main__":
    unittest.main()
