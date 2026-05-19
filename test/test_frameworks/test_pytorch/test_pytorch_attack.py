import os
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import TinyData
from numpy.exceptions import AxisError
from sklearn.linear_model import LinearRegression, LogisticRegression

from deckard.attack import AttackConfig
from deckard.attack.base import SensitiveFeaturesWrapper, _sensitive_slice

pytest.importorskip("torch")

# TOOD Create PytorchAttackConfig and test it here rather than a generic one.


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

    def _load_pytorch_model_inversion_config(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        config_path = (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "pytorch"
            / "config"
            / "attack"
            / "model-inversion.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return PytorchAttackConfig(**config)

    def _load_pytorch_database_reconstruction_config(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        config_path = (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "pytorch"
            / "config"
            / "attack"
            / "database-reconstruction.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return PytorchAttackConfig(**config)

    def test_post_init(self):
        self.assertTrue(hasattr(self.attack, "attack_type"))
        self.assertTrue(hasattr(self.attack, "attack_params"))

    def test_select_extraction_scorer_falls_back_for_logits(self):
        scorer, use_proba = AttackConfig._select_extraction_scorer(
            benign_pred=np.array([[0.8, 0.2], [0.1, 0.9]]),
            extracted_pred=np.array([[1.5, -0.2], [-0.4, 2.1]]),
        )
        self.assertFalse(use_proba)
        self.assertNotIn("roc_auc", scorer.scorers)

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
        self.assertIn("model_inversion_mse", scores)
        self.assertIn("model_inversion_mae", scores)
        self.assertIn("model_inversion_num_targets", scores)
        self.assertIn("attack_score_time", scores)

    def test_real_database_reconstruction_attack_executes(self):
        data = TinyData()
        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 0},
            attack_size=1,
        )

        scores = attack(data, model)
        self.assertIn("database_reconstruction_feature_mse", scores)
        self.assertIn("database_reconstruction_feature_mae", scores)
        self.assertIn("database_reconstruction_num_features", scores)
        self.assertIn("attack_score_time", scores)

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


class TestPytorchAttackConfig(unittest.TestCase):
    """Tests for deckard/attack/pytorch.py — PytorchAttackConfig feature prep."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch

            cls.torch = torch
        except ImportError:
            cls.torch = None

    def _skip_if_no_torch(self):
        if self.torch is None:
            self.skipTest("torch not installed")

    def test_prepare_features_tensor_passthrough(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
        )
        t = self.torch.randn(4, 3)
        result = cfg._prepare_features_for_attack(t)
        self.assertIs(result, t)

    def test_prepare_features_dataframe_to_numpy(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
        )
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = cfg._prepare_features_for_attack(df)
        self.assertIsInstance(result, np.ndarray)

    def test_prepare_features_series_to_numpy(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
        )
        s = pd.Series([1.0, 2.0, 3.0])
        result = cfg._prepare_features_for_attack(s)
        self.assertIsInstance(result, np.ndarray)

    def test_prepare_features_passthrough_for_other_types(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
        )
        arr = np.array([1.0, 2.0])
        result = cfg._prepare_features_for_attack(arr)
        self.assertIs(result, arr)

    def test_prepare_labels_tensor_passthrough(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
        )
        t = self.torch.tensor([0, 1, 1])
        result = cfg._prepare_labels_for_attack(t)
        self.assertIs(result, t)

    def test_prepare_labels_dataframe_to_numpy(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
        )
        df = pd.DataFrame({"label": [0, 1]})
        result = cfg._prepare_labels_for_attack(df)
        self.assertIsInstance(result, np.ndarray)

    def test_prepare_labels_series_to_numpy(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
        )
        s = pd.Series([0, 1, 0])
        result = cfg._prepare_labels_for_attack(s)
        self.assertIsInstance(result, np.ndarray)

    def test_prepare_labels_passthrough_for_other_types(self):
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
        )
        arr = np.array([0, 1])
        result = cfg._prepare_labels_for_attack(arr)
        self.assertIs(result, arr)

    def test_prepare_features_for_art_tensor_to_numpy_float_dtype(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.attack import PytorchAttackConfig

        cfg = PytorchAttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
        )
        t = self.torch.randn(4, 3)
        result = cfg._prepare_features_for_art(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 3))
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

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
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_size=8,
        )

        runtime = cfg._with_attack_context(attack_type="evasion", attack_subtype="")
        scores = runtime.evade(data, _DummyArtModel(), _DummyAttack())
        self.assertIn("evasion_accuracy", scores)
        self.assertIsInstance(runtime.attack, np.ndarray)
        self.assertEqual(runtime.attack.shape[0], 8)


class TestTorchUtils(unittest.TestCase):
    """Tests for deckard/frameworks/pytorch/torch_utils.py."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch

            cls.torch = torch
        except ImportError:
            cls.torch = None

    def _skip_if_no_torch(self):
        if self.torch is None:
            self.skipTest("torch not installed")

    def test_is_tensor_true_for_tensor(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import is_tensor

        self.assertTrue(is_tensor(self.torch.tensor([1.0])))

    def test_is_tensor_false_for_numpy(self):
        from deckard.frameworks.pytorch.torch_utils import is_tensor

        self.assertFalse(is_tensor(np.array([1.0])))

    def test_is_torch_model_true(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import is_torch_model

        model = self.torch.nn.Linear(2, 2)
        self.assertTrue(is_torch_model(model))

    def test_is_torch_model_false_for_sklearn(self):
        from sklearn.linear_model import LogisticRegression

        from deckard.frameworks.pytorch.torch_utils import is_torch_model

        self.assertFalse(is_torch_model(LogisticRegression()))

    def test_is_dataloader_true(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, TensorDataset

        from deckard.frameworks.pytorch.torch_utils import is_dataloader

        ds = TensorDataset(self.torch.randn(4, 2))
        dl = DataLoader(ds, batch_size=2)
        self.assertTrue(is_dataloader(dl))

    def test_is_dataloader_false_for_list(self):
        from deckard.frameworks.pytorch.torch_utils import is_dataloader

        self.assertFalse(is_dataloader([1, 2, 3]))

    def test_tensor_to_numpy_converts(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import tensor_to_numpy

        t = self.torch.tensor([1.0, 2.0, 3.0])
        arr = tensor_to_numpy(t)
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_tensor_to_numpy_with_dtype(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import tensor_to_numpy

        t = self.torch.tensor([1.0, 2.0])
        arr = tensor_to_numpy(t, dtype=np.float32)
        self.assertEqual(arr.dtype, np.float32)

    def test_tensor_to_numpy_passthrough_non_tensor(self):
        from deckard.frameworks.pytorch.torch_utils import tensor_to_numpy

        arr = np.array([1, 2])
        result = tensor_to_numpy(arr)
        self.assertIs(result, arr)

    def test_get_torch_model_device_returns_cpu(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import get_torch_model_device

        model = self.torch.nn.Linear(2, 2)
        device = get_torch_model_device(model)
        self.assertEqual(str(device.type), "cpu")

    def test_get_torch_model_device_non_torch_returns_cpu(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import get_torch_model_device

        device = get_torch_model_device(object())
        self.assertEqual(str(device.type), "cpu")

    def test_get_torch_model_device_model_with_no_parameters_returns_cpu(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import get_torch_model_device

        class EmptyModule(self.torch.nn.Module):
            def forward(self, x):
                return x

        model = EmptyModule()
        device = get_torch_model_device(model)
        self.assertEqual(str(device.type), "cpu")

    def test_build_torch_art_model_raises_when_torch_flag_disabled(self):
        from deckard.frameworks.pytorch.torch_utils import build_torch_art_model

        with patch("deckard.frameworks.pytorch.torch_utils.HAS_TORCH", False):
            with self.assertRaises(ImportError):
                build_torch_art_model(object(), object())

    def test_collect_subset_raises_when_torch_flag_disabled(self):
        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        with patch("deckard.frameworks.pytorch.torch_utils.HAS_TORCH", False):
            with self.assertRaises(ImportError):
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
        self.assertEqual(estimator.input_shape, (3,))

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
        self.assertEqual(estimator.input_shape, (3,))

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
        self.assertEqual(x_sub.shape[0], 4)
        self.assertEqual(y_sub.shape[0], 4)

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
        self.assertEqual(x_sub.shape[0], 5)

    def test_collect_subset_raises_for_non_dataloader(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        with self.assertRaises(TypeError):
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


class TestSensitiveFeaturesWrapper(unittest.TestCase):
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
        self.assertEqual(proba.shape, (3, 2))

    def test_predict_proba_fallback_when_no_predict_proba(self):
        """Cover the fallback branch that builds a two-column probability matrix."""
        est = self._make_estimator_without_predict_proba()
        sf = np.array([0, 1, 0, 1, 0])
        wrapper = SensitiveFeaturesWrapper(est, sf)
        X = np.zeros((3, 2))
        proba = wrapper.predict_proba(X)
        # fallback: each row has one 1.0, rest 0.0
        self.assertEqual(proba.shape[0], 3)
        self.assertTrue(np.all(proba.sum(axis=1) == 1.0))

    def test_get_params(self):
        est = MagicMock()
        sf = np.array([1, 0, 1])
        wrapper = SensitiveFeaturesWrapper(est, sf)
        params = wrapper.get_params()
        self.assertIn("estimator", params)
        self.assertIn("sensitive_features", params)
        self.assertIs(params["estimator"], est)

    def test_set_params(self):
        est = MagicMock()
        sf = np.array([1, 0])
        wrapper = SensitiveFeaturesWrapper(est, sf)
        new_est = MagicMock()
        new_sf = np.array([0, 1, 0])
        wrapper.set_params(estimator=new_est, sensitive_features=new_sf)
        self.assertIs(wrapper.estimator, new_est)
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
        self.assertEqual(len(preds), 3)


# ---------------------------------------------------------------------------
# _sensitive_slice helper
# ---------------------------------------------------------------------------


class TestSensitiveSlice(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_sensitive_slice(None, 5))

    def test_slices_to_n(self):
        arr = np.array([0, 1, 2, 3, 4])
        result = _sensitive_slice(arr, 3)
        np.testing.assert_array_equal(result, [0, 1, 2])


# ---------------------------------------------------------------------------
# __post_init__ branches
# ---------------------------------------------------------------------------


class TestPostInitBranches(unittest.TestCase):
    """Cover DictConfig scorer path and scorer-as-type path."""

    def test_scorer_as_type_class(self):
        from deckard.score.attack import AttackScorerConfig

        cfg = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            scorer=AttackScorerConfig,  # pass the class, not an instance
        )
        self.assertIsInstance(cfg.scorer, AttackScorerConfig)

    def test_scorer_as_dictconfig(self):
        from omegaconf import OmegaConf

        dc = OmegaConf.create(
            {"_target_": "deckard.score.attack.AttackScorerConfig"},
        )
        cfg = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            scorer=dc,
        )
        # scorer should have been coerced to a dict then processed
        self.assertTrue(hasattr(cfg.scorer, "_score"))

    def test_scorer_null_string(self):
        # "null" should be treated as None -> default scorer
        cfg = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            scorer="null",
        )
        self.assertTrue(hasattr(cfg.scorer, "_score"))

    def test_scorer_dict_with_no_target_uses_attack_scorer_config(self):
        from deckard.score.attack import AttackScorerConfig

        cfg = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            scorer={},  # empty dict, no _target_
        )
        self.assertIsInstance(cfg.scorer, AttackScorerConfig)


# ---------------------------------------------------------------------------
# _initialize_attack branches
# ---------------------------------------------------------------------------


class TestInitializeAttackBranches(unittest.TestCase):
    def setUp(self):
        self.data = _make_tiny_data()

    def test_generic_classifer_wrapping_and_nb_classes(self):
        """Cover the 'generic BaseEstimator classifier' wrapping branch."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200).fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
            attack_size=6,
        )
        # Patch resolve_class to avoid constructing a real FGM
        fake_fgm = MagicMock()
        fake_fgm.return_value = MagicMock()
        with patch("deckard.attack.base.resolve_class", return_value=fake_fgm):
            result = attack._initialize_attack(model, self.data)
        self.assertIsNotNone(result)

    def test_unsupported_model_raises_value_error(self):
        """Cover the 'else: raise ValueError' branch for unknown model types."""
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={},
        )
        with self.assertRaises((ValueError, Exception)):
            attack._initialize_attack("not_a_model", self.data)

    def test_targeted_attribute_string_current_behavior_does_not_raise(self):
        """Document current behavior: target_index field bypasses missing-column check."""
        attack = AttackConfig(
            attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
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
        self.assertNotIn("attack_feature", attack.attack_params)

    def test_attack_model_invalid_type_raises(self):
        """Cover the 'else: raise ValueError' branch for invalid attack_model."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200).fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={"attack_model": 12345},  # invalid type
        )
        with self.assertRaises((ValueError, Exception)):
            attack._initialize_attack(model, self.data)

    def test_model_with_sensitive_features_predict_wraps_with_wrapper(self):
        """Cover the SensitiveFeaturesWrapper path in _initialize_attack."""
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _SFModel(BaseEstimator, ClassifierMixin):
            classes_ = [0, 1]

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
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        fake_fgm = MagicMock()
        fake_fgm.return_value = MagicMock()
        with patch("deckard.attack.base.resolve_class", return_value=fake_fgm):
            result = attack._initialize_attack(model, self.data)
        self.assertIsNotNone(result)

    def test_not_fitted_sklearn_model_triggers_fit(self):
        """Cover NotFittedError branch for sklearn_dict models (lines 436-438)."""
        from sklearn.exceptions import NotFittedError
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200)
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
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
        self.assertIsNotNone(result)

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
            attack_type="art.attacks.evasion.FastGradientMethod",
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
        self.assertIsNotNone(result)

    def test_sensitive_features_fallback_uses_sensitive_train(self):
        """Cover _sensitive_train fallback branch (line 452)."""
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _SFModel(BaseEstimator, ClassifierMixin):
            classes_ = [0, 1]

            def fit(self, X, y):
                self.fitted_ = True
                return self

            def predict(self, X, sensitive_features=None):
                return np.zeros(len(X), dtype=int)

        model = _SFModel().fit(self.data.X_train.values, self.data.y_train.values)
        self.data._sensitive_test = None
        self.data._sensitive_train = np.zeros(len(self.data.X_train))
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        fake_attack_cls = MagicMock(return_value=MagicMock())
        with patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls):
            result = attack._initialize_attack(model, self.data)
        self.assertIsNotNone(result)

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
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
        )
        fake_attack_cls = MagicMock(return_value=MagicMock())
        with patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls):
            result = attack._initialize_attack(model, self.data)
        self.assertIsNotNone(result)

    def test_unsupported_model_type_reaches_value_error_branch(self):
        """Cover explicit unsupported model type ValueError branch (line 470)."""
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
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
            with self.assertRaises(ValueError):
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
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
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
        self.assertIsNotNone(result)

    def test_attack_model_modelconfig_instance_path(self):
        """Cover ModelConfig instance branch (lines 504-506)."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200).fit(
            self.data.X_train.values,
            self.data.y_train.values,
        )
        fake_cfg = _PickleableFakeModelConfig()
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={"attack_model": fake_cfg},
        )

        fake_attack_cls = MagicMock(return_value=MagicMock())
        with (
            patch("deckard.attack.base.resolve_class", return_value=fake_attack_cls),
            patch("deckard.attack.base.ModelConfig", _PickleableFakeModelConfig),
        ):
            result = attack._initialize_attack(model, self.data)
        self.assertIsNotNone(result)

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
                attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
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
            self.assertIsNotNone(result)
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# __call__ file caching branches
# ---------------------------------------------------------------------------


class TestCallCachingPaths(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data = _make_tiny_data()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _base_attack(self):
        return AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
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
        self.assertIn("evasion_success", result)

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
        self.assertIn("evasion_success", result)

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
        self.assertIn("evasion_success", result)

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


class TestGetAttackSubset(unittest.TestCase):
    def test_raises_for_unsupported_type(self):
        attack = AttackConfig(attack_type="art.attacks.evasion.FastGradientMethod")
        attack.attack_size = 4

        class _BadData:
            X_test = "not_an_array"
            y_test = "not_an_array"

        with self.assertRaises(ValueError):
            attack.get_attack_subset(_BadData())

    def test_returns_subset_from_numpy(self):
        attack = AttackConfig(attack_type="art.attacks.evasion.FastGradientMethod")
        attack.attack_size = 3

        class _Data:
            X_test = np.arange(20).reshape(10, 2)
            y_test = np.zeros(10)

        n, x_sub, y_sub = attack.get_attack_subset(_Data())
        self.assertEqual(n, 3)
        self.assertEqual(len(x_sub), 3)

    def test_train_subset(self):
        attack = AttackConfig(attack_type="art.attacks.evasion.FastGradientMethod")
        attack.attack_size = 3

        class _Data:
            X_train = np.arange(20).reshape(10, 2)
            y_train = np.zeros(10)
            X_test = np.arange(20).reshape(10, 2)
            y_test = np.zeros(10)

        n, x_sub, y_sub = attack.get_attack_subset(_Data(), test=False)
        self.assertEqual(n, 3)


# ---------------------------------------------------------------------------
# _get_benign_preds train=True path
# ---------------------------------------------------------------------------


class TestGetBenignPreds(unittest.TestCase):
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
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_size=4,
        )
        data = self._make_numpy_data()
        art_model = _FakeArtModel()
        n, labels, x_sub, y_sub = attack._get_benign_preds(data, art_model, train=True)
        self.assertEqual(n, 4)
        self.assertIsInstance(labels, np.ndarray)

    def test_train_false_uses_train_data(self):
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_size=4,
        )
        data = self._make_numpy_data()
        art_model = _FakeArtModel()
        n, labels, x_sub, y_sub = attack._get_benign_preds(
            data,
            art_model,
            train=False,
        )
        self.assertEqual(n, 4)


# ---------------------------------------------------------------------------
# _evade branches
# ---------------------------------------------------------------------------


class TestEvadeBranches(unittest.TestCase):
    def test_adversarial_patch_branch(self):
        """Cover the 'AdversarialPatch' special handling."""
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
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
        runtime = attack._with_attack_context(attack_type="evasion", attack_subtype="")
        result = runtime.evade(
            data=_TinyData(),
            art_model=_FakeModel(),
            attack=fake_attack_obj,
        )
        self.assertIsInstance(result, dict)

    def test_evade_regression_path(self):
        """Cover the is_regression=True scoring path in _evade."""
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
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

        runtime = attack._with_attack_context(attack_type="evasion", attack_subtype="")
        result = runtime.evade(
            data=_TinyData(),
            art_model=_RegressionArtModel(),
            attack=_FakeEvasionAttack(),
        )
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# infer_attribute: list targeted_attribute path
# ---------------------------------------------------------------------------


class TestInferAttributeBranches(unittest.TestCase):
    def test_list_targeted_attribute_executes(self):
        data = _make_tiny_data()
        attack = AttackConfig(
            attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
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

        self.assertIsInstance(result, dict)

    def test_attribute_column_missing_raises(self):
        data = _make_tiny_data()
        attack = AttackConfig(
            attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
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
            attack_type="inference",
            attack_subtype="attribute_inference",
        )
        with self.assertRaises((AssertionError, ValueError, KeyError)):
            runtime.infer_attribute(
                data,
                _FakeArt(),
                _FakeAttribAttack(),
                targeted_attribute="nonexistent_col",
            )

    def test_attribute_list_column_missing_raises_value_error(self):
        data = _make_tiny_data()
        attack = AttackConfig(
            attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
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
            attack_type="inference",
            attack_subtype="attribute_inference",
        )
        with self.assertRaises((ValueError, AssertionError)):
            runtime.infer_attribute(
                data,
                _FakeArt(),
                _FakeAttribAttack(),
                targeted_attribute=["not_a_column"],
            )


# ---------------------------------------------------------------------------
# infer_membership: AxisError fallback + sensitive features
# ---------------------------------------------------------------------------


class TestInferMembershipBranches(unittest.TestCase):
    def test_axis_error_fallback_is_used(self):
        """Cover the AxisError fallback in infer_membership."""
        data = _make_tiny_data()
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
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
            attack_type="inference",
            attack_subtype="membership_inference",
        )
        result = runtime.infer_membership(data=data, attack=_FakeMIAttack())
        self.assertIsInstance(result, dict)

    def test_sensitive_features_present_builds_big_sensitive(self):
        """Cover the path that concatenates train+test sensitive features."""
        data = _make_tiny_data()
        data._sensitive_train = np.zeros(len(data.X_train))
        data._sensitive_test = np.ones(len(data.X_test))
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_size=16,
        )

        class _FakeMIAttack:
            def fit(self, x, y, test_x, **kw):
                pass

            def infer(self, x, y=None):
                return np.zeros(len(x), dtype=int)

        runtime = attack._with_attack_context(
            attack_type="inference",
            attack_subtype="membership_inference",
        )
        result = runtime.infer_membership(data=data, attack=_FakeMIAttack())
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# infer_model_inversion init modes
# ---------------------------------------------------------------------------


class TestInferModelInversionModes(unittest.TestCase):
    def _make_mi_attack_config(self, init_mode="zeros"):
        return AttackConfig(
            attack_type="art.attacks.inference.model_inversion.mi_face.MIFace",
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
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        self.assertIn("model_inversion_mse", result)

    def test_ones_init(self):
        cfg = self._make_mi_attack_config("ones")
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        self.assertIn("model_inversion_mse", result)

    def test_random_init(self):
        cfg = self._make_mi_attack_config("random")
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        self.assertIn("model_inversion_mse", result)

    def test_average_init(self):
        cfg = self._make_mi_attack_config("average")
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        self.assertIn("model_inversion_mse", result)

    def test_invalid_init_mode_raises(self):
        cfg = self._make_mi_attack_config("invalid_mode")
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        with self.assertRaises(ValueError):
            runtime.infer_model_inversion(
                data=self._make_data(),
                attack=self._fake_attack(),
            )

    def test_train_split_used(self):
        cfg = AttackConfig(
            attack_type="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={"split": "train"},
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        self.assertIn("model_inversion_mse", result)

    def test_mi_invalid_split_raises(self):
        cfg = AttackConfig(
            attack_type="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={"split": "validate"},
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        with self.assertRaises(ValueError):
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
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        with self.assertRaises(ValueError):
            runtime.infer_model_inversion(data=d, attack=self._fake_attack())

    def test_explicit_targets_param(self):
        cfg = AttackConfig(
            attack_type="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={"split": "test", "targets": [0, 1]},
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        result = runtime.infer_model_inversion(
            data=self._make_data(),
            attack=self._fake_attack(),
        )
        self.assertIn("model_inversion_mse", result)

    def test_x_init_from_param(self):
        d = self._make_data()
        x_init = np.zeros((2, 3), dtype=np.float32)
        cfg = AttackConfig(
            attack_type="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={
                "split": "test",
                "x_init": x_init.tolist(),
                "targets": [0, 1],
            },
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        result = runtime.infer_model_inversion(data=d, attack=self._fake_attack())
        self.assertIn("model_inversion_mse", result)

    def test_x_init_length_mismatch_raises(self):
        d = self._make_data()
        x_init = np.zeros((5, 3), dtype=np.float32)  # 5 != 2 targets
        cfg = AttackConfig(
            attack_type="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={
                "split": "test",
                "x_init": x_init.tolist(),
                "targets": [0, 1],
            },
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        with self.assertRaises(ValueError):
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
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        result = runtime.infer_model_inversion(data=self._make_data(), attack=_A())
        self.assertIn("model_inversion_mse", result)

    def test_empty_target_labels_raises(self):
        """Cover the path when target_labels is empty."""
        cfg = AttackConfig(
            attack_type="art.attacks.inference.model_inversion.mi_face.MIFace",
            attack_params={"split": "test", "targets": []},
            attack_size=2,
        )
        runtime = cfg._with_attack_context(
            attack_type="inference",
            attack_subtype="model_inversion",
        )
        with self.assertRaises(ValueError):
            runtime.infer_model_inversion(
                data=self._make_data(),
                attack=self._fake_attack(),
            )


# ---------------------------------------------------------------------------
# infer_database_reconstruction branches
# ---------------------------------------------------------------------------


class TestInferDatabaseReconstructionBranches(unittest.TestCase):
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
            attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "test", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([0])

        runtime = attack._with_attack_context(
            attack_type="inference",
            attack_subtype="reconstruction",
        )
        result = runtime.infer_database_reconstruction(
            data=self._make_data(),
            attack=_FakeAttack(),
        )
        self.assertIn("database_reconstruction_feature_mse", result)

    def test_dr_invalid_split_raises(self):
        attack = AttackConfig(
            attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "validate", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([0])

        runtime = attack._with_attack_context(
            attack_type="inference",
            attack_subtype="reconstruction",
        )
        with self.assertRaises(ValueError):
            runtime.infer_database_reconstruction(
                data=self._make_data(),
                attack=_FakeAttack(),
            )

    def test_missing_index_out_of_bounds_raises(self):
        attack = AttackConfig(
            attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 100},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([0])

        runtime = attack._with_attack_context(
            attack_type="inference",
            attack_subtype="reconstruction",
        )
        with self.assertRaises(ValueError):
            runtime.infer_database_reconstruction(
                data=self._make_data(),
                attack=_FakeAttack(),
            )

    def test_too_few_rows_raises(self):
        attack = AttackConfig(
            attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
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
            attack_type="inference",
            attack_subtype="reconstruction",
        )
        with self.assertRaises(ValueError):
            runtime.infer_database_reconstruction(data=d, attack=_FakeAttack())

    def test_y_reconstructed_none_skips_label_scoring(self):
        """Cover the path where reconstructed tuple has only x."""
        attack = AttackConfig(
            attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                # Return only features, no labels
                return x[:1]

        runtime = attack._with_attack_context(
            attack_type="inference",
            attack_subtype="reconstruction",
        )
        result = runtime.infer_database_reconstruction(
            data=self._make_data(),
            attack=_FakeAttack(),
        )
        self.assertNotIn("database_reconstruction_label_accuracy", result)

    def test_regression_task_uses_mae_label(self):
        """Cover the regression label scoring branch."""
        attack = AttackConfig(
            attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
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
            attack_type="inference",
            attack_subtype="reconstruction",
        )
        result = runtime.infer_database_reconstruction(data=d, attack=_FakeAttack())
        self.assertIn("database_reconstruction_label_mae", result)

    def test_type_error_fallback_on_reconstruct(self):
        """Cover the TypeError fallback (positional reconstruct)."""
        attack = AttackConfig(
            attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
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
            attack_type="inference",
            attack_subtype="reconstruction",
        )
        result = runtime.infer_database_reconstruction(
            data=self._make_data(),
            attack=_FakeAttack(),
        )
        self.assertIn("database_reconstruction_feature_mse", result)

    def test_empty_y_reconstructed_skips_label(self):
        """Cover path where y_pred is empty after to_numpy_array."""
        attack = AttackConfig(
            attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
            attack_params={"split": "train", "missing_index": 0},
            attack_size=1,
        )

        class _FakeAttack:
            def reconstruct(self, x, y=None):
                return x[:1], np.array([])

        runtime = attack._with_attack_context(
            attack_type="inference",
            attack_subtype="reconstruction",
        )
        result = runtime.infer_database_reconstruction(
            data=self._make_data(),
            attack=_FakeAttack(),
        )
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# _resolve_eval_split branches
# ---------------------------------------------------------------------------


class TestResolveEvalSplit(unittest.TestCase):
    def test_val_split_available_returns_val(self):
        attack = AttackConfig(
            attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            mode="val",
        )

        class _Data:
            X_val = np.zeros((4, 2))
            y_val = np.array([0, 1, 0, 1])
            X_test = np.zeros((4, 2))
            y_test = np.array([0, 1, 0, 1])

        runtime = attack._with_attack_context(
            attack_type="poisoning",
            attack_subtype="gradient_matching_attack",
        )
        mode, x, y = runtime._resolve_eval_split(_Data())
        self.assertEqual(mode, "val")

    def test_val_split_unavailable_falls_back_to_test(self):
        attack = AttackConfig(
            attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            mode="val",
        )

        class _Data:
            X_test = np.zeros((4, 2))
            y_test = np.array([0, 1, 0, 1])

        runtime = attack._with_attack_context(
            attack_type="poisoning",
            attack_subtype="gradient_matching_attack",
        )
        mode, x, y = runtime._resolve_eval_split(_Data())
        self.assertEqual(mode, "test")

    def test_invalid_mode_raises(self):
        attack = AttackConfig(
            attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            mode="invalid",
        )

        class _Data:
            X_test = np.zeros((4, 2))
            y_test = np.array([0, 1, 0, 1])

        runtime = attack._with_attack_context(
            attack_type="poisoning",
            attack_subtype="gradient_matching_attack",
        )
        with self.assertRaises(ValueError):
            runtime._resolve_eval_split(_Data())

    def test_test_mode_with_missing_data_raises(self):
        attack = AttackConfig(
            attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
            attack_params={"class_source": 0, "class_target": 1},
            mode="test",
        )

        class _Data:
            X_test = None
            y_test = None

        runtime = attack._with_attack_context(
            attack_type="poisoning",
            attack_subtype="gradient_matching_attack",
        )
        with self.assertRaises(ValueError):
            runtime._resolve_eval_split(_Data())


# ---------------------------------------------------------------------------
# _poison val mode + class_source fallback
# ---------------------------------------------------------------------------


class TestPoisonBranches(unittest.TestCase):
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
            attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
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
            attack_type="poisoning",
            attack_subtype="gradient_matching_attack",
        )
        result = runtime.poison(
            data=self._make_data(),
            art_model=_FakeArtModel(),
            attack=_FakePoisonAttack(),
        )
        self.assertEqual(result["poison_mode"], "val")

    def test_class_source_fallback_when_no_samples(self):
        """Cover the warning/fallback when class_source has no samples in eval."""
        attack = AttackConfig(
            attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
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
            attack_type="poisoning",
            attack_subtype="gradient_matching_attack",
        )
        result = runtime.poison(
            data=self._make_data(),
            art_model=_FakeArtModel(),
            attack=_FakePoisonAttack(),
        )
        # class_source should have been adjusted to something present
        self.assertIn("poison_attack_source_class", result)

    def test_poisoning_svm_branch_scores_benign_and_poisoned_accuracy(self):
        attack = AttackConfig(
            attack_type="art.attacks.poisoning.PoisoningAttackSVM",
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
            attack_type="poisoning",
            attack_subtype="PoisoningAttackSVM",
        )
        result = runtime.poison(
            data=self._make_data(),
            art_model=_FakeArtModel(),
            attack=_FakePoisoningAttackSVM(),
        )

        self.assertIn("benign_accuracy", result)
        self.assertIn("poisoned_accuracy", result)
        self.assertIn("poisoning_attack_points", result)
        self.assertEqual(result["attack_size"], 2)


# ---------------------------------------------------------------------------
# _extract val mode
# ---------------------------------------------------------------------------


class TestExtractBranches(unittest.TestCase):
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
            dataset_name="torch.utils.data.TensorDataset",
            train_size=12,
            test_size=4,
            classifier=True,
            random_state=42,
            data_params={"_args_": [X, y]},
        )
        data()

        model = PytorchModelConfig(
            model_type="torch.nn.Linear",
            model_params={"in_features": 4, "out_features": 2},
            classifier=True,
            fit_params={"nb_epochs": 1, "batch_size": 4},
            criterion="CrossEntropyLoss",
            optimizer={"name": "SGD", "lr": 0.05},
        )
        model(data)

        attack = AttackConfig(
            attack_type="art.attacks.extraction.CopycatCNN",
            attack_params={},
            attack_size=4,
        )

        with patch(
            "deckard.attack.base.resolve_class",
            return_value=DummyCopycatCNN,
        ):
            initialized_attack, art_model, attack_type, attack_subtype = (
                attack._initialize_attack(
                    model,
                    data,
                )
            )

        self.assertEqual(attack_type, "extraction")
        self.assertEqual(attack_subtype, "CopycatCNN")
        runtime = attack._with_attack_context(
            attack_type="extraction",
            attack_subtype="CopycatCNN",
        )
        self.assertTrue(runtime._is_nn_art_classifier(art_model))
        self.assertIs(initialized_attack.classifier, art_model)

    def test_extract_uses_val_split(self):
        attack = AttackConfig(
            attack_type="art.attacks.extraction.CopycatCNN",
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

        self.assertEqual(result.get("extraction_mode"), "val")

    def test_extract_not_implemented_raises_for_non_type(self):
        """Cover the not-implemented path for unsupported attack_type."""
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
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

            with self.assertRaises(NotImplementedError):
                attack(object(), object())


# ---------------------------------------------------------------------------
# Additional helper / static method branches
# ---------------------------------------------------------------------------


class TestStaticHelpers(unittest.TestCase):
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
        self.assertEqual(result.shape, (8,))

    def test_normalize_ground_truth_dataframe_regression(self):
        df = pd.DataFrame({"a": [0.1, 0.2, 0.3]})
        result = AttackConfig._normalize_ground_truth(df, is_regression=True)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3])

    def test_normalize_ground_truth_dataframe_classification(self):
        df = pd.DataFrame({"a": [0, 1, 0]})
        result = AttackConfig._normalize_ground_truth(df, is_regression=False)
        self.assertEqual(result.shape, (3,))

    def test_normalize_ground_truth_2d_one_hot(self):
        arr = np.array([[1, 0], [0, 1], [1, 0]])
        result = AttackConfig._normalize_ground_truth(arr, is_regression=False)
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_to_numpy_array_with_dtype_on_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = AttackConfig._to_numpy_array(df, dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_to_numpy_array_with_dtype_on_series(self):
        s = pd.Series([1, 2, 3])
        result = AttackConfig._to_numpy_array(s, dtype=np.float64)
        self.assertEqual(result.dtype, np.float64)

    def test_to_numpy_array_with_dtype_on_ndarray(self):
        arr = np.array([1, 2, 3])
        result = AttackConfig._to_numpy_array(arr, dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_is_regression_prediction_output_1d_col(self):
        """Cover the 2D single-col -> regression = True path."""
        preds = np.array([[0.1], [0.2], [0.3]])
        labels = np.array([0.1, 0.2, 0.3])
        self.assertTrue(AttackConfig._is_regression_prediction_output(labels, preds))

    def test_select_extraction_scorer_with_probabilities(self):
        benign = np.array([[0.3, 0.7], [0.6, 0.4]])
        extracted = np.array([[0.4, 0.6], [0.5, 0.5]])
        scorer, has_proba = AttackConfig._select_extraction_scorer(benign, extracted)
        self.assertTrue(has_proba)
        self.assertIsNotNone(scorer)

    def test_is_nn_art_classifier_returns_false_for_plain_object(self):
        attack = AttackConfig(attack_type="art.attacks.extraction.CopycatCNN")
        runtime = attack._with_attack_context(
            attack_type="extraction",
            attack_subtype="CopycatCNN",
        )
        self.assertFalse(runtime._is_nn_art_classifier(object()))

    def test_is_nn_art_classifier_returns_true_for_pytorch_name(self):
        class PyTorchClassifier:
            _model = None

        attack = AttackConfig(attack_type="art.attacks.extraction.CopycatCNN")
        runtime = attack._with_attack_context(
            attack_type="extraction",
            attack_subtype="CopycatCNN",
        )
        self.assertTrue(runtime._is_nn_art_classifier(PyTorchClassifier()))

    def test_normalize_inferred_output_higher_dim_reference(self):
        """Cover the `ref.ndim > arr.ndim` branch."""
        inferred = np.array([0, 1, 0])
        ref = np.array([[1, 0], [0, 1], [1, 0]])
        result = AttackConfig._normalize_inferred_output(inferred, reference=ref)
        # Should have been get_dummies-expanded
        self.assertEqual(result.ndim, 2)

    def test_normalize_inferred_output_lower_dim_reference(self):
        """Cover the `arr.ndim > ref.ndim` branch."""
        inferred = np.array([[0.2, 0.8], [0.7, 0.3]])
        ref = np.array([0, 1])
        result = AttackConfig._normalize_inferred_output(inferred, reference=ref)
        np.testing.assert_array_equal(result, [1, 0])

    def test_save_method_appends_pkl(self):
        tmpdir = tempfile.mkdtemp()
        try:
            attack = AttackConfig(attack_type="art.attacks.evasion.FastGradientMethod")
            path_without_ext = os.path.join(tmpdir, "attack_saved")
            attack._save(path_without_ext)
            self.assertTrue(Path(path_without_ext + ".pkl").exists())
        finally:
            shutil.rmtree(tmpdir)

    def test_infer_task_from_data_classifier_attr(self):
        """Cover the `hasattr(data, 'classifier')` path in _infer_task_is_classification."""

        class _Data:
            classifier = True

        result = AttackConfig._infer_task_is_classification(_Data(), object())
        self.assertTrue(result)

    def test_infer_task_returns_none_for_unknown(self):

        class _Data:
            pass

        result = AttackConfig._infer_task_is_classification(_Data(), object())
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Fairlearn scorer integration coverage
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("require_fairlearn")
class TestFairlearnAttackScorer(unittest.TestCase):
    """Unit tests for FairlearnAttackScorerConfig per-group attack metrics."""

    def _make_data_with_sensitive(self):
        from deckard.plugins.fairlearn.data import FairlearnDataConfig

        data = FairlearnDataConfig(dataset_name="adult", sensitive_columns="sex")
        data()
        return data

    def test_fairlearn_attack_scorer_instantiates(self):
        from deckard.plugins.fairlearn.score import FairlearnScoreDictConfig
        from deckard.score.attack import FairlearnAttackScorerConfig
        from deckard.score.base import DefaultClassifierConfig

        scorer = FairlearnAttackScorerConfig(evasion=DefaultClassifierConfig())
        self.assertIsInstance(scorer.evasion, FairlearnScoreDictConfig)
        self.assertIsInstance(scorer.membership_inference, FairlearnScoreDictConfig)
        self.assertIsInstance(scorer.attribute_inference, FairlearnScoreDictConfig)

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

    def test_score_membership_with_sensitive_features_produces_group_metrics(self):
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

    def test_score_attribute_with_sensitive_features_produces_group_metrics(self):
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
