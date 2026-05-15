import unittest
import tempfile
import shutil
from pathlib import Path
import pytest
from torch.utils.data import Dataset
from unittest.mock import patch


torch = pytest.importorskip("torch")
Tensor = pytest.importorskip("torch").Tensor
PytorchDataConfig = pytest.importorskip(
    "deckard.frameworks.pytorch.data",
).PytorchDataConfig
PytorchCustomDataConfig = pytest.importorskip(
    "deckard.frameworks.pytorch.data",
).PytorchCustomDataConfig


class TensorWithSensitiveDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        sensitive = int(y.item() % 2)
        return x, y, sensitive


class TensorWithDatasetSensitiveAttr(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self._sensitive = [int(i % 2) for i in range(len(y))]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ArrayFallbackSample:
    def __init__(self, values):
        self.values = values

    def __array__(self, dtype=None):
        import numpy as np

        return np.asarray(self.values, dtype=dtype)


class ArrayFallbackDataset(Dataset):
    def __init__(self):
        self.samples = [
            (ArrayFallbackSample([[0, 255], [128, 64]]), [0, 1]) for _ in range(4)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class IntImageDataset(Dataset):
    def __init__(self):
        self.samples = [
            (torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), 1) for _ in range(6)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class InvalidSampleDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return torch.tensor([idx])


class MismatchedSensitiveDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self._sensitive = [0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SensitiveBatchDataset(Dataset):
    def __init__(self, n=6):
        self.X = torch.randn(n, 4)
        self.y = torch.randint(0, 2, (n,))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], torch.tensor([idx % 2, (idx + 1) % 2])


class InvalidBatchDataset(Dataset):
    def __init__(self, n=4):
        self.values = [torch.tensor([i]) for i in range(n)]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]
    


class TestPytorchDataConfig(unittest.TestCase):

    def setUp(self):
        X = torch.randn(300, 4)
        y = torch.randint(0, 2, (300,))
        self.config = PytorchDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            data_dir=self.temp_dir,
            test_size=100,
            train_size=100,
            random_state=42,
            data_params={"_args_": [X, y]},
        )

    @classmethod
    def setUpClass(cls):
        # Create temporary directory for data storage
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_initialization(self):
        self.assertEqual(
            self.config.dataset_name,
            "torch.utils.data.TensorDataset",
        )
        self.assertEqual(self.config.data_dir, self.temp_dir)
        self.assertEqual(self.config.test_size, 100)
        self.assertEqual(self.config.train_size, 100)
        self.assertEqual(self.config.random_state, 42)
        self.assertTrue(self.config.stratify)

    def test_load_data(self):
        self.config._load_data()
        self.assertIsInstance(self.config._X, (Tensor, Dataset))
        self.assertIsInstance(self.config._y, (Tensor, Dataset))
        self.assertGreater(self.config.data_load_time, 0)

    def test_private_max_samples_caps_loaded_dataset(self):
        X = torch.randn(300, 4)
        y = torch.randint(0, 2, (300,))
        with patch.dict("os.environ", {"DECKARD_TEST_MAX_SAMPLES": "120"}):
            cfg = PytorchDataConfig(
                dataset_name="torch.utils.data.TensorDataset",
                data_dir=self.temp_dir,
                train_size=80,
                test_size=40,
                random_state=42,
                data_params={"_args_": [X, y]},
            )

            cfg._load_data()

        self.assertEqual(cfg._X.shape[0], 120)
        self.assertEqual(cfg._y.shape[0], 120)

    def test_sample(self):
        self.config._load_data()
        self.config._sample()
        from torch.utils.data import Subset

        self.assertIsInstance(self.config.X_train, (Tensor, Dataset, Subset))
        self.assertIsInstance(self.config.y_train, (Tensor, Dataset))
        self.assertIsInstance(self.config.X_test, (Tensor, Dataset, Subset))
        self.assertIsInstance(self.config.y_test, (Tensor, Dataset))

    def test_sample_allows_stratify_false(self):
        self.config.stratify = False
        self.config._load_data()
        self.config._sample()
        self.assertEqual(len(self.config.X_train), 100)
        self.assertEqual(len(self.config.X_test), 100)

    def test_call(self):
        scores = self.config(data_file=str(Path(self.temp_dir) / "data.pkl"))
        self.assertIn("data_load_time", scores)
        self.assertIn("data_sample_time", scores)
        self.assertGreater(scores["data_load_time"], 0)
        self.assertGreater(scores["data_sample_time"], 0)

    def test_invalid_dataset_name(self):
        self.config.dataset_name = "invalid_dataset"
        with self.assertRaises(Exception):
            self.config._load_data()

    def test_hash_method(self):
        h1 = hash(self.config)
        h2 = hash(self.config)
        self.assertEqual(h1, h2)

    def test_load_data_collects_sensitive_from_third_tuple_item(self):
        X = torch.randn(120, 4)
        y = torch.randint(0, 2, (120,))
        ds = TensorWithSensitiveDataset(X, y)
        cfg = PytorchDataConfig(
            dataset_name="dummy.dataset",
            data_dir=self.temp_dir,
            train_size=60,
            test_size=40,
            stratify=True,
            data_params={},
        )

        with patch("deckard.frameworks.pytorch.data.load_class", return_value=ds):
            cfg._load_data()
        self.assertTrue(hasattr(cfg, "_sensitive"))
        self.assertEqual(len(cfg._sensitive), len(y))

        cfg._sample()
        self.assertTrue(hasattr(cfg, "_sensitive_train"))
        self.assertTrue(hasattr(cfg, "_sensitive_test"))
        self.assertEqual(len(cfg._sensitive_train), 60)
        self.assertEqual(len(cfg._sensitive_test), 40)

    def test_load_data_collects_sensitive_from_dataset_attribute(self):
        X = torch.randn(110, 4)
        y = torch.randint(0, 2, (110,))
        ds = TensorWithDatasetSensitiveAttr(X, y)
        cfg = PytorchDataConfig(
            dataset_name="dummy.dataset",
            data_dir=self.temp_dir,
            train_size=70,
            test_size=20,
            stratify=True,
            data_params={},
        )

        with patch("deckard.frameworks.pytorch.data.load_class", return_value=ds):
            cfg._load_data()
        self.assertTrue(hasattr(cfg, "_sensitive"))
        self.assertEqual(len(cfg._sensitive), len(y))

        cfg._sample()
        self.assertTrue(hasattr(cfg, "_sensitive_train"))
        self.assertTrue(hasattr(cfg, "_sensitive_test"))
        self.assertEqual(len(cfg._sensitive_train), 70)
        self.assertEqual(len(cfg._sensitive_test), 20)

    def test_pytorch_data_hash_stable_after_call_and_runtime_mutation(self):
        original_hash = hash(self.config)
        self.config()
        self.config.score_dict["runtime"] = 1
        self.config.data_load_time = 999.0
        self.assertEqual(hash(self.config), original_hash)

    def test_pytorch_data_score_persistence_roundtrip(self):
        self.config._load_data()
        self.config._sample()
        from torch.utils.data import Subset

        self.assertIsInstance(self.config.X_train, (Tensor, Dataset, Subset))
        self.assertIsInstance(self.config.X_test, (Tensor, Dataset, Subset))
        scores = self.config._score()

        score_path = Path(self.temp_dir) / "pytorch_data_scores.json"
        self.config.save_scores(scores, str(score_path))
        loaded = self.config.load_scores(str(score_path))

        self.assertTrue(len(loaded) > 0)
        self.assertIn("num_classes", loaded)


    def test_call_with_score_file_saves_json(self):
        score_path = str(Path(self.temp_dir) / "pytorch_scores.json")
        scores = self.config(score_file=score_path)
        self.assertTrue(Path(score_path).exists())
        self.assertIn("data_load_time", scores)

    def test_float_train_test_sizes(self):
        X = torch.randn(100, 4)
        y = torch.randint(0, 2, (100,))
        cfg = PytorchDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            data_dir=self.temp_dir,
            train_size=0.7,
            test_size=0.2,
            data_params={"_args_": [X, y]},
        )
        cfg._load_data()
        cfg._sample()
        self.assertEqual(len(cfg.X_train), 70)
        self.assertEqual(len(cfg.X_test), 20)

    def test_train_test_exceed_total_raises(self):
        X = torch.randn(100, 4)
        y = torch.randint(0, 2, (100,))
        cfg = PytorchDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            data_dir=self.temp_dir,
            train_size=70,
            test_size=70,
            data_params={"_args_": [X, y]},
        )
        cfg._load_data()
        with self.assertRaises(ValueError):
            cfg._sample()

    def test_invalid_stratify_raises(self):
        X = torch.randn(80, 4)
        y = torch.randint(0, 2, (80,))
        cfg = PytorchDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            data_dir=self.temp_dir,
            train_size=50,
            test_size=30,
            stratify="column_name",
            data_params={"_args_": [X, y]},
        )
        cfg._load_data()
        with self.assertRaises(ValueError):
            cfg._sample()

    def test_normalize_sensitive_item_variants(self):
        cfg = self.config
        # tensor scalar
        t_scalar = torch.tensor(3)
        self.assertEqual(cfg._normalize_sensitive_item(t_scalar), 3)
        # tensor vector
        t_vec = torch.tensor([1, 2])
        self.assertEqual(cfg._normalize_sensitive_item(t_vec), (1, 2))
        # numpy scalar
        import numpy as np

        arr_scalar = np.array(5)
        self.assertEqual(cfg._normalize_sensitive_item(arr_scalar), 5)
        # numpy vector
        arr_vec = np.array([1, 2])
        self.assertEqual(cfg._normalize_sensitive_item(arr_vec), (1, 2))
        # list
        self.assertEqual(cfg._normalize_sensitive_item([1, 2]), (1, 2))
        # dict
        result = cfg._normalize_sensitive_item({"b": 2, "a": 1})
        self.assertEqual(result, (("a", 1), ("b", 2)))
        # passthrough for str
        self.assertEqual(cfg._normalize_sensitive_item("hello"), "hello")

    def test_post_init_normalizes_data_dir_and_torchvision_root(self):
        cfg = PytorchDataConfig(
            dataset_name="mnist",
            data_dir=None,
            train_size=4,
            test_size=2,
            data_params=None,
        )

        self.assertIsInstance(cfg.data_dir, str)
        self.assertEqual(cfg.data_params["root"], cfg.data_dir)

    def test_load_data_alias_and_numpy_fallback_paths(self):
        cfg = PytorchDataConfig(
            dataset_name="torch_mnist",
            data_dir=self.temp_dir,
            train_size=2,
            test_size=2,
            data_params={"batch_size": 8, "_args_": ["ignored"]},
        )

        seen = {}

        def fake_load_class(name, **kwargs):
            seen["name"] = name
            seen["kwargs"] = kwargs
            return ArrayFallbackDataset()

        with patch("deckard.frameworks.pytorch.data.load_class", side_effect=fake_load_class):
            cfg._load_data()

        self.assertEqual(seen["name"], "torchvision.datasets.MNIST")
        self.assertNotIn("batch_size", seen["kwargs"])
        self.assertEqual(tuple(cfg._X.shape), (4, 1, 2, 2))
        self.assertTrue(torch.is_floating_point(cfg._X))
        self.assertEqual(tuple(cfg._y.shape), (4, 2))

    def test_load_data_converts_nonfloating_tensor_inputs(self):
        cfg = PytorchDataConfig(
            dataset_name="dummy.dataset",
            data_dir=self.temp_dir,
            train_size=3,
            test_size=2,
            data_params={},
        )

        with patch("deckard.frameworks.pytorch.data.load_class", return_value=IntImageDataset()):
            cfg._load_data()

        self.assertEqual(tuple(cfg._X.shape), (6, 1, 2, 2))
        self.assertEqual(cfg._X.dtype, torch.float32)

    def test_load_data_rejects_invalid_samples_and_mismatched_sensitive_lengths(self):
        cfg = PytorchDataConfig(
            dataset_name="dummy.dataset",
            data_dir=self.temp_dir,
            train_size=1,
            test_size=1,
            data_params={},
        )

        with patch(
            "deckard.frameworks.pytorch.data.load_class",
            return_value=InvalidSampleDataset(),
        ):
            with self.assertRaises(ValueError):
                cfg._load_data()

        X = torch.randn(4, 2)
        y = torch.randint(0, 2, (4,))
        bad_sensitive = MismatchedSensitiveDataset(X, y)
        with patch("deckard.frameworks.pytorch.data.load_class", return_value=bad_sensitive):
            with self.assertRaises((ValueError, RuntimeError)):
                cfg._load_data()

    def test_sample_requires_loaded_data_and_supports_derived_split_sizes(self):
        cfg = PytorchDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            data_dir=self.temp_dir,
            train_size=6,
            test_size=2,
            data_params={"_args_": [torch.randn(10, 3), torch.randint(0, 2, (10,))]},
        )

        with self.assertRaises(ValueError):
            cfg._sample()

        cfg._load_data()
        cfg.train_size = None
        cfg.test_size = 0.2
        cfg._sample()
        self.assertEqual(len(cfg.X_train), 8)
        self.assertEqual(len(cfg.X_test), 2)

        cfg.test_size = None
        cfg.train_size = 0.5
        cfg.data_sample_time = None
        cfg._sample()
        self.assertEqual(len(cfg.X_train), 5)
        self.assertEqual(len(cfg.X_test), 5)

        cfg.train_size = None
        cfg.test_size = None
        cfg.data_sample_time = None
        with self.assertRaises(ValueError):
            cfg._sample()


    def test_call_accepts_existing_data_and_score_paths(self):
        data_path = Path(self.temp_dir) / "existing_data.pkl"
        score_path = Path(self.temp_dir) / "existing_scores.json"
        data_path.write_text("cached")
        score_path.write_text("cached")

        with patch.object(
            self.config,
            "load_scores",
            return_value={"cached": 1},
        ) as load_scores:
            with patch.object(self.config, "save_scores") as save_scores:
                scores = self.config(
                    data_file=str(data_path),
                    score_file=str(score_path),
                )

        load_scores.assert_not_called()
        save_scores.assert_called_once()
        self.assertIn("data_load_time", scores)
        


class TestPytorchCustomDataConfig(unittest.TestCase):
    """Tests for PytorchCustomDataConfig — covers DataLoader-based loading paths."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _make_simple_dataset(self, n=40):
        """Return a simple TensorDataset to use as both train and test."""
        X = torch.randn(n, 4)
        y = torch.randint(0, 2, (n,))
        return torch.utils.data.TensorDataset(X, y)

    def test_load_data_creates_dataloaders(self):
        from deckard.frameworks.pytorch.data import PytorchCustomDataConfig

        ds = self._make_simple_dataset()

        cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset=ds,
            data_dir=str(self.temp_dir),
            train_size=20,
            test_size=10,
            data_params={"batch_size": 4},
            val=False,
        )

        with patch.object(cfg, "_as_dataset", side_effect=[ds, ds]):
            cfg._load_data()

        self.assertIsNotNone(cfg.data_load_time)
        self.assertIsInstance(cfg._X, (Tensor, Dataset, tuple, list))
        # Accept both tuple/list of length 2 or a dataset of length 40
        if isinstance(cfg._X, (tuple, list)):
            self.assertEqual(len(cfg._X), 2)
        else:
            self.assertEqual(len(cfg._X), 40)

    def test_sample_creates_train_test_loaders(self):
        from deckard.frameworks.pytorch.data import PytorchCustomDataConfig

        ds_train = self._make_simple_dataset(40)
        ds_test = self._make_simple_dataset(20)

        cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="dummy",
            data_dir=str(self.temp_dir),
            train_size=40,
            test_size=20,
            data_params={"batch_size": 8},
            val=False,
        )
        # Manually inject _X to simulate _load_data output
        cfg._X = (ds_train, ds_test)
        cfg._y = (ds_train, ds_test)
        cfg.data_load_time = 0.0
        cfg.data_sample_time = None

        cfg._sample()

        from torch.utils.data import DataLoader

        self.assertIsInstance(cfg.X_train, DataLoader)
        self.assertIsInstance(cfg.X_test, DataLoader)
        self.assertIsInstance(cfg.y_train, torch.Tensor)
        self.assertIsInstance(cfg.y_test, torch.Tensor)

    def test_truncate_dataset(self):
        from deckard.frameworks.pytorch.data import PytorchCustomDataConfig

        ds = self._make_simple_dataset(40)
        cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="dummy",
            data_dir=str(self.temp_dir),
            train_size=10,
            test_size=10,
            data_params={},
        )
        subset = cfg._truncate_dataset(ds, 10)
        self.assertEqual(len(subset), 10)

    def test_as_dataset_with_string_raises_on_invalid(self):
        from deckard.frameworks.pytorch.data import PytorchCustomDataConfig

        cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="not.a.real.Dataset",
            data_dir=str(self.temp_dir),
            train_size=10,
            test_size=10,
            data_params={},
        )
        with self.assertRaises(Exception):
            cfg._as_dataset("not.a.real.Dataset", split="train", transform=None)

    def test_as_dataset_with_invalid_type_raises(self):
        from deckard.frameworks.pytorch.data import PytorchCustomDataConfig

        cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="dummy",
            data_dir=str(self.temp_dir),
            train_size=10,
            test_size=10,
            data_params={},
        )
        with self.assertRaises(TypeError):
            cfg._as_dataset(12345, split="train", transform=None)

    def test_custom_config_post_init_and_hash_defaults(self):
        from deckard.frameworks.pytorch.data import PytorchCustomDataConfig

        cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="dummy",
            data_dir=str(self.temp_dir),
            train_size=4,
            test_size=2,
            data_params=None,
        )

        self.assertEqual(cfg.data_params, {})
        self.assertTrue(cfg.shuffle)
        self.assertEqual(hash(cfg), hash(cfg))

    def test_custom_load_data_handles_callable_transforms_and_default_lengths(self):

        train_ds = self._make_simple_dataset(7)
        test_ds = self._make_simple_dataset(5)

        def train_transform(value):
            return value

        def test_transform(value):
            return value

        cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="dummy",
            data_dir=str(self.temp_dir),
            train_size=None,
            test_size=None,
            train_transform=train_transform,
            test_transform=test_transform,
            val=True,
            data_params={},
        )

        with patch.object(
            cfg,
            "_as_dataset",
            side_effect=[train_ds, test_ds],
        ) as as_dataset:
            cfg._load_data()

        self.assertEqual(cfg.train_n, 7)
        self.assertEqual(cfg.test_n, 5)
        self.assertIs(cfg.train_transform, train_transform)
        self.assertIs(cfg.test_transform, test_transform)
        self.assertEqual(as_dataset.call_args_list[1].kwargs["split"], "test")

    def test_custom_sample_handles_sensitive_batches_and_invalid_batch_shapes(self):

        cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="dummy",
            data_dir=str(self.temp_dir),
            train_size=6,
            test_size=4,
            data_params={"batch_size": 2},
        )
        cfg._X = (SensitiveBatchDataset(6), SensitiveBatchDataset(4))
        cfg._y = cfg._X
        cfg._sample()

        self.assertEqual(len(cfg._sensitive_train), 6)
        self.assertEqual(len(cfg._sensitive_test), 4)
        self.assertEqual(len(cfg._sensitive_all), 10)

        bad_cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="dummy",
            data_dir=str(self.temp_dir),
            train_size=4,
            test_size=4,
            data_params={"batch_size": 2},
        )
        bad_cfg._X = (InvalidBatchDataset(4), InvalidBatchDataset(4))
        bad_cfg._y = bad_cfg._X

        with self.assertRaises(ValueError):
            bad_cfg._sample()

    def test_custom_call_uses_cached_paths_and_persists_outputs(self):
        from deckard.frameworks.pytorch.data import PytorchCustomDataConfig
        import json

        data_path = Path(self.temp_dir) / "custom_data.pkl"
        score_path = Path(self.temp_dir) / "custom_scores.json"
        data_path.write_text("cached")
        score_path.write_text(json.dumps({"cached": 1}))

        cfg = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="dummy",
            data_dir=str(self.temp_dir),
            train_size=4,
            test_size=2,
            data_params={},
        )

        cached = self._make_simple_dataset(6)
        loaded = PytorchCustomDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            dataset="dummy",
            data_dir=str(self.temp_dir),
            train_size=4,
            test_size=2,
            data_params={},
        )
        loaded._X = (cached, cached)
        loaded._y = (cached, cached)
        loaded.X_train = object()
        loaded.score_dict = {"existing": 1}

        with patch.object(cfg, "load_object", return_value=loaded) as load_object:
            with patch.object(loaded, "save_scores") as save_scores:
                with patch.object(loaded, "save_object") as save_object:
                    scores = cfg(data_file=str(data_path), score_file=str(score_path), mode="pre-sample")

        load_object.assert_called_once()
        save_scores.assert_called_once()
        save_object.assert_called_once()
        self.assertEqual(scores, {"cached": 1})


class TestPytorchCustomDatasetConfig(unittest.TestCase):
    # TODO Implement mixin for using a dataloader object and test it here
    pass

class TestPytorchCustomDataLoaderConfig(unittest.TestCase):
    # TODO Implement mixin for using a dataloader object and test it here
    pass

class TestPytorchCustomTensorSetConfig(unittest.TestCase):
    # TODO Implement mixin for using a custom tensor data object and test it here
    pass



if __name__ == "__main__":
    unittest.main()
    # Remove temporary directory after tests
