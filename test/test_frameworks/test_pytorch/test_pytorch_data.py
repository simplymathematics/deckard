import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from torch.utils.data import Dataset, IterableDataset

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


class StreamingIterableDataset(IterableDataset):
    def __iter__(self):
        for i in range(4):
            yield torch.randn(4), torch.tensor(i % 2)


def _split_sampler(train_size, test_size, **overrides):
    sampler = {
        "name": "split",
        "train_size": train_size,
        "test_size": test_size,
    }
    sampler.update(overrides)
    return sampler


def _tensor_binary_data(n=300, features=4):
    return torch.randn(n, features), torch.randint(0, 2, (n,))


def _assert_train_test_dataloaders(cfg):
    from torch.utils.data import DataLoader

    assert isinstance(cfg.X_train, DataLoader)
    assert isinstance(cfg.X_test, DataLoader)


def _assert_sensitive_split(cfg, *, total_len, train_len, test_len):
    assert hasattr(cfg, "_sensitive")
    assert len(cfg._sensitive) == total_len
    assert hasattr(cfg, "_sensitive_train")
    assert hasattr(cfg, "_sensitive_test")
    assert len(cfg._sensitive_train) == train_len
    assert len(cfg._sensitive_test) == test_len


class TestPytorchDataConfig:

    def _make_config(self, **overrides):
        config = {
            "name": "torch.utils.data.TensorDataset",
            "data_dir": self.temp_dir,
            "sampler": _split_sampler(100, 100, random_state=42),
            "data_params": None,
            "data_args": _tensor_binary_data(),
        }
        config.update(overrides)

        data_dir = (
            config["data_dir"] if config["data_dir"] is not None else self.temp_dir
        )
        data_args = config["data_args"]
        data_params = config["data_params"]
        if data_params is None:
            data_params = {"_args_": [*data_args]}
        else:
            data_params = dict(data_params)
            data_params.setdefault("_args_", [*data_args])
        return PytorchDataConfig(
            name=config["name"],
            data_dir=data_dir,
            sampler=config["sampler"],
            data_params=data_params,
        )

    def setup_method(self):
        X, y = _tensor_binary_data()
        self.config = self._make_config(data_args=(X, y))

    @classmethod
    def setup_class(cls):
        # Create temporary directory for data storage
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_initialization(self):
        assert self.config.data_dir == self.temp_dir
        assert self.config._get_sampler_option("test_size", None) == 100
        assert self.config._get_sampler_option("train_size", None) == 100
        assert self.config._get_sampler_option("random_state", None) == 42
        assert self.config._get_sampler_option("stratify", True)

    def test_load_data(self):
        self.config.load_dataset()
        assert isinstance(self.config._X, (Tensor, Dataset))
        assert isinstance(self.config._y, (Tensor, Dataset))
        assert self.config.data_load_time > 0

    def test_private_max_samples_caps_loaded_dataset(self):
        X, y = _tensor_binary_data()
        with patch.dict("os.environ", {"DECKARD_TEST_MAX_SAMPLES": "120"}):
            cfg = self._make_config(
                sampler=_split_sampler(80, 40, random_state=42),
                data_args=(X, y),
            )

            cfg.load_dataset()

        assert cfg._X.shape[0] == 120
        assert cfg._y.shape[0] == 120

    def test_sample(self):
        self.config.load_dataset()
        self.config.fit()
        from torch.utils.data import Subset

        assert isinstance(self.config.X_train, (Tensor, Dataset, Subset))
        assert isinstance(self.config.y_train, (Tensor, Dataset))
        assert isinstance(self.config.X_test, (Tensor, Dataset, Subset))
        assert isinstance(self.config.y_test, (Tensor, Dataset))

    def test_sample_allows_stratify_false(self):
        self.config._set_sampler_option("stratify", False)
        self.config.load_dataset()
        self.config.fit()
        assert len(self.config.X_train) == 100
        assert len(self.config.X_test) == 100

    def test_call(self):
        scores = self.config(
            files={"data_file": str(Path(self.temp_dir) / "data.pkl")},
        )
        assert "data_load_time" in scores
        assert "data_sample_time" in scores
        assert scores["data_load_time"] > 0
        assert scores["data_sample_time"] > 0

    def test_invalid_dataset_name(self):
        self.config.data_params["name"] = "invalid_dataset"
        with pytest.raises(Exception):
            self.config.load_dataset()

    @pytest.mark.parametrize(
        "dataset,expected",
        [
            (IntImageDataset(), "map"),
            (StreamingIterableDataset(), "iterable"),
        ],
    )
    def test_resolve_dataset_type(self, dataset, expected):
        assert self.config.resolve_dataset_type(dataset) == expected

    def test_hash_method(self):
        h1 = hash(self.config)
        h2 = hash(self.config)
        assert h1 == h2

    def test_load_data_collects_sensitive_from_third_tuple_item(self):
        X, y = _tensor_binary_data(120)
        ds = TensorWithSensitiveDataset(X, y)
        cfg = self._make_config(
            name="dummy.dataset",
            sampler=_split_sampler(60, 40, stratify=True),
            data_params={},
            data_args=(X, y),
        )

        with patch("deckard.frameworks.pytorch.data.load_class", return_value=ds):
            cfg.load_dataset()
        cfg.fit()
        _assert_sensitive_split(cfg, total_len=len(y), train_len=60, test_len=40)

    def test_load_data_collects_sensitive_from_dataset_attribute(self):
        X, y = _tensor_binary_data(110)
        ds = TensorWithDatasetSensitiveAttr(X, y)
        cfg = self._make_config(
            name="dummy.dataset",
            sampler=_split_sampler(70, 20, stratify=True),
            data_params={},
            data_args=(X, y),
        )

        with patch("deckard.frameworks.pytorch.data.load_class", return_value=ds):
            cfg.load_dataset()
        cfg.fit()
        _assert_sensitive_split(cfg, total_len=len(y), train_len=70, test_len=20)

    def test_pytorch_data_hash_stable_after_call_and_runtime_mutation(self):
        original_hash = hash(self.config)
        self.config()
        self.config.score_dict["runtime"] = 1
        self.config.data_load_time = 999.0
        assert hash(self.config) == original_hash

    def test_pytorch_data_score_persistence_roundtrip(self):
        self.config.load_dataset()
        self.config.fit()
        from torch.utils.data import Subset

        assert isinstance(self.config.X_train, (Tensor, Dataset, Subset))
        assert isinstance(self.config.X_test, (Tensor, Dataset, Subset))
        scores = self.config.score()

        score_path = Path(self.temp_dir) / "pytorch_data_scores.json"
        self.config.save_scores(scores, str(score_path))
        loaded = self.config.load_scores(str(score_path))

        assert len(loaded) > 0
        assert "test" in loaded
        assert "num_classes" in loaded["test"]

    def test_call_with_score_file_saves_json(self):
        score_path = str(Path(self.temp_dir) / "pytorch_scores.json")
        scores = self.config(files={"score_file": score_path})
        assert Path(score_path).exists()
        assert "data_load_time" in scores

    def test_float_train_test_sizes(self):
        X, y = _tensor_binary_data(100)
        cfg = PytorchDataConfig(
            name="torch.utils.data.TensorDataset",
            data_dir=self.temp_dir,
            sampler=_split_sampler(0.7, 0.2),
            data_params={"_args_": [X, y]},
        )
        cfg.load_dataset()
        cfg.fit()
        assert len(cfg.X_train) == 70
        assert len(cfg.X_test) == 20

    def test_train_test_exceed_total_raises(self):
        X, y = _tensor_binary_data(100)
        cfg = self._make_config(
            sampler=_split_sampler(70, 70),
            data_args=(X, y),
        )
        cfg.load_dataset()
        with pytest.raises(ValueError):
            cfg.fit()

    def test_invalid_stratify_raises(self):
        X, y = _tensor_binary_data(80)
        cfg = self._make_config(
            sampler=_split_sampler(50, 30, stratify="column_name"),
            data_args=(X, y),
        )
        cfg.load_dataset()
        with pytest.raises(ValueError):
            cfg.fit()

    def test_normalize_sensitive_item_variants(self):
        cfg = self.config
        # tensor scalar
        t_scalar = torch.tensor(3)
        assert cfg._normalize_sensitive_item(t_scalar) == 3
        # tensor vector
        t_vec = torch.tensor([1, 2])
        assert cfg._normalize_sensitive_item(t_vec) == (1, 2)
        # numpy scalar
        import numpy as np

        arr_scalar = np.array(5)
        assert cfg._normalize_sensitive_item(arr_scalar) == 5
        # numpy vector
        arr_vec = np.array([1, 2])
        assert cfg._normalize_sensitive_item(arr_vec) == (1, 2)
        # list
        assert cfg._normalize_sensitive_item([1, 2]) == (1, 2)
        # dict
        result = cfg._normalize_sensitive_item({"b": 2, "a": 1})
        assert result == (("a", 1), ("b", 2))
        # passthrough for str
        assert cfg._normalize_sensitive_item("hello") == "hello"

    def test_post_init_normalizes_data_dir_and_torchvision_root(self):
        cfg = PytorchDataConfig(
            name="torchvision.datasets.MNIST",
            data_dir=None,
            sampler=_split_sampler(4, 2),
            data_params=None,
        )

        assert isinstance(cfg.data_dir, str)
        assert cfg.data_params["root"] == cfg.data_dir

    def test_load_data_uses_canonical_torchvision_name_and_numpy_fallback_paths(self):
        cfg = PytorchDataConfig(
            name="torchvision.datasets.MNIST",
            data_dir=self.temp_dir,
            sampler=_split_sampler(2, 2),
            data_params={"batch_size": 8, "_args_": ["ignored"]},
        )

        seen = {}

        def fake_load_class(name, **kwargs):
            seen["name"] = name
            seen["kwargs"] = kwargs
            return ArrayFallbackDataset()

        with patch(
            "deckard.frameworks.pytorch.data.load_class",
            side_effect=fake_load_class,
        ):
            cfg.load_dataset()

        assert seen["name"] == "torchvision.datasets.MNIST"
        assert "batch_size" not in seen["kwargs"]
        assert tuple(cfg._X.shape) == (4, 1, 2, 2)
        assert torch.is_floating_point(cfg._X)
        assert tuple(cfg._y.shape) == (4, 2)

    def test_load_data_converts_nonfloating_tensor_inputs(self):
        cfg = self._make_config(
            name="dummy.dataset",
            sampler=_split_sampler(3, 2),
            data_params={},
        )

        with patch(
            "deckard.frameworks.pytorch.data.load_class",
            return_value=IntImageDataset(),
        ):
            cfg.load_dataset()

        assert tuple(cfg._X.shape) == (6, 1, 2, 2)
        assert cfg._X.dtype == torch.float32

    def test_load_data_rejects_invalid_samples_and_mismatched_sensitive_lengths(self):
        cfg = self._make_config(
            name="dummy.dataset",
            sampler=_split_sampler(1, 1),
            data_params={},
        )

        with patch(
            "deckard.frameworks.pytorch.data.load_class",
            return_value=InvalidSampleDataset(),
        ):
            with pytest.raises(ValueError):
                cfg.load_dataset()

        X, y = _tensor_binary_data(4, 2)
        bad_sensitive = MismatchedSensitiveDataset(X, y)
        with patch(
            "deckard.frameworks.pytorch.data.load_class",
            return_value=bad_sensitive,
        ):
            with pytest.raises((ValueError, RuntimeError)):
                cfg.load_dataset()

    def test_sample_requires_loaded_data_and_supports_derived_split_sizes(self):
        cfg = self._make_config(
            sampler=_split_sampler(6, 2),
            data_args=_tensor_binary_data(10, 3),
        )

        with pytest.raises(ValueError):
            cfg.fit()

        cfg.load_dataset()
        cfg._set_sampler_option("train_size", None)
        cfg._set_sampler_option("test_size", 0.2)
        cfg.fit()
        assert len(cfg.X_train) == 8
        assert len(cfg.X_test) == 2

        cfg._set_sampler_option("test_size", None)
        cfg._set_sampler_option("train_size", 0.5)
        cfg.data_sample_time = None
        cfg.fit()
        assert len(cfg.X_train) == 5
        assert len(cfg.X_test) == 5

        cfg._set_sampler_option("train_size", None)
        cfg._set_sampler_option("test_size", None)
        cfg.data_sample_time = None
        with pytest.raises(ValueError):
            cfg.fit()

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
                    files={
                        "data_file": str(data_path),
                        "score_file": str(score_path),
                    },
                )

        load_scores.assert_not_called()
        save_scores.assert_called_once()
        assert "data_load_time" in scores


class TestPytorchCustomDataConfig:
    """Tests for PytorchCustomDataConfig — covers DataLoader-based loading paths."""

    @classmethod
    def setup_class(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _make_config(self, **kwargs):
        config = {
            "name": "torch.utils.data.TensorDataset",
            "dataset": "dummy",
            "data_dir": str(self.temp_dir),
            "sampler": _split_sampler(20, 10),
            "data_params": {},
            "val": False,
        }
        config.update(kwargs)
        return PytorchCustomDataConfig(**config)

    def _make_simple_dataset(self, n=40):
        """Return a simple TensorDataset to use as both train and test."""
        X = torch.randn(n, 4)
        y = torch.randint(0, 2, (n,))
        return torch.utils.data.TensorDataset(X, y)

    def _cache_paths(self, data_name, score_name):
        data_path = Path(self.temp_dir) / data_name
        score_path = Path(self.temp_dir) / score_name
        files = {
            "data_file": str(data_path),
            "score_file": str(score_path),
        }
        return data_path, score_path, files

    def _prepare_scoring_cfg(self, **kwargs):
        cfg = self._make_config(**kwargs)
        cfg.data_load_time = 0.0
        cfg.data_sample_time = 0.0
        cfg.score_dict = {}
        return cfg

    def test_load_data_creates_dataloaders(self):

        ds = self._make_simple_dataset()

        cfg = self._make_config(
            sampler=_split_sampler(20, 10),
            data_params={"batch_size": 4},
        )

        with patch.object(cfg, "_as_dataset", side_effect=[ds, ds]):
            cfg.load_dataset()

        assert cfg.data_load_time is not None
        assert isinstance(cfg._X, (Tensor, Dataset, tuple, list))
        # Accept both tuple/list of length 2 or a dataset of length 40
        if isinstance(cfg._X, (tuple, list)):
            assert len(cfg._X) == 2
        else:
            assert len(cfg._X) == 40

    def test_sample_creates_train_test_loaders(self):

        ds_train = self._make_simple_dataset(40)
        ds_test = self._make_simple_dataset(20)

        cfg = self._make_config(
            sampler=_split_sampler(40, 20),
            data_params={"batch_size": 8},
        )
        # Manually inject _X to simulate _load_data output
        cfg._X = (ds_train, ds_test)
        cfg._y = (ds_train, ds_test)
        cfg.data_load_time = 0.0
        cfg.data_sample_time = None

        cfg.fit()
        _assert_train_test_dataloaders(cfg)
        assert isinstance(cfg.y_train, torch.Tensor)
        assert isinstance(cfg.y_test, torch.Tensor)

    def test_truncate_dataset(self):

        ds = self._make_simple_dataset(40)
        cfg = self._make_config(
            sampler=_split_sampler(10, 10),
        )
        subset = cfg._truncate_dataset(ds, 10)
        assert len(subset) == 10

        capped_subset = cfg._truncate_dataset(ds, 100)
        assert len(capped_subset) == 40

    @pytest.mark.parametrize(
        "dataset_arg,expected_exc",
        [
            ("not.a.real.Dataset", Exception),
            (12345, TypeError),
        ],
    )
    def test_as_dataset_invalid_inputs_raise(self, dataset_arg, expected_exc):

        cfg = self._make_config(
            dataset="dummy",
            sampler=_split_sampler(10, 10),
        )
        with pytest.raises(expected_exc):
            cfg._as_dataset(dataset_arg, split="train", transform=None)

    def test_as_dataset_accepts_pre_split_dataset_for_unknown_split_tag(self):
        cfg = self._make_config(sampler=_split_sampler(10, 10))
        ds = self._make_simple_dataset(8)

        resolved = cfg._as_dataset(ds, split="holdout", transform=None)

        assert resolved is ds

    def test_as_dataset_forwards_unknown_split_tag_to_loader(self):
        cfg = self._make_config(sampler=_split_sampler(10, 10))
        ds = self._make_simple_dataset(6)

        with patch(
            "deckard.frameworks.pytorch.data.load_class",
            return_value=ds,
        ) as loader:
            resolved = cfg._as_dataset(
                "dummy.dataset",
                split="holdout",
                transform=None,
            )

        assert resolved is ds
        assert loader.call_args.kwargs["split"] == "holdout"

    def test_custom_config_post_init_and_hash_defaults(self):
        cfg = self._make_config(sampler=_split_sampler(4, 2), data_params=None)

        assert cfg.data_params == {}
        assert cfg.shuffle
        assert hash(cfg) == hash(cfg)

    def test_custom_load_data_handles_callable_transforms_and_default_lengths(self):

        train_ds = self._make_simple_dataset(7)
        test_ds = self._make_simple_dataset(5)

        def train_transform(value):
            return value

        def test_transform(value):
            return value

        cfg = self._make_config(
            sampler=_split_sampler(1, 1),
            train_transform=train_transform,
            test_transform=test_transform,
            val=True,
        )

        with patch.object(
            cfg,
            "_as_dataset",
            side_effect=[train_ds, test_ds],
        ) as as_dataset:
            cfg.load_dataset()

        assert cfg.train_n == 1
        assert cfg.test_n == 1
        assert cfg.train_transform is train_transform
        assert cfg.test_transform is test_transform
        assert as_dataset.call_args_list[0].kwargs["split"] == "train"
        assert as_dataset.call_args_list[1].kwargs["split"] == "val"

    @pytest.mark.parametrize(
        "use_val_split,expected_second_split",
        [
            (False, "evaluation"),
            (True, "holdout"),
        ],
    )
    def test_custom_load_data_supports_sampler_split_key_overrides(
        self,
        use_val_split,
        expected_second_split,
    ):

        train_ds = self._make_simple_dataset(6)
        test_ds = self._make_simple_dataset(4)

        cfg = self._make_config(
            sampler=_split_sampler(2, 2),
            sampler_params={
                "train_split_key": "training",
                "test_split_key": "evaluation",
                "val_split_key": "holdout",
            },
            val=use_val_split,
        )

        with patch.object(
            cfg,
            "_as_dataset",
            side_effect=[train_ds, test_ds],
        ) as as_dataset:
            cfg.load_dataset()

        assert as_dataset.call_args_list[0].kwargs["split"] == "training"
        assert as_dataset.call_args_list[1].kwargs["split"] == expected_second_split

    def test_custom_sample_handles_sensitive_batches_and_invalid_batch_shapes(self):

        cfg = self._make_config(
            sampler=_split_sampler(6, 4),
            data_params={"batch_size": 2},
        )
        cfg._X = (SensitiveBatchDataset(6), SensitiveBatchDataset(4))
        cfg._y = cfg._X
        cfg.fit()

        assert len(cfg._sensitive_train) == 6
        assert len(cfg._sensitive_test) == 4
        assert len(cfg._sensitive_all) == 10

        bad_cfg = self._make_config(
            sampler=_split_sampler(4, 4),
            data_params={"batch_size": 2},
        )
        bad_cfg._X = (InvalidBatchDataset(4), InvalidBatchDataset(4))
        bad_cfg._y = bad_cfg._X

        with pytest.raises(ValueError):
            bad_cfg.fit()

    def test_custom_call_uses_cached_paths_and_persists_outputs(self):
        import json

        data_path, score_path, files = self._cache_paths(
            "custom_data.pkl",
            "custom_scores.json",
        )
        data_path.write_text("cached")
        score_path.write_text(json.dumps({"cached": 1}))

        cfg = self._make_config(sampler=_split_sampler(4, 2))

        cached = self._make_simple_dataset(6)
        loaded = self._make_config(sampler=_split_sampler(4, 2))
        loaded._X = (cached, cached)
        loaded._y = (cached, cached)
        loaded.X_train = object()
        loaded.score_dict = {"existing": 1}

        with patch.object(cfg, "load_object", return_value=loaded) as load_object:
            with patch.object(loaded, "save_scores") as save_scores:
                with patch.object(loaded, "save_object") as save_object:
                    scores = cfg(
                        files=files,
                        mode="pre-sample",
                    )

        load_object.assert_called_once()
        save_scores.assert_called_once()
        save_object.assert_called_once()
        assert scores == {"cached": 1}

    def test_custom_call_continues_when_data_cache_pickle_fails(self, caplog):
        data_path, _score_path, files = self._cache_paths(
            "uncacheable_custom_data.pkl",
            "uncacheable_custom_scores.json",
        )

        cfg = self._prepare_scoring_cfg(sampler=_split_sampler(4, 2))

        with patch.object(cfg, "score", return_value={"ok": 1}):
            with patch.object(
                cfg,
                "save_object",
                side_effect=pickle.PicklingError("cannot pickle custom data"),
            ) as save_object:
                scores = cfg(files=files)

        save_object.assert_called_once()
        assert scores["ok"] == 1
        assert not data_path.exists()
        assert "Failed to cache data object" in caplog.text

    def test_file_backed_custom_dataset_skips_pickle_cache(self):
        data_path, _score_path, files = self._cache_paths(
            "file_backed_custom_data.pkl",
            "file_backed_custom_scores.json",
        )

        cfg = self._prepare_scoring_cfg(
            dataset="custom_dataset.py:ExampleDataset",
            sampler=_split_sampler(4, 2),
        )

        with patch.object(cfg, "score", return_value={"ok": 1}):
            with patch.object(cfg, "save_object") as save_object:
                scores = cfg(files=files)

        save_object.assert_not_called()
        assert scores["ok"] == 1
        assert not data_path.exists()

    @pytest.mark.parametrize(
        "dataset_size,train_size,test_size,set_data_load_time",
        [
            (12, 6, 6, True),
            (16, 8, 8, False),
        ],
    )
    def test_pre_split_dataset_paths_create_dataloaders(
        self,
        dataset_size,
        train_size,
        test_size,
        set_data_load_time,
    ):
        ds = torch.utils.data.TensorDataset(
            torch.randn(dataset_size, 4),
            torch.randint(0, 2, (dataset_size,)),
        )
        cfg = self._make_config(
            sampler=_split_sampler(train_size, test_size),
            data_params={"batch_size": 4},
        )
        cfg._X = (ds, ds)
        cfg._y = (ds, ds)
        if set_data_load_time:
            cfg.data_load_time = 0.0
            cfg.data_sample_time = None

        cfg.fit()
        _assert_train_test_dataloaders(cfg)

    def test_custom_tensor_dataset_path(self):
        X, y = _tensor_binary_data(10)
        ds = torch.utils.data.TensorDataset(X, y)
        cfg = self._make_config(
            dataset=ds,
            sampler=_split_sampler(5, 5),
            data_params={"batch_size": 2},
        )

        cfg.load_dataset()
        assert cfg.resolve_dataset_type(ds) == "tensor"
