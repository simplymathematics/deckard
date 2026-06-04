from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_fairness_dataset_module():
    root = Path(__file__).resolve().parents[3]
    module_path = root / "examples" / "pytorch" / "torch_fairness_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "torch_fairness_dataset",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_image_dataset_materializes_only_active_split():
    module = _load_fairness_dataset_module()

    dataset = module.SyntheticImageDataset(
        num_samples=12,
        image_size=8,
        num_channels=1,
        split="validation",
        random_state=123,
    )

    assert dataset.split == "val"
    assert not hasattr(dataset, "_split_payloads")
    assert len(dataset) == 12

    image, label, sensitive = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)
    assert isinstance(sensitive, int)


def test_synthetic_tabular_dataset_materializes_only_active_split():
    module = _load_fairness_dataset_module()

    dataset = module.SyntheticTabularFairnessDataset(
        num_samples=10,
        n_features=5,
        split="valid",
        random_state=55,
    )

    assert dataset.split == "val"
    assert not hasattr(dataset, "_split_payloads")
    assert len(dataset) == 10

    features, label, sensitive = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(label, int)
    assert isinstance(sensitive, int)


def test_celeba_wrapper_uses_flexible_hf_loader(monkeypatch):
    module = _load_fairness_dataset_module()

    class _FakeImage:
        mode = "RGB"

        def convert(self, _mode):
            return self

        def __array__(self, dtype=None):
            tensor = torch.zeros((8, 8, 3), dtype=torch.uint8)
            arr = tensor.numpy()
            return arr.astype(dtype) if dtype is not None else arr

    calls = []

    class _FakeHFDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            _ = idx
            return {"image": _FakeImage(), "Smiling": 1, "Male": 0}

    def _fake_loader(name, dataset_split, data_params=None):
        calls.append((name, dataset_split, dict(data_params or {})))
        return _FakeHFDataset()

    monkeypatch.setattr(
        module.FlexibleHuggingFaceDataset,
        "load_huggingface_dataset",
        _fake_loader,
    )

    dataset = module.CelebASmileDataset(
        dataset_name="flwrlabs/celeba",
        subset="img_align+identity+attr",
        split="valid",
    )

    image, label, sensitive = dataset[0]

    assert calls == [
        (
            "flwrlabs/celeba",
            "validation",
            {"dataset_config_name": "img_align+identity+attr"},
        ),
    ]
    assert isinstance(image, torch.Tensor)
    assert label == 1
    assert sensitive == 0
