import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CelebASmileDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "flwrlabs/celeba",
        subset: str = "img_align+identity+attr",
        smile_attribute: str = "Smiling",
        sensitive_attribute: str = "Male",
        sensitive_attributes=None,
        transform=None,
        split: str = "train",
        **kwargs,
    ):
        self.smile_attribute = smile_attribute
        self.sensitive_attribute = sensitive_attribute
        self.sensitive_attributes = sensitive_attributes or [
            sensitive_attribute
        ]
        self.transform = transform

        try:
            from datasets import load_dataset  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "CelebASmileDataset requires the 'datasets' package. Install deckard[pytorch,fairlearn] extras.",
            ) from exc

        self.dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
        )
        self._sensitive = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(image, torch.Tensor):
            # Ensure tensors are channel-first for PyTorch models.
            if (
                image.ndim == 3
                and image.shape[0] not in {1, 3}
                and image.shape[-1] in {1, 3}
            ):
                image = image.permute(2, 0, 1)
            image = image.float()
            if torch.max(image) > 1.0:
                image = image / 255.0

        smile_label = int(item[self.smile_attribute])

        if len(self.sensitive_attributes) == 1:
            sensitive = item[self.sensitive_attributes[0]]
        else:
            sensitive = tuple(item[attr] for attr in self.sensitive_attributes)

        self._sensitive.append(sensitive)
        return image, smile_label, sensitive


class SyntheticImageDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        image_size: int = 28,
        num_channels: int = 1,
        num_classes: int = 2,
        random_state: int = 42,
        transform=None,
        split: str = "train",
        **kwargs,
    ):
        self.transform = transform
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.num_channels = int(num_channels)
        self.num_classes = int(num_classes)

        split_offsets = {"train": 0, "valid": 1, "test": 2}
        seed = int(random_state) + split_offsets.get(split, 3)
        rng = np.random.default_rng(seed)

        images = rng.random(
            (
                self.num_samples,
                self.num_channels,
                self.image_size,
                self.image_size,
            ),
            dtype=np.float32,
        )
        labels = rng.integers(
            0,
            self.num_classes,
            size=self.num_samples,
            dtype=np.int64,
        )
        sensitive = rng.integers(0, 2, size=self.num_samples, dtype=np.int64)

        self._X = torch.from_numpy(images)
        self._y = torch.from_numpy(labels)
        self._sensitive = sensitive.tolist()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self._X[idx]
        if self.transform:
            image = self.transform(image)
        return image, int(self._y[idx].item()), int(self._sensitive[idx])


class SyntheticTabularFairnessDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        n_features: int = 16,
        num_classes: int = 2,
        random_state: int = 42,
        **kwargs,
    ):
        self.num_samples = int(num_samples)
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)

        rng = np.random.default_rng(int(random_state))
        X = rng.standard_normal(
            (self.num_samples, self.n_features), dtype=np.float32
        )
        y = rng.integers(
            0, self.num_classes, size=self.num_samples, dtype=np.int64
        )
        sensitive = rng.integers(0, 2, size=self.num_samples, dtype=np.int64)

        self._X = torch.from_numpy(X)
        self._y = torch.from_numpy(y)
        self._sensitive = sensitive.tolist()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._X[idx], int(self._y[idx].item()), int(self._sensitive[idx])


def build_celeba_smile_loaders(cfg):
    try:
        from smile_detector.transforms import (  # type: ignore[reportMissingImports]
            get_train_transforms,
            get_val_transforms,
        )
    except ImportError as exc:
        raise ImportError(
            "build_celeba_smile_loaders requires smile_detector.transforms in examples/pytorch/smile_detector.",
        ) from exc

    img_size = tuple(cfg.dataset.img_size)
    batch_size = int(cfg.dataset.batch_size)
    num_workers = int(cfg.dataset.num_workers)

    train_dataset = CelebASmileDataset(
        dataset_name=cfg.dataset.dataset_name,
        subset=cfg.dataset.subset,
        smile_attribute=cfg.dataset.smile_attribute,
        sensitive_attribute=cfg.dataset.sensitive_attribute,
        sensitive_attributes=getattr(cfg.dataset, "sensitive_attributes", None),
        transform=get_train_transforms(img_size),
        split="train",
    )

    val_dataset = CelebASmileDataset(
        dataset_name=cfg.dataset.dataset_name,
        subset=cfg.dataset.subset,
        smile_attribute=cfg.dataset.smile_attribute,
        sensitive_attribute=cfg.dataset.sensitive_attribute,
        sensitive_attributes=getattr(cfg.dataset, "sensitive_attributes", None),
        transform=get_val_transforms(img_size),
        split="valid",
    )

    test_dataset = CelebASmileDataset(
        dataset_name=cfg.dataset.dataset_name,
        subset=cfg.dataset.subset,
        smile_attribute=cfg.dataset.smile_attribute,
        sensitive_attribute=cfg.dataset.sensitive_attribute,
        sensitive_attributes=getattr(cfg.dataset, "sensitive_attributes", None),
        transform=get_val_transforms(img_size),
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
