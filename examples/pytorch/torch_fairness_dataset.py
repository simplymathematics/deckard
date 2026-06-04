import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


from deckard.plugins.datasets.declarations import FlexibleHuggingFaceDataset


def _normalize_hf_split(split: str) -> str:
    token = str(split).strip().lower()
    if token in {"val", "valid", "validation"}:
        return "validation"
    return token


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
            sensitive_attribute,
        ]
        self.transform = transform
        self.split = _normalize_hf_split(split)

        loader_params = dict(kwargs)
        if subset is not None:
            subset_token = str(subset).strip()
            if subset_token != "":
                loader_params.setdefault("dataset_config_name", subset_token)

        self.dataset = FlexibleHuggingFaceDataset.load_huggingface_dataset(
            name=dataset_name,
            dataset_split=self.split,
            data_params=loader_params,
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
        dataset_name=cfg.dataset.name,
        subset=cfg.dataset.subset,
        smile_attribute=cfg.dataset.smile_attribute,
        sensitive_attribute=cfg.dataset.sensitive_attribute,
        sensitive_attributes=getattr(cfg.dataset, "sensitive_attributes", None),
        transform=get_train_transforms(img_size),
        split="train",
    )

    val_dataset = CelebASmileDataset(
        dataset_name=cfg.dataset.name,
        subset=cfg.dataset.subset,
        smile_attribute=cfg.dataset.smile_attribute,
        sensitive_attribute=cfg.dataset.sensitive_attribute,
        sensitive_attributes=getattr(cfg.dataset, "sensitive_attributes", None),
        transform=get_val_transforms(img_size),
        split="valid",
    )

    test_dataset = CelebASmileDataset(
        dataset_name=cfg.dataset.name,
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
