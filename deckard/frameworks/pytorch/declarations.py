import numpy as np

try:
    import torch
    import torchvision.models as models
    from torch import nn
    from torch.utils.data import Dataset as TorchDataset
except Exception:
    torch = None
    nn = None
    models = None
    TorchDataset = object


# TinyNet: Minimal torch model for binary classification
class TinyNet(nn.Module if nn else object):
    """A minimal torch model for binary classification (2-layer MLP)."""

    def __init__(self, input_dim=10, hidden_dim=16, output_dim=2):
        if nn is None:
            raise ImportError("TinyNet requires torch to be installed.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Run forward pass through TinyNet.

        Args:
            x: Input tensor batch.

        Returns:
            Logit tensor output.
        """
        return self.net(x)


class FlexNet(nn.Module if nn else object):
    """
    A customizable model that loads pre-trained weights and configures:
    - Input channels (via feature adapter)
    - Base feature extractor backbone
    - Output classes (via classification head)
    """

    def __init__(
        self,
        backbone_name="resnet18",
        num_channels=3,
        num_classes=2,
        pretrained=True,
    ):
        if nn is None or models is None:
            raise ImportError(
                "FlexibleNet requires torch and torchvision to be installed.",
            )
        super().__init__()

        # 1. Load the pre-trained backbone dynamically
        if not hasattr(models, backbone_name):
            raise ValueError(
                f"Backbone '{backbone_name}' not found in torchvision.models.",
            )

        weights = "DEFAULT" if pretrained else None
        # Call the model builder factory (e.g., models.resnet18(weights='DEFAULT'))
        model_fn = getattr(models, backbone_name)
        self.backbone = model_fn(weights=weights)

        # 2. Configure Input Channels
        # Most torchvision backbones expect 3 channels in their first convolution (conv1 or features[0])
        if num_channels != 3:
            self._adapt_input_channels(num_channels)

        # 3. Configure Output Classes
        # Dynamically find the classification head (fc, classifier, or heads) and replace it
        self.in_features = self._adapt_output_classes(num_classes)

    def _adapt_input_channels(self, num_channels):
        """Modifies the first convolutional layer to accept custom channel counts."""
        if hasattr(self.backbone, "conv1"):  # ResNet style
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                num_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
        elif hasattr(self.backbone, "features") and isinstance(
            self.backbone.features[0],
            nn.Conv2d,
        ):  # VGG / ConvNeXt style
            old_conv = self.backbone.features[0]
            self.backbone.features[0] = nn.Conv2d(
                num_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
        else:
            raise NotImplementedError(
                "Channel adaptation not automatically supported for this backbone architecture.",
            )

    def _adapt_output_classes(self, num_classes) -> int:
        """Replaces the final linear layer to output the correct number of classes."""
        if hasattr(self.backbone, "fc"):  # ResNet style
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, "classifier") and isinstance(
            self.backbone.classifier,
            nn.Sequential,
        ):  # VGG style
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, "classifier") and isinstance(
            self.backbone.classifier,
            nn.Linear,
        ):  # MobileNet style
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, "heads") and hasattr(
            self.backbone.heads,
            "head",
        ):  # ViT style
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes)
        else:
            raise NotImplementedError(
                "Classification head adaptation not supported for this backbone architecture.",
            )
        return in_features

    def forward(self, x):
        """Run forward pass through configured backbone/classification head.

        Args:
            x: Input tensor batch.

        Returns:
            Model logits for configured classes.
        """
        return self.backbone(x)


def _split_seed_from_random_state(random_state: int, split: str) -> int:
    split_offsets = {
        "train": 0,
        "val": 1,
        "valid": 1,
        "validation": 1,
        "test": 2,
    }
    split_token = str(split).strip().lower()
    return int(random_state) + split_offsets.get(split_token, 3)


def _normalize_synthetic_split(split: str) -> str:
    token = str(split).strip().lower()
    if token in {"val", "valid", "validation"}:
        return "val"
    if token in {"train", "test"}:
        return token
    return token


class SyntheticImageDataset(TorchDataset):
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
        _ = kwargs
        if torch is None:
            raise ImportError(
                "SyntheticImageDataset requires torch to be installed.",
            )
        self.transform = transform
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.num_channels = int(num_channels)
        self.num_classes = int(num_classes)
        self.split = _normalize_synthetic_split(split)
        active_split = (
            self.split if self.split in {"train", "val", "test"} else "train"
        )
        seed = _split_seed_from_random_state(
            random_state=random_state,
            split=active_split,
        )
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


class SyntheticTabularFairnessDataset(TorchDataset):
    def __init__(
        self,
        num_samples: int = 256,
        n_features: int = 16,
        num_classes: int = 2,
        random_state: int = 42,
        split: str = "train",
        **kwargs,
    ):
        _ = kwargs
        if torch is None:
            raise ImportError(
                "SyntheticTabularFairnessDataset requires torch to be installed.",
            )
        self.num_samples = int(num_samples)
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.split = _normalize_synthetic_split(split)
        active_split = (
            self.split if self.split in {"train", "val", "test"} else "train"
        )
        seed = _split_seed_from_random_state(
            random_state=random_state,
            split=active_split,
        )
        rng = np.random.default_rng(seed)
        X = rng.standard_normal(
            (self.num_samples, self.n_features),
            dtype=np.float32,
        )
        y = rng.integers(
            0,
            self.num_classes,
            size=self.num_samples,
            dtype=np.int64,
        )
        sensitive = rng.integers(0, 2, size=self.num_samples, dtype=np.int64)
        self._X = torch.from_numpy(X)
        self._y = torch.from_numpy(y)
        self._sensitive = sensitive.tolist()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._X[idx], int(self._y[idx].item()), int(self._sensitive[idx])


__all__ = [
    "TinyNet",
    "FlexNet",
    "SyntheticImageDataset",
    "SyntheticTabularFairnessDataset",
]
