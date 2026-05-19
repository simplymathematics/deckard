try:
    import torchvision.models as models
    from torch import nn
except:
    nn = None
    models = None


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
        return self.backbone(x)
