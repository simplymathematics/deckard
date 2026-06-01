from __future__ import annotations

from types import SimpleNamespace

import pytest

import deckard.frameworks.pytorch.declarations as declarations


def test_tinynet_requires_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(declarations, "nn", None)

    with pytest.raises(ImportError):
        declarations.TinyNet()


def test_tinynet_forward_shape_when_torch_available() -> None:
    torch = pytest.importorskip("torch")

    model = declarations.TinyNet(input_dim=4, hidden_dim=8, output_dim=3)
    out = model(torch.randn(5, 4))

    assert out.shape == (5, 3)


def test_flexnet_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="Backbone 'missing_backbone' not found"):
        declarations.FlexNet(backbone_name="missing_backbone", pretrained=False)


def test_flexnet_adapts_resnet_style_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(16, 2)

        def forward(self, x):
            batch = x.shape[0]
            logits = self.fc(torch.ones((batch, 16), dtype=x.dtype, device=x.device))
            return logits

    def _factory(*, weights=None):
        assert weights is None
        return _Backbone()

    monkeypatch.setattr(
        declarations,
        "models",
        SimpleNamespace(resnet18=_factory),
    )

    model = declarations.FlexNet(
        backbone_name="resnet18",
        num_channels=1,
        num_classes=5,
        pretrained=False,
    )
    out = model(torch.randn(2, 1, 6, 6))

    assert model.backbone.conv1.in_channels == 1
    assert model.backbone.fc.out_features == 5
    assert model.in_features == 16
    assert out.shape == (2, 5)


def test_flexnet_input_adaptation_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

    monkeypatch.setattr(
        declarations,
        "models",
        SimpleNamespace(resnet18=lambda **_: _Backbone()),
    )

    with pytest.raises(NotImplementedError, match="Channel adaptation"):
        declarations.FlexNet(
            backbone_name="resnet18",
            num_channels=2,
            pretrained=False,
        )


def test_flexnet_output_adaptation_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)

    monkeypatch.setattr(
        declarations,
        "models",
        SimpleNamespace(resnet18=lambda **_: _Backbone()),
    )

    with pytest.raises(NotImplementedError, match="Classification head adaptation"):
        declarations.FlexNet(backbone_name="resnet18", pretrained=False)
