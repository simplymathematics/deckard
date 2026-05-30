from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import deckard.frameworks.transformers.declarations as decl  # noqa: E402


class _FakeConfig:
    hidden_size = 8
    pad_token_id = 0


class _FakeBackbone(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError("input_ids is required for fake backbone")
        if input_ids.ndim != 2:
            raise ValueError("expected 2D input_ids")
        batch, seq_len = input_ids.shape
        hidden = torch.randn(batch, seq_len, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=None)


class _FakeAutoConfig:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeConfig()


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeBackbone(kwargs["config"])

    @staticmethod
    def from_config(config):
        return _FakeBackbone(config)


def _patch_hf(monkeypatch):
    monkeypatch.setattr(decl, "nn", torch.nn)
    monkeypatch.setattr(decl, "AutoConfig", _FakeAutoConfig)
    monkeypatch.setattr(decl, "AutoModel", _FakeAutoModel)


def test_transformer_wrapper_requires_optional_dependencies(monkeypatch):
    _patch_hf(monkeypatch)
    monkeypatch.setattr(decl, "nn", None)

    with pytest.raises(ImportError, match="requires 'torch' and 'transformers'"):
        decl.GenericFlexibleTransformer(
            model_name="local-model",
            model_revision="main",
        )


def test_transformer_wrapper_requires_revision_for_remote_models(monkeypatch):
    _patch_hf(monkeypatch)

    with pytest.raises(ValueError, match="model_revision must be provided"):
        decl.GenericFlexibleTransformer(model_name="bert-base-uncased")


def test_transformer_wrapper_returns_logits_by_default(monkeypatch):
    _patch_hf(monkeypatch)

    model = decl.GenericFlexibleTransformer(
        model_name="bert-base-uncased",
        model_revision="main",
        pretrained=False,
        out_features=4,
        num_classes=3,
    )

    batch_input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    logits = model.forward(batch_input_ids)

    assert torch.is_tensor(logits)
    assert logits.shape == (2, 3)


def test_transformer_wrapper_can_return_features_dict(monkeypatch):
    _patch_hf(monkeypatch)

    model = decl.GenericFlexibleTransformer(
        model_name="bert-base-uncased",
        model_revision="main",
        pretrained=True,
        out_features=5,
        num_classes=2,
        return_features=True,
    )

    output = model.forward(
        input_ids=torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long),
    )

    assert isinstance(output, dict)
    assert set(output.keys()) == {"features", "logits"}
    assert output["features"].shape == (2, 5)
    assert output["logits"].shape == (2, 2)


def test_transformer_wrapper_coerces_float_token_inputs_to_long(monkeypatch):
    _patch_hf(monkeypatch)

    model = decl.GenericFlexibleTransformer(
        model_name="bert-base-uncased",
        model_revision="main",
        pretrained=False,
        out_features=4,
        num_classes=3,
    )

    float_input_ids = torch.tensor([[1.0, 2.0, 3.0, 0.0]], dtype=torch.float32)
    logits = model.forward(float_input_ids)

    assert torch.is_tensor(logits)
    assert logits.shape == (1, 3)


def test_huggingface_art_model_uses_art_safe_model_copy(monkeypatch):
    class _RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.last_input_device = None

        def forward(self, input_ids):
            self.last_input_device = input_ids.device
            logits = torch.zeros((input_ids.shape[0], 2), device=input_ids.device)
            return logits

    inner = _RecordingModel()
    wrapper = decl.HuggingFaceArtModelWrapper(inner)

    output = wrapper.forward(torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32))

    assert output.logits.device == inner.weight.device
    assert inner.last_input_device == inner.weight.device
