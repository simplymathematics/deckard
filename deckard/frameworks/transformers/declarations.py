import torch
from pathlib import Path

try:
    from torch import nn
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
except ImportError:
    nn = None
    AutoModel = None
    AutoConfig = None
    AutoModelForSequenceClassification = None


class GenericFlexibleTransformer(nn.Module if nn else object):
    """
    A generic transformer wrapper that accepts any Hugging Face backbone
    (Text, Vision, Audio) and handles arbitrary input and output shapes.
    """

    def __init__(
        self,
        model_name="bert-base-uncased",
        model_revision: str | None = None,
        out_features=256,
        num_classes=2,
        pretrained=True,
        return_features: bool = False,
    ):
        if nn is None or AutoModel is None:
            raise ImportError(
                "GenericFlexibleTransformer requires 'torch' and 'transformers'.",
            )
        super().__init__()

        # 1. Load generic backbone config and model
        # Require an immutable model revision for remote HF Hub downloads.
        is_local_model = Path(str(model_name)).exists()
        if not is_local_model and model_revision is None:
            raise ValueError(
                "model_revision must be provided for non-local Hugging Face model_name",
            )

        self.config = AutoConfig.from_pretrained(
            model_name,
            revision=model_revision,
        )
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                config=self.config,
                revision=model_revision,
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Extract the hidden feature dimension of the backbone dynamically
        self.hidden_dim = getattr(
            self.config,
            "hidden_size",
            getattr(self.config, "d_model", getattr(self.config, "embed_dim", None)),
        )

        if self.hidden_dim is None:
            raise ValueError(
                "Could not automatically determine backbone hidden_size dimension.",
            )

        # 2. Configure downstream task headers (Features & Classes)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, out_features),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.classifier = nn.Linear(out_features, num_classes)
        self.return_features = bool(return_features)
        self.pad_token_id = getattr(self.config, "pad_token_id", 0)

    @staticmethod
    def _coerce_index_tensor(value):
        if not torch.is_tensor(value):
            return value
        if torch.is_floating_point(value):
            return value.long()
        return value

    def _pool_hidden_states(self, outputs, attention_mask=None) -> torch.Tensor:
        """
        Generic pooling engine. Automatically detects sequence shapes
        and extracts the context vector (CLS token, mean pooling, or pooled_output).
        """
        # Strategy A: Use built-in pooler output if present (e.g., BERT/RoBERTa)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        # Strategy B: Extract sequential hidden states (Batch, Seq_Len, Hidden_Dim)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                last_hidden_state = outputs[0]
            else:
                raise ValueError(
                    "Backbone output does not include last_hidden_state or tuple[0] tensor.",
                )

        # If text attention mask is provided, perform mean pooling over valid tokens
        if attention_mask is not None:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask

        # Strategy C: Standard Vision/Audio sequence pooling (Mean over time/patches)
        if len(last_hidden_state.shape) == 3:
            return torch.mean(last_hidden_state, dim=1)

        return last_hidden_state

    def forward(self, *args, **kwargs):
        """
        Accepts arbitrary keyword arguments (**kwargs) to adapt natively to
        tokenizers, image processors, or audio feature extractors.

        Args:
            **kwargs: Backbone-ready model inputs (token/image/audio features).

        Returns:
            Dictionary containing pooled features and classifier logits.
        """
        # Support both trainer-style positional tensors and keyword HF inputs.
        if args:
            if len(args) != 1:
                raise ValueError(
                    "GenericFlexibleTransformer accepts at most one positional input tensor",
                )
            if "input_ids" in kwargs:
                raise ValueError(
                    "Pass either positional tensor or input_ids, not both",
                )
            kwargs["input_ids"] = args[0]

        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if key in kwargs:
                kwargs[key] = self._coerce_index_tensor(kwargs[key])

        # Auto-build an attention mask when input_ids are provided without one.
        if "attention_mask" not in kwargs and "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
            if torch.is_tensor(input_ids):
                kwargs["attention_mask"] = (input_ids != self.pad_token_id).long()

        # Safely extract attention mask if passing tokenized text inputs
        attention_mask = kwargs.get("attention_mask", None)

        # Forward pass through the HF backbone
        outputs = self.backbone(**kwargs)

        # Pool downstream outputs into a flat context vector
        pooled_features = self._pool_hidden_states(
            outputs,
            attention_mask=attention_mask,
        )

        # Pass through the target projection layers
        features = self.feature_extractor(pooled_features)
        logits = self.classifier(features)

        if self.return_features:
            return {
                "features": features,
                "logits": logits,
            }

        # Default to logits for compatibility with generic torch trainers.
        return logits


class PretrainedSequenceClassificationTransformer(nn.Module if nn else object):
    """Loads a Hugging Face sequence-classification model with its tuned head."""

    def __init__(
        self,
        model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        model_revision: str | None = "main",
        pretrained=True,
        return_features: bool = False,
        **kwargs,
    ):
        _ = kwargs
        if nn is None or AutoModelForSequenceClassification is None:
            raise ImportError(
                "PretrainedSequenceClassificationTransformer requires 'torch' and 'transformers'.",
            )
        super().__init__()

        is_local_model = Path(str(model_name)).exists()
        if not is_local_model and model_revision is None:
            raise ValueError(
                "model_revision must be provided for non-local Hugging Face model_name",
            )

        if pretrained:
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                revision=model_revision,
            )
        else:
            config = AutoConfig.from_pretrained(model_name, revision=model_revision)
            self.backbone = AutoModelForSequenceClassification.from_config(config)

        self.return_features = bool(return_features)
        self.pad_token_id = getattr(self.backbone.config, "pad_token_id", 0)

    @staticmethod
    def _coerce_index_tensor(value):
        if not torch.is_tensor(value):
            return value
        if torch.is_floating_point(value):
            return value.long()
        return value

    def forward(self, *args, **kwargs):
        if args:
            if len(args) != 1:
                raise ValueError(
                    "PretrainedSequenceClassificationTransformer accepts at most one positional input tensor",
                )
            if "input_ids" in kwargs:
                raise ValueError(
                    "Pass either positional tensor or input_ids, not both",
                )
            kwargs["input_ids"] = args[0]

        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if key in kwargs:
                kwargs[key] = self._coerce_index_tensor(kwargs[key])

        if "attention_mask" not in kwargs and "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
            if torch.is_tensor(input_ids):
                kwargs["attention_mask"] = (input_ids != self.pad_token_id).long()

        outputs = self.backbone(**kwargs)
        logits = outputs.logits

        if self.return_features:
            hidden_states = getattr(outputs, "hidden_states", None)
            features = None
            if hidden_states is not None and len(hidden_states) > 0:
                features = hidden_states[-1][:, 0, :]
            return {"features": features, "logits": logits}

        return logits


class HuggingFaceArtModelWrapper(nn.Module if nn else object):
    """Wraps GenericFlexibleTransformer for use with ART's HuggingFaceClassifierPyTorch.

    ART's InputFilter converts inputs to numpy arrays and then back to tensors.
    When integer input_ids go through this path they may arrive as float32.
    This wrapper:
    - Casts float input tensors back to ``torch.long`` before forwarding.
    - Returns a namespace object with a ``.logits`` attribute, as expected by
      ``art.estimators.classification.HuggingFaceClassifierPyTorch``.
    """

    def __init__(self, model: "GenericFlexibleTransformer"):
        if nn is None:
            raise ImportError(
                "HuggingFaceArtModelWrapper requires 'torch' and 'transformers'.",
            )
        super().__init__()
        self._inner = model

    def _inner_device(self):
        first_param = next(self._inner.parameters(), None)
        if first_param is None:
            return torch.device("cpu")
        return first_param.device

    def forward(self, x: "torch.Tensor"):
        from types import SimpleNamespace

        if torch.is_tensor(x) and x.dtype in (torch.float32, torch.float64):
            x = x.long()
        if torch.is_tensor(x):
            x = x.to(self._inner_device())
        result = self._inner(x)
        if isinstance(result, dict):
            logits = result["logits"]
        else:
            logits = result
        return SimpleNamespace(logits=logits)

    def parameters(self, recurse: bool = True):
        return self._inner.parameters(recurse=recurse)

    def named_parameters(self, *args, **kwargs):
        return self._inner.named_parameters(*args, **kwargs)

    def train(self, mode: bool = True):
        self._inner.train(mode)
        return self

    def eval(self):
        self._inner.eval()
        return self
