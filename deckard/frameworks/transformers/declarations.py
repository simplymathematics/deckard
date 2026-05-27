import torch
from pathlib import Path

try:
    from torch import nn
    from transformers import AutoConfig, AutoModel
except ImportError:
    nn = None
    AutoModel = None
    AutoConfig = None


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

    def _pool_hidden_states(self, outputs, attention_mask=None) -> torch.Tensor:
        """
        Generic pooling engine. Automatically detects sequence shapes
        and extracts the context vector (CLS token, mean pooling, or pooled_output).
        """
        # Strategy A: Use built-in pooler output if present (e.g., BERT/RoBERTa)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        # Strategy B: Extract sequential hidden states (Batch, Seq_Len, Hidden_Dim)
        last_hidden_state = outputs.last_hidden_state

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

    def forward(self, **kwargs):
        """
        Accepts arbitrary keyword arguments (**kwargs) to adapt natively to
        tokenizers, image processors, or audio feature extractors.

        Args:
            **kwargs: Backbone-ready model inputs (token/image/audio features).

        Returns:
            Dictionary containing pooled features and classifier logits.
        """
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

        return {
            "features": features,
            "logits": logits,
        }
