"""Shared text-runtime helpers for optional attack libraries.

This module centralizes text-dataset extraction and model adapter behavior used
by multiple plugin attack families (for example TextAttack and OpenAttack).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..artifacts import ScoreDict

try:
    import torch
    from torch.utils.data import Subset
except Exception:  # pragma: no cover
    torch = None
    Subset = object


def is_library_attack_name(attack_name: str, library_name: str) -> bool:
    """Return whether an attack declaration resolves to a library namespace."""
    normalized_name = str(attack_name or "").strip().lower()
    normalized_library = str(library_name or "").strip().lower()
    if normalized_library == "":
        return False
    return normalized_name.startswith(f"{normalized_library}.")


def resolve_runtime_model(model: Any):
    """Return concrete runtime model, unwrapping configs that store ``_model``."""
    runtime_model = getattr(model, "_model", None)
    return runtime_model if runtime_model is not None else model


def resolve_text_batch(
    data: Any,
    attack_config: Any,
    split: str | None = None,
) -> tuple[list[str], np.ndarray, Any]:
    """Extract text samples, labels, and tokenizer from runtime datasets."""
    effective_split = split or attack_config.resolve_mode_for_attack_kind("evasion")
    dataset = _resolve_split_dataset(data, effective_split)
    texts, labels, tokenizer = _dataset_texts_targets(dataset)
    limit = max(1, int(getattr(attack_config, "attack_size", 1) or 1))
    return texts[:limit], labels[:limit], tokenizer


def resolve_text_max_length(data: Any) -> int | None:
    """Resolve optional max token length from data runtime params."""
    dataset_params = getattr(data, "dataset_params", {}) or {}
    if isinstance(dataset_params, dict):
        return dataset_params.get("max_length")
    return None


def apply_attack_runtime_outputs(
    runtime: Any,
    *,
    records: list[dict[str, Any]],
    library: str,
    attack_name: str,
    successful_examples: int,
    count: int,
    error: str | None,
) -> None:
    """Write canonical attack outputs back onto a runtime attack config."""
    runtime.results = records
    runtime.attack = [row.get("adversarial_text") for row in records]
    runtime.attack_predictions = np.asarray(
        [row.get("adversarial_prediction") for row in records],
    )
    runtime.attack_time = 0.0
    runtime.attack_prediction_time = 0.0
    runtime.attack_score_time = 0.0
    runtime.score_dict = ScoreDict.from_payload(
        {
            "library": library,
            "attack_name": attack_name,
            "attack_size": count,
            "successful_examples": successful_examples,
            "attack_success_rate": (float(successful_examples) / max(count, 1)),
            "error": error,
        },
    )


def _resolve_split_dataset(data: Any, split: str):
    payload = getattr(data, "_X", None)
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        return payload[0] if split == "train" else payload[1]
    attr_name = f"{split}_dataset"
    dataset = getattr(data, attr_name, None)
    if dataset is not None:
        return dataset
    raise ValueError(
        "Expected data config with split datasets in _X or <split>_dataset attributes.",
    )


def _dataset_texts_targets(dataset: Any) -> tuple[list[str], np.ndarray, Any]:
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = list(dataset.indices)
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))

    texts = getattr(base_dataset, "texts", None)
    targets = getattr(base_dataset, "targets", None)
    tokenizer = getattr(base_dataset, "tokenizer", None)

    if texts is None or targets is None or tokenizer is None:
        raise ValueError(
            "Text attack plugins require datasets with texts, targets, and tokenizer attributes.",
        )

    selected_texts = [str(texts[index]) for index in indices]
    selected_targets = np.asarray(
        [int(targets[index]) for index in indices],
        dtype=int,
    )
    return selected_texts, selected_targets, tokenizer


@dataclass
class TransformerTextAdapter:
    """Tokenizer-driven text inference adapter for model wrappers."""

    model: Any
    tokenizer: Any
    max_length: int | None = None
    batch_size: int = 8

    def __post_init__(self):
        if torch is None:
            raise ImportError(
                "Transformer text attack helpers require torch to be installed.",
            )
        if hasattr(self.model, "eval"):
            self.model.eval()

    @property
    def device(self):
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return torch.device("cpu")
        return first_param.device

    def predict_logits(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, 0), dtype=np.float32)

        all_logits: list[np.ndarray] = []
        max_length = self.max_length or getattr(
            self.tokenizer,
            "model_max_length",
            128,
        )
        if max_length is None or int(max_length) <= 0:
            max_length = 128

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
                if torch.is_tensor(value)
            }
            with torch.no_grad():
                outputs = self.model(**encoded)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            if torch.is_tensor(logits):
                logits = logits.detach().cpu().numpy()
            all_logits.append(np.asarray(logits))
        return np.concatenate(all_logits, axis=0)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        logits = self.predict_logits(texts)
        if logits.size == 0:
            return logits
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)


__all__ = [
    "apply_attack_runtime_outputs",
    "TransformerTextAdapter",
    "is_library_attack_name",
    "resolve_runtime_model",
    "resolve_text_batch",
    "resolve_text_max_length",
]
