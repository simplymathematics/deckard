from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_split(split: str, *, validation_fallback: str = "test") -> str:
    token = str(split).strip().lower()
    if token == "valid":
        return "validation"
    if token == "validation":
        return "validation"
    if token in {"train", "test"}:
        return token
    return validation_fallback if token else "train"


@dataclass
class _HFTextRuntime:
    tokenizer: Any
    max_length: int


class PythonProgrammingAppsDataset(Dataset):
    """Binary/ordinal-friendly programming dataset built from codeparrot/apps."""

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "codeparrot/apps",
        dataset_config: str | None = None,
        model_name: str = "microsoft/codebert-base",
        text_field: str = "question",
        label_field: str = "difficulty",
        max_length: int = 192,
        limit: int | None = 2000,
        transform=None,
        **kwargs,
    ):
        _ = kwargs
        self.transform = transform

        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "PythonProgrammingAppsDataset requires 'datasets' and 'transformers'.",
            ) from exc

        hf_split = _normalize_split(split)
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, split=hf_split)
        else:
            ds = load_dataset(dataset_name, split=hf_split)

        if limit is not None:
            ds = ds.select(range(min(int(limit), len(ds))))

        self.texts = [str(v) for v in ds[text_field]]
        raw_labels = [str(v).lower() for v in ds[label_field]]
        order = ["introductory", "interview", "competition"]
        known = {name: idx for idx, name in enumerate(order)}
        if all(v in known for v in raw_labels):
            self.targets = [known[v] for v in raw_labels]
        else:
            values = sorted(set(raw_labels))
            mapping = {name: idx for idx, name in enumerate(values)}
            self.targets = [mapping[v] for v in raw_labels]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.runtime = _HFTextRuntime(tokenizer=tokenizer, max_length=int(max_length))
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        encoded = self.runtime.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.runtime.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        if self.transform is not None:
            input_ids = self.transform(input_ids)
        return input_ids, int(self.targets[idx])


class FrenchEnglishTranslationDirectionDataset(Dataset):
    """Creates a two-class language-direction task from an en-fr translation corpus."""

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "opus_books",
        dataset_config: str = "en-fr",
        model_name: str = "distilbert-base-multilingual-cased",
        source_lang: str = "en",
        target_lang: str = "fr",
        translation_field: str = "translation",
        max_length: int = 160,
        limit: int | None = 5000,
        transform=None,
        **kwargs,
    ):
        _ = kwargs
        self.transform = transform

        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "FrenchEnglishTranslationDirectionDataset requires 'datasets' and 'transformers'.",
            ) from exc

        hf_split = _normalize_split(split)
        ds = load_dataset(dataset_name, dataset_config, split=hf_split)
        if limit is not None:
            ds = ds.select(range(min(int(limit), len(ds))))

        source_texts: list[str] = []
        target_texts: list[str] = []
        for item in ds:
            pair = item.get(translation_field, {})
            source_texts.append(str(pair.get(source_lang, "")))
            target_texts.append(str(pair.get(target_lang, "")))

        self.texts = source_texts + target_texts
        self.targets = [0] * len(source_texts) + [1] * len(target_texts)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.runtime = _HFTextRuntime(tokenizer=tokenizer, max_length=int(max_length))
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        encoded = self.runtime.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.runtime.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        if self.transform is not None:
            input_ids = self.transform(input_ids)
        return input_ids, int(self.targets[idx])


class ArithmeticMathQADataset(Dataset):
    """Arithmetic/category classification view of the MathQA dataset."""

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "math_qa",
        dataset_config: str | None = None,
        model_name: str = "distilbert-base-uncased",
        text_field: str = "Problem",
        label_field: str = "category",
        max_length: int = 192,
        limit: int | None = 6000,
        transform=None,
        **kwargs,
    ):
        _ = kwargs
        self.transform = transform

        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "ArithmeticMathQADataset requires 'datasets' and 'transformers'.",
            ) from exc

        hf_split = _normalize_split(split)
        try:
            if dataset_config:
                ds = load_dataset(dataset_name, dataset_config, split=hf_split)
            else:
                ds = load_dataset(dataset_name, split=hf_split)
        except ValueError as exc:
            # Some Hub datasets only expose a train split. Fall back so custom
            # data pipelines can still construct train/test subsets downstream.
            if "Unknown split" not in str(exc) or hf_split == "train":
                raise
            if dataset_config:
                ds = load_dataset(dataset_name, dataset_config, split="train")
            else:
                ds = load_dataset(dataset_name, split="train")

        if limit is not None:
            ds = ds.select(range(min(int(limit), len(ds))))

        # Build a split-stable class index so train/test labels are aligned.
        try:
            if dataset_config:
                ref_ds = load_dataset(dataset_name, dataset_config, split="train")
            else:
                ref_ds = load_dataset(dataset_name, split="train")
            ref_labels = [str(v) for v in ref_ds[label_field]]
        except Exception:
            ref_labels = [str(v) for v in ds[label_field]]

        mapping = {name: idx for idx, name in enumerate(sorted(set(ref_labels)))}

        self.texts = [str(v) for v in ds[text_field]]
        raw_labels = [str(v) for v in ds[label_field]]
        for value in raw_labels:
            if value not in mapping:
                mapping[value] = len(mapping)
        self.targets = [mapping[v] for v in raw_labels]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.runtime = _HFTextRuntime(tokenizer=tokenizer, max_length=int(max_length))
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        encoded = self.runtime.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.runtime.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        if self.transform is not None:
            input_ids = self.transform(input_ids)
        return input_ids, int(self.targets[idx])


class MedicalImagingPathMNISTDataset(Dataset):
    """Image classification dataset using MedMNIST PathMNIST from Hugging Face datasets."""

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "flwrlabs/medmnist",
        dataset_config: str = "pathmnist",
        model_name: str = "google/vit-base-patch16-224",
        image_field: str = "image",
        label_field: str = "label",
        limit: int | None = 12000,
        transform=None,
        **kwargs,
    ):
        _ = kwargs
        self.transform = transform

        try:
            from datasets import load_dataset
            from transformers import AutoImageProcessor
        except ImportError as exc:
            raise ImportError(
                "MedicalImagingPathMNISTDataset requires 'datasets' and 'transformers'.",
            ) from exc

        hf_split = _normalize_split(split, validation_fallback="test")
        ds = load_dataset(dataset_name, dataset_config, split=hf_split)
        if limit is not None:
            ds = ds.select(range(min(int(limit), len(ds))))

        self.dataset = ds
        self.image_field = image_field
        self.label_field = label_field
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        image = item[self.image_field]

        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)

        if self.transform is not None:
            pixel_values = self.transform(pixel_values)

        raw_label = item[self.label_field]
        if isinstance(raw_label, (list, tuple, np.ndarray)):
            label = int(raw_label[0])
        elif torch.is_tensor(raw_label):
            label = int(raw_label.detach().cpu().reshape(-1)[0].item())
        else:
            label = int(raw_label)

        return pixel_values, label
