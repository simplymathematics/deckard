from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class _DatasetSource:
    repo: str

    def raw_url(self, relative_path: str) -> str:
        rel = relative_path.lstrip("/")
        return f"https://raw.githubusercontent.com/{self.repo}/main/{rel}"


class DemiMathAnalysisDataset(Dataset):
    """Torch dataset backed by DEMI-MathAnalysis CSV files.

    Expected split behavior:
    - train: divided/pretraining_data.csv
    - valid: deterministic subset from benchmark_data.csv
    - test: full benchmark_data.csv
    """

    def __init__(
        self,
        repo: str = "ziye2chen/DEMI-MathAnalysis",
        split: str = "train",
        model_name: str = "distilbert-base-uncased",
        max_length: int = 96,
        cache_dir: str = "raw_data/demi_math_analysis",
        transform=None,
        **kwargs,
    ):
        self.source = _DatasetSource(repo=repo)
        self.split = str(split).strip().lower()
        self.max_length = int(max_length)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transform = transform

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "DemiMathAnalysisDataset requires 'transformers'. Install deckard[transformers].",
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        frame = self._load_split_frame(self.split)
        text_column = self._resolve_text_column(frame)
        label_column = self._resolve_label_column(frame)

        self.texts = frame[text_column].astype(str).fillna("").tolist()
        labels = frame[label_column].astype(str).fillna("unknown").tolist()

        ordered_labels = sorted(set(labels))
        self.label_to_idx = {name: idx for idx, name in enumerate(ordered_labels)}
        self.targets = [self.label_to_idx[name] for name in labels]

    def _cache_path(self, relative_path: str) -> Path:
        digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
        stem = Path(relative_path).name
        return self.cache_dir / f"{digest}_{stem}"

    def _read_remote_csv(self, relative_path: str) -> pd.DataFrame:
        target = self._cache_path(relative_path)
        if not target.exists():
            url = self.source.raw_url(relative_path)
            frame = pd.read_csv(url)
            frame.to_csv(target, index=False)
        return pd.read_csv(target)

    def _load_split_frame(self, split: str) -> pd.DataFrame:
        if split == "train":
            return self._read_remote_csv("divided/pretraining_data.csv")

        benchmark = self._read_remote_csv("divided/benchmark_data.csv")
        if split == "valid":
            sample_n = max(1, int(round(len(benchmark) * 0.25)))
            return benchmark.sample(n=sample_n, random_state=42)
        if split == "test":
            return benchmark

        raise ValueError("split must be one of: train, valid, test")

    @staticmethod
    def _resolve_text_column(frame: pd.DataFrame) -> str:
        candidates = ["problem", "question", "text", "prompt"]
        normalized = {str(col).strip().lower(): col for col in frame.columns}
        for col in candidates:
            if col in normalized:
                return normalized[col]
        return frame.columns[0]

    @staticmethod
    def _resolve_label_column(frame: pd.DataFrame) -> str:
        candidates = ["problem_type", "problemtype", "topic", "label", "category"]
        normalized = {str(col).strip().lower(): col for col in frame.columns}
        for col in candidates:
            if col in normalized:
                return normalized[col]
        return frame.columns[-1]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)

        if self.transform is not None:
            input_ids = self.transform(input_ids)

        label = int(self.targets[idx])
        return input_ids, label
