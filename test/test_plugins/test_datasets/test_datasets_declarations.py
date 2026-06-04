from __future__ import annotations

import pandas as pd
import pytest

import deckard.plugins.datasets.declarations as decl


def test_flexible_huggingface_dataset_requires_keywords_only():
    with pytest.raises(TypeError):
        decl.FlexibleHuggingFaceDataset(
            "imdb",
            "label",
            ["text"],
            "train",
        )


def test_flexible_huggingface_dataset_loads_explicit_columns(monkeypatch):
    frame = pd.DataFrame(
        {
            "text": ["one", "two", "three"],
            "label": [0, 1, 0],
            "ignored": [10, 20, 30],
        },
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return frame

    monkeypatch.setattr(decl, "load_dataset", _fake_load_dataset)

    config = decl.FlexibleHuggingFaceDataset(
        name="imdb",
        target="label",
        keep=["text"],
        dataset_split="validation",
        data_params={"cache_dir": "/tmp/deckard-hf"},
        limit=2,
    )

    loaded = config.load_dataset()

    assert loaded is config
    assert config._X.equals(frame.loc[:1, ["text"]].reset_index(drop=True))
    assert config._y.equals(frame.loc[:1, "label"].reset_index(drop=True))
    assert calls == [
        (("imdb",), {"split": "validation", "cache_dir": "/tmp/deckard-hf"}),
    ]
    assert config.data_load_time is not None


def test_flexible_huggingface_dataset_uses_data_params_for_dataset_identity(
    monkeypatch,
):
    frame = pd.DataFrame(
        {
            "text": ["one", "two"],
            "label": [0, 1],
        },
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return frame

    monkeypatch.setattr(decl, "load_dataset", _fake_load_dataset)

    config = decl.FlexibleHuggingFaceDataset(
        name="fallback-dataset-name",
        target="label",
        keep=["text"],
        dataset_split="train",
        data_params={
            "dataset_name": "flwrlabs/celeba",
            "subset": "img_align+identity+attr",
            "cache_dir": "/tmp/deckard-hf",
        },
    )

    config.load_dataset()

    assert calls == [
        (
            ("flwrlabs/celeba", "img_align+identity+attr"),
            {"split": "train", "cache_dir": "/tmp/deckard-hf"},
        ),
    ]


def test_flexible_huggingface_dataset_requires_explicit_columns(monkeypatch):
    monkeypatch.setattr(
        decl,
        "load_dataset",
        lambda *args, **kwargs: pd.DataFrame({"text": ["one"]}),
    )

    config = decl.FlexibleHuggingFaceDataset(
        name="imdb",
        target="label",
        keep=["text"],
        dataset_split="train",
    )

    with pytest.raises(KeyError, match="missing required columns"):
        config.load_dataset()
