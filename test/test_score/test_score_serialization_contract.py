import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deckard.artifacts import ArtifactLoaderConfig, SCORE_PAYLOAD_SCHEMA


def _sample_scores() -> dict:
    return {
        "test": {
            "accuracy": 0.91,
            "confusion": [10, 2, 1, 12],
            "thresholds": np.array([0.1, 0.3, 0.6]),
        },
        "post-pipeline": {
            "k_anonymity": 5,
            "nested": {"l_diversity": 2.0},
        },
        "files": {"score_file": "scores.json"},
    }


def test_json_score_contract_persists_payload_flat_and_dotlist(tmp_path: Path):
    cfg = ArtifactLoaderConfig()
    scores = _sample_scores()
    score_file = tmp_path / "scores.json"

    cfg.save_scores(scores, str(score_file))

    raw = json.loads(score_file.read_text(encoding="utf-8"))
    assert raw["_schema"] == SCORE_PAYLOAD_SCHEMA
    assert isinstance(raw["payload"], dict)
    assert isinstance(raw["flat"], dict)
    assert isinstance(raw["dotlist"], dict)
    assert isinstance(raw["flat_by_scope"], dict)
    assert "test.accuracy" in raw["flat"]
    assert "test.thresholds" in raw["dotlist"]
    assert "test" in raw["flat_by_scope"]

    loaded = cfg.load_scores(str(score_file))
    assert "test" in loaded
    assert loaded["test"]["accuracy"] == 0.91
    assert loaded["post-pipeline"]["k_anonymity"] == 5
    assert loaded["test"]["thresholds"] == [0.1, 0.3, 0.6]


def test_yaml_score_contract_roundtrip(tmp_path: Path):
    pytest.importorskip("yaml")
    cfg = ArtifactLoaderConfig()
    scores = _sample_scores()
    score_file = tmp_path / "scores.yaml"

    cfg.save_scores(scores, str(score_file))
    loaded = cfg.load_scores(str(score_file))

    assert loaded["test"]["confusion"] == [10, 2, 1, 12]
    assert loaded["post-pipeline"]["nested"]["l_diversity"] == 2.0


def test_csv_vector_values_are_parseable(tmp_path: Path):
    cfg = ArtifactLoaderConfig()
    scores = {
        "train": {
            "vector": [1, 2, 3],
            "scalar": 4.5,
        },
    }
    score_file = tmp_path / "scores.csv"

    cfg.save_scores(scores, str(score_file))
    loaded = cfg.load_scores(str(score_file))

    assert "train.vector" in loaded
    assert loaded["train.vector"] == [1, 2, 3]
    assert loaded["train.scalar"] == 4.5


def test_json_contract_handles_pandas_payloads(tmp_path: Path):
    cfg = ArtifactLoaderConfig()
    scores = {
        "test": {
            "series": pd.Series([1.0, 2.0], index=["a", "b"]),
            "frame": pd.DataFrame({"x": [3, 4], "y": [5, 6]}),
        }
    }
    score_file = tmp_path / "pandas_scores.json"

    cfg.save_scores(scores, str(score_file))
    loaded = cfg.load_scores(str(score_file))

    assert loaded["test"]["series"] == {"a": 1.0, "b": 2.0}
    assert loaded["test"]["frame"]["columns"] == ["x", "y"]
    assert len(loaded["test"]["frame"]["records"]) == 2
