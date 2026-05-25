from __future__ import annotations

import json
from pathlib import Path

from deckard.artifacts import ArtifactLoaderConfig, ScoreDict
from deckard.experiment.canon import (
    build_experiment_stage_cache_key,
    build_experiment_stage_params_subset,
)


def _optimize_manifest(*, score_mode: str = "test", plot_backend: str = "seaborn") -> dict:
    return {
        "schema_version": "deckard.experiment.runtime.v1",
        "experiment_name": "phase6-demo",
        "library": "sklearn",
        "classifier": True,
        "evaluation_mode": "standard",
        "score_mode": score_mode,
        "random_state": 42,
        "data": {"alias": "adult"},
        "model": {"alias": "logreg"},
        "score": {"alias": "classification"},
        "plot": {"backend": plot_backend},
        "files": {
            "params_file": "outputs/params.yaml",
            "score_file": "outputs/scores.json",
        },
    }


def test_optimize_notebook_stage_cache_keys_support_selective_invalidation() -> None:
    manifest_a = _optimize_manifest(score_mode="test", plot_backend="seaborn")
    manifest_b = _optimize_manifest(score_mode="test", plot_backend="yellowbrick")
    manifest_c = _optimize_manifest(score_mode="val", plot_backend="seaborn")

    key_a = build_experiment_stage_cache_key(
        params_manifest=manifest_a,
        stage="score",
        component="score",
        identity={"run_idx": 0},
    )
    key_b = build_experiment_stage_cache_key(
        params_manifest=manifest_b,
        stage="score",
        component="score",
        identity={"run_idx": 0},
    )
    key_c = build_experiment_stage_cache_key(
        params_manifest=manifest_c,
        stage="score",
        component="score",
        identity={"run_idx": 0},
    )
    persist_key = build_experiment_stage_cache_key(
        params_manifest=manifest_a,
        stage="persist",
        component="experiment",
        identity={"run_idx": 0},
    )

    # Plot-only changes should not invalidate score-stage cache identity.
    assert key_a == key_b
    # Stage or mode changes should invalidate the cache key.
    assert key_a != key_c
    assert key_a != persist_key

    subset = build_experiment_stage_params_subset(
        params_manifest=manifest_a,
        stage="score",
        component="score",
    )
    assert subset["score_mode"] == "test"
    assert "score" in subset
    assert "plot" not in subset


def test_scoring_notebook_dotlist_projection_is_stable(tmp_path: Path) -> None:
    score_file = tmp_path / "scores.json"
    loader = ArtifactLoaderConfig(payload_kind="score")

    scores = ScoreDict.from_payload(
        {
            "accuracy": 0.91,
            "thresholds": [0.2, 0.5, 0.8],
        },
    )
    scores.update_score(
        {"precision": 0.9, "recall": 0.88},
        stage="post-pipeline",
        mode="test",
    )
    scores.update_score(
        0.12,
        key="demographic_parity_difference",
        stage="post-defense",
        mode="val",
        split="fold-0",
    )

    scores(score_file=str(score_file), artifact_loader=loader, persist=True)

    raw = json.loads(score_file.read_text(encoding="utf-8"))
    dotlist = raw["dotlist"]
    flat = raw["flat"]

    assert dotlist["accuracy"] == 0.91
    assert dotlist["post-pipeline.test.precision"] == 0.9
    assert dotlist["post-pipeline.test.recall"] == 0.88
    assert dotlist["post-defense.val.fold-0.demographic_parity_difference"] == 0.12
    assert flat["post-pipeline.test.precision"] == 0.9
    assert flat["post-defense.val.fold-0.demographic_parity_difference"] == 0.12