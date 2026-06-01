import json
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deckard.artifacts import ArtifactLoaderMixin, ScoreDict
from deckard.artifacts import (
    SCORE_PAYLOAD_SCHEMA,
    _deserialize_scores_payload,
    _flat_by_scope,
    _flatten_score_payload,
    _serialize_scores_payload,
)
from deckard.utils import BaseConfig, load_data, save_data


class _ScoreLoaderStub:
    def __init__(self):
        self.saved = None

    def load_scores(self, filepath):
        _ = filepath
        return {"existing": 1}

    def save_scores(self, scores, filepath):
        self.saved = (dict(scores), filepath)


class _BaseCfg(BaseConfig):
    def __call__(self):
        return {"ok": 1}


@dataclass(eq=False, kw_only=True)
class _RequiredArtifactLoader(ArtifactLoaderMixin):
    required_value: str


def test_scoredict_from_payload_scalar_wraps_value():
    payload = ScoreDict.from_payload(3)
    assert payload == {"value": 3}


def test_artifact_loader_init_warns_and_preserves_unknown_kwargs():
    with pytest.warns(UserWarning, match="unknown init kwargs"):
        cfg = ArtifactLoaderMixin(payload_kind="score", extra_token="kept")
    assert cfg.extra_token == "kept"


def test_artifact_loader_init_rejects_too_many_positional_args():
    with pytest.raises(TypeError, match="init argument error"):
        ArtifactLoaderMixin("a", "b", "c", "d", "e")


def test_artifact_loader_required_field_fast_failure():
    with pytest.raises(TypeError, match="required_value"):
        _RequiredArtifactLoader()


def test_scoredict_update_get_and_flatten_nested_paths():
    scores = ScoreDict()
    scores.update_score(0.95, key="acc", stage="eval", mode="test", split="holdout")
    scores.update_score({"loss": 0.12}, stage="eval", mode="test")

    assert scores.get_scores(stage="eval", mode="test", split="holdout") == {
        "acc": 0.95,
    }
    assert scores.get_scores(stage="eval", mode="test")["loss"] == 0.12

    flat = scores.flatten()
    assert flat["eval.test.holdout.acc"] == 0.95
    assert flat["eval.test.loss"] == 0.12


def test_scoredict_call_merges_disk_and_persists_when_loader_present(tmp_path):
    loader = _ScoreLoaderStub()
    score_file = tmp_path / "scores.json"
    score_file.write_text("{}", encoding="utf-8")

    scores = ScoreDict({"new": 2})
    result = scores(score_file=str(score_file), artifact_loader=loader, persist=True)

    assert result == {"new": 2, "existing": 1}
    assert loader.saved is not None
    saved_scores, saved_path = loader.saved
    assert saved_scores == {"new": 2, "existing": 1}
    assert saved_path == str(score_file)


def test_save_and_load_scores_json_round_trip(tmp_path):
    loader = ArtifactLoaderMixin(payload_kind="score")
    score_path = tmp_path / "scores.json"

    payload = {
        "metrics": {"acc": np.float64(0.91)},
        "list_values": [1, 2, 3],
        "path": Path("foo/bar"),
    }
    loader.save_scores(payload, str(score_path))

    loaded = loader.load_scores(str(score_path))
    assert isinstance(loaded, ScoreDict)
    assert loaded["metrics"]["acc"] == 0.91
    assert loaded["list_values"] == [1, 2, 3]
    assert loaded["path"] == "foo/bar"


def test_artifact_loader_save_load_metadata_envelope_json(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    loader = ArtifactLoaderMixin(
        id="abc123",
        payload_kind="data",
        metadata={"owner": "deckard"},
    )

    loader.save(filepath=str(artifact_path))

    reloaded = ArtifactLoaderMixin(path=str(artifact_path))
    result = reloaded.load()

    assert result is reloaded
    assert reloaded.id == "abc123"
    assert reloaded.payload_kind == "data"
    assert reloaded.metadata == {"owner": "deckard"}


def test_save_and_load_data_csv_round_trip(tmp_path):
    loader = ArtifactLoaderMixin(payload_kind="data")
    data_path = tmp_path / "data.csv"
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    loader.save_data(data, str(data_path))
    loaded = loader.load_data(str(data_path))

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == (2, 2)
    assert loaded.to_dict(orient="list") == {"a": [1, 2], "b": [3, 4]}


def test_load_object_ignore_and_delete_corrupt_file(tmp_path):
    loader = ArtifactLoaderMixin(payload_kind="model")
    bad_path = tmp_path / "bad.pkl"
    bad_path.write_text("not-a-pickle", encoding="utf-8")

    loaded = loader.load_object(
        str(bad_path),
        ignore_corrupt=True,
        delete_corrupt=True,
    )

    assert loaded is None
    assert not bad_path.exists()


def test_save_load_generic_pickle_object_via_dispatch(tmp_path):
    payload = {"k": "v", "n": 2}
    loader = ArtifactLoaderMixin(payload_kind="data")
    path = tmp_path / "obj.pkl"

    loader.save(payload, filepath=str(path))
    loaded = loader.load(filepath=str(path))

    assert loaded == payload
    assert json.loads(json.dumps(loaded)) == payload


def test_baseconfig_read_or_initialize_scores_merges_existing_file(tmp_path):
    cfg = _BaseCfg(score_dict={"base": 1})
    score_path = tmp_path / "scores.json"
    score_path.write_text(json.dumps({"new": 2}), encoding="utf-8")

    merged = cfg.read_or_initialize_scores(str(score_path))

    assert merged["base"] == 1
    assert merged["new"] == 2


def test_baseconfig_read_or_initialize_scores_creates_parent(tmp_path):
    cfg = _BaseCfg(score_dict={"base": 1})
    score_path = tmp_path / "nested" / "scores.json"

    out = cfg.read_or_initialize_scores(str(score_path))

    assert score_path.parent.exists()
    assert out == {"base": 1}


def test_save_scores_csv_and_load_scores_csv_round_trip(tmp_path):
    cfg = _BaseCfg()
    score_path = tmp_path / "scores.csv"

    cfg.save_scores({"acc": 0.9}, score_path)
    loaded = cfg.load_scores(str(score_path))

    assert isinstance(loaded, ScoreDict)
    assert "acc" in loaded


@pytest.mark.skipif(
    importlib.util.find_spec("openpyxl") is None,
    reason="openpyxl not installed",
)
def test_save_scores_xlsx_and_load_scores_xlsx_round_trip(tmp_path):
    cfg = _BaseCfg()
    score_path = tmp_path / "scores.xlsx"

    cfg.save_scores({"acc": 0.9}, score_path)
    loaded = cfg.load_scores(str(score_path))

    assert isinstance(loaded, ScoreDict)
    assert "acc" in loaded


def test_save_scores_unsupported_extension_raises(tmp_path):
    cfg = _BaseCfg()
    with pytest.raises(ValueError):
        cfg.save_scores({"acc": 1.0}, tmp_path / "scores.txt")


def test_load_scores_unsupported_extension_raises(tmp_path):
    cfg = _BaseCfg()
    path = tmp_path / "scores.txt"
    path.write_text("nothing", encoding="utf-8")
    with pytest.raises(ValueError):
        cfg.load_scores(str(path))


def test_save_data_html(tmp_path):
    cfg = _BaseCfg()
    path = tmp_path / "data.html"
    cfg.save_data(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), path)
    assert path.exists()


@pytest.mark.skipif(
    importlib.util.find_spec("openpyxl") is None,
    reason="openpyxl not installed",
)
def test_save_data_xlsx(tmp_path):
    cfg = _BaseCfg()
    path = tmp_path / "data.xlsx"
    cfg.save_data(pd.DataFrame({"a": [1, 2]}), path)
    assert path.exists()


def test_save_data_parquet(tmp_path):
    cfg = _BaseCfg()
    path = tmp_path / "data.parquet"
    cfg.save_data(pd.DataFrame({"a": [1, 2]}), path)
    assert path.exists()


def test_save_data_unsupported_extension_raises(tmp_path):
    cfg = _BaseCfg()
    with pytest.raises(ValueError):
        cfg.save_data(pd.DataFrame({"a": [1]}), tmp_path / "data.xyz")


def test_load_data_delegates_csv(tmp_path):
    cfg = _BaseCfg()
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    out = cfg.load_data(str(path))
    assert isinstance(out, pd.DataFrame)


def test_save_object_unsupported_extension_raises(tmp_path):
    cfg = _BaseCfg()
    with pytest.raises(ValueError):
        cfg.save_object(cfg, str(tmp_path / "obj.txt"))


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".json", ".pkl"])
def test_top_level_save_load_round_trip_core_formats(tmp_path, suffix):
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / f"data{suffix}"

    save_data(df, path)
    out = load_data(str(path))

    assert isinstance(out, pd.DataFrame)


def test_top_level_save_load_html(tmp_path):
    if importlib.util.find_spec("lxml") is None:
        pytest.skip("lxml not installed")
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "data.html"
    save_data(df, path)
    out = load_data(str(path))
    assert isinstance(out, pd.DataFrame)


def test_top_level_save_load_xlsx(tmp_path):
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("openpyxl not installed")
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "data.xlsx"
    save_data(df, path)
    out = load_data(str(path))
    assert isinstance(out, pd.DataFrame)


def test_top_level_load_data_none_raises():
    with pytest.raises(FileNotFoundError):
        load_data(None)


def test_top_level_load_data_npz(tmp_path):
    path = tmp_path / "arr.npz"
    np.savez(path, data=np.array([[1, 2], [3, 4]]))
    out = load_data(str(path))
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (2, 2)


def test_top_level_save_data_unsupported_raises(tmp_path):
    with pytest.raises(ValueError):
        save_data(pd.DataFrame({"x": [1]}), tmp_path / "data.xyz")


def test_top_level_load_data_unsupported_raises(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("junk", encoding="utf-8")
    with pytest.raises(ValueError):
        load_data(str(path))


def test_top_level_save_data_converts_non_dataframe(tmp_path):
    path = tmp_path / "data.csv"
    save_data({"a": [1, 2], "b": [3, 4]}, path)
    assert path.exists()


def test_scoredict_contract_envelope_includes_flattened_views():
    scores = ScoreDict({"eval": {"test": {"acc": 0.9}}})
    envelope = scores.to_contract_envelope()

    assert envelope["_schema"] == SCORE_PAYLOAD_SCHEMA
    assert envelope["payload"]["eval"]["test"]["acc"] == 0.9
    assert envelope["flat"]["eval.test.acc"] == 0.9
    assert envelope["flat_by_scope"]["eval"]["test.acc"] == 0.9
    assert "eval.test.acc=0.9" in envelope["dotlist_items"]


def test_flatten_helper_with_prefix_and_scope_groups():
    payload = {"a": {"b": 1}, "root": 2}
    flat = _flatten_score_payload(payload, prefix="stage")

    assert flat["stage.a.b"] == 1
    assert flat["stage.root"] == 2

    grouped = _flat_by_scope(flat)
    assert grouped["stage"]["a.b"] == 1
    assert grouped["stage"]["root"] == 2


def test_serialize_and_deserialize_score_payload_round_trip():
    raw = {"eval": {"test": {"acc": 0.97}}}
    envelope = _serialize_scores_payload(raw)
    restored = _deserialize_scores_payload(dict(envelope))

    assert envelope["_schema"] == SCORE_PAYLOAD_SCHEMA
    assert restored == raw


def test_deserialize_legacy_payload_preserves_files_and_params():
    raw = {
        "accuracy": 0.9,
        "files": {"score_file": "scores.json"},
        "params": {"mode": "test"},
    }

    restored = _deserialize_scores_payload(dict(raw))

    assert restored["accuracy"] == 0.9
    assert restored["files"] == {"score_file": "scores.json"}
    assert restored["params"] == {"mode": "test"}


def test_deserialize_non_mapping_payload_wraps_in_data_key():
    assert _deserialize_scores_payload(3) == {"data": 3}
