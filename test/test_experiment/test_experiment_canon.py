from types import SimpleNamespace

import pytest

from deckard.plugins import HookPlugin
from deckard.plugins.base import HookBundle, compose_hook_plugins

from deckard.experiment.canon import (
    CANONICAL_EXPERIMENT_COMPONENT_STAGES,
    CANONICAL_EXPERIMENT_SCORE_MODES,
    CANONICAL_EXPERIMENT_STAGES,
    CANONICAL_EXPERIMENT_TIMES,
    build_experiment_hook_bundle,
    build_experiment_hook_graph,
    build_experiment_hook_plugins,
    build_experiment_params_manifest,
    build_experiment_stage_param_key_paths,
    build_experiment_stage_params_subset,
    build_experiment_stage_cache_key,
    ensure_canonical_experiment_times,
    ensure_experiment_runtime_contract,
    normalize_experiment_score_mode,
    normalize_experiment_score_modes,
    normalize_experiment_stage,
)


def test_experiment_canon_times_contains_required_keys():
    times = ensure_canonical_experiment_times()
    for key in CANONICAL_EXPERIMENT_TIMES:
        assert key in times


def test_experiment_canon_times_preserves_extensions():
    times = ensure_canonical_experiment_times(
        {
            "experiment_total_time": 1.0,
            "custom_hook_time": 0.5,
        }
    )
    assert times["experiment_total_time"] == 1.0
    assert times["custom_hook_time"] == 0.5


@pytest.mark.parametrize(
    "value,expected",
    [
        ("train", "train"),
        ("training", "train"),
        ("test", "test"),
        ("eval", "test"),
        ("val", "val"),
        ("pre_sample", "pre-sample"),
        (None, "test"),
    ],
)
def test_experiment_canon_score_mode_normalization(value, expected):
    assert normalize_experiment_score_mode(value) == expected


def test_experiment_canon_score_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        normalize_experiment_score_mode("post-pipeline")


def test_experiment_canon_score_modes_normalize_list():
    modes = normalize_experiment_score_modes(["train", "eval", "validation"])
    assert modes == ["train", "test", "val"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("load", "load"),
        ("sampling", "sample"),
        ("training", "train"),
        ("scoring", "score"),
        (None, "all"),
    ],
)
def test_experiment_canon_stage_normalization(value, expected):
    assert normalize_experiment_stage(value) == expected


def test_experiment_canon_stage_rejects_unknown_value():
    with pytest.raises(ValueError):
        normalize_experiment_stage("post-pipeline")


def test_experiment_canon_runtime_contract_populates_missing_fields():
    runtime = ensure_experiment_runtime_contract(SimpleNamespace(score_dict=None))
    assert isinstance(runtime.score_dict, dict)
    assert isinstance(runtime.times, dict)
    assert isinstance(runtime.outputs, dict)
    assert isinstance(runtime.params, dict)


def test_experiment_canon_declares_score_modes_and_stages():
    assert set(CANONICAL_EXPERIMENT_SCORE_MODES) == {"pre-sample", "train", "test", "val"}
    assert set(CANONICAL_EXPERIMENT_STAGES) == {
        "load",
        "sample",
        "train",
        "defense",
        "attack",
        "score",
        "persist",
        "all",
    }


def test_build_experiment_params_manifest_includes_component_metadata():
    class _Component:
        alias = "component-alias"

    target = SimpleNamespace(
        experiment_name="exp",
        library="sklearn",
        classifier=True,
        evaluation_mode="standard",
        score_mode="test",
        random_state=7,
        data=_Component(),
        model=_Component(),
        defense=None,
        attack=None,
        detector=None,
        score=None,
    )

    manifest = build_experiment_params_manifest(target, runtime_kwargs={"k": 1})
    assert manifest["experiment_name"] == "exp"
    assert manifest["data"]["alias"] == "component-alias"
    assert manifest["runtime_kwargs"]["k"] == 1


def test_experiment_hook_graph_is_built_from_component_stage_contracts():
    graph = build_experiment_hook_graph()
    assert set(graph) == set(CANONICAL_EXPERIMENT_COMPONENT_STAGES)
    assert any(node["stage"] == "pre-load" for node in graph["data"])
    assert all("before" in node and "after" in node for nodes in graph.values() for node in nodes)


def test_experiment_hook_plugin_and_bundle_generation_have_expected_shape():
    plugins = build_experiment_hook_plugins()
    assert len(plugins) > 0
    assert all(hasattr(plugin, "hook_name") for plugin in plugins)
    bundle = build_experiment_hook_bundle(name="test-bundle")
    assert bundle.name == "test-bundle"
    assert len(bundle.hooks) == len(plugins)


def test_experiment_stage_cache_key_is_deterministic_and_identity_sensitive():
    params_manifest = {"experiment_name": "exp", "random_state": 42}
    key_a = build_experiment_stage_cache_key(
        params_manifest=params_manifest,
        stage="train",
        component="model",
        identity={"run_idx": 0},
    )
    key_b = build_experiment_stage_cache_key(
        params_manifest=params_manifest,
        stage="train",
        component="model",
        identity={"run_idx": 0},
    )
    key_c = build_experiment_stage_cache_key(
        params_manifest=params_manifest,
        stage="train",
        component="model",
        identity={"run_idx": 1},
    )
    assert key_a == key_b
    assert key_a != key_c


def test_experiment_stage_param_key_paths_are_stage_scoped():
    load_paths = set(build_experiment_stage_param_key_paths(stage="load", component="data"))
    model_paths = set(build_experiment_stage_param_key_paths(stage="model_score", component="model"))

    assert "data" in load_paths
    assert "model" not in load_paths
    assert "model" in model_paths


def test_experiment_stage_param_subset_filters_manifest_by_stage_component():
    manifest = {
        "schema_version": "deckard.experiment.runtime.v1",
        "experiment_name": "exp",
        "library": "sklearn",
        "classifier": True,
        "evaluation_mode": "standard",
        "score_mode": "test",
        "random_state": 42,
        "data": {"alias": "data-a"},
        "model": {"alias": "model-a"},
        "attack": {"alias": "attack-a"},
    }

    subset = build_experiment_stage_params_subset(
        params_manifest=manifest,
        stage="load",
        component="data",
    )
    assert "data" in subset
    assert "model" not in subset
    assert subset["random_state"] == 42


def test_compose_hook_plugins_preserves_order_and_dedupes_bundle_entries():
    first = HookPlugin(hook_name="before_load", method_name="_experiment_stage_hook")
    duplicate = HookPlugin(hook_name="before_load", method_name="_experiment_stage_hook")
    second = HookPlugin(hook_name="after_load", method_name="_experiment_stage_hook")

    bundle_a = HookBundle(name="bundle-a", hooks=(first,))
    bundle_b = HookBundle(name="bundle-b", hooks=(duplicate, second))

    composed = compose_hook_plugins(bundle_a, bundle_b)

    assert [(p.hook_name, p.method_name) for p in composed] == [
        ("before_load", "_experiment_stage_hook"),
        ("after_load", "_experiment_stage_hook"),
    ]
