from dataclasses import dataclass, field
from types import SimpleNamespace

from deckard.data.canon import ScoringOrchestratorMixin


@dataclass(eq=False, kw_only=True)
class _RuntimeHarness(ScoringOrchestratorMixin):
    score_split: str = "test"
    scorer: object | None = None
    plugins: list = field(default_factory=list)
    score_dict: dict = field(default_factory=dict)
    _plugin_objects: list | None = None

    def score(self, *args, mode=None, stage=None, **kwargs):
        _ = (args, kwargs)
        return {str(mode): {"stage": stage, "base": 1.0}}


def test_expand_canonical_score_stages_handles_auto_and_all():
    runtime = _RuntimeHarness()
    stages = runtime._expand_canonical_score_stages(["auto", "all"])
    assert stages == ["pre-load", "pre-sample", "post-sample", "post-pipeline"]


def test_configure_score_orchestration_plugins_maps_stage_hooks():
    runtime = _RuntimeHarness(
        scorer=SimpleNamespace(
            configured_scorers={
                "a": SimpleNamespace(stage="pre-sample"),
                "b": SimpleNamespace(stage="post-pipeline"),
            },
        ),
    )
    runtime._plugin_objects = []

    runtime._configure_score_orchestration_plugins()

    hook_names = [p.hook_name for p in runtime._plugin_objects]
    assert "before_sample" in hook_names
    assert "after_pipeline" in hook_names


def test_score_orchestration_hook_runs_stage_hooks_and_merges_scores():
    class _Plugin:
        def before_score_post_pipeline(self, runtime, **kwargs):
            runtime.score_dict["before_seen"] = kwargs["stage"]

        def after_score(self, runtime, **kwargs):
            _ = runtime
            return {"plugin_metric": 7.0}

    runtime = _RuntimeHarness(plugins=[_Plugin()])
    runtime._plugin_objects = [_Plugin()]

    result = runtime._score_orchestration_hook(stage="post-pipeline")

    assert "test" in result
    assert result["plugin_metric"] == 7.0
    assert runtime.score_dict["before_seen"] == "post-pipeline"
    assert runtime.score_dict["test"]["stage"] == "post-pipeline"
