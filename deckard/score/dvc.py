"""DVC-aware scorer helpers for experiment runtime monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ScorerConfig, ScorerDictConfig, _DataScorerMarker, safe_store

DVC_SYSTEM_SCORE_STAGES: tuple[str, ...] = (
    "data-score",
    "model-score",
    "attack-score",
    "detector-score",
)

_POWER_NAMESPACE_BY_COMPONENT: dict[str, str] = {
    "data": "data",
    "model": "model",
    "attack": "attack",
    "defense": "detector",
}


def _sanitize_metric_key(key: Any) -> str:
    token = str(key).strip().lower().replace("/", "_").replace("-", "_")
    token = "_".join(part for part in token.split("_") if part)
    if token in {"ram", "memory_used", "mem"}:
        return "memory"
    if token.endswith("_ram"):
        return token[: -len("_ram")] + "_memory"
    return token


def _collect_component_power_stats(
    experiment: Any,
    component: str,
) -> dict[str, float]:
    score_dict = getattr(experiment, "score_dict", None)
    if not isinstance(score_dict, dict):
        return {}

    namespace = _POWER_NAMESPACE_BY_COMPONENT.get(component, component)
    prefix = f"power/{namespace}/"
    collected: dict[str, float] = {}
    metric_map = {
        "cpu_watts": "cpu_power",
        "gpu_watts": "gpu_power",
        "total_watts": "power",
        "energy_wh": "energy",
    }
    for key, value in score_dict.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if not isinstance(value, (int, float)):
            continue
        raw_metric = key[len(prefix) :]
        metric = metric_map.get(raw_metric, _sanitize_metric_key(raw_metric))
        collected[metric] = float(value)
    return collected


def dvc_component_stats_score(
    y_true: Any,
    y_pred: Any,
    *,
    experiment: Any = None,
    component: str | None = None,
    stage: str | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Return concise component-scoped DVC runtime stats.

    Args:
        y_true: Required scorer dependent payload (unused).
        y_pred: Required scorer independent payload (unused).
        experiment: Active experiment runtime object.
        component: Component name prefix for emitted keys.
        stage: Runtime stage token.
        **kwargs: Additional scorer kwargs.

    Returns:
        Flat numeric score payload keyed as ``<component>_<stat>``.
    """
    _ = (y_true, y_pred)

    stage_token = str(stage or "").strip().lower().replace("_", "-")
    component_token = (
        str(
            component
            or kwargs.get("component")
            or stage_token.removesuffix("-score")
            or "score",
        )
        .strip()
        .lower()
    )
    if component_token.startswith("attack:"):
        component_token = "attack"
    if component_token == "detector":
        component_token = "defense"

    from ..experiment import dvc as dvc_module

    plugin_cfg = dvc_module.coerce_dvc_experiment_plugin(
        getattr(experiment, "dvc_plugin", None),
    )
    monitor_scores: dict[str, float] = {}
    if experiment is not None and plugin_cfg.enabled:
        try:
            monitor_scores = dvc_module._collect_system_monitor_scores(
                experiment,
                plugin_cfg,
            )
        except Exception:
            monitor_scores = {}
    power_scores = (
        _collect_component_power_stats(experiment, component_token)
        if experiment is not None
        else {}
    )

    score_dict: dict[str, float] = {}

    for key, value in monitor_scores.items():
        metric_token = str(key).strip().lower()
        if metric_token.startswith("system_monitor/"):
            metric_token = metric_token[len("system_monitor/") :]
        metric_key = _sanitize_metric_key(metric_token)
        score_dict[metric_key] = float(value)

    score_dict.update(power_scores)

    if not score_dict:
        score_dict["available"] = 0.0

    return score_dict


def dvc_system_snapshot_score(
    y_true: Any,
    y_pred: Any,
    *,
    experiment: Any = None,
    component: str | None = None,
    stage: str | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Backward-compatible alias for component stats scorer."""
    return dvc_component_stats_score(
        y_true,
        y_pred,
        experiment=experiment,
        component=component,
        stage=stage,
        **kwargs,
    )


@dataclass(eq=False, kw_only=True)
class DVCSystemScorerDictConfig(_DataScorerMarker, ScorerDictConfig):
    """Stage-scoped DVC system scorer used by experiment DVC hooks.

    By default, this scorer is configured to execute only during component score
    hook stages: data-score, model-score, attack-score, detector-score.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    scorers: dict[str, ScorerConfig] = field(
        default_factory=dict,
        metadata={
            "help": "Optional stage-scoped DVC scorer overrides keyed by component name.",
        },
    )

    def __post_init__(self) -> None:
        if not getattr(self, "scorers", None):
            self.scorers = {
                "data": ScorerConfig(
                    score_name="data",
                    score_function=dvc_component_stats_score,
                    needs_labels=False,
                    stage=["data-score"],
                    score_params={"component": "data"},
                ),
                "model": ScorerConfig(
                    score_name="model",
                    score_function=dvc_component_stats_score,
                    needs_labels=False,
                    stage=["model-score"],
                    score_params={"component": "model"},
                ),
                "attack": ScorerConfig(
                    score_name="attack",
                    score_function=dvc_component_stats_score,
                    needs_labels=False,
                    stage=["attack-score"],
                    score_params={"component": "attack"},
                ),
                "defense": ScorerConfig(
                    score_name="defense",
                    score_function=dvc_component_stats_score,
                    needs_labels=False,
                    stage=["detector-score"],
                    score_params={"component": "defense"},
                ),
            }
        super().__post_init__()


safe_store(
    group="score",
    name="dvc-system",
    node={"_target_": "deckard.score.dvc.DVCSystemScorerDictConfig"},
)


__all__ = [
    "DVC_SYSTEM_SCORE_STAGES",
    "dvc_component_stats_score",
    "dvc_system_snapshot_score",
    "DVCSystemScorerDictConfig",
]
