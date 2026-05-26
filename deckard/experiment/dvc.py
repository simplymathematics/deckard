"""DVC pipeline autogeneration helpers for experiment runtime contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Literal, cast

import yaml
from omegaconf import DictConfig, OmegaConf

from .canon import (
    CANONICAL_EXPERIMENT_PIPELINE_STAGES,
    CANONICAL_EXPERIMENT_COMPONENT_STAGES,
    CANONICAL_EXPERIMENT_RUN_MODE_ALIASES,
    CANONICAL_EXPERIMENT_STAGE_OUTPUT_KEYS,
    CANONICAL_EXPERIMENT_STAGE_COMPONENTS,
    CANONICAL_EXPERIMENT_STAGE_PRIMARY_COMPONENTS,
    build_experiment_params_manifest,
    build_experiment_stage_param_key_paths,
    normalize_experiment_stage,
)
from ..plugins import HookPlugin

if TYPE_CHECKING:
    from .base import ExperimentConfig

_DEFAULT_STAGE_ORDER: tuple[str, ...] = (*CANONICAL_EXPERIMENT_PIPELINE_STAGES,)

_DVC_STAGE_CANONICAL_MAP: dict[str, str] = {
    "generation": "attack",
    "data-score": "score",
    "model-score": "score",
    "attack-score": "score",
    "data-persist": "persist",
    "model-persist": "persist",
    "attack-persist": "persist",
    "detector-persist": "persist",
    "detector-score": "score",
    "apply-fit-defense": "defense",
    "apply-predict-defense": "defense",
    "detector-train": "defense",
    "detector-defense": "defense",
}

_DVC_STAGE_NAME_TOKEN_OVERRIDES: dict[tuple[str, str], str] = {
    ("detector", "detector-train"): "train",
    ("detector", "detector-defense"): "defense",
}

_DVC_STAGE_COMPONENT_OVERRIDES: dict[str, str] = {
    "load": "data",
    "sample": "data",
    "pipeline": "data",
    "train": "model",
    "detector-train": "detector",
    "detector-defense": "detector",
    "apply-fit-defense": "defense",
    "apply-predict-defense": "defense",
    "data-score": "data",
    "model-score": "model",
    "attack-score": "attack",
    "data-persist": "data",
    "model-persist": "model",
    "attack-persist": "attack",
    "detector-persist": "detector",
    "detector-score": "detector",
    "generation": "attack",
    "score": "experiment",
    "persist": "experiment",
}

_STAGE_OUTPUT_KEYS: dict[str, tuple[str, ...]] = CANONICAL_EXPERIMENT_STAGE_OUTPUT_KEYS

_VEGA_PLOT_FILENAMES: tuple[str, ...] = (
    "roc_auc.vl.json",
    "<attack_alias>_<attack_param>_vs_<metric>.vl.json",
    "<defense_alias>_<defense_param>_vs_<metric>.vl.json",
    "adversarial_vs_benign_<metric>.vl.json",
    "attack_vs_defense_<metric>_heatmap.vl.json",
    "epochs_vs_loss.vl.json",
    "feature_importance.vl.json",
    "covariance.vl.json",
)

DVCScalar = str | int | float | bool | None
DVCValue = DVCScalar | list["DVCValue"] | dict[str, "DVCValue"]


@dataclass(eq=False, kw_only=True)
class DVCExperimentPlugin:
    """Runtime DVC + DVCLive hook policy configuration."""

    enabled: bool = False
    dvclive_dir: str | None = None
    mode: str = "single"
    pull_dependencies: bool = True
    push_outputs: bool = True
    make_summary: bool = True
    make_report: bool = True
    make_dvcyaml: bool = False
    report_mode: str = "html"
    resume: bool = True
    save_dvc_exp: bool = False
    cache_images: bool = False
    monitor_system: bool = False
    fail_on_dvc_error: bool = False
    dvc_file: str = "dvc.yaml"
    params_file: str = "params.yaml"

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, DVCValue]:
        """Return normalized plugin settings for runtime hook composition.

        Args:
            *args: Unused positional args for plugin-call compatibility.
            **kwargs: Unused keyword args for plugin-call compatibility.

        Returns:
            Normalized DVC plugin policy mapping.
        """
        _ = args, kwargs
        return self.to_dict()

    def to_dict(self) -> dict[str, DVCValue]:
        """Serialize DVC plugin policy fields into a plain runtime dictionary.

        Returns:
            Normalized DVC plugin policy mapping.
        """
        return {
            "enabled": bool(self.enabled),
            "dvclive_dir": self.dvclive_dir,
            "mode": str(self.mode),
            "pull_dependencies": bool(self.pull_dependencies),
            "push_outputs": bool(self.push_outputs),
            "make_summary": bool(self.make_summary),
            "make_report": bool(self.make_report),
            "make_dvcyaml": bool(self.make_dvcyaml),
            "report_mode": str(self.report_mode),
            "resume": bool(self.resume),
            "save_dvc_exp": bool(self.save_dvc_exp),
            "cache_images": bool(self.cache_images),
            "monitor_system": bool(self.monitor_system),
            "fail_on_dvc_error": bool(self.fail_on_dvc_error),
            "dvc_file": str(self.dvc_file),
            "params_file": str(self.params_file),
        }


@dataclass(eq=False, kw_only=True)
class DVCExperimentConfig:
    """Lightweight wrapper around ExperimentConfig with native DVC plugin policy."""

    experiment: Any
    dvc_plugin: Any = None
    _target_: str | None = None
    _experiment_obj: Any = None

    def __post_init__(self) -> None:
        if self._target_ in [None, ""]:
            self._target_ = "deckard.experiment.dvc.DVCExperimentConfig"
        plugin_cfg = coerce_dvc_experiment_plugin(self.dvc_plugin)
        self.dvc_plugin = plugin_cfg.to_dict()
        self._experiment_obj = self._coerce_experiment(self.experiment)
        setattr(self._experiment_obj, "dvc_plugin", dict(self.dvc_plugin))
        if plugin_cfg.enabled and hasattr(
            self._experiment_obj,
            "_initialize_hook_orchestration",
        ):
            self._experiment_obj._initialize_hook_orchestration()

    def _coerce_experiment(self, value: Any):
        from hydra.utils import instantiate

        from .base import ExperimentConfig

        if isinstance(value, ExperimentConfig):
            return value
        payload = value
        if isinstance(payload, DictConfig):
            payload = OmegaConf.to_container(payload, resolve=True)
        if isinstance(payload, Mapping):
            payload = dict(payload)
            payload.setdefault("_target_", "deckard.ExperimentConfig")
            payload = instantiate(payload)
        if not isinstance(payload, ExperimentConfig):
            raise TypeError(
                "DVCExperimentConfig.experiment must resolve to ExperimentConfig, "
                f"got {type(payload)}",
            )
        return payload

    def to_experiment_config(self) -> ExperimentConfig:
        """Return the normalized wrapped ExperimentConfig runtime object.

        Returns:
            Wrapped ExperimentConfig runtime object.
        """
        return self._experiment_obj

    def to_dict(self, *, for_hash: bool = False) -> dict[str, DVCValue]:
        """Serialize DVC experiment wrapper into a plain declaration dictionary.

        Args:
            for_hash: Request stable payload suitable for hashing.

        Returns:
            Serialized DVC experiment wrapper mapping.
        """
        experiment_payload: dict[str, Any]
        if hasattr(self._experiment_obj, "to_dict") and callable(
            getattr(self._experiment_obj, "to_dict"),
        ):
            try:
                raw = self._experiment_obj.to_dict(for_hash=for_hash)
            except TypeError:
                raw = self._experiment_obj.to_dict()
            experiment_payload = dict(raw) if isinstance(raw, Mapping) else {}
        else:
            experiment_payload = {}
        return {
            "_target_": "deckard.experiment.dvc.DVCExperimentConfig",
            "experiment": experiment_payload,
            "dvc_plugin": dict(self.dvc_plugin),
        }

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._experiment_obj, name)

    def __call__(self) -> dict[str, DVCValue]:
        """Execute wrapped experiment runtime with DVC plugin policy applied.

        Returns:
            Experiment runtime output payload.
        """
        return cast(dict[str, DVCValue], self._experiment_obj())


def coerce_dvc_experiment_plugin(plugin: Any) -> DVCExperimentPlugin:
    """Normalize plugin declarations from bool/dict/object forms."""
    if isinstance(plugin, DVCExperimentPlugin):
        return plugin
    if plugin in [None, False]:
        return DVCExperimentPlugin(enabled=False)
    if plugin is True:
        return DVCExperimentPlugin(enabled=True)
    if isinstance(plugin, Mapping):
        payload = dict(plugin)
        return DVCExperimentPlugin(
            enabled=bool(payload.get("enabled", True)),
            dvclive_dir=payload.get("dvclive_dir"),
            mode=str(payload.get("mode", "single")),
            pull_dependencies=bool(payload.get("pull_dependencies", True)),
            push_outputs=bool(payload.get("push_outputs", True)),
            make_summary=bool(payload.get("make_summary", True)),
            make_report=bool(payload.get("make_report", True)),
            make_dvcyaml=bool(payload.get("make_dvcyaml", False)),
            report_mode=str(payload.get("report_mode", "html")),
            resume=bool(payload.get("resume", True)),
            save_dvc_exp=bool(payload.get("save_dvc_exp", False)),
            cache_images=bool(payload.get("cache_images", False)),
            monitor_system=bool(payload.get("monitor_system", False)),
            fail_on_dvc_error=bool(payload.get("fail_on_dvc_error", False)),
            dvc_file=str(payload.get("dvc_file", "dvc.yaml")),
            params_file=str(payload.get("params_file", "params.yaml")),
        )
    raise TypeError(
        "dvc_plugin must be a bool, mapping, DVCExperimentPlugin, or None.",
    )


def configure_dvclive_runtime(
    experiment: Any,
    *,
    enabled: bool = True,
    dir: str | None = None,
    monitor_system: bool = True,
    make_dvcyaml: bool = False,
    make_report: bool = False,
    make_summary: bool = False,
) -> dict[str, list[HookPlugin]]:
    """Attach a DVCLive runtime policy to an experiment instance."""
    plugin = DVCExperimentPlugin(
        enabled=enabled,
        dvclive_dir=dir,
        monitor_system=monitor_system,
        make_dvcyaml=make_dvcyaml,
        make_report=make_report,
        make_summary=make_summary,
    )
    setattr(experiment, "dvc_plugin", plugin.to_dict())
    first_hooks, last_hooks = build_dvc_experiment_plugin_hooks(plugin)
    payload = {"first": first_hooks, "last": last_hooks}
    setattr(experiment, "_dvc_plugin_hooks", payload)
    return payload


def _normalize_token(value: str) -> str:
    token = str(value).strip().lower().replace("_", "-")
    while "--" in token:
        token = token.replace("--", "-")
    return token.strip("-")


def _sanitize_identity(value: str | None) -> str:
    if value is None:
        raise ValueError(
            "Missing experiment identity: provide experiment.experiment_name when mode='single'.",
        )
    token = "".join(ch if ch.isalnum() else "-" for ch in str(value).strip().lower())
    while "--" in token:
        token = token.replace("--", "-")
    token = token.strip("-")
    if not token:
        raise ValueError(
            "Invalid experiment identity: experiment_name resolves to an empty token.",
        )
    return token


def _normalize_mode(mode: str) -> str:
    token = _normalize_token(mode)
    normalized = CANONICAL_EXPERIMENT_RUN_MODE_ALIASES.get(token)
    if normalized is None:
        allowed = ", ".join(
            sorted(set(CANONICAL_EXPERIMENT_RUN_MODE_ALIASES.values())),
        )
        raise ValueError(
            f"Unsupported run mode '{mode}'. Expected one of: {allowed}.",
        )
    return normalized


def _strip_hook_event_prefix(stage: str) -> str:
    token = str(stage).strip().lower()
    for prefix in ("before_", "after_", "before-", "after-"):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return token


def _resolve_stage_token(stage: Any) -> str:
    raw = str(stage).strip()
    if not raw:
        raise ValueError("Stage token is empty.")
    token = _normalize_token(_strip_hook_event_prefix(raw))
    base_token, _ = _split_stage_alias(token)
    if token in _DVC_STAGE_CANONICAL_MAP or base_token in _DVC_STAGE_CANONICAL_MAP:
        return token
    if token == "attack":
        return "generation"
    if token == "pipeline":
        return "pipeline"
    try:
        return normalize_experiment_stage(raw)
    except ValueError:
        if not token:
            raise ValueError(f"Unsupported stage token '{stage}'.") from None
        if token == "attack":
            return "generation"
        if token in _STAGE_COMPONENT_ALIASES or base_token in _STAGE_COMPONENT_ALIASES:
            return token
        raise ValueError(
            f"Unsupported stage token '{stage}'. Provide a canonical stage or known hook stage alias.",
        ) from None


def _resolve_runtime_stage(stage: str) -> str:
    token = _resolve_stage_token(stage)
    base_token, _ = _split_stage_alias(token)
    return _DVC_STAGE_CANONICAL_MAP.get(
        base_token,
        _DVC_STAGE_CANONICAL_MAP.get(token, token),
    )


def _resolve_stage_for_naming(stage: str) -> str:
    stage_token = _resolve_stage_token(stage)
    if stage_token == "all":
        raise ValueError("Stage 'all' cannot be used as a concrete DVC stage name.")
    return stage_token


def _split_stage_alias(stage: str) -> tuple[str, str | None]:
    token = _normalize_token(stage)
    if "-" not in token:
        return token, None
    for base in sorted(_DVC_STAGE_CANONICAL_MAP.keys(), key=len, reverse=True):
        if token == base:
            return base, None
        if token.startswith(f"{base}-"):
            suffix = token[len(base) + 1 :]
            return base, suffix if suffix else None
    return token, None


def _iter_component_aliases(component: Any) -> list[str]:
    aliases: list[str] = []
    if component is None:
        return aliases
    alias = getattr(component, "alias", None)
    if isinstance(alias, str) and alias.strip() != "":
        aliases.append(_normalize_token(alias))
    return aliases


def _dedupe_tokens(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        token = _normalize_token(value)
        if token == "" or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _resolve_data_scorer_aliases(experiment: Any) -> list[str]:
    data_cfg = getattr(experiment, "data", None)
    scorer_cfg = getattr(data_cfg, "scorer", None)
    return _resolve_scorer_stage_group_aliases(scorer_cfg)


def _resolve_model_scorer_aliases(experiment: Any) -> list[str]:
    model_cfg = getattr(experiment, "model", None)
    scorer_cfg = getattr(model_cfg, "scorer", None)
    return _resolve_scorer_stage_group_aliases(scorer_cfg)


def _iter_stage_group_tokens(stage_value: Any) -> list[str]:
    if stage_value in [None, ""]:
        return []
    if isinstance(stage_value, str):
        return [
            token
            for token in [_normalize_token(part) for part in stage_value.split(",")]
            if token != ""
        ]
    if isinstance(stage_value, Iterable) and not isinstance(
        stage_value,
        (str, bytes, Mapping),
    ):
        tokens: list[str] = []
        for item in stage_value:
            tokens.extend(_iter_stage_group_tokens(item))
        return tokens
    token = _normalize_token(str(stage_value))
    return [token] if token != "" else []


def _resolve_scorer_stage_group_aliases(scorer_cfg: Any) -> list[str]:
    aliases: list[str] = []
    if scorer_cfg is None:
        return aliases

    aliases.extend(_iter_stage_group_tokens(getattr(scorer_cfg, "stage", None)))

    scorers = getattr(scorer_cfg, "scorers", None)
    if isinstance(scorers, Mapping):
        for scorer in scorers.values():
            scorer_stage = getattr(scorer, "stage", None)
            if scorer_stage is None and isinstance(scorer, Mapping):
                scorer_stage = scorer.get("stage")
            aliases.extend(_iter_stage_group_tokens(scorer_stage))

    aliases = _dedupe_tokens(aliases)
    if aliases:
        return aliases

    component_aliases = _iter_component_aliases(scorer_cfg)
    if component_aliases:
        return _dedupe_tokens(component_aliases)

    profile_attr = getattr(scorer_cfg, "_profile_attr", None)
    if isinstance(profile_attr, str) and profile_attr.strip() != "":
        return [_normalize_token(profile_attr)]

    return []


def _resolve_attack_aliases(experiment: Any) -> list[str]:
    aliases: list[str] = []
    chain = getattr(experiment, "_attack_chain", None)
    if isinstance(chain, (list, tuple)) and len(chain) > 0:
        for index, attack_cfg in enumerate(chain):
            attack_aliases = _iter_component_aliases(attack_cfg)
            if attack_aliases:
                aliases.extend(attack_aliases)
            else:
                aliases.append(f"attack{index + 1}")
    else:
        aliases.extend(_iter_component_aliases(getattr(experiment, "attack", None)))
    return _dedupe_tokens(aliases)


def _resolve_attack_score_aliases(experiment: Any) -> list[str]:
    aliases: list[str] = []
    chain = getattr(experiment, "_attack_chain", None)
    if isinstance(chain, (list, tuple)) and len(chain) > 0:
        for attack_cfg in chain:
            aliases.extend(
                _resolve_scorer_stage_group_aliases(
                    getattr(attack_cfg, "scorer", None),
                ),
            )
    else:
        attack_cfg = getattr(experiment, "attack", None)
        aliases.extend(
            _resolve_scorer_stage_group_aliases(getattr(attack_cfg, "scorer", None)),
        )
    return _dedupe_tokens(aliases)


def _resolve_defense_aliases(
    experiment: Any,
    *,
    require_attr: str | None = None,
) -> list[str]:
    aliases: list[str] = []
    defense_cfg = getattr(experiment, "defense", None)
    defenses = getattr(defense_cfg, "defenses", None)
    if isinstance(defenses, (list, tuple)) and len(defenses) > 0:
        for index, defense_step in enumerate(defenses):
            if require_attr is not None and not bool(
                getattr(defense_step, require_attr, False),
            ):
                continue
            step_aliases = _iter_component_aliases(
                getattr(defense_step, "defense", defense_step),
            )
            if step_aliases:
                aliases.extend(step_aliases)
            else:
                aliases.append(f"defense{index + 1}")
    else:
        aliases.extend(_iter_component_aliases(defense_cfg))
    return _dedupe_tokens(aliases)


def _resolve_detector_aliases(experiment: Any) -> list[str]:
    return _dedupe_tokens(
        _iter_component_aliases(getattr(experiment, "detector", None)),
    )


def _resolve_stage_aliases(experiment: Any, stage: str) -> list[str]:
    stage_token = _resolve_stage_token(stage)
    base_stage, _ = _split_stage_alias(stage_token)
    alias_map: dict[str, list[str]] = {
        "data-score": _resolve_data_scorer_aliases(experiment),
        "model-score": _resolve_model_scorer_aliases(experiment),
        "attack-score": _resolve_attack_score_aliases(experiment),
        "generation": _resolve_attack_aliases(experiment),
        "apply-fit-defense": _resolve_defense_aliases(
            experiment,
            require_attr="apply_fit",
        ),
        "apply-predict-defense": _resolve_defense_aliases(
            experiment,
            require_attr="apply_predict",
        ),
        "detector-train": _resolve_detector_aliases(experiment),
        "detector-defense": _resolve_detector_aliases(experiment),
        "detector-score": _resolve_detector_aliases(experiment),
    }
    return alias_map.get(base_stage, [])


def _expand_stage_with_aliases(experiment: Any, stage: str) -> list[str]:
    stage_token = _resolve_stage_token(stage)
    base_stage, alias = _split_stage_alias(stage_token)
    if alias is not None:
        return [stage_token]
    aliases = _resolve_stage_aliases(experiment, stage_token)
    if not aliases:
        return [stage_token]
    return [f"{base_stage}-{token}" for token in aliases]


def _resolve_stage_sequence(experiment: Any, stage_selection: Any) -> list[str]:
    selected = _resolve_stage_selection(stage_selection)
    expanded: list[str] = []
    for stage in selected:
        expanded.extend(_expand_stage_with_aliases(experiment, stage))
    return _dedupe_tokens(expanded)


def _resolve_params_manifest(experiment: Any) -> Mapping[str, Any]:
    params = getattr(experiment, "params", None)
    if isinstance(params, Mapping):
        return params
    return build_experiment_params_manifest(experiment)


def _resolve_run_identity(
    experiment: Any,
    *,
    mode: str,
    stage: str,
    params_manifest: Mapping[str, Any],
) -> str:
    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "single":
        experiment_name = getattr(experiment, "experiment_name", None)
        if not experiment_name and isinstance(params_manifest, Mapping):
            experiment_name = params_manifest.get("experiment_name")
        return _sanitize_identity(experiment_name)

    payload = {
        "stage": stage,
        "params": dict(params_manifest),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _resolve_dvclive_dir(experiment: Any, plugin: DVCExperimentPlugin) -> Path:
    if plugin.dvclive_dir not in [None, ""]:
        return Path(str(plugin.dvclive_dir)).resolve()
    identity = _resolve_run_identity(
        experiment,
        mode=plugin.mode,
        stage="persist",
        params_manifest=_resolve_params_manifest(experiment),
    )
    return (Path("outputs") / "logs" / identity).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_repo_relative_path(path_value: str | Path) -> str:
    path = Path(str(path_value))
    if not path.is_absolute():
        return path.as_posix()

    resolved = path.resolve()
    root = _repo_root().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        # Avoid persisting host-specific absolute paths in serialized outputs.
        return Path(os.path.relpath(resolved.as_posix(), root.as_posix())).as_posix()


def _sanitize_plugin_path_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    for key in ("dvclive_dir", "dvc_file", "params_file"):
        value = sanitized.get(key)
        if isinstance(value, str) and value.strip() != "":
            sanitized[key] = _to_repo_relative_path(value)
    return sanitized


def _normalize_report_mode(value: str | None) -> str | None:
    if value in [None, "none", ""]:
        return None
    token = str(value).strip().lower()
    if token in {"md", "notebook", "html"}:
        return token
    raise ValueError(
        "Unsupported report_mode for DVCExperimentPlugin. "
        "Expected one of: md, notebook, html, None.",
    )


def _load_dvclive_live_class():
    try:
        module = importlib.import_module("dvclive")
        Live = getattr(module, "Live", None)
        if Live is None:
            live_module = importlib.import_module("dvclive.live")
            Live = getattr(live_module, "Live", None)
        if Live is None:
            raise ImportError("Could not resolve dvclive.Live")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DVCLive is required when dvc_plugin.enabled=True.",
        ) from exc
    return Live


def _ensure_output_buckets(experiment: Any) -> dict[str, Any]:
    outputs = getattr(experiment, "outputs", None)
    if not isinstance(outputs, dict):
        outputs = {}
        setattr(experiment, "outputs", outputs)
    return outputs.setdefault("dvclive", {})


def _get_runtime_state(experiment: Any) -> dict[str, Any]:
    runtime = getattr(experiment, "_dvc_plugin_runtime", None)
    if isinstance(runtime, dict):
        return runtime
    runtime = {
        "live": None,
        "dir": None,
        "params_logged": False,
        "scores_logged": False,
        "deps_pulled": False,
        "outputs_pushed": False,
    }
    setattr(experiment, "_dvc_plugin_runtime", runtime)
    return runtime


def _ensure_live_instance(experiment: Any, plugin: DVCExperimentPlugin):
    runtime = _get_runtime_state(experiment)
    if runtime.get("live") is not None:
        return runtime["live"]

    Live = _load_dvclive_live_class()
    dvclive_dir = _resolve_dvclive_dir(experiment, plugin)
    dvclive_dir.mkdir(parents=True, exist_ok=True)
    report_mode = cast(
        Literal["md", "notebook", "html"] | None,
        _normalize_report_mode(plugin.report_mode if plugin.make_report else None),
    )

    live = Live(
        dir=dvclive_dir.as_posix(),
        dvcyaml=plugin.make_dvcyaml,
        resume=plugin.resume,
        save_dvc_exp=plugin.save_dvc_exp,
        report=report_mode,
        cache_images=plugin.cache_images,
        monitor_system=plugin.monitor_system,
    )
    runtime["live"] = live
    runtime["dir"] = dvclive_dir
    return live


def _safe_run_dvc_cmd(
    plugin: DVCExperimentPlugin,
    command: list[str],
    *,
    cwd: Path | None = None,
) -> dict[str, Any]:
    if not command:
        return {"ok": False, "command": []}
    try:
        completed = subprocess.run(
            command,
            cwd=(cwd or _repo_root()).as_posix(),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        if plugin.fail_on_dvc_error:
            raise RuntimeError(
                f"Failed to execute {' '.join(command)}: {exc}",
            ) from exc
        return {
            "ok": False,
            "command": command,
            "error": str(exc),
        }

    payload = {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0 and plugin.fail_on_dvc_error:
        raise RuntimeError(
            f"DVC command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}",
        )
    return payload


def _coerce_file_mapping(files: Any) -> dict[str, str]:
    if files is None:
        return {}
    if isinstance(files, DictConfig):
        files = OmegaConf.to_container(files, resolve=True)
    if hasattr(files, "as_dict") and callable(getattr(files, "as_dict")):
        files = files.as_dict()
    if not isinstance(files, Mapping):
        raise TypeError(
            f"files must be a mapping-like object, got {type(files).__name__}.",
        )
    coerced: dict[str, str] = {}
    for key, value in files.items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip() != "":
            coerced[str(key)] = value
            continue
        raise TypeError(
            f"File alias '{key}' must be a non-empty string or None, got {type(value).__name__}.",
        )
    return coerced


def _resolve_params_file_path(
    experiment: Any,
    plugin: DVCExperimentPlugin,
) -> Path:
    file_aliases = _coerce_file_mapping(getattr(experiment, "files", None))
    params_file = (
        file_aliases.get("params_file") or plugin.params_file or "params.yaml"
    )
    return Path(str(params_file))


def _mapping_get_path(payload: Mapping[str, Any] | None, *path: str) -> Any:
    current: Any = payload
    for key in path:
        if isinstance(current, DictConfig):
            current = OmegaConf.to_container(current, resolve=True)
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _has_configured_optuna_pruner(experiment_payload: Mapping[str, Any]) -> bool:
    pruner_value = _mapping_get_path(experiment_payload, "hydra", "sweeper", "pruner")
    if pruner_value in [None, False, "", {}]:
        return False
    if isinstance(pruner_value, Mapping):
        target = pruner_value.get("_target_")
        if isinstance(target, str) and target.strip() == "":
            return False
    return True


def _normalize_dvclive_enabled(
    experiment_payload: dict[str, Any],
    *,
    plugin: DVCExperimentPlugin,
) -> bool:
    current = experiment_payload.get("dvclive_enabled")
    if current is None:
        enabled = bool(plugin.enabled)
        experiment_payload["dvclive_enabled"] = enabled
        return enabled
    enabled = bool(current)
    experiment_payload["dvclive_enabled"] = enabled
    return enabled


def _serialize_stage_selection(stage: Any) -> str | list[str]:
    if isinstance(stage, str):
        token = stage.strip()
        return token or "all"
    if isinstance(stage, Iterable):
        values = [str(item).strip() for item in stage if str(item).strip()]
        if len(values) == 0:
            return "all"
        return values[0] if len(values) == 1 else values
    if stage in [None, ""]:
        return "all"
    return str(stage)


def _build_dvc_params_payload(
    experiment: Any,
    *,
    plugin: DVCExperimentPlugin,
    stage: str,
) -> dict[str, Any]:
    manifest = dict(_resolve_params_manifest(experiment))
    experiment_payload: dict[str, Any] = {}

    if hasattr(experiment, "to_dict") and callable(getattr(experiment, "to_dict")):
        try:
            candidate = experiment.to_dict(for_hash=True)
        except TypeError:
            candidate = experiment.to_dict()
        if isinstance(candidate, Mapping):
            experiment_payload = dict(candidate)

    if (
        experiment_payload
        and hasattr(experiment, "_sanitize_runtime_instantiation_payload")
        and callable(getattr(experiment, "_sanitize_runtime_instantiation_payload"))
    ):
        sanitized = experiment._sanitize_runtime_instantiation_payload(
            experiment_payload,
        )
        if isinstance(sanitized, Mapping):
            experiment_payload = dict(sanitized)

    if not experiment_payload:
        experiment_payload = {
            "experiment_name": getattr(experiment, "experiment_name", None),
            "library": getattr(experiment, "library", None),
            "classifier": getattr(experiment, "classifier", None),
            "evaluation_mode": getattr(experiment, "evaluation_mode", None),
            "score_mode": getattr(experiment, "score_mode", None),
            "random_state": getattr(experiment, "random_state", None),
        }

    dvclive_enabled = _normalize_dvclive_enabled(
        experiment_payload,
        plugin=plugin,
    )
    pruning_enabled = bool(experiment_payload.get("pruning_enabled", False))
    pruner_configured = _has_configured_optuna_pruner(experiment_payload)

    base_target = experiment_payload.get("_target_")
    if not isinstance(base_target, str) or base_target.strip() == "":
        fallback_target = getattr(experiment, "_target_", None)
        if not isinstance(fallback_target, str) or fallback_target.strip() == "":
            fallback_target = "deckard.experiment.ExperimentConfig"
        experiment_payload["_target_"] = fallback_target

    if isinstance(experiment_payload.get("dvc_plugin"), Mapping):
        experiment_payload["dvc_plugin"] = _sanitize_plugin_path_fields(
            cast(Mapping[str, Any], experiment_payload["dvc_plugin"]),
        )
    experiment_payload.pop("dvc_plugin", None)

    plugin_payload = _sanitize_plugin_path_fields(plugin.to_dict())
    wrapper_target = "deckard.experiment.dvc.DVCExperimentConfig"
    return {
        "_target_": wrapper_target,
        "experiment": experiment_payload,
        "dvc_plugin": plugin_payload,
        "_dvc": {
            "stage_selection": _serialize_stage_selection(stage),
            "run_mode": _normalize_mode(plugin.mode),
            "params_manifest": manifest,
            "dvclive": {
                "enabled": dvclive_enabled,
            },
            "pruning": {
                "enabled": pruning_enabled,
                "pruner_configured": pruner_configured,
                "active": bool(pruning_enabled and pruner_configured),
            },
        },
    }


def _write_dvc_params_file(
    experiment: Any,
    *,
    plugin: DVCExperimentPlugin,
    stage: str,
) -> str:
    params_path = _resolve_params_file_path(experiment, plugin)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_dvc_params_payload(
        experiment,
        plugin=plugin,
        stage=stage,
    )
    params_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return params_path.as_posix()


def _write_generated_pipeline_params_file(
    experiment: Any,
    *,
    plugin: DVCExperimentPlugin,
    params_file: str,
    stage_selection: Any,
) -> dict[str, Any]:
    params_path = Path(str(params_file))
    params_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_dvc_params_payload(
        experiment,
        plugin=plugin,
        stage=stage_selection,
    )
    params_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return payload


def _resolve_stage_outputs(
    experiment: Any,
    *,
    stage: str,
    include_identity_dir: bool = False,
    plugin: DVCExperimentPlugin | None = None,
) -> list[str]:
    aliases = _coerce_file_mapping(getattr(experiment, "files", None))
    stage_token = _resolve_stage_token(stage)
    base_stage, _ = _split_stage_alias(stage_token)
    keys = _STAGE_OUTPUT_KEYS.get(stage_token, _STAGE_OUTPUT_KEYS.get(base_stage, ()))
    outputs = [aliases[key] for key in keys if key in aliases]
    if include_identity_dir:
        cfg = plugin or DVCExperimentPlugin(enabled=False)
        identity = _resolve_run_identity(
            experiment,
            mode=cfg.mode,
            stage="persist",
            params_manifest=_resolve_params_manifest(experiment),
        )
        outputs.append((Path("outputs") / "logs" / identity).as_posix())
    unique: list[str] = []
    seen: set[str] = set()
    for path in outputs:
        value = str(path)
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _log_dvclive_params(experiment: Any, plugin: DVCExperimentPlugin) -> None:
    runtime = _get_runtime_state(experiment)
    if runtime.get("params_logged"):
        return
    live = _ensure_live_instance(experiment, plugin)
    if not hasattr(live, "log_params"):
        return
    live.log_params(dict(_resolve_params_manifest(experiment)))
    runtime["params_logged"] = True


def _log_dvclive_scores(experiment: Any, plugin: DVCExperimentPlugin) -> None:
    runtime = _get_runtime_state(experiment)
    live = _ensure_live_instance(experiment, plugin)
    score_dict = getattr(experiment, "score_dict", None)
    if not isinstance(score_dict, Mapping):
        return
    for key, value in score_dict.items():
        if isinstance(value, (int, float)) and hasattr(live, "log_metric"):
            live.log_metric(str(key), float(value))
    runtime["scores_logged"] = True


def _log_dvclive_artifacts_and_plots(
    experiment: Any,
    plugin: DVCExperimentPlugin,
) -> None:
    live = _ensure_live_instance(experiment, plugin)
    log_artifact = getattr(live, "log_artifact", None)
    log_image = getattr(live, "log_image", None)
    if not callable(log_artifact) and not callable(log_image):
        return

    image_suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    for output in _resolve_stage_outputs(
        experiment,
        stage="persist",
        include_identity_dir=True,
        plugin=plugin,
    ):
        path = Path(output)
        if path.exists():
            if path.suffix.lower() in image_suffixes and callable(log_image):
                log_image(path.name, path.as_posix())
            elif callable(log_artifact):
                log_artifact(path.as_posix())


def _run_dvc_pull(experiment: Any, plugin: DVCExperimentPlugin) -> dict[str, Any]:
    runtime = _get_runtime_state(experiment)
    if runtime.get("deps_pulled"):
        return {"executed": False}
    if not plugin.pull_dependencies:
        return {"executed": False, "reason": "pull_dependencies_disabled"}
    payload = _safe_run_dvc_cmd(plugin, ["dvc", "pull"], cwd=_repo_root())
    runtime["deps_pulled"] = bool(payload.get("ok"))
    return payload


def _run_dvc_push(experiment: Any, plugin: DVCExperimentPlugin) -> dict[str, Any]:
    runtime = _get_runtime_state(experiment)
    if runtime.get("outputs_pushed"):
        return {"executed": False}
    if not plugin.push_outputs:
        return {"executed": False, "reason": "push_outputs_disabled"}

    tracked_paths = [
        path
        for path in _resolve_stage_outputs(
            experiment,
            stage="persist",
            include_identity_dir=True,
            plugin=plugin,
        )
        if Path(path).exists()
    ]
    add_payload = None
    if tracked_paths:
        add_payload = _safe_run_dvc_cmd(
            plugin,
            ["dvc", "add", *tracked_paths],
            cwd=_repo_root(),
        )
    push_payload = _safe_run_dvc_cmd(plugin, ["dvc", "push"], cwd=_repo_root())
    runtime["outputs_pushed"] = bool(push_payload.get("ok"))
    return {
        "add": add_payload,
        "push": push_payload,
    }


def render_dvclive_report(
    experiment: Any,
    *,
    plugin: Any = None,
) -> dict[str, Any]:
    """Render DVCLive summary/report artifacts for an experiment runtime."""
    plugin_cfg = coerce_dvc_experiment_plugin(plugin)
    if not plugin_cfg.enabled:
        return {"enabled": False}

    live = _ensure_live_instance(experiment, plugin_cfg)
    runtime = _get_runtime_state(experiment)
    dvclive_dir = Path(
        str(runtime.get("dir") or _resolve_dvclive_dir(experiment, plugin_cfg)),
    )
    mode = _normalize_report_mode(
        plugin_cfg.report_mode if plugin_cfg.make_report else None,
    )

    if plugin_cfg.make_summary and hasattr(live, "make_summary"):
        live.make_summary()
    if plugin_cfg.make_report and hasattr(live, "make_report"):
        live.make_report()

    summary_json = dvclive_dir / "summary.json"
    mode_to_name = {
        "md": "report.md",
        "notebook": "report.ipynb",
        "html": "report.html",
        None: None,
    }
    report_name = mode_to_name.get(mode)
    report_file = (dvclive_dir / report_name) if report_name else None

    summary_path = summary_json.as_posix() if summary_json.exists() else None
    report_path = (
        report_file.as_posix()
        if report_file is not None and report_file.exists()
        else None
    )

    return {
        "enabled": True,
        "dvclive_dir": _to_repo_relative_path(dvclive_dir),
        "report_mode": mode,
        "summary_json": _to_repo_relative_path(summary_path) if summary_path else None,
        "report_file": _to_repo_relative_path(report_path) if report_path else None,
        "report_html": (
            _to_repo_relative_path(report_path)
            if mode == "html" and report_path
            else None
        ),
    }


def _build_dvc_plugin_hook_wrappers(
    plugin_cfg: DVCExperimentPlugin,
    *,
    method_name: str,
) -> tuple[list[HookPlugin], list[HookPlugin]]:
    first_hooks: list[HookPlugin] = []
    last_hooks: list[HookPlugin] = []
    plugin_payload = plugin_cfg.to_dict()

    for stage in CANONICAL_EXPERIMENT_PIPELINE_STAGES:
        stage_token = _normalize_token(stage).replace("-", "_")
        for event in ("before", "after"):
            hook_name = f"{event}_{stage_token}"
            first_hooks.append(
                HookPlugin(
                    hook_name=hook_name,
                    method_name=method_name,
                    method_kwargs={
                        "dvc_plugin": plugin_payload,
                        "plugin_position": "first",
                    },
                ),
            )
            last_hooks.append(
                HookPlugin(
                    hook_name=hook_name,
                    method_name=method_name,
                    method_kwargs={
                        "dvc_plugin": plugin_payload,
                        "plugin_position": "last",
                    },
                ),
            )

    return first_hooks, last_hooks


def build_dvc_experiment_plugin_hooks(
    plugin: Any,
    *,
    method_name: str = "_dvc_experiment_plugin_hook",
) -> tuple[list[HookPlugin], list[HookPlugin]]:
    """Construct first/last DVC hook wrappers for experiment orchestration."""
    plugin_cfg = coerce_dvc_experiment_plugin(plugin)
    if not plugin_cfg.enabled:
        return [], []
    first_hooks, last_hooks = _build_dvc_plugin_hook_wrappers(
        plugin_cfg,
        method_name=method_name,
    )
    return first_hooks, last_hooks


def run_dvc_experiment_plugin_hook(
    experiment: Any,
    *,
    dvc_plugin: Any,
    plugin_position: str,
    component: str,
    stage: str,
    event: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one DVCExperimentPlugin hook callback."""
    _ = kwargs
    plugin_cfg = coerce_dvc_experiment_plugin(dvc_plugin)
    stage_token = _resolve_stage_token(stage)
    event_token = str(event).strip().lower()
    position_token = str(plugin_position).strip().lower()

    result: dict[str, Any] = {
        "enabled": bool(plugin_cfg.enabled),
        "position": position_token,
        "component": str(component),
        "stage": stage_token,
        "event": event_token,
        "executed": False,
    }

    dvclive_bucket = _ensure_output_buckets(experiment)
    hook_trace = dvclive_bucket.setdefault("hook_trace", [])
    hook_trace.append(dict(result))

    if not plugin_cfg.enabled:
        return result

    if position_token == "first" and event_token == "before" and stage_token == "load":
        result["params_file"] = _write_dvc_params_file(
            experiment,
            plugin=plugin_cfg,
            stage=stage_token,
        )
        _ensure_live_instance(experiment, plugin_cfg)
        _log_dvclive_params(experiment, plugin_cfg)
        result["pull"] = _run_dvc_pull(experiment, plugin_cfg)
        result["executed"] = True

    if position_token == "last" and event_token == "after" and stage_token == "score":
        _log_dvclive_scores(experiment, plugin_cfg)
        live = _ensure_live_instance(experiment, plugin_cfg)
        if hasattr(live, "next_step"):
            live.next_step()
        result["executed"] = True

    if (
        position_token == "last"
        and event_token == "after"
        and stage_token == "persist"
    ):
        result["params_file"] = _write_dvc_params_file(
            experiment,
            plugin=plugin_cfg,
            stage=stage_token,
        )
        _log_dvclive_params(experiment, plugin_cfg)
        _log_dvclive_scores(experiment, plugin_cfg)
        _log_dvclive_artifacts_and_plots(experiment, plugin_cfg)
        result.update(render_dvclive_report(experiment, plugin=plugin_cfg))
        result["push"] = _run_dvc_push(experiment, plugin_cfg)
        live = _ensure_live_instance(experiment, plugin_cfg)
        if hasattr(live, "end"):
            live.end()
        result["executed"] = True

    dvclive_bucket.update(
        {k: v for k, v in result.items() if k not in {"component", "stage", "event"}},
    )
    return result


def _build_stage_component_aliases() -> dict[str, str]:
    def _token(value: str) -> str:
        token = str(value).strip().lower().replace("_", "-")
        while "--" in token:
            token = token.replace("--", "-")
        return token.strip("-")

    aliases: dict[str, str] = {}
    for component, stages in CANONICAL_EXPERIMENT_COMPONENT_STAGES.items():
        for stage in stages:
            aliases[_token(stage)] = component
    for stage, component in CANONICAL_EXPERIMENT_STAGE_PRIMARY_COMPONENTS.items():
        aliases[_token(stage)] = component
    for stage, component in _DVC_STAGE_COMPONENT_OVERRIDES.items():
        aliases[_token(stage)] = component
    for stage, components in CANONICAL_EXPERIMENT_STAGE_COMPONENTS.items():
        stage_token = _token(stage)
        for component in components:
            aliases.setdefault(stage_token, component)
    return aliases


_STAGE_COMPONENT_ALIASES: dict[str, str] = _build_stage_component_aliases()


def _resolve_stage_component(stage: str) -> str:
    stage_token = _resolve_stage_token(stage)
    base_stage, _ = _split_stage_alias(stage_token)
    if stage_token == "all":
        raise ValueError("Stage 'all' cannot be resolved to a concrete DVC component.")
    if stage_token in _DVC_STAGE_COMPONENT_OVERRIDES:
        return _DVC_STAGE_COMPONENT_OVERRIDES[stage_token]
    if base_stage in _DVC_STAGE_COMPONENT_OVERRIDES:
        return _DVC_STAGE_COMPONENT_OVERRIDES[base_stage]
    component = _STAGE_COMPONENT_ALIASES.get(stage_token)
    if component:
        return component
    component = _STAGE_COMPONENT_ALIASES.get(base_stage)
    if component:
        return component
    raise ValueError(
        f"No component mapping found for stage '{stage}' (normalized='{stage_token}').",
    )


def _sanitize_filename_token(value: Any, *, default: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    token = "".join(ch if ch.isalnum() else "_" for ch in raw)
    while "__" in token:
        token = token.replace("__", "_")
    token = token.strip("_")
    return token or default


def _first_mapping_key(mapping: Any) -> str | None:
    if isinstance(mapping, Mapping):
        for key in mapping.keys():
            token = str(key).strip()
            if token:
                return token
    return None


def _resolve_plot_filename_tokens(
    experiment: Any,
    params_manifest: Mapping[str, Any],
) -> dict[str, str]:
    attack_cfg = getattr(experiment, "attack", None)
    defense_cfg = getattr(experiment, "defense", None)

    attack_alias = _sanitize_filename_token(
        getattr(attack_cfg, "alias", None),
        default="attack",
    )
    defense_alias = _sanitize_filename_token(
        getattr(defense_cfg, "alias", None),
        default="defense",
    )

    attack_params = getattr(attack_cfg, "attack_params", None)
    defense_params = getattr(defense_cfg, "defense_params", None)
    attack_param = _sanitize_filename_token(
        _first_mapping_key(attack_params),
        default="strength",
    )
    defense_param = _sanitize_filename_token(
        _first_mapping_key(defense_params),
        default="setting",
    )

    metric = "accuracy"
    configured_optimizers = getattr(experiment, "optimizers", None)
    if (
        isinstance(configured_optimizers, (list, tuple))
        and len(configured_optimizers) > 0
    ):
        metric = _sanitize_filename_token(configured_optimizers[0], default="accuracy")
    if isinstance(params_manifest, Mapping):
        runtime = params_manifest.get("runtime_kwargs")
        if isinstance(runtime, Mapping):
            optimizers = runtime.get("optimizers")
            if isinstance(optimizers, (list, tuple)) and len(optimizers) > 0:
                metric = _sanitize_filename_token(optimizers[0], default="accuracy")

    return {
        "attack_alias": attack_alias,
        "attack_param": attack_param,
        "defense_alias": defense_alias,
        "defense_param": defense_param,
        "metric": metric,
    }


def _resolve_vega_plot_filenames(
    experiment: Any,
    params_manifest: Mapping[str, Any],
) -> list[str]:
    tokens = _resolve_plot_filename_tokens(experiment, params_manifest)
    rendered: list[str] = []
    for template in _VEGA_PLOT_FILENAMES:
        name = template
        for key, value in tokens.items():
            name = name.replace(f"<{key}>", value)
        rendered.append(name)
    seen: set[str] = set()
    unique: list[str] = []
    for name in rendered:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _default_runtime_cache_path(file_aliases: Mapping[str, str]) -> str | None:
    params_file = file_aliases.get("params_file")
    if not params_file:
        return None
    params_path = Path(params_file)
    if params_path.suffix:
        return params_path.with_name(
            f"{params_path.stem}.runtime_cache.pkl",
        ).as_posix()
    return params_path.with_suffix(".runtime_cache.pkl").as_posix()


def _resolve_stage_selection(stage_selection: Any) -> list[str]:
    if stage_selection is None:
        return list(_DEFAULT_STAGE_ORDER)

    raw: list[Any]
    if isinstance(stage_selection, str):
        raw = [token.strip() for token in stage_selection.split(",") if token.strip()]
    elif isinstance(stage_selection, Iterable):
        raw = list(stage_selection)
    else:
        raw = [stage_selection]

    selected: list[str] = []
    for stage in raw:
        canonical = _resolve_stage_token(stage)
        if canonical == "all":
            return list(_DEFAULT_STAGE_ORDER)
        if canonical not in selected:
            selected.append(canonical)
    if not selected:
        raise ValueError("stage_selection resolved to an empty stage set.")
    return selected


def _stage_enabled(experiment: Any, stage: str) -> bool:
    stage_token = _resolve_stage_token(stage)
    base_stage, _ = _split_stage_alias(stage_token)
    runtime_stage = _resolve_runtime_stage(stage_token)
    if base_stage in {"load", "sample", "score", "persist"}:
        return True
    if base_stage == "data-persist":
        return getattr(experiment, "data", None) is not None
    if base_stage == "model-persist":
        return getattr(experiment, "model", None) is not None
    if base_stage == "attack-persist":
        return (
            getattr(experiment, "attack", None) is not None
            or len(getattr(experiment, "_attack_chain", []) or []) > 0
        )
    if base_stage == "attack-score":
        return (
            getattr(experiment, "attack", None) is not None
            or len(getattr(experiment, "_attack_chain", []) or []) > 0
        )
    if base_stage == "detector-persist":
        return getattr(experiment, "detector", None) is not None
    if base_stage == "pipeline":
        return True
    if base_stage in {"detector-train", "detector-defense", "detector-score"}:
        return getattr(experiment, "detector", None) is not None
    if base_stage in {"data-score", "model-score"}:
        component_name = "data" if base_stage == "data-score" else "model"
        return getattr(experiment, component_name, None) is not None
    if base_stage in {"apply-fit-defense", "apply-predict-defense"}:
        return getattr(experiment, "defense", None) is not None
    if runtime_stage == "train":
        return getattr(experiment, "model", None) is not None
    if runtime_stage == "attack":
        chain = getattr(experiment, "_attack_chain", None)
        if isinstance(chain, (list, tuple)) and len(chain) > 0:
            return True
        return getattr(experiment, "attack", None) is not None
    if runtime_stage == "defense":
        return (
            getattr(experiment, "detector", None) is not None
            or getattr(experiment, "defense", None) is not None
        )
    return True


def build_dvc_stage_name(component: str, stage: str) -> str:
    """Build canonical DVC stage name as <component>__<stage>."""
    component_token = _normalize_token(component)
    raw_stage_token = _resolve_stage_for_naming(stage)
    base_stage, alias = _split_stage_alias(raw_stage_token)
    base_name = _DVC_STAGE_NAME_TOKEN_OVERRIDES.get(
        (component_token, base_stage),
        base_stage,
    )
    stage_token = f"{base_name}-{alias}" if alias else base_name
    return f"{component_token}__{stage_token}"


def extract_dvc_file_aliases(
    file_dict: Mapping[str, Any] | DictConfig | None,
    cache_path: str | None = None,
) -> dict[str, str]:
    """Normalize file aliases used by DVC deps/outs wiring."""
    aliases = _coerce_file_mapping(file_dict)
    if cache_path:
        aliases["runtime_cache_file"] = str(cache_path)
    return aliases


def _stage_cmd(
    *,
    stage: str,
    mode: str,
    multirun_count: int | None = None,
    params_file: str | None = None,
    dvc_file: str | None = None,
    runtime_overrides: list[str] | None = None,
) -> str:
    cmd = ["deckard optimize", f"+stage={stage}"]
    _normalize_mode(mode)
    if multirun_count is not None and int(multirun_count) <= 0:
        raise ValueError("multirun_count must be a positive integer when provided.")
    if params_file not in [None, ""]:
        cmd.append(f"+params_file={params_file}")
    if dvc_file not in [None, ""]:
        cmd.append(f"+dvc_file={dvc_file}")
    for override in runtime_overrides or []:
        token = str(override).strip()
        if token:
            cmd.append(token)
    return " ".join(cmd)


def build_dvc_cmd(
    experiment: Any,
    stage_plan: Mapping[str, Any],
    mode: str,
    multirun_count: int | None = None,
) -> str:
    """Emit reproducible optimize command for one stage plan entry."""
    _ = experiment
    if "stage" not in stage_plan:
        raise KeyError("stage_plan must include a 'stage' key.")
    stage = str(stage_plan.get("runtime_stage", stage_plan["stage"]))
    return _stage_cmd(
        stage=stage,
        mode=mode,
        multirun_count=multirun_count,
        params_file=stage_plan.get("params_file"),
        dvc_file=stage_plan.get("dvc_file"),
        runtime_overrides=stage_plan.get("runtime_overrides"),
    )


def build_dvc_stage_plan(
    experiment: Any,
    stage_selection: Any = None,
    include_cache_aliases: bool = True,
    *,
    mode: str = "single",
    multirun_count: int | None = None,
    params_file: str | None = None,
    dvc_file: str = "dvc.yaml",
) -> list[dict[str, Any]]:
    """Build canonical stage plan entries for DVC autogeneration."""
    mode = _normalize_mode(mode)
    selected_stages = [
        stage
        for stage in _resolve_stage_sequence(experiment, stage_selection)
        if _stage_enabled(experiment, stage)
    ]
    params_manifest = _resolve_params_manifest(experiment)

    file_aliases = _coerce_file_mapping(getattr(experiment, "files", None))
    if params_file not in [None, ""]:
        file_aliases["params_file"] = str(params_file)

    cache_path = None
    if include_cache_aliases:
        cache_path = _default_runtime_cache_path(file_aliases)
        if cache_path is None:
            raise ValueError(
                "include_cache_aliases=True requires files.params_file to derive runtime cache path.",
            )
    aliases = extract_dvc_file_aliases(file_aliases, cache_path=cache_path)

    stage_plan: list[dict[str, Any]] = []
    params_file_alias = aliases.get("params_file")
    stage_name_to_outs: dict[str, list[str]] = {}
    artifact_owners: dict[str, dict[str, Any]] = {}

    def _claim_stage_artifacts(stage_entry: dict[str, Any]) -> None:
        for key in ("outs", "metrics", "plots"):
            claimed: list[str] = []
            for path in stage_entry.get(key, []):
                owner = artifact_owners.get(path)
                if owner is not None and owner is not stage_entry:
                    for owner_key in ("outs", "metrics", "plots"):
                        owner_values = owner.get(owner_key, [])
                        if path in owner_values:
                            owner[owner_key] = [
                                value for value in owner_values if value != path
                            ]
                    owner_name = str(owner.get("name", "")).strip()
                    if owner_name:
                        stage_name_to_outs[owner_name] = list(owner.get("outs", []))
                if path not in claimed:
                    claimed.append(path)
                    artifact_owners[path] = stage_entry
            stage_entry[key] = claimed

    def _dedupe_values(values: list[Any]) -> list[Any]:
        seen: set[str] = set()
        unique: list[Any] = []
        for value in values:
            if value in [None, ""]:
                continue
            marker = json.dumps(value, sort_keys=True, default=str)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(value)
        return unique

    for idx, stage in enumerate(selected_stages):
        runtime_stage = _resolve_runtime_stage(stage)
        base_stage, stage_alias = _split_stage_alias(stage)
        component = _resolve_stage_component(stage)
        stage_name = build_dvc_stage_name(component, stage)
        deps: list[str] = [".deckard_rc", "deckard"]
        if idx > 0:
            prev_stage = selected_stages[idx - 1]
            prev_name = build_dvc_stage_name(
                _resolve_stage_component(prev_stage),
                prev_stage,
            )
            deps.extend(stage_name_to_outs.get(prev_name, []))

        stage_entry: dict[str, Any] = {
            "name": stage_name,
            "stage": stage,
            "runtime_stage": runtime_stage,
            "component": component,
            "runtime_overrides": [],
            "deps": deps,
            "outs": [],
            "params": [],
            "param_key_paths": [],
            "metrics": [],
            "plots": [],
            "params_file": params_file_alias,
            "dvc_file": dvc_file,
        }

        outs = stage_entry["outs"]
        metrics = stage_entry["metrics"]

        if params_file_alias:
            deps.append(params_file_alias)
            stage_entry["param_key_paths"] = list(
                build_experiment_stage_param_key_paths(
                    stage=stage,
                    component=component,
                ),
            )
            if stage_entry["param_key_paths"]:
                stage_entry["params"] = [
                    {params_file_alias: stage_entry["param_key_paths"]},
                ]

        if runtime_stage in {"load", "sample", "train"} and "data_file" in aliases:
            deps.append(aliases["data_file"])
            if base_stage in {"load", "sample", "pipeline"}:
                outs.append(aliases["data_file"])

        if runtime_stage == "train":
            for key in _STAGE_OUTPUT_KEYS["train"]:
                value = aliases.get(key)
                if value:
                    outs.append(value)

        if base_stage == "detector-train":
            runtime_overrides = ["detector.mode=train"]
            if stage_alias:
                runtime_overrides.append(f"detector.alias={stage_alias}")
            stage_entry["runtime_overrides"] = runtime_overrides
            for key in _STAGE_OUTPUT_KEYS["detector-train"]:
                value = aliases.get(key)
                if value:
                    outs.append(value)

        if runtime_stage == "defense" and base_stage != "detector-train":
            runtime_overrides: list[str] = []
            if base_stage == "detector-defense":
                runtime_overrides.append("detector.mode=filter")
            if base_stage == "apply-fit-defense":
                runtime_overrides.append("defense.apply=fit")
            if base_stage == "apply-predict-defense":
                runtime_overrides.append("defense.apply=predict")
            if stage_alias:
                runtime_overrides.append(f"stage_alias={stage_alias}")
            stage_entry["runtime_overrides"] = runtime_overrides
            for key in _STAGE_OUTPUT_KEYS["defense"]:
                value = aliases.get(key)
                if value:
                    outs.append(value)

        if runtime_stage == "attack":
            if stage_alias:
                stage_entry["runtime_overrides"] = [
                    *stage_entry["runtime_overrides"],
                    f"attack.alias={stage_alias}",
                ]
            for key in _STAGE_OUTPUT_KEYS["generation"]:
                value = aliases.get(key)
                if value:
                    outs.append(value)

        if runtime_stage == "score":
            score_file = aliases.get("score_file")
            if not score_file:
                raise ValueError(
                    "Stage 'score' requires files.score_file to be configured.",
                )
            outs.append(score_file)
            metrics.append(score_file)
            if base_stage in {
                "data-score",
                "model-score",
                "attack-score",
                "detector-score",
            }:
                stage_entry["runtime_overrides"].append(f"score.scope={component}")

        if runtime_stage == "persist":
            if base_stage in {
                "data-persist",
                "model-persist",
                "attack-persist",
                "detector-persist",
            }:
                for key in _STAGE_OUTPUT_KEYS.get(base_stage, ()):
                    value = aliases.get(key)
                    if value:
                        outs.append(value)
                stage_entry["runtime_overrides"].append(f"persist.scope={component}")
                if aliases.get("params_file") and stage_entry["param_key_paths"]:
                    stage_entry["params"] = [
                        {aliases["params_file"]: stage_entry["param_key_paths"]},
                    ]
                stage_entry["cmd"] = build_dvc_cmd(
                    experiment,
                    stage_entry,
                    mode=mode,
                    multirun_count=multirun_count,
                )
                for key in ("deps", "outs", "params", "metrics", "plots"):
                    stage_entry[key] = _dedupe_values(stage_entry[key])
                _claim_stage_artifacts(stage_entry)
                tracked = set(stage_entry.get("outs", []))
                tracked.update(stage_entry.get("metrics", []))
                tracked.update(stage_entry.get("plots", []))
                stage_entry["deps"] = [
                    value for value in stage_entry["deps"] if value not in tracked
                ]
                stage_name_to_outs[stage_name] = list(stage_entry["outs"])
                stage_plan.append(stage_entry)
                continue

            required_aliases = ("params_file", "score_file", "log_file", "error_file")
            missing_aliases = [
                name for name in required_aliases if not aliases.get(name)
            ]
            if missing_aliases:
                missing = ", ".join(missing_aliases)
                raise ValueError(
                    "Stage 'persist' requires file aliases: "
                    f"{missing}. Configure these under experiment.files.",
                )

            run_identity = _resolve_run_identity(
                experiment,
                mode=mode,
                stage=stage,
                params_manifest=params_manifest,
            )
            root = Path("outputs") / "logs" / run_identity
            score_file = aliases["score_file"]
            stage_entry["identity"] = run_identity
            if aliases.get("params_file") and stage_entry["param_key_paths"]:
                stage_entry["params"] = [
                    {aliases["params_file"]: stage_entry["param_key_paths"]},
                ]
            outs.append(root.as_posix())
            if include_cache_aliases and aliases.get("runtime_cache_file"):
                outs.append(aliases["runtime_cache_file"])

            metrics.extend(
                [
                    score_file,
                ],
            )
            # When tracking the run root directory as an out, DVC forbids nested
            # tracked outputs (metrics/plots) under the same directory.

        stage_entry["cmd"] = build_dvc_cmd(
            experiment,
            stage_entry,
            mode=mode,
            multirun_count=multirun_count,
        )

        for key in ("deps", "outs", "params", "metrics", "plots"):
            stage_entry[key] = _dedupe_values(stage_entry[key])

        _claim_stage_artifacts(stage_entry)
        tracked = set(stage_entry.get("outs", []))
        tracked.update(stage_entry.get("metrics", []))
        tracked.update(stage_entry.get("plots", []))
        stage_entry["deps"] = [
            value for value in stage_entry["deps"] if value not in tracked
        ]

        stage_name_to_outs[stage_name] = list(stage_entry["outs"])
        stage_plan.append(stage_entry)

    return stage_plan


def generate_dvc_pipeline(
    experiment: Any,
    output_file: str = "dvc.yaml",
    params_file: str = "params.yaml",
    stage_selection: Any = None,
    include_cache_aliases: bool = True,
    mode: str = "single",
    multirun_count: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate and write a DVC pipeline from experiment runtime metadata."""
    output_path = Path(output_file)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_file}")

    stage_plan = build_dvc_stage_plan(
        experiment,
        stage_selection=stage_selection,
        include_cache_aliases=include_cache_aliases,
        mode=mode,
        multirun_count=multirun_count,
        params_file=params_file,
        dvc_file=output_file,
    )
    plugin_cfg = coerce_dvc_experiment_plugin(getattr(experiment, "dvc_plugin", None))
    params_payload = _write_generated_pipeline_params_file(
        experiment,
        plugin=plugin_cfg,
        params_file=params_file,
        stage_selection=(stage_selection if stage_selection is not None else "all"),
    )

    stages: dict[str, dict[str, Any]] = {}
    for entry in stage_plan:
        stage_name = entry["name"]
        stage_payload: dict[str, Any] = {"cmd": entry["cmd"]}
        for key in ("deps", "outs", "params", "metrics", "plots"):
            values = entry.get(key) or []
            if values:
                stage_payload[key] = values
        stages[stage_name] = stage_payload

    payload = {
        "params_file": params_file,
        "params_payload": params_payload,
        "stages": stages,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    return payload


def generate_vega_lite_plot_spec(
    *,
    output_file: str,
    title: str,
    x_field: str,
    y_field: str,
    mark: str = "line",
    color_field: str | None = None,
    data_values: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write one Vega-Lite spec file and return the in-memory payload.

    Any .yaml/.yml output path is normalized to a hydra-resolvable .vl.json file.
    """
    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": title,
        "title": title,
        "mark": str(mark),
        "data": {"values": data_values or []},
        "encoding": {
            "x": {"field": x_field, "type": "quantitative"},
            "y": {"field": y_field, "type": "quantitative"},
        },
    }
    if color_field:
        spec["encoding"]["color"] = {
            "field": str(color_field),
            "type": "nominal",
        }

    output_path = Path(output_file)
    if output_path.suffix.lower() in {".yaml", ".yml"}:
        output_path = output_path.with_suffix("")
        output_path = output_path.with_suffix(".vl.json")
    elif output_path.suffix.lower() == ".json" and not output_path.name.endswith(
        ".vl.json",
    ):
        output_path = output_path.with_name(f"{output_path.stem}.vl.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)
    spec["output_file"] = output_path.as_posix()
    return spec


__all__ = [
    "configure_dvclive_runtime",
    "DVCExperimentConfig",
    "DVCExperimentPlugin",
    "build_dvc_cmd",
    "build_dvc_experiment_plugin_hooks",
    "build_dvc_stage_name",
    "build_dvc_stage_plan",
    "coerce_dvc_experiment_plugin",
    "extract_dvc_file_aliases",
    "render_dvclive_report",
    "generate_vega_lite_plot_spec",
    "generate_dvc_pipeline",
    "run_dvc_experiment_plugin_hook",
]
