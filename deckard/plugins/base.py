"""Shared plugin runtime and score orchestration mixins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from . import HookPlugin
from ..utils import (
	coerce_to_list,
	instantiate_plugin_spec,
	load_class,
	normalize_plugin_specs,
)


def _clone_hook_plugin(plugin: HookPlugin) -> HookPlugin:
	"""Return a detached hook plugin instance for safe per-runtime composition."""
	return HookPlugin(
		hook_name=plugin.hook_name,
		method_name=plugin.method_name,
		method_kwargs=dict(plugin.method_kwargs),
		init_params=dict(plugin.init_params),
	)


@dataclass(frozen=True)
class HookBundle:
	"""Named reusable collection of hook plugins for runtime composition."""

	name: str
	hooks: tuple[HookPlugin, ...]

	def clone_plugins(self) -> list[HookPlugin]:
		return [_clone_hook_plugin(plugin) for plugin in self.hooks]


def compose_hook_plugins(*parts: Any) -> list[HookPlugin]:
	"""Compose named bundles and standalone hooks into an ordered plugin list."""
	plugins: list[HookPlugin] = []
	seen: set[tuple[str, str]] = set()

	def _append(plugin: HookPlugin) -> None:
		key = (plugin.hook_name, plugin.method_name)
		if key in seen:
			return
		seen.add(key)
		plugins.append(plugin)

	for part in parts:
		if part is None:
			continue
		if isinstance(part, HookBundle):
			for plugin in part.clone_plugins():
				_append(plugin)
			continue
		if isinstance(part, HookPlugin):
			_append(_clone_hook_plugin(part))
			continue
		for item in coerce_to_list(part):
			if isinstance(item, HookBundle):
				for plugin in item.clone_plugins():
					_append(plugin)
			elif isinstance(item, HookPlugin):
				_append(_clone_hook_plugin(item))
			else:
				raise TypeError(
					"compose_hook_plugins accepts HookPlugin, HookBundle, or lists of them",
				)
	return plugins


@dataclass(eq=False, kw_only=True)
class RuntimeBase:
	"""Reusable plugin instantiation and hook dispatch behavior."""

	def _instantiate_plugin(self, plugin_spec: Any):
		return instantiate_plugin_spec(plugin_spec, loader=load_class)

	def _get_plugins(self) -> list:
		if not hasattr(self, "_plugin_objects") or self._plugin_objects is None:
			plugin_specs = normalize_plugin_specs(getattr(self, "plugins", []))
			self._plugin_objects = [
				self._instantiate_plugin(spec) for spec in plugin_specs
			]
		return self._plugin_objects

	def _run_plugin_hook(self, hook_name: str, **kwargs: Any) -> list[Any]:
		hook_outputs: list[Any] = []
		for plugin in self._get_plugins():
			hook = getattr(plugin, hook_name, None)
			if callable(hook):
				hook_outputs.append(hook(self, **kwargs))
		return hook_outputs


@dataclass(eq=False, kw_only=True)
class OrchestratorBase(RuntimeBase):
	"""Centralized stage-driven score orchestration behavior for runtimes."""

	default_stage: str = "post-pipeline"
	score_stage_aliases: ClassVar[dict[str, str]] = {
		"pre-load": "pre-load",
		"preload": "pre-load",
		"before-load": "pre-load",
		"before_load": "pre-load",
		"pre-sample": "pre-sample",
		"pre_sample": "pre-sample",
		"presample": "pre-sample",
		"post-sample": "post-sample",
		"post_sample": "post-sample",
		"postsample": "post-sample",
		"post-pipeline": "post-pipeline",
		"post_pipeline": "post-pipeline",
		"postpipeline": "post-pipeline",
	}
	score_stage_order: ClassVar[tuple[str, ...]] = (
		"pre-load",
		"pre-sample",
		"post-sample",
		"post-pipeline",
	)
	score_event_aliases: ClassVar[dict[str, str]] = {
		"pre": "before",
		"before": "before",
		"post": "after",
		"after": "after",
	}
	score_stage_to_hook: ClassVar[dict[str, str]] = {
		"pre-load": "before_load_data",
		"pre-sample": "before_sample",
		"post-sample": "after_sample",
		"post-pipeline": "after_pipeline",
	}
	_score_orchestration_active: bool = True

	def _normalize_score_mode(self, mode: str) -> str:
		return str(mode)

	def _stage_hook_token(self, stage: str) -> str:
		key = str(stage).strip().lower().replace(" ", "-")
		if key in self.score_stage_aliases:
			return self.score_stage_aliases[key].replace("-", "_")
		raise ValueError(
			f"Unknown score hook stage '{stage}'. "
			f"Must be one of {list(self.score_stage_order) + ['all', 'auto']}",
		)

	def _iter_configured_score_stages(self) -> list[str]:
		scorer = getattr(self, "scorer", None)
		configured = getattr(scorer, "configured_scorers", None)
		if not isinstance(configured, dict) or len(configured) == 0:
			return [self.default_stage]

		raw_stages: list[str] = []
		for scorer_cfg in configured.values():
			stage_value = getattr(scorer_cfg, "stage", None)
			if stage_value in [None, "", []]:
				raw_stages.append(self.default_stage)
				continue
			if isinstance(stage_value, str):
				raw_stages.append(stage_value)
				continue
			for token in coerce_to_list(stage_value):
				raw_stages.append(str(token))

		if len(raw_stages) == 0:
			return [self.default_stage]
		return raw_stages

	def _expand_canonical_score_stages(self, raw_stages: list[str]) -> list[str]:
		ordered = list(self.score_stage_order)
		expanded: list[str] = []

		for token in raw_stages:
			normalized = str(token).strip().lower().replace("_", "-")
			if normalized in {"", "auto"}:
				expanded.append(self.default_stage)
				continue
			if normalized == "all":
				expanded.extend(ordered)
				continue
			if normalized in ordered:
				expanded.append(normalized)
				continue
			raise ValueError(
				f"Unsupported score stage '{token}'. "
				f"Expected one of {ordered + ['all', 'auto']}",
			)

		deduped: list[str] = []
		for stage in ordered:
			if stage in expanded and stage not in deduped:
				deduped.append(stage)
		return deduped or [self.default_stage]

	def _configure_score_orchestration_plugins(self) -> None:
		stage_tokens = self._expand_canonical_score_stages(
			self._iter_configured_score_stages(),
		)
		score_plugins = [
			HookPlugin(
				hook_name=self.score_stage_to_hook[stage],
				method_name="_score_orchestration_hook",
				method_kwargs={"stage": stage},
			)
			for stage in stage_tokens
			if stage in self.score_stage_to_hook
		]
		if len(score_plugins) == 0:
			return
		if not hasattr(self, "_plugin_objects") or self._plugin_objects is None:
			self._plugin_objects = []
		self._plugin_objects.extend(score_plugins)

	def _score_orchestration_hook(self, stage: str, **kwargs: Any):
		if not self._score_orchestration_active:
			return None
		try:
			self._stage_hook_token(stage)
		except ValueError:
			return None
		mode = kwargs.pop("mode", None)
		mode = self._normalize_score_mode(mode or getattr(self, "score_split", "test"))
		score_kwargs = kwargs.pop("score_kwargs", None) or {}
		self._run_score_stage_hooks("before", stage, score_kwargs=score_kwargs)
		score_fn = getattr(self, "score", None)
		if not callable(score_fn):
			raise AttributeError(f"{type(self).__name__} has no callable 'score' method")
		result = score_fn(mode=mode, stage=stage, **score_kwargs)
		plugin_scores = self._run_score_stage_hooks("after", stage, scores=result)
		if isinstance(result, dict):
			for plugin_score in plugin_scores:
				if isinstance(plugin_score, dict):
					result.update(plugin_score)
			if getattr(self, "score_dict", None) is None:
				self.score_dict = {}
			for key, value in result.items():
				if (
					key in self.score_dict
					and isinstance(self.score_dict.get(key), dict)
					and isinstance(value, dict)
				):
					self.score_dict[key].update(value)
				else:
					self.score_dict[key] = value
		return result

	def _run_score_stage_hooks(
		self,
		when: str,
		stage: str,
		**kwargs: Any,
	) -> list[Any]:
		event = str(when).strip().lower()
		if event not in self.score_event_aliases:
			raise ValueError(f"Score hook event must be 'before' or 'after', got {when}")
		event = self.score_event_aliases[event]
		stage_token = self._stage_hook_token(stage)
		stage = stage_token.replace("_", "-")
		stage_kwargs = {"stage": stage, **kwargs}
		outputs: list[Any] = []
		outputs.extend(
			self._run_plugin_hook(
				f"{event}_score_{stage_token}",
				**stage_kwargs,
			),
		)
		outputs.extend(self._run_plugin_hook(f"{event}_score", **stage_kwargs))
		return outputs


__all__ = [
	"HookBundle",
	"compose_hook_plugins",
	"RuntimeBase",
	"OrchestratorBase",
]
