"""Model runtime mixins for pretrained, pruning, training, and plugin-hook flows."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from ..frameworks.types import ArrayLike, EstimatorLike, MatrixLike
from ..plugins import HookPlugin
from ..utils import instantiate_plugin_spec, load_class, normalize_plugin_specs


class PruneTrialProtocol(Protocol):
    """Minimal trial protocol for pruning integrations."""

    def report(self, value: float, step: int) -> None:
        """Report intermediate metric values to the pruning backend.

        Args:
            value: Intermediate scalar metric.
            step: Iteration or epoch step index.
        """
        ...

    def should_prune(self) -> bool:
        """Return whether current trial should be pruned.

        Returns:
            ``True`` when backend requests pruning.
        """
        ...


class ModelTrainingMixin:
    """Reusable model training behavior for non-pretrained model flows."""

    def train_model(self, X: MatrixLike, y: ArrayLike) -> None:
        """Fit the underlying estimator.

        Args:
            X: Training features.
            y: Training targets.

        Raises:
            ValueError: If runtime model is missing or non-trainable.
        """
        model = getattr(self, "_model", None)
        if model is None or not hasattr(model, "fit"):
            raise ValueError("Model is not initialized and cannot be trained.")
        fit_params = getattr(self, "fit_params", {}) or {}
        model.fit(X, y, **fit_params)

    def train(self, X: MatrixLike, y: ArrayLike) -> None:
        """Public training entrypoint that delegates to the model implementation.

        Args:
            X: Training features.
            y: Training targets.
        """
        self.train_model(X, y)

    def _train(self, X: MatrixLike, y: ArrayLike) -> None:
        """Protected training hook used by model runtime orchestration."""
        self.train_model(X, y)


class PretrainedModelMixin:
    """Reusable pretrained-model loading behavior."""

    def load_cached(self, path: str) -> EstimatorLike:
        """Load a persisted model from ``path`` using the config's loader.

        Args:
            path: Filepath or serialized model location.

        Returns:
            The loaded config or estimator instance.

        Raises:
            NotImplementedError: If neither config nor model exposes ``load(path)``.
        """
        loader = getattr(self, "load", None)
        if callable(loader):
            return loader(path)
        model = getattr(self, "_model", None)
        if model is not None and hasattr(model, "load"):
            return model.load(path)
        raise NotImplementedError(
            "Pretrained loading requires a load(path) method on the config or model.",
        )


class ModelPrunerMixin:
    """Reusable Optuna-style pruning behavior."""

    def check_prune(
        self,
        trial: PruneTrialProtocol | None,
        value: float | int | None = None,
        step: int | None = None,
    ) -> bool:
        """Report an intermediate value and ask a trial whether it should prune.

        Args:
            trial: Optuna trial-like object.
            value: Optional intermediate metric value.
            step: Optional reporting step.

        Returns:
            ``True`` when the trial requests pruning, otherwise ``False``.
        """
        if trial is None:
            return False
        if value is not None and hasattr(trial, "report"):
            trial.report(value, 0 if step is None else step)
        should_prune = getattr(trial, "should_prune", None)
        if callable(should_prune):
            return bool(should_prune())
        return False


class ModelHookRuntimeMixin:
    """Reusable plugin orchestration and runtime-state copy behavior."""

    def _instantiate_plugin(self, plugin_spec: Any):
        """Create one plugin instance from a normalized plugin specification."""
        return instantiate_plugin_spec(plugin_spec, loader=load_class)

    def _get_plugins(self) -> list:
        """Lazily instantiate and cache configured plugin objects."""
        if self._plugin_objects is None:
            plugin_specs = normalize_plugin_specs(self.plugins)
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs: Any):
        """Execute one named hook across all configured plugins."""
        hook_outputs = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_outputs.append(hook(self, **kwargs))
        return hook_outputs

    def _merge_plugin_scores(self, hook_outputs: Iterable[Any]) -> None:
        """Merge dictionary hook outputs into ``score_dict`` in-order."""
        if self.score_dict is None:
            self.score_dict = {}
        for output in hook_outputs:
            if isinstance(output, dict):
                self.score_dict.update(output)

    def _copy_runtime_state_to(self, target: Any) -> None:
        """Copy standard runtime attributes from this object to ``target``."""
        runtime_fields = [
            "_model",
            "score_dict",
            "training_predictions",
            "predictions",
            "val_predictions",
            "training_probabilities",
            "probabilities",
            "val_probabilities",
            "training_time",
            "prediction_time",
            "val_prediction_time",
            "training_prediction_time",
            "training_score_time",
            "prediction_score_time",
            "val_score_time",
            "defense_application_time",
            "training_n",
            "prediction_n",
            "val_n",
        ]
        for attr in runtime_fields:
            setattr(target, attr, getattr(self, attr, None))


__all__ = [
    "HookPlugin",
    "ModelHookRuntimeMixin",
    "ModelTrainingMixin",
    "PretrainedModelMixin",
    "ModelPrunerMixin",
]
