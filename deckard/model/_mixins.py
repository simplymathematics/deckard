"""Model runtime mixins for pretrained loading and pruning helpers."""

from __future__ import annotations

from typing import Protocol

from ..frameworks.types import EstimatorLike


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


class PretrainedModelMixin:
    """Reusable pretrained-model loading behavior.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

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
    """Reusable Optuna-style pruning behavior.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

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


__all__ = [
    "PretrainedModelMixin",
    "ModelPrunerMixin",
]
