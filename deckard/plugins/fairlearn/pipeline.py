from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig

from deckard.plugins import HookPlugin
from deckard.plugins.base import HookBundle

FAIRLEARN_PIPELINE_HOOKS = HookBundle(
    name="fairlearn.data.pipeline_hooks",
    hooks=(
        HookPlugin(
            hook_name="before_sample",
            method_name="apply_defense",
            init_params={
                "library": "fairlearn",
                "type": "data",
                "class": "CorrelationRemover",
                "phase": "pipeline",
            },
        ),
        HookPlugin(
            hook_name="after_pipeline",
            method_name="_run_fairlearn_post_pipeline_hook",
            init_params={
                "library": "fairlearn",
                "type": "data",
                "class": "post_pipeline_policy",
                "phase": "pipeline",
            },
        ),
    ),
)


class FairlearnPipelineHooksMixin:
    """Pipeline-stage hook implementations for fairlearn data runtimes.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def apply_defense(self) -> None:
        """Canonical public entrypoint for fairlearn pipeline defense injection."""
        if self.fairness_defense in [None, False]:
            return
        if self.fairness_defense is True:
            raise ValueError(
                "fairness_defense=True is ambiguous. Provide a config dict with at least a 'name' key.",
            )
        if not isinstance(self.fairness_defense, (dict, DictConfig)):
            raise TypeError(
                "fairness_defense must be a dict/DictConfig, False, or None. "
                f"Got {type(self.fairness_defense)}",
            )
        if self.sensitive_columns is None:
            raise ValueError("sensitive_columns must be configured")
        if (
            not hasattr(self, "_X")
            or self._X is None
            or not isinstance(self._X, pd.DataFrame)
        ):
            return

        sensitive_columns = [
            col for col in self.sensitive_columns if col in self._X.columns
        ]
        if not sensitive_columns:
            raise RuntimeError(
                f"Sensitive features not found for {self.sensitive_columns}.",
            )

        step_config: dict[str, Any] = {
            "sensitive_feature_ids": list(sensitive_columns),
        }
        step_name = "fairness_correlation_remover"
        custom = dict(self.fairness_defense)
        step_name = custom.pop("step_name", step_name)
        step_config.update(custom)
        if "name" not in step_config:
            raise ValueError(
                "fairness_defense config must include a 'name' key",
            )

        from ...data.pipeline.base import DataPipeline

        pipeline_runtime = getattr(self, "pipeline", None)
        if isinstance(pipeline_runtime, DataPipeline):
            pipeline_dict = dict(pipeline_runtime.pipeline or {})
            if step_name in pipeline_dict:
                return
            updated = {step_name: step_config, **pipeline_dict}
            pipeline_runtime.pipeline = updated
            pipeline_runtime.clear()
            pipeline_runtime.update(updated)
            self.pipeline = pipeline_runtime
            return

        if pipeline_runtime in [None, False]:
            self.pipeline = DataPipeline(pipeline={step_name: step_config})
            return

        pipeline_dict = dict(pipeline_runtime)
        if step_name in pipeline_dict:
            return
        self.pipeline = DataPipeline(
            pipeline={step_name: step_config, **pipeline_dict},
        )

    def _run_fairlearn_post_pipeline_hook(self, **kwargs):
        """Post-pipeline policy hook for fairlearn runtime stage alignment."""
        _ = kwargs
        self._fairlearn_post_pipeline_seen = True
        return None


__all__ = ["FAIRLEARN_PIPELINE_HOOKS", "FairlearnPipelineHooksMixin"]
