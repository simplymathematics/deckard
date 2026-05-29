from __future__ import annotations

import inspect

import pandas as pd
from omegaconf import DictConfig

from deckard.plugins import HookPlugin
from deckard.plugins.base import HookBundle

from ...utils import resolve_class as _default_resolve_class

ANJANA_PIPELINE_HOOKS = HookBundle(
    name="anjana.data.pipeline_hooks",
    hooks=(
        HookPlugin(
            hook_name="before_sample",
            method_name="apply_defense",
            init_params={
                "library": "anjana",
                "type": "data",
                "class": "pre_sample_anonymization",
                "phase": "pipeline",
            },
        ),
    ),
)


class AnjanaPipelineHooksMixin:
    """Pipeline-stage hook implementations for ANJANA data runtimes.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def apply_defense(self) -> None:
        """Canonical public entrypoint for ANJANA pipeline defense application."""
        if self.anjana_defense in [None, False]:
            return
        if self.anjana_defense is True:
            raise ValueError(
                "anjana_defense=True is ambiguous. Provide a config dict with at least a 'name' key.",
            )
        if not isinstance(self.anjana_defense, (dict, DictConfig)):
            raise TypeError(
                "anjana_defense must be a dict/DictConfig, False, or None. "
                f"Got {type(self.anjana_defense)}",
            )

        defense_name, defense_cfg = self._normalize_runtime_defense_config(
            dict(self.anjana_defense),
            field_name="anjana_defense",
        )

        from . import data as anjana_data_module

        resolver = getattr(anjana_data_module, "resolve_class", _default_resolve_class)
        defense_fn = resolver(defense_name)
        if not callable(defense_fn):
            raise TypeError(
                f"Configured ANJANA defense '{defense_name}' is not callable",
            )

        frame = self._build_privacy_frame()
        call_kwargs = dict(defense_cfg)
        call_kwargs.setdefault("data", frame)
        call_kwargs.setdefault("ident", self.identifiers or [])
        if self.quasi_identifiers is not None:
            call_kwargs.setdefault("quasi_ident", self.quasi_identifiers)
        if self.sensitive_attribute is not None:
            call_kwargs.setdefault("sens_att", self.sensitive_attribute)
        call_kwargs.setdefault("supp_level", 100)
        if self.hierarchies is not None:
            call_kwargs.setdefault("hierarchies", self.hierarchies)
        elif self.quasi_identifiers is not None:
            call_kwargs.setdefault(
                "hierarchies",
                self.generate_anjana_hierarchy_dict(frame=frame),
            )

        signature = inspect.signature(defense_fn)
        supports_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not supports_var_kwargs:
            accepted = {
                name
                for name, p in signature.parameters.items()
                if p.kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            }
            call_kwargs = {
                key: value for key, value in call_kwargs.items() if key in accepted
            }

        transformed = defense_fn(**call_kwargs)
        if not isinstance(transformed, pd.DataFrame):
            raise TypeError(
                f"ANJANA defense '{defense_name}' must return pandas.DataFrame, got {type(transformed)}",
            )

        target_col = self._resolve_anjana_target_column()
        if target_col not in transformed.columns:
            retained_index = transformed.index.intersection(frame.index)
            transformed = transformed.loc[retained_index].copy()
            self._y = pd.Series(self._y, index=frame.index).loc[retained_index]
        else:
            self._y = transformed[target_col]
            transformed = transformed.drop(columns=[target_col])

        self._X = transformed

    @staticmethod
    def _normalize_runtime_defense_config(
        defense_cfg: dict,
        *,
        field_name: str,
    ) -> tuple[str, dict]:
        """Normalize runtime defense mappings to canonical defense fields.

        Requires canonical ``name`` and merges ``defense_params`` into call kwargs.
        """
        normalized = dict(defense_cfg)
        defense_params = normalized.pop("defense_params", {})
        if isinstance(defense_params, DictConfig):
            defense_params = dict(defense_params)
        elif not isinstance(defense_params, dict):
            raise TypeError(
                f"{field_name}.defense_params must be a dict/DictConfig when provided. Got {type(defense_params)}",
            )

        defense_name = normalized.pop("name", None)
        if not isinstance(defense_name, str):
            raise ValueError(
                f"{field_name} config must include 'name'",
            )

        return defense_name, {**defense_params, **normalized}


__all__ = ["ANJANA_PIPELINE_HOOKS", "AnjanaPipelineHooksMixin"]
