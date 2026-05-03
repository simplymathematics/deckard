"""ANJANA-specific scoring helpers and default scorer configuration."""

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from .base import ScorerConfig, ScorerDictConfig, safe_store

__all__ = [
    "anjana_k_anonymity_score",
    "anjana_l_diversity_score",
    "anjana_t_closeness_score",
    "DefaultAnjanaDataScoreConfig",
    "DefaultAnjanaModelScoreConfig",
]


def _resolve_frame_and_context(data=None, y_pred=None, **kwargs):
    frame = None
    if isinstance(y_pred, pd.DataFrame):
        frame = y_pred
    elif data is not None and isinstance(
        getattr(data, "_X", None), pd.DataFrame
    ):
        frame = data._X
    if frame is None:
        raise ValueError(
            "ANJANA scorers require a pandas.DataFrame via y_pred or data._X",
        )

    quasi_ident = kwargs.get(
        "quasi_ident", getattr(data, "quasi_identifiers", None)
    )
    sens_att = kwargs.get(
        "sens_att", getattr(data, "sensitive_attribute", None)
    )
    if isinstance(quasi_ident, str):
        quasi_ident = [quasi_ident]
    if not isinstance(quasi_ident, list) or len(quasi_ident) == 0:
        raise ValueError("ANJANA scorers require quasi_ident identifiers")
    return frame, quasi_ident, sens_att


def anjana_k_anonymity_score(y_true=None, y_pred=None, data=None, **kwargs):
    try:
        from pycanon import anonymity as pycanon_anonymity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ANJANA scorers require optional dependency pycanon/anjana",
        ) from exc

    frame, quasi_ident, _ = _resolve_frame_and_context(
        data=data,
        y_pred=y_pred,
        **kwargs,
    )
    return float(pycanon_anonymity.k_anonymity(frame, quasi_ident))


def anjana_l_diversity_score(y_true=None, y_pred=None, data=None, **kwargs):
    try:
        from pycanon import anonymity as pycanon_anonymity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ANJANA scorers require optional dependency pycanon/anjana",
        ) from exc

    frame, quasi_ident, sens_att = _resolve_frame_and_context(
        data=data,
        y_pred=y_pred,
        **kwargs,
    )
    if sens_att is None:
        raise ValueError(
            "ANJANA l-diversity scorer requires sens_att/sensitive_attribute",
        )
    return float(pycanon_anonymity.l_diversity(frame, quasi_ident, [sens_att]))


def anjana_t_closeness_score(y_true=None, y_pred=None, data=None, **kwargs):
    try:
        from pycanon import anonymity as pycanon_anonymity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ANJANA scorers require optional dependency pycanon/anjana",
        ) from exc

    frame, quasi_ident, sens_att = _resolve_frame_and_context(
        data=data,
        y_pred=y_pred,
        **kwargs,
    )
    if sens_att is None:
        raise ValueError(
            "ANJANA t-closeness scorer requires sens_att/sensitive_attribute",
        )
    return float(pycanon_anonymity.t_closeness(frame, quasi_ident, [sens_att]))


@dataclass(eq=False)
class DefaultAnjanaDataScoreConfig(ScorerDictConfig):
    scorers: Dict[str, ScorerConfig] = field(
        default_factory=lambda: {
            "k_anonymity": ScorerConfig(
                score_name="anjana.anonymity.k_anonymity",
                score_function=anjana_k_anonymity_score,
                greater_is_better=True,
            ),
            "l_diversity": ScorerConfig(
                score_name="anjana.anonymity.l_diversity",
                score_function=anjana_l_diversity_score,
                greater_is_better=True,
            ),
            "t_closeness": ScorerConfig(
                score_name="anjana.anonymity.t_closeness",
                score_function=anjana_t_closeness_score,
                greater_is_better=True,
            ),
        },
    )


@dataclass(eq=False)
class DefaultAnjanaModelScoreConfig(ScorerDictConfig):
    scorers: Dict[str, ScorerConfig] = field(
        default_factory=lambda: {
            "k_anonymity": ScorerConfig(
                score_name="k_anonymity",
                score_function=anjana_k_anonymity_score,
                greater_is_better=True,
            ),
            "l_diversity": ScorerConfig(
                score_name="l_diversity",
                score_function=anjana_l_diversity_score,
                greater_is_better=True,
            ),
            "t_closeness": ScorerConfig(
                score_name="t_closeness",
                score_function=anjana_t_closeness_score,
                greater_is_better=True,
            ),
        },
    )


safe_store(group="score", name="anjana_data", node=DefaultAnjanaDataScoreConfig)
safe_store(
    group="score", name="anjana_model", node=DefaultAnjanaModelScoreConfig
)
