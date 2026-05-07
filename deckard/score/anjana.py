"""ANJANA-specific scoring helpers and default scorer configuration."""

from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from .base import ScorerConfig, ScorerDictConfig, _DataScorerMarker, safe_store
from ..data import DataConfig

__all__ = [
    "anjana_k_anonymity_score",
    "anjana_l_diversity_score",
    "anjana_t_closeness_score",
    "_AnjanaScorerMixin",
    "DefaultAnjanaScoreConfig",
    "DefaultAnjanaDataScoreConfig",
    "DefaultAnjanaModelScoreConfig",
]

def _resolve_frame_and_context(
    data: DataConfig | None = None,
    y_pred: Any | None = None,
    y_true: Any | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, list[str], str | None]:
    frame = None
    if isinstance(y_pred, pd.DataFrame):
        frame = y_pred
    elif data is not None and isinstance(
        getattr(data, "_X", None),
        pd.DataFrame,
    ):
        frame = data._X
    if frame is None:
        raise ValueError(
            "ANJANA scorers require a pandas.DataFrame via y_pred or data._X",
        )

    quasi_ident = kwargs.get(
        "quasi_ident",
        getattr(data, "quasi_identifiers", None),
    )
    sens_att = kwargs.get(
        "sens_att",
        getattr(data, "sensitive_attribute", None),
    )

    if isinstance(sens_att, str) and sens_att not in frame.columns and y_true is not None:
        frame = frame.copy()
        labels = pd.Series(cast(Any, y_true)).reset_index(drop=True)
        if len(labels) == len(frame):
            frame[sens_att] = labels

    if sens_att is not None and not isinstance(sens_att, str):
        sens_att = str(sens_att)

    if isinstance(quasi_ident, str):
        quasi_ident = [quasi_ident]
    if not isinstance(quasi_ident, list) or len(quasi_ident) == 0:
        raise ValueError("ANJANA scorers require quasi_ident identifiers")
    quasi_ident = [str(identifier) for identifier in quasi_ident]
    return cast(pd.DataFrame, frame), quasi_ident, sens_att


def anjana_k_anonymity_score(
    y_true: Any = None,
    y_pred: Any = None,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
    """Compute k-anonymity score for an anonymized dataset.

    Parameters
    ----------
    y_true : array-like, optional
        Ground-truth labels (unused; present for scorer interface compatibility).
    y_pred : pd.DataFrame, optional
        The (possibly anonymized) feature matrix.  When provided as a DataFrame
        it is used directly; otherwise the function falls back to ``data._X``.
    data : AnjanaDataConfig, optional
        Data configuration carrying ``quasi_identifiers`` and
        ``sensitive_attribute``.
    **kwargs
        Forwarded to :func:`_resolve_frame_and_context`; supports
        ``quasi_ident`` and ``sens_att`` overrides.

    Returns
    -------
    float
        The k-anonymity value of the dataset.

    Raises
    ------
    ImportError
        If ``pycanon`` is not installed.
    ValueError
        If no quasi-identifier columns are available.
    """
    try:
        from pycanon import anonymity as pycanon_anonymity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ANJANA scorers require optional dependency pycanon/anjana",
        ) from exc

    frame, quasi_ident, _ = _resolve_frame_and_context(
        data=data,
        y_pred=y_pred,
        y_true=y_true,
        **kwargs,
    )
    return float(pycanon_anonymity.k_anonymity(frame, quasi_ident))


def anjana_l_diversity_score(
    y_true: Any = None,
    y_pred: Any = None,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
    """Compute l-diversity score for an anonymized dataset.

    Parameters
    ----------
    y_true : array-like, optional
        Ground-truth labels (unused; present for scorer interface compatibility).
    y_pred : pd.DataFrame, optional
        The (possibly anonymized) feature matrix.
    data : AnjanaDataConfig, optional
        Data configuration carrying ``quasi_identifiers`` and
        ``sensitive_attribute``.
    **kwargs
        Forwarded to :func:`_resolve_frame_and_context`; supports
        ``quasi_ident`` and ``sens_att`` overrides.

    Returns
    -------
    float
        The l-diversity value of the dataset.

    Raises
    ------
    ImportError
        If ``pycanon`` is not installed.
    ValueError
        If no quasi-identifier columns or sensitive attribute is available.
    """
    try:
        from pycanon import anonymity as pycanon_anonymity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ANJANA scorers require optional dependency pycanon/anjana",
        ) from exc

    frame, quasi_ident, sens_att = _resolve_frame_and_context(
        data=data,
        y_pred=y_pred,
        y_true=y_true,
        **kwargs,
    )
    if sens_att is None:
        raise ValueError(
            "ANJANA l-diversity scorer requires sens_att/sensitive_attribute",
        )
    return float(pycanon_anonymity.l_diversity(frame, quasi_ident, [sens_att]))


def anjana_t_closeness_score(
    y_true: Any = None,
    y_pred: Any = None,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
    """Compute t-closeness score for an anonymized dataset.

    Parameters
    ----------
    y_true : array-like, optional
        Ground-truth labels (unused; present for scorer interface compatibility).
    y_pred : pd.DataFrame, optional
        The (possibly anonymized) feature matrix.
    data : AnjanaDataConfig, optional
        Data configuration carrying ``quasi_identifiers`` and
        ``sensitive_attribute``.
    **kwargs
        Forwarded to :func:`_resolve_frame_and_context`; supports
        ``quasi_ident`` and ``sens_att`` overrides.

    Returns
    -------
    float
        The t-closeness value of the dataset.

    Raises
    ------
    ImportError
        If ``pycanon`` is not installed.
    ValueError
        If no quasi-identifier columns or sensitive attribute is available.
    """
    try:
        from pycanon import anonymity as pycanon_anonymity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ANJANA scorers require optional dependency pycanon/anjana",
        ) from exc

    frame, quasi_ident, sens_att = _resolve_frame_and_context(
        data=data,
        y_pred=y_pred,
        y_true=y_true,
        **kwargs,
    )
    if sens_att is None:
        raise ValueError(
            "ANJANA t-closeness scorer requires sens_att/sensitive_attribute",
        )
    return float(pycanon_anonymity.t_closeness(frame, quasi_ident, [sens_att]))


class _AnjanaScorerMixin(_DataScorerMarker):
    """Marker mixin for ANJANA privacy scorers.

    Inherits :class:`_DataScorerMarker` so that
    ``_initialize_component_scorers`` routes these scorers to ``data.scorer``.
    Subclass this to add ANJANA-specific call-time behaviour; the
    ``"anjana_scores"`` output wrapping is enforced by the data layer
    (:meth:`AnjanaDataConfig._score`).
    """


@dataclass(eq=False)
class DefaultAnjanaScoreConfig(_AnjanaScorerMixin, ScorerDictConfig):
    scorers: dict[str, ScorerConfig] = field(
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
class DefaultAnjanaDataScoreConfig(DefaultAnjanaScoreConfig):
    pass


@dataclass(eq=False)
class DefaultAnjanaModelScoreConfig(DefaultAnjanaScoreConfig):
    pass


safe_store(group="score", name="anjana", node=DefaultAnjanaScoreConfig)
safe_store(group="score", name="anjana_data", node=DefaultAnjanaDataScoreConfig)
safe_store(
    group="score",
    name="anjana_model",
    node=DefaultAnjanaModelScoreConfig,
)
