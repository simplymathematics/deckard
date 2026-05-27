"""ANJANA-specific scoring helpers and default scorer configuration."""

from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from ...artifacts import ScoreDict
from ...data import DataConfig
from ...data.canon import normalize_data_score_mode
from ...orchestration import resolve_data_split_payload
from ...plugins import HookPlugin
from ...plugins.base import HookBundle
from ...score.base import (
    ScorerConfig,
    ScorerDictConfig,
    _DataScorerMarker,
    TaskAwareScorerMixin,
    safe_store,
)
from ...utils import is_default_config_value, load_class

__all__ = [
    "ANJANA_SCORING_HOOKS",
    "AnjanaDataScoreHooksMixin",
    "anjana_k_anonymity_score",
    "anjana_l_diversity_score",
    "anjana_t_closeness_score",
    "AnjanaScorerMixin",
    "DefaultAnjanaScorerDictConfig",
    "DefaultAnjanaDataScorerDictConfig",
    "DefaultAnjanaModelScorerDictConfig",
]


ANJANA_SCORING_HOOKS = HookBundle(
    name="anjana.data.scoring_hooks",
    hooks=(
        HookPlugin(
            hook_name="after_score_post_pipeline",
            method_name="_append_anjana_tail_scores",
            init_params={
                "library": "anjana",
                "type": "data",
                "class": "tail_score",
                "phase": "scoring",
            },
        ),
    ),
)


class AnjanaDataScoreHooksMixin:
    """Data-runtime ANJANA scoring hooks and split-scoped score adapter.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def score(self, *args: Any, mode: str | None = None, **kwargs: Any) -> ScoreDict:
        """Run base data scoring with ANJANA defaults and split-aware fallback.

        Args:
            *args: Positional score payloads forwarded to base scorer execution.
            mode: Optional data split mode to score.
            **kwargs: Keyword payloads forwarded to scorer execution.

        Returns:
            Normalized score payload from the configured scorer.

        Raises:
            TypeError: If the configured scorer is not callable.
            Exception: Re-raises non-data-profile TypeError from parent scoring.
        """
        if is_default_config_value(self.scorer, include_best=False):
            from . import data as anjana_data_module

            loader = getattr(anjana_data_module, "load_class", load_class)
            scorer_obj = loader(
                "deckard.plugins.anjana.score.DefaultAnjanaScorerDictConfig",
            )
            self.scorer = scorer_obj() if isinstance(scorer_obj, type) else scorer_obj

        if self.scorer is None:
            return ScoreDict()
        if not callable(self.scorer):
            raise TypeError(
                f"AnjanaDataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )

        resolved_mode = normalize_data_score_mode(
            mode if mode is not None else getattr(self, "score_mode", "test"),
        )

        try:
            return ScoreDict.from_payload(
                super().score(*args, mode=resolved_mode, **kwargs),
            )
        except TypeError as exc:
            if "data-profile scorer" not in str(exc):
                raise
            y, X = resolve_data_split_payload(
                self,
                resolved_mode,
                fallback_to_all=False,
            )
            return ScoreDict.from_payload(
                self.scorer(
                    *args,
                    y=y,
                    X=X,
                    mode=resolved_mode,
                    data=self,
                    **kwargs,
                ),
            )

    def _append_anjana_tail_scores(
        self,
        stage: str,
        scores: dict | None = None,
        **kwargs,
    ) -> ScoreDict:
        """Run ANJANA score hook after base/core scores and append last."""
        _ = kwargs
        if self.scorer is None:
            return ScoreDict()
        if not callable(self.scorer):
            raise TypeError(
                f"AnjanaDataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )

        hook_stage = str(stage).strip().lower()
        if hook_stage != "post-pipeline":
            return ScoreDict()

        resolved_mode = normalize_data_score_mode(getattr(self, "score_mode", "test"))
        y, X = resolve_data_split_payload(self, resolved_mode, fallback_to_all=False)
        tail_scores = self.scorer(
            y=y,
            X=X,
            mode=resolved_mode,
            data=self,
        )
        if isinstance(tail_scores, dict) and len(tail_scores) == 1:
            only_key = next(iter(tail_scores))
            only_val = tail_scores[only_key]
            if isinstance(only_val, dict):
                tail_scores = only_val
        if not isinstance(tail_scores, dict):
            tail_scores = {"anjana_score": tail_scores}
        existing = dict(scores or {})
        if len(existing) == 0:
            return ScoreDict.from_payload(tail_scores)
        merged_tail = {}
        for key, value in tail_scores.items():
            if key in existing:
                merged_tail[f"anjana_{key}"] = value
            else:
                merged_tail[key] = value
        return ScoreDict.from_payload(merged_tail)


def _resolve_frame_and_context(
    data: DataConfig | None = None,
    X: Any | None = None,
    y: Any | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, list[str], str | None]:
    if X is None:
        X = kwargs.pop("y_pred", None)
    if y is None:
        y = kwargs.pop("y_true", None)

    frame = None
    if isinstance(X, pd.DataFrame):
        frame = X
    elif data is not None and isinstance(
        getattr(data, "_X", None),
        pd.DataFrame,
    ):
        frame = data._X
    if frame is None:
        raise ValueError(
            "ANJANA scorers require a pandas.DataFrame via X or data._X",
        )

    quasi_ident = kwargs.get(
        "quasi_ident",
        getattr(data, "quasi_identifiers", None),
    )
    sens_att = kwargs.get(
        "sens_att",
        getattr(data, "sensitive_attribute", None),
    )

    if isinstance(sens_att, str) and sens_att not in frame.columns and y is not None:
        frame = frame.copy()
        labels = pd.Series(cast(Any, y)).reset_index(drop=True)
        if len(labels) == len(frame):
            frame[sens_att] = labels

    if sens_att is not None and not isinstance(sens_att, str):
        sens_att = str(sens_att)

    if isinstance(quasi_ident, str):
        quasi_ident = [quasi_ident]
    if not isinstance(quasi_ident, list) or len(quasi_ident) == 0:
        if sens_att is not None:
            quasi_ident = sens_att
        else:
            raise ValueError("ANJANA scorers require quasi_ident identifiers")

    quasi_ident = [str(identifier) for identifier in quasi_ident]
    return cast(pd.DataFrame, frame), quasi_ident, sens_att


def anjana_k_anonymity_score(
    y: Any = None,
    X: Any = None,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
    """Compute k-anonymity score for an anonymized dataset.

    Args:
        y: Ground-truth labels retained for scorer interface compatibility.
        X: Possibly anonymized feature matrix. When passed as a DataFrame it is
            used directly; otherwise the scorer falls back to ``data._X``.
        data: Data configuration carrying quasi-identifier and sensitive-attribute metadata.
        **kwargs: Forwarded to ``_resolve_frame_and_context``. Supports
            ``quasi_ident`` and ``sens_att`` overrides.

    Returns:
        K-anonymity value of the dataset.

    Raises:
        ImportError: If ``pycanon`` is not installed.
        ValueError: If no quasi-identifier columns are available.
    """
    try:
        from pycanon import anonymity as pycanon_anonymity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ANJANA scorers require optional dependency pycanon/anjana",
        ) from exc

    frame, quasi_ident, _ = _resolve_frame_and_context(
        data=data,
        X=X,
        y=y,
        **kwargs,
    )
    return float(pycanon_anonymity.k_anonymity(frame, quasi_ident))


def anjana_l_diversity_score(
    y: Any = None,
    X: Any = None,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
    """Compute l-diversity score for an anonymized dataset.

    Args:
        y: Ground-truth labels retained for scorer interface compatibility.
        X: Possibly anonymized feature matrix.
        data: Data configuration carrying quasi-identifiers and sensitive-attribute metadata.
        **kwargs: Forwarded to ``_resolve_frame_and_context``. Supports
            ``quasi_ident`` and ``sens_att`` overrides.

    Returns:
        L-diversity value of the dataset.

    Raises:
        ImportError: If ``pycanon`` is not installed.
        ValueError: If no quasi-identifier columns or sensitive attribute is available.
    """
    try:
        from pycanon import anonymity as pycanon_anonymity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ANJANA scorers require optional dependency pycanon/anjana",
        ) from exc

    frame, quasi_ident, sens_att = _resolve_frame_and_context(
        data=data,
        X=X,
        y=y,
        **kwargs,
    )
    if sens_att is None:
        raise ValueError(
            "ANJANA l-diversity scorer requires sens_att/sensitive_attribute",
        )
    return float(pycanon_anonymity.l_diversity(frame, quasi_ident, [sens_att]))


def anjana_t_closeness_score(
    y: Any = None,
    X: Any = None,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
    """Compute t-closeness score for an anonymized dataset.

    Args:
        y: Ground-truth labels retained for scorer interface compatibility.
        X: Possibly anonymized feature matrix.
        data: Data configuration carrying quasi-identifiers and sensitive-attribute metadata.
        **kwargs: Forwarded to ``_resolve_frame_and_context``. Supports
            ``quasi_ident`` and ``sens_att`` overrides.

    Returns:
        T-closeness value of the dataset.

    Raises:
        ImportError: If ``pycanon`` is not installed.
        ValueError: If no quasi-identifier columns or sensitive attribute is available.
    """
    try:
        from pycanon import anonymity as pycanon_anonymity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ANJANA scorers require optional dependency pycanon/anjana",
        ) from exc

    frame, quasi_ident, sens_att = _resolve_frame_and_context(
        data=data,
        X=X,
        y=y,
        **kwargs,
    )
    if sens_att is None:
        raise ValueError(
            "ANJANA t-closeness scorer requires sens_att/sensitive_attribute",
        )
    return float(pycanon_anonymity.t_closeness(frame, quasi_ident, [sens_att]))


class AnjanaScorerMixin(_DataScorerMarker):
    """Marker mixin for ANJANA privacy scorers.

    Inherits :class:`_DataScorerMarker` so that
    ``_initialize_component_scorers`` routes these scorers to ``data.scorer``.
    Subclass this to add ANJANA-specific call-time behaviour; the
    ``"anjana_scores"`` output wrapping is enforced by the data layer
    (:meth:`AnjanaDataConfig._score`).

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """


@dataclass(eq=False, kw_only=True)
class DefaultAnjanaScorerDictConfig(AnjanaScorerMixin, ScorerDictConfig):
    """Default privacy scorer set for ANJANA anonymization analysis.

    This config composes ANJANA privacy ``ScorerConfig`` objects into one
    ``ScorerDictConfig`` that emits a ``ScoreDict`` for anonymization
    evaluation, including k-anonymity, l-diversity, and t-closeness.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

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


@dataclass(eq=False, kw_only=True)
class DefaultAnjanaDataScorerDictConfig(TaskAwareScorerMixin, ScorerDictConfig):
    """Default data-analysis scorers plus ANJANA privacy scorers.

    This config composes base data-analysis and ANJANA privacy
    ``ScorerConfig`` objects into one ``ScorerDictConfig`` that emits a
    ``ScoreDict`` covering both utility and anonymization signals.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    classifier: bool | None = None
    scorers: dict[str, ScorerConfig] = field(default_factory=dict)

    def _build_default_scorers(self, classifier: bool) -> dict[str, ScorerConfig]:
        from ...score.data import (
            DefaultDataClassificationScorerDictConfig,
            DefaultDataRegressionScorerDictConfig,
        )

        base_scorers = (
            DefaultDataClassificationScorerDictConfig().scorers
            if classifier
            else DefaultDataRegressionScorerDictConfig().scorers
        )
        privacy_scorers = dict(DefaultAnjanaScorerDictConfig().scorers)
        return {**base_scorers, **privacy_scorers}

    def __post_init__(self):
        if not getattr(self, "scorers", None):
            self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


@dataclass(eq=False, kw_only=True)
class DefaultAnjanaModelScorerDictConfig(DefaultAnjanaScorerDictConfig):
    """Model-scope privacy scorer set for ANJANA anonymization analysis.

    This specialization reuses the default ANJANA privacy scorer dict for
    explicit model-scope routing and still emits a ``ScoreDict``.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    pass


safe_store(group="score", name="anjana", node=DefaultAnjanaScorerDictConfig)
safe_store(group="score", name="anjana_data", node=DefaultAnjanaDataScorerDictConfig)
safe_store(
    group="score",
    name="anjana_model",
    node=DefaultAnjanaModelScorerDictConfig,
)
