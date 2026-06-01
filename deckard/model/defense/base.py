# A BaseConfig class for Configuration of Models using adversarial-robustness-toolbox (ART)
# https://adversarial-robustness-toolbox.readthedocs.io/en/latest

import logging
import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Union, cast, ClassVar, Final

from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ...artifacts import ScoreDict
from ...frameworks.types import ArtEsimtator, EstimatorLike, StringifiedClass
from ...data import DataConfig
from ...utils import (
    BaseConfig,
    coerce_config,
    coerce_to_list,
    instantiate_plugin_spec,
    is_null_config_value,
    normalize_plugin_specs,
    resolve_class,
)
from ..canon import defense_stage_priority, resolve_model_defense_stage
from ..base import ModelConfig

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

DefenseScoreValue = str | int | float | bool | None
DefenseInitParamValue = (
    EstimatorLike | tuple[int, ...] | tuple[float, float] | str | int | None
)


CANON_DEFENSE_LIBRARIES: Final[tuple[str, ...]] = (
    "art",
    "fairlearn",
    "anjana",
)

CANON_DEFENSE_PREFIXES: Final[tuple[str, ...]] = tuple(
    f"{lib}." for lib in CANON_DEFENSE_LIBRARIES
)


def _looks_like_defense_class_path(path: Any) -> bool:
    if not isinstance(path, str):
        return False

    lowered = path.strip().lower()
    return bool(lowered) and lowered.startswith(CANON_DEFENSE_PREFIXES)


def _split_defense_name(
    defense_name: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not isinstance(defense_name, str) or defense_name.strip() == "":
        return None, None, None

    try:
        module_name, class_name = defense_name.rsplit(".", 1)
    except ValueError as exc:
        raise ImportError(
            f"Could not parse defense type from defense name {defense_name}",
        ) from exc
    module_parts = module_name.split(".")
    if len(module_parts) < 3:
        raise ImportError(
            f"Could not parse defense type from defense name {defense_name}",
        )

    defense_type = module_parts[2]
    defense_subtype = module_parts[3] if len(module_parts) >= 4 else None
    if defense_type not in supported_defense_types:
        raise AssertionError(
            f"Unsupported defense type: {defense_type}. Supported types are: {supported_defense_types}",
        )
    return defense_type, defense_subtype, class_name


@lru_cache(maxsize=1)
def _get_art_symbols() -> dict[str, Any]:
    from art.estimators.classification import PyTorchClassifier
    from art.estimators.classification.scikitlearn import (
        ScikitlearnAdaBoostClassifier,
        ScikitlearnBaggingClassifier,
        ScikitlearnClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnRandomForestClassifier,
        ScikitlearnSVC,
    )
    from art.estimators.regression import PyTorchRegressor
    from art.estimators.regression.scikitlearn import (
        ScikitlearnDecisionTreeRegressor,
        ScikitlearnRegressor,
    )

    classifier_dict = {
        "SVC": ScikitlearnSVC,
        "LogisticRegression": ScikitlearnLogisticRegression,
        "RandomForestClassifier": ScikitlearnRandomForestClassifier,
        "GradientBoostingClassifier": ScikitlearnGradientBoostingClassifier,
        "ExtraTreesClassifier": ScikitlearnExtraTreesClassifier,
        "AdaBoostClassifier": ScikitlearnAdaBoostClassifier,
        "BaggingClassifier": ScikitlearnBaggingClassifier,
        "DecisionTreeClassifier": ScikitlearnDecisionTreeClassifier,
        "sklearn-classifier": ScikitlearnClassifier,
    }

    regressor_dict = {
        "DecisionTreeRegressor": ScikitlearnDecisionTreeRegressor,
        "sklearn-regressor": ScikitlearnRegressor,
    }

    sklearn_dict = {**classifier_dict, **regressor_dict}
    return {
        "classifier_dict": classifier_dict,
        "regressor_dict": regressor_dict,
        "sklearn_dict": sklearn_dict,
        "sklearn_models": list(sklearn_dict.keys()),
        "torch_wrapper_types": (PyTorchClassifier, PyTorchRegressor),
        "torch_classifier": PyTorchClassifier,
        "torch_regressor": PyTorchRegressor,
    }


def _is_art_torch_wrapper(model_obj: Any) -> bool:
    try:
        torch_wrapper_types = _get_art_symbols()["torch_wrapper_types"]
    except Exception:
        return False
    return isinstance(model_obj, torch_wrapper_types)


def _is_art_wrapper_instance(model_obj: Any) -> bool:
    if model_obj is None:
        return False
    try:
        art_symbols = _get_art_symbols()
    except Exception:
        return False
    sklearn_wrapper_types = tuple(art_symbols["sklearn_dict"].values())
    torch_wrapper_types = art_symbols["torch_wrapper_types"]
    return isinstance(model_obj, sklearn_wrapper_types + torch_wrapper_types)


def _get_wrapper_state(model_obj: Any) -> dict[str, Any] | None:
    state = getattr(model_obj, "_deckard_art_wrapper_state", None)
    if not isinstance(state, dict):
        return None
    if state.get("wrapped_by_deckard") is not True:
        return None
    return state


supported_defense_types = [
    "detector",
    "preprocessor",
    "postprocessor",
    "trainer",
    "regularizer",
    "transformer",
    None,
]


@dataclass(eq=False)
class DefenseStep:
    """One defense-chain step with explicit fit/predict application flags.

    Attributes:
        name: Canonical defense class path.
        defense_params: Defense parameters passed during class instantiation.
        apply_fit: Whether or not to apply before training (might trigger a retrain).
        apply_predict: Whether or not to apply during prediction.
    """

    name: str | None
    defense_params: dict[str, Any]
    apply_fit: bool
    apply_predict: bool
    _runtime_defense: Any = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        metadata={"help": "Runtime defense instance cache for this step."},
    )

    def __post_init__(self) -> None:
        if isinstance(self.name, str) and self.name.strip() == "":
            self.name = None
        if not isinstance(self.defense_params, dict):
            self.defense_params = dict(self.defense_params or {})

    def _get_runtime_defense(self) -> Any:
        runtime_defense = object.__getattribute__(self, "_runtime_defense")
        if runtime_defense is not None:
            return runtime_defense
        if not isinstance(self.name, str) or self.name.strip() == "":
            return None
        runtime_defense = resolve_class(self.name)(**dict(self.defense_params or {}))
        object.__setattr__(self, "_runtime_defense", runtime_defense)
        return runtime_defense

    @staticmethod
    def _derive_default_flags(defense_obj: Any) -> tuple[bool, bool]:
        defense_name = str(getattr(defense_obj, "name", None) or "").lower()
        if defense_name.startswith("fairlearn.") or defense_name.startswith("anjana."):
            return True, True
        return True, True

    @classmethod
    def from_defense(
        cls,
        defense_obj: Any,
        *,
        apply_fit: bool | None = None,
        apply_predict: bool | None = None,
    ) -> "DefenseStep":
        """Build a defense step from a raw defense object.

        Args:
                defense_obj: Concrete defense object.
                apply_fit: Optional fit-time application override.
                apply_predict: Optional predict-time application override.

        Returns:
                Defense step with resolved application flags.
        """
        default_fit, default_predict = cls._derive_default_flags(defense_obj)
        resolved_apply_fit = default_fit if apply_fit is None else bool(apply_fit)
        resolved_apply_predict = (
            default_predict if apply_predict is None else bool(apply_predict)
        )

        defense_name = str(getattr(defense_obj, "name", None) or "").lower()
        if (
            defense_name.startswith("art.")
            and apply_fit is None
            and apply_predict is None
        ):
            logger.warning(
                "ART defense step '%s' is missing explicit apply_fit/apply_predict flags; "
                "defaulting to apply_fit=True, apply_predict=True.",
                getattr(defense_obj, "name", None) or type(defense_obj).__name__,
            )

        step = cls(
            name=getattr(defense_obj, "name", None),
            defense_params=dict(getattr(defense_obj, "defense_params", {}) or {}),
            apply_fit=resolved_apply_fit,
            apply_predict=resolved_apply_predict,
        )
        object.__setattr__(step, "_runtime_defense", defense_obj)
        return step

    def __getattr__(self, name: str) -> Any:
        runtime_defense = self._get_runtime_defense()
        if runtime_defense is None:
            raise AttributeError(name)
        return getattr(runtime_defense, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "name",
            "defense_params",
            "apply_fit",
            "apply_predict",
            "_runtime_defense",
        }:
            object.__setattr__(self, name, value)
            return
        runtime_defense = self._get_runtime_defense()
        if runtime_defense is None:
            object.__setattr__(self, name, value)
            return
        setattr(runtime_defense, name, value)


class DefenseHookRuntimeMixin:
    """Shared plugin-hook runtime behavior for defense pipeline and defense configs.

    This mixin centralizes plugin instantiation, hook dispatch, and score-dict merge
    behavior so all defense runtime owners expose consistent hook semantics.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    plugins: list
    _plugin_objects: Union[list, None]
    score_dict: ScoreDict

    def _instantiate_plugin(self, plugin_spec: Any):
        """Instantiate one runtime plugin from a normalized plugin spec."""

        def _resolve_and_instantiate(path: str, **kwargs):
            return resolve_class(path)(**kwargs)

        return instantiate_plugin_spec(
            plugin_spec,
            loader=_resolve_and_instantiate,
        )

    def _get_plugins(self) -> list:
        """Lazily instantiate and cache runtime plugins for defense dispatch."""
        if getattr(self, "_plugin_objects", None) is None:
            plugin_specs = normalize_plugin_specs(getattr(self, "plugins", []))
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return list(self._plugin_objects or [])

    def _run_plugin_hook(self, hook_name: str, **kwargs) -> list[Any]:
        """Execute one plugin hook across all instantiated plugins."""
        hook_outputs = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_outputs.append(hook(self, **kwargs))
        return hook_outputs

    def _merge_plugin_scores(self, hook_outputs):
        """Merge dictionary hook outputs into the runtime ``score_dict``."""
        if self.score_dict is None:
            self.score_dict = ScoreDict()
        for output in hook_outputs:
            if isinstance(output, dict):
                self.score_dict.update(output)

    def _finalize_defense_application(
        self,
        *,
        defended_estimator,
        defense_signature=None,
        defense=None,
        apply_fit: bool = True,
        elapsed: float | None = None,
    ) -> None:
        """Persist runtime defense application state on the active config."""
        if elapsed is not None:
            self.defense_application_time = float(elapsed)
        if defense_signature is not None and defended_estimator is not None:
            mark_signature = getattr(
                self,
                "_mark_applied_defense_signature",
                None,
            )
            if callable(mark_signature):
                mark_signature(defended_estimator, defense_signature)
        if defense is not None:
            self._last_applied_defense = defense
        self._last_defense_apply_fit = bool(apply_fit)


class DefensePipelineConfigBehaviorMixin(DefenseHookRuntimeMixin):
    """Pipeline-owner behavior for composing, normalizing, and applying defense chains.

    This mixin owns pipeline-level orchestration semantics: chain coercion,
    stage resolution, and ordered application of multi-step defenses.

    Note:
        Pipeline hook names include ``resolve_defense_stage``,
        ``before_apply_defense``, ``before_apply_defense_step``,
        ``after_apply_defense_step``, and ``after_apply_defense``.

        Additional runtime-dispatch hooks include
        ``resolve_defense_mixins``, ``resolve_defense_handler``,
        ``before_defense_dispatch``, and ``after_defense_dispatch``.
        Dictionary outputs from post hooks are merged into ``score_dict``.

    Attributes:
        defenses: Ordered defense chain to apply to runtime estimators.
        plugins: Runtime defense plugins used for hook-based extension.
        score_dict: Runtime score payload merged from defense hooks.
    """

    defenses: list
    plugins: list
    score_dict: ScoreDict
    _plugin_objects: Union[list, None]

    @classmethod
    def _is_pipeline_target(cls, target: Any) -> bool:
        if not isinstance(target, str):
            return False
        return target.rsplit(".", 1)[-1] == cls.__name__

    @classmethod
    def _looks_like_single_defense_spec(
        cls,
        defense_spec: dict[str, Any],
    ) -> bool:
        _ = (cls, defense_spec)
        return False

    @classmethod
    def coerce(
        cls,
        defense_config: "DefenseConfig | dict | list | tuple | None",
    ) -> "DefenseConfig | None":
        """Normalize user defense config input into ``DefenseConfig``.

        Args:
                defense_config: Raw defense config payload.

        Returns:
                Normalized pipeline config payload.

        Raises:
                TypeError: If input payload cannot be interpreted as defense config.
        """
        if defense_config is None or isinstance(defense_config, cls):
            return cast("DefenseConfig | None", defense_config)
        if hasattr(defense_config, "apply_to"):
            return DefenseConfig(
                defenses=[DefenseStep.from_defense(defense_config)],
            )
        defense_config = coerce_config(defense_config)
        if isinstance(defense_config, (list, ListConfig)):
            return DefenseConfig(defenses=coerce_to_list(defense_config))
        if isinstance(defense_config, dict):
            defense_dict = cast(dict[str, Any], dict(defense_config))
            target = defense_dict.pop("_target_", None)
            if target is not None:
                if cls._is_pipeline_target(target):
                    return DefenseConfig(**defense_dict)
                return DefenseConfig(
                    defenses=[
                        DefenseStep.from_defense(resolve_class(target)(**defense_dict)),
                    ],
                )
            if "defenses" in defense_dict:
                return DefenseConfig(**defense_dict)
            if cls._looks_like_single_defense_spec(defense_dict):
                return DefenseConfig(defenses=[defense_dict])
        raise TypeError(
            "Defense config must be a DefenseConfig, a single defense spec, or None",
        )

    def _coerce_single_defense(self, defense_obj):
        if isinstance(defense_obj, DefenseStep):
            return defense_obj
        if hasattr(defense_obj, "apply_to"):
            return DefenseStep.from_defense(defense_obj)

        defense_obj = coerce_config(defense_obj)

        if isinstance(defense_obj, dict):
            defense_dict = cast(dict[str, Any], dict(defense_obj))
            apply_fit = defense_dict.pop("apply_fit", None)
            apply_predict = defense_dict.pop("apply_predict", None)
            raw_params = defense_dict.get("defense_params", None)
            if isinstance(raw_params, dict):
                if apply_fit is None and "apply_fit" in raw_params:
                    apply_fit = raw_params.get("apply_fit")
                if apply_predict is None and "apply_predict" in raw_params:
                    apply_predict = raw_params.get("apply_predict")
            nested_defense = defense_dict.pop("defense", None)
            if nested_defense is not None:
                step_defense = self._coerce_single_defense(nested_defense)
                if apply_fit is not None:
                    setattr(step_defense, "apply_fit", bool(apply_fit))
                if apply_predict is not None:
                    setattr(step_defense, "apply_predict", bool(apply_predict))
                return step_defense
            target = defense_dict.pop("_target_", None)
            if target is not None:
                defense_instance = resolve_class(target)(**defense_dict)
                step = DefenseStep.from_defense(
                    defense_instance,
                    apply_fit=apply_fit,
                    apply_predict=apply_predict,
                )
                if not isinstance(step.name, str) or step.name.strip() == "":
                    step.name = str(defense_dict.get("name", None) or target)
                return step

            if "defense_name" in defense_dict:
                raise ValueError(
                    "defense_name is no longer supported in defense specs. Use 'name' for defense class path.",
                )

            raise TypeError(
                "Defense specs must use an explicit _target_ or wrap a concrete defense object.",
            )

        raise TypeError(
            f"Unsupported defense specification in pipeline: {type(defense_obj)}",
        )

    def _inherit_model_context(self, defense_obj, estimator) -> None:
        base_estimator = getattr(estimator, "model", estimator)

        blank_values = {None, "", "None", "null", "Null", "NULL"}

        if (
            hasattr(defense_obj, "name")
            and getattr(
                defense_obj,
                "name",
                None,
            )
            in blank_values
        ):
            defense_obj.name = (
                f"{base_estimator.__class__.__module__}."
                f"{base_estimator.__class__.__name__}"
            )

        if (
            hasattr(defense_obj, "classifier")
            and getattr(
                defense_obj,
                "classifier",
                None,
            )
            is None
        ):
            estimator_type = getattr(base_estimator, "_estimator_type", None)
            if estimator_type == "classifier":
                defense_obj.classifier = True
            elif estimator_type == "regressor":
                defense_obj.classifier = False

        if (
            hasattr(defense_obj, "model_params")
            and not getattr(
                defense_obj,
                "model_params",
                None,
            )
            and hasattr(base_estimator, "get_params")
        ):
            defense_obj.model_params = base_estimator.get_params()

        if hasattr(defense_obj, "probability") and hasattr(
            base_estimator,
            "predict_proba",
        ):
            defense_obj.probability = True

    def normalize_defenses(
        self,
        defenses: list | tuple | dict | BaseConfig | None,
    ) -> list[DefenseStep]:
        """Return a normalized list of defense objects from raw defense specs.

        Args:
                defenses: Raw defense payload(s).

        Returns:
                Normalized defense object list.
        """
        if defenses is None:
            return []
        if isinstance(defenses, (tuple, list)):
            defense_list = list(defenses)
        else:
            defense_list = [defenses]
        return [self._coerce_single_defense(item) for item in defense_list]

    def requires_fit_application(self) -> bool:
        """Return ``True`` when any defense step requests fit-time application.

        Returns:
                Whether at least one configured defense applies during fit.
        """
        return any(
            bool(getattr(step, "apply_fit", True))
            for step in self.normalize_defenses(getattr(self, "defenses", []))
        )

    @staticmethod
    def _unwrap_defense(defense_obj: Any) -> Any:
        if isinstance(defense_obj, DefenseStep):
            return defense_obj._get_runtime_defense()
        return defense_obj

    def resolve_stage(
        self,
        default_stage: str = "post_fit_pre_predict",
        **context: Any,
    ) -> str:
        """Resolve defense stage using plugin hooks and fallback defaults.

        Args:
                default_stage: Fallback stage when plugins do not override.
                **context: Optional runtime context for plugin stage resolution.

        Returns:
                Resolved defense stage token.
        """
        stage = default_stage
        hook_outputs = self._run_plugin_hook(
            "resolve_defense_stage",
            default_stage=default_stage,
            current_stage=stage,
            **context,
        )
        for output in hook_outputs:
            if isinstance(output, str) and output.strip():
                stage = output.strip()
            elif isinstance(output, dict):
                candidate = output.get(
                    "defense_stage",
                    output.get("stage", None),
                )
                if isinstance(candidate, str) and candidate.strip():
                    stage = candidate.strip()
        return stage

    def _is_art_defense(self, defense_obj) -> bool:
        """Return True if defense_obj is an ART defense (wraps model, must be last)."""
        defense_obj = self._unwrap_defense(defense_obj)
        defense_name = getattr(defense_obj, "name", None)
        if isinstance(defense_name, str) and defense_name.startswith("art."):
            return True
        try:
            from ...plugins.fairlearn.model import FairlearnDefenseConfig

            if isinstance(defense_obj, FairlearnDefenseConfig):
                return False
        except Exception:
            pass
        if isinstance(defense_obj, DefenseConfig):
            return True
        return False

    def _is_model_wrapper_defense(self, defense_obj) -> bool:
        """Return True for defenses that wrap/transform estimators rather than raw data."""
        defense_obj = self._unwrap_defense(defense_obj)
        if self._is_art_defense(defense_obj):
            return True
        try:
            from ...plugins.fairlearn.model import FairlearnDefenseConfig

            if isinstance(defense_obj, FairlearnDefenseConfig):
                return True
        except Exception:
            pass
        return False

    def _is_retraining_defense(self, defense_obj) -> bool:
        defense_obj = self._unwrap_defense(defense_obj)
        defense_name = getattr(defense_obj, "name", None)
        if not isinstance(defense_name, str):
            return False
        lowered = defense_name.lower()
        return ".trainer." in lowered and (
            "adversarialtrainer" in lowered
            or "retraining" in lowered
            or "madry" in lowered
        )

    def apply(
        self,
        estimator: BaseEstimator,
        data: DataConfig | None,
        stage: str = "post_fit_pre_predict",
    ) -> BaseEstimator:
        """Apply configured defense chain to ``estimator`` for the given stage.

        Args:
                estimator: Base estimator to transform.
                data: Runtime data context.
                stage: Defense stage token.

        Returns:
                Defended estimator.

        Raises:
                ValueError: If estimator is missing.
                TypeError: If a configured defense does not expose ``apply_to``.
        """
        if estimator is None:
            raise ValueError(
                "estimator must be provided before applying defenses",
            )
        defense_chain = self.normalize_defenses(self.defenses)
        if len(defense_chain) == 0:
            return estimator

        staged_chain = []
        for step in defense_chain:
            defense_name = step.name if isinstance(step, DefenseStep) else None
            resolved_stage = resolve_model_defense_stage(
                defense_name,
                default_stage=stage,
            )
            staged_chain.append((resolved_stage, step))

        sorted_chain = sorted(
            staged_chain,
            key=lambda item: defense_stage_priority(item[0]),
        )
        if [obj for _, obj in sorted_chain] != defense_chain:
            logger.warning(
                "Defense chain stage order adjusted to canonical model defense staging. "
                "Original=%s Reordered=%s",
                [
                    (d.name if isinstance(d, DefenseStep) else None)
                    or type(d).__name__
                    for d in defense_chain
                ],
                [
                    f"{(d.name if isinstance(d, DefenseStep) else None) or type(d).__name__}@{s}"
                    for s, d in sorted_chain
                ],
            )
            defense_chain = [obj for _, obj in sorted_chain]

        wrapper_defenses = [
            d for d in defense_chain if self._is_model_wrapper_defense(d)
        ]
        data_defenses = [
            d for d in defense_chain if not self._is_model_wrapper_defense(d)
        ]
        if wrapper_defenses and data_defenses:
            first_wrapper_idx = next(
                i
                for i, d in enumerate(defense_chain)
                if self._is_model_wrapper_defense(d)
            )
            last_data_idx = max(
                i
                for i, d in enumerate(defense_chain)
                if not self._is_model_wrapper_defense(d)
            )
            if first_wrapper_idx < last_data_idx:
                logger.warning(
                    "Defense chain contains model-wrapper defenses (ART/fairlearn) before "
                    "data-transform defenses. Wrapper defenses only transform the estimator, "
                    "so they are automatically reordered to run last. "
                    "Data defenses: %s; wrapper defenses: %s.",
                    [
                        (d.name if isinstance(d, DefenseStep) else None)
                        or type(d).__name__
                        for d in data_defenses
                    ],
                    [
                        (d.name if isinstance(d, DefenseStep) else None)
                        or type(d).__name__
                        for d in wrapper_defenses
                    ],
                )
                defense_chain = data_defenses + wrapper_defenses

        retraining_defenses = [
            d for d in defense_chain if self._is_retraining_defense(d)
        ]
        if retraining_defenses:
            non_retraining_defenses = [
                d for d in defense_chain if not self._is_retraining_defense(d)
            ]
            first_retraining_idx = next(
                i
                for i, d in enumerate(defense_chain)
                if self._is_retraining_defense(d)
            )
            last_non_retraining_idx = max(
                (
                    i
                    for i, d in enumerate(defense_chain)
                    if not self._is_retraining_defense(d)
                ),
                default=-1,
            )
            if first_retraining_idx < last_non_retraining_idx:
                warning_msg = (
                    "Adversarial retraining defenses must run last in the defense chain. "
                    "deckard will automatically move retraining defenses to the end."
                )
                logger.warning(warning_msg)
                warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
                defense_chain = non_retraining_defenses + retraining_defenses

        self._run_plugin_hook(
            "before_apply_defense",
            estimator=estimator,
            data=data,
            stage=stage,
            defense_chain=defense_chain,
        )

        defended_estimator = estimator
        total_defense_time = 0.0
        applied_defenses = []
        for defense_step in defense_chain:
            defense_obj = self._unwrap_defense(defense_step)
            stage_token = str(stage).strip().lower()
            if stage_token in {"pre_fit", "pre_art_defense"}:
                should_apply = bool(getattr(defense_step, "apply_fit", True))
            else:
                should_apply = bool(getattr(defense_step, "apply_predict", True))
            if not should_apply:
                continue
            self._inherit_model_context(defense_obj, defended_estimator)
            if (
                hasattr(defense_obj, "data")
                and getattr(defense_obj, "data", None) is None
            ):
                setattr(defense_obj, "data", data)
            apply_to = getattr(defense_obj, "apply_to", None)
            if not callable(apply_to):
                raise TypeError(
                    "Configured defenses must implement apply_to(estimator, data)",
                )
            self._run_plugin_hook(
                "before_apply_defense_step",
                estimator=defended_estimator,
                data=data,
                stage=stage,
                defense=defense_step,
                applied_defenses=applied_defenses,
            )
            started = time.perf_counter()
            defended_estimator = apply_to(
                estimator=defended_estimator,
                data=data,
            )
            elapsed = getattr(defense_obj, "defense_application_time", None)
            if elapsed is None:
                elapsed = time.perf_counter() - started
            total_defense_time += float(elapsed)
            applied_defenses.append(defense_step)
            step_outputs = self._run_plugin_hook(
                "after_apply_defense_step",
                estimator=defended_estimator,
                data=data,
                stage=stage,
                defense=defense_step,
                applied_defenses=applied_defenses,
                step_defense_time=float(elapsed),
            )
            self._merge_plugin_scores(step_outputs)

        self.defense_application_time = total_defense_time
        hook_outputs = self._run_plugin_hook(
            "after_apply_defense",
            estimator=defended_estimator,
            data=data,
            stage=stage,
            defense_chain=defense_chain,
            applied_defenses=applied_defenses,
            applied_defense_types=[
                type(self._unwrap_defense(d)).__name__ for d in applied_defenses
            ],
            defense_application_time=total_defense_time,
        )
        self._merge_plugin_scores(hook_outputs)
        return cast(BaseEstimator, defended_estimator)

    def apply_to(
        self,
        estimator: BaseEstimator,
        data: DataConfig | None,
        stage: str = "post_fit_pre_predict",
    ) -> BaseEstimator:
        """Apply the configured defense chain to an estimator runtime payload.

        Args:
                estimator: Base estimator to transform.
                data: Runtime data context.
                stage: Defense stage token.

        Returns:
                Defended estimator.
        """
        return self.apply(estimator=estimator, data=data, stage=stage)

    def apply_defense(
        self,
        estimator: BaseEstimator,
        data: DataConfig | None,
        stage: str = "post_fit_pre_predict",
    ) -> BaseEstimator:
        """Canonical public runtime alias for applying defense chains.

        Args:
                estimator: Base estimator to transform.
                data: Runtime data context.
                stage: Defense stage token.

        Returns:
                Defended estimator.
        """
        return self.apply(estimator=estimator, data=data, stage=stage)


def _is_torch_model_instance(model_obj) -> bool:
    try:
        import torch
    except Exception:  # pragma: no cover - optional dependency import may fail at runtime
        return False
    return isinstance(model_obj, torch.nn.Module)


@dataclass(eq=True)
class DefenseMixin:
    """Base callable defense handler used by runtime defense context resolution.

    The ``runtime`` attribute is the active defense config instance owned by
    defense orchestration. Attribute access is delegated to that runtime object.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    runtime: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "runtime":
            object.__setattr__(self, name, value)
            return
        runtime = object.__getattribute__(self, "runtime")
        if runtime is None:
            object.__setattr__(self, name, value)
            return
        setattr(runtime, name, value)

    def __call__(
        self,
        *,
        data: DataConfig | None,
        defense_type: StringifiedClass | None,
        defense_subtype: Union[str, None],
        defense_class: type | None,
        art_class: ArtEsimtator,
        init_params: dict,
        base_estimator: EstimatorLike,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Execute one defense handler.

        Args:
                data: Data runtime containing train/test/val splits.
                defense_type: Parsed defense family.
                defense_subtype: Parsed defense subtype.
                defense_class: Concrete defense class resolved from ``defense_name``.
                art_class: ART estimator wrapper class selected for model type.
                init_params: Runtime ART estimator initialization kwargs.
                base_estimator: Unwrapped model estimator used as defense target.
                existing_preprocessors: Existing preprocessor defenses already attached.
                existing_postprocessors: Existing postprocessor defenses already attached.

        Returns:
                Defense artifact and defended estimator.

        Raises:
                NotImplementedError: Always raised by the base defense mixin.
        """
        raise NotImplementedError("Defense handlers must implement __call__")

    def defend(
        self,
        *,
        data: DataConfig | None,
        defense_type: StringifiedClass | None,
        defense_subtype: Union[str, None],
        defense_class: type | None,
        art_class: ArtEsimtator,
        init_params: dict,
        base_estimator: EstimatorLike,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Public verb-form alias for applying a defense handler.

        Args:
                data: Data runtime containing train/test/val splits.
                defense_type: Parsed defense family.
                defense_subtype: Parsed defense subtype.
                defense_class: Concrete defense class resolved from ``defense_name``.
                art_class: ART estimator wrapper class selected for model type.
                init_params: Runtime ART estimator initialization kwargs.
                base_estimator: Unwrapped model estimator used as defense target.
                existing_preprocessors: Existing preprocessor defenses already attached.
                existing_postprocessors: Existing postprocessor defenses already attached.

        Returns:
                Defense artifact and defended estimator.
        """
        return self(
            data=data,
            defense_type=defense_type,
            defense_subtype=defense_subtype,
            defense_class=defense_class,
            art_class=art_class,
            init_params=init_params,
            base_estimator=base_estimator,
            existing_preprocessors=existing_preprocessors,
            existing_postprocessors=existing_postprocessors,
        )


class PassthroughDefenseMixin(DefenseMixin):
    """Default handler for no-op/passthrough ART wrapping.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def __call__(
        self,
        *,
        data: DataConfig | None,
        defense_type: StringifiedClass | None,
        defense_subtype: Union[str, None],
        defense_class: type | None,
        art_class: ArtEsimtator,
        init_params: dict,
        base_estimator: EstimatorLike,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Build and return an ART wrapper without adding defense transforms.

        Args:
                data: Data runtime containing train/test/val splits.
                defense_type: Parsed defense family.
                defense_subtype: Parsed defense subtype.
                defense_class: Concrete defense class resolved from ``defense_name``.
                art_class: ART estimator wrapper class selected for model type.
                init_params: Runtime ART estimator initialization kwargs.
                base_estimator: Unwrapped model estimator used as defense target.
                existing_preprocessors: Existing preprocessor defenses already attached.
                existing_postprocessors: Existing postprocessor defenses already attached.

        Returns:
                ``None`` and the wrapped estimator.
        """
        defended_estimator = self._build_art_wrapper(
            art_class=art_class,
            base_estimator=base_estimator,
            init_params={**self.defense_params, **init_params},
            preprocessing_defences=existing_preprocessors,
            postprocessing_defences=existing_postprocessors,
        )
        return None, defended_estimator


@dataclass(init=False, eq=False)
class ARTDefenseBehaviorMixin(DefenseHookRuntimeMixin):
    """Single-defense dispatch behavior mixed into concrete defense config dataclasses.

    This mixin owns per-defense runtime dispatch semantics: resolving handler/mixins,
    building ART-compatible wrappers, and applying a single defense spec.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    name: StringifiedClass | None = None
    classifier: Union[bool, str, None] = True
    model_params: dict = field(
        default_factory=dict,
        metadata={"help": "Model construction parameters for defense application."},
    )
    probability: bool = False
    alias: str = ""
    defense_name: Union[str, None] = None
    defense_params: dict = field(
        default_factory=dict,
        metadata={"help": "Defense constructor parameters."},
    )
    model_name: StringifiedClass | None = None
    _model: Union[BaseEstimator, None] = field(default=None, init=False, repr=False, compare=False, metadata={'help': 'Configuration field: _model.'})
    score_dict: ScoreDict = field(default_factory=ScoreDict, init=False, repr=False, metadata={'help': 'Configuration field: score_dict.'})
    _target_: Union[str, None] = field(default='deckard.model.defense.base.ARTDefenseBehaviorMixin', init=True, repr=True, metadata={'help': 'Hydra target path used when this defense behavior mixin is serialized through a concrete config.'})
    _model_config: Union[ModelConfig, None] = field(default=None, init=False, repr=False, compare=False, metadata={'help': 'Configuration field: _model_config.'})
    plugins: list = field(default_factory=list, repr=True, metadata={'help': 'Configuration field: plugins.'})
    _plugin_objects: Union[list, None] = field(default=None, init=False, repr=False, compare=False, metadata={'help': 'Configuration field: _plugin_objects.'})

    _BUILTIN_DEFENSE_HANDLER_TYPES: ClassVar[dict[str, str]] = {
        "detector": "deckard.model.defense.detector.DetectorDefenseConfig",
        "preprocessor": "deckard.model.defense.preprocessor.PreprocessorDefenseConfig",
        "postprocessor": "deckard.model.defense.postprocessor.PostprocessorDefenseConfig",
        "trainer": "deckard.model.defense.trainer.TrainerDefenseConfig",
        "transformer": "deckard.model.defense.transformer.TransformerDefenseConfig",
        "regularizer": "deckard.model.defense.regularizer.RegularizerDefenseConfig",
    }

    def _instantiate_runtime_handler(self, handler_type: type):
        """Instantiate a handler config class with runtime context copied from self."""
        handler = handler_type()
        shared_runtime_attrs = (
            "name",
            "model_name",
            "classifier",
            "model_params",
            "probability",
            "clip_values",
            "defense_name",
            "defense_params",
            "alias",
            "plugins",
            "_model",
            "model_config",
            "_target_",
            "_plugin_objects",
            "score_dict",
            "_apply_fit",
        )
        for attr in shared_runtime_attrs:
            if hasattr(self, attr):
                setattr(handler, attr, getattr(self, attr))
        return handler

    def _resolve_runtime_defense_mixins(
        self,
        defense_type: StringifiedClass | None,
        defense_subtype: Union[str, None],
    ) -> tuple[type, ...]:
        mixins: list[type] = []
        dtype = (defense_type or "").lower() if defense_type is not None else None
        if dtype is None:
            mixins.append(type(self))
        elif dtype in self._BUILTIN_DEFENSE_HANDLER_TYPES:
            handler_type = resolve_class(self._BUILTIN_DEFENSE_HANDLER_TYPES[dtype])
            if isinstance(handler_type, type):
                mixins.append(handler_type)

        plugin_outputs = self._run_plugin_hook(
            "resolve_defense_mixins",
            defense_type=defense_type,
            defense_subtype=defense_subtype,
            default_mixins=tuple(mixins),
        )
        for output in plugin_outputs:
            if isinstance(output, type):
                mixins.append(output)
            elif isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, type):
                        mixins.append(item)

        deduped: list[type] = []
        for mixin in mixins:
            if mixin not in deduped:
                deduped.append(mixin)
        return tuple(deduped)

    def _resolve_defense_handler(
        self,
        defense_type: StringifiedClass | None,
        defense_subtype: Union[str, None],
    ):
        mixins = self._resolve_runtime_defense_mixins(defense_type, defense_subtype)
        default_handler = None
        for mixin in mixins:
            if isinstance(mixin, type) and mixin in type(self).mro():
                default_handler = self
                break
            if isinstance(mixin, type) and issubclass(mixin, BaseConfig):
                default_handler = self._instantiate_runtime_handler(mixin)
                break

        hook_outputs = self._run_plugin_hook(
            "resolve_defense_handler",
            defense_type=defense_type,
            defense_subtype=defense_subtype,
            default_handler=default_handler,
            default_mixins=mixins,
        )
        for output in hook_outputs:
            if callable(output):
                return output
            if isinstance(output, type) and issubclass(output, BaseConfig):
                return self._instantiate_runtime_handler(output)
            if isinstance(output, type) and issubclass(output, DefenseMixin):
                return output(runtime=self)

        return default_handler

    def _get_model_config(self) -> ModelConfig:
        if getattr(self, "model_config", None) is None:
            self.model_config = ModelConfig(
                name=self.name,
                classifier=cast(
                    bool | str,
                    self.classifier if self.classifier is not None else True,
                ),
                model_params=self.model_params,
                probability=self.probability,
                alias=self.alias,
            )
            self.model_config.defense = None
        assert self.model_config is not None
        return self.model_config

    def __post_init__(self):
        if not is_null_config_value(
            getattr(self, "defense_name", None),
            allow_empty=True,
        ):
            raise ValueError(
                "defense_name is no longer supported. Use 'name' for defense class path.",
            )

        if _looks_like_defense_class_path(getattr(self, "name", None)):
            self.defense_name = str(self.name)
            self.name = self.model_name
        else:
            self.defense_name = None

        if not is_null_config_value(getattr(self, "name", None), allow_empty=True):
            model_cfg = self._get_model_config()
            self.classifier = model_cfg.classifier
            self.model_params = model_cfg.model_params
            self.model = model_cfg._model
        elif not hasattr(self, "_model"):
            self.model = None

        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = ScoreDict()
        if not hasattr(self, "_target_") or self._target_ in {None, ""}:
            cls = type(self)
            self._target_ = f"{cls.__module__}.{cls.__name__}"

        self.defense_training_time = None
        self.defense_application_time = None
        self.defense_prediction_time = None
        self.defense_scoring_time = None
        self.defense_params = getattr(self, "defense_params", None) or {}
        self._apply_fit = True
        self.plugins = normalize_plugin_specs(getattr(self, "plugins", []))
        try:
            BaseConfig._after_post_init(cast(Any, self))
        except Exception:
            pass

    def parse_defense_name(
        self,
    ) -> tuple[str | None, str | None, type | None]:
        """Parse the configured defense name into type, subtype, and class.

        Returns:
            Defense family token, subtype token, and resolved defense class.

        Raises:
            ImportError: If the configured defense class cannot be imported.
        """
        candidate_name = getattr(self, "defense_name", None)
        if not is_null_config_value(candidate_name, allow_empty=True):
            defense_type, defense_subtype, class_name = _split_defense_name(
                str(candidate_name),
            )
            try:
                defense_class = resolve_class(str(candidate_name))
            except Exception as exc:
                raise ImportError(
                    f"Could not import defense class from defense name {candidate_name}",
                ) from exc
            return defense_type, defense_subtype, defense_class

        return None, None, None

    def __call__(
        self,
        *,
        data: DataConfig | None,
        defense_type: StringifiedClass | None,
        defense_subtype: Union[str, None],
        defense_class: type | None,
        art_class: ArtEsimtator,
        init_params: dict,
        base_estimator: EstimatorLike,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Abstract single-defense dispatch contract.

        Args:
            data: Optional runtime data config.
            defense_type: Canonical defense family token.
            defense_subtype: Canonical defense subtype token.
            defense_class: Resolved defense class.
            art_class: ART wrapper class for defense execution.
            init_params: ART wrapper initialization kwargs.
            base_estimator: Base estimator to defend.
            existing_preprocessors: Existing preprocessor defenses.
            existing_postprocessors: Existing postprocessor defenses.

        Returns:
            Tuple of configured defense object and defended estimator.

        Raises:
            NotImplementedError: Subclasses must implement runtime defense dispatch.
        """
        raise NotImplementedError("Defense handlers must implement __call__")

    def __hash__(self) -> int:
        return super().__hash__()

    def _freeze_defense_value(self, value):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (key, self._freeze_defense_value(val))
                    for key, val in value.items()
                ),
            )
        if isinstance(value, (list, tuple, ListConfig)):
            return tuple(self._freeze_defense_value(item) for item in value)
        return value

    def _defense_signature(self):
        return (
            getattr(self, "defense_name", None),
            self._freeze_defense_value(
                getattr(self, "defense_params", {}) or {},
            ),
        )

    def _get_applied_defense_signatures(self, estimator) -> list:
        return list(
            getattr(estimator, "_deckard_applied_defense_signatures", []) or [],
        )

    def _mark_applied_defense_signature(self, estimator, signature) -> None:
        existing = self._get_applied_defense_signatures(estimator)
        if signature not in existing:
            existing.append(signature)
        setattr(estimator, "_deckard_applied_defense_signatures", existing)

    def _extract_art_wrapper_context(self, art_class, init_params):
        base_estimator = self.model
        art_params = dict(init_params or {})
        existing_preprocessors = []
        existing_postprocessors = []
        if _is_art_wrapper_instance(self.model):
            wrapper_state = _get_wrapper_state(self.model)
            state_base = (
                None if wrapper_state is None else wrapper_state.get("base_estimator")
            )
            if state_base is not None:
                base_estimator = state_base
            else:
                wrapped_model = getattr(self.model, "model", None)
                if wrapped_model is not None:
                    base_estimator = wrapped_model
            art_class = self.model.__class__
            raw_preprocessors = getattr(self.model, "preprocessing_defences", None)
            raw_postprocessors = getattr(self.model, "postprocessing_defences", None)
            if isinstance(raw_preprocessors, (list, tuple)):
                existing_preprocessors = list(raw_preprocessors)
            if isinstance(raw_postprocessors, (list, tuple)):
                existing_postprocessors = list(raw_postprocessors)
            art_params["preprocessing"] = getattr(
                self.model,
                "preprocessing",
                art_params.get("preprocessing"),
            )
            clip_values = getattr(self.model, "clip_values", None)
            if clip_values is not None:
                art_params["clip_values"] = clip_values
        return (
            base_estimator,
            art_class,
            art_params,
            existing_preprocessors,
            existing_postprocessors,
        )

    def _build_art_wrapper(
        self,
        art_class,
        base_estimator,
        init_params,
        preprocessing_defences,
        postprocessing_defences,
    ):
        art_params = dict(init_params or {})
        art_params["preprocessing_defences"] = preprocessing_defences or None
        art_params["postprocessing_defences"] = postprocessing_defences or None
        wrapped_estimator = art_class(base_estimator, **art_params)
        setattr(
            wrapped_estimator,
            "_deckard_art_wrapper_state",
            {
                "wrapped_by_deckard": True,
                "base_estimator": base_estimator,
                "wrapper_type": type(wrapped_estimator).__name__,
            },
        )
        return wrapped_estimator

    @property
    def model(self) -> BaseEstimator | None:
        """Public accessor for the runtime estimator payload.

        Returns:
                Runtime estimator payload.
        """
        return getattr(self, "_model", None)

    @model.setter
    def model(self, value: BaseEstimator | None) -> None:
        """Set the runtime estimator payload.

        Args:
                value: Estimator payload.
        """
        self._model = value

    @property
    def model_config(self) -> ModelConfig | None:
        """Public accessor for the lazily built model config shell.

        Returns:
                Lazily built model config shell.
        """
        return getattr(self, "_model_config", None)

    @model_config.setter
    def model_config(self, value: ModelConfig | None) -> None:
        """Set the lazily built model config shell.

        Args:
                value: Model config shell.
        """
        self._model_config = value

    def get_model(self) -> BaseEstimator:
        """Get the model's estimator.

        Returns:
                The model's estimator.

        Raises:
                ValueError: If model estimator is not available.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model

    def apply_to(
        self,
        estimator: Union["BaseEstimator", None],
        data: DataConfig | None,
    ) -> "BaseEstimator":
        """Apply this defense to a pre-fitted estimator.

        Args:
                estimator: Fitted estimator payload.
                data: Runtime data payload.

        Returns:
                Defended estimator.

        Raises:
                ValueError: If estimator is missing.
        """
        if estimator is None:
            raise ValueError(
                "estimator must be provided before applying defense",
            )
        self.model = estimator
        model_cfg = self.model_config
        if model_cfg is not None and hasattr(model_cfg, "set_estimator"):
            cast(Any, model_cfg).set_estimator(estimator)
        elif model_cfg is not None and hasattr(model_cfg, "_model"):
            setattr(model_cfg, "_model", estimator)
        return self.apply_defense(data)

    def apply(
        self,
        estimator: Union["BaseEstimator", None],
        data: DataConfig | None,
    ) -> "BaseEstimator":
        """Unified public entrypoint for applying a single defense config.

        Args:
                estimator: Fitted estimator payload to defend.
                data: Runtime data payload.

        Returns:
                Defended estimator.

        Notes:
                This method is equivalent to ``apply_to`` and exists to align the
                public API with pipeline-level ``DefenseConfig.apply``.
        """
        return self.apply_to(estimator=estimator, data=data)

    def apply_defense(self, data: DataConfig | None) -> "BaseEstimator":
        """Apply the configured defense to the current estimator.

        Args:
                data: Runtime data payload used to build defense context.

        Returns:
                The defended estimator.

        Raises:
                ValueError: If the model is not fitted before applying defense.
        """

        if self.model is None:
            raise ValueError(
                "ModelConfig must have a fitted estimator before applying defense",
            )
        elif (
            not isinstance(self.model, BaseEstimator)
            and not _is_torch_model_instance(self.model)
            and not hasattr(
                self.model,
                "model",
            )
        ):
            assert isinstance(
                self.model,
                BaseEstimator,
            ), "ModelConfig's _model must be a scikit-learn BaseEstimator"

        defense_signature = self._defense_signature()
        if defense_signature in self._get_applied_defense_signatures(
            self.model,
        ):
            self._finalize_defense_application(
                defended_estimator=self.model,
                defense_signature=defense_signature,
                apply_fit=False,
                elapsed=0.0,
            )
            return self.model

        defense_type, defense_subtype, defense_class = self.parse_defense_name()
        art_class, init_params = self.get_art_class(data)
        (
            base_estimator,
            art_class,
            init_params,
            existing_preprocessors,
            existing_postprocessors,
        ) = self._extract_art_wrapper_context(
            art_class=art_class,
            init_params=init_params,
        )
        if not _is_torch_model_instance(base_estimator):
            try:
                check_is_fitted(cast(Any, base_estimator))
            except NotFittedError as e:
                raise ValueError(
                    "ModelConfig must have a fitted estimator before applying defense",
                ) from e
        start = time.perf_counter()
        handler = self._resolve_defense_handler(
            defense_type=defense_type,
            defense_subtype=defense_subtype,
        )
        if handler is None:
            raise NotImplementedError(
                f"Defense type '{defense_type}' has no registered callable handler.",
            )

        self._run_plugin_hook(
            "before_defense_dispatch",
            data=data,
            defense_type=defense_type,
            defense_subtype=defense_subtype,
            art_class=art_class,
            init_params=init_params,
            base_estimator=base_estimator,
            existing_preprocessors=existing_preprocessors,
            existing_postprocessors=existing_postprocessors,
            handler=handler,
        )

        defense, defended_estimator = cast(Any, handler)(
            data=data,
            defense_type=defense_type,
            defense_subtype=defense_subtype,
            defense_class=defense_class,
            art_class=art_class,
            init_params=init_params,
            base_estimator=base_estimator,
            existing_preprocessors=existing_preprocessors,
            existing_postprocessors=existing_postprocessors,
        )
        if defended_estimator is None:
            raise RuntimeError(
                "Defense application did not produce an estimator",
            )
        end = time.perf_counter()
        self._finalize_defense_application(
            defended_estimator=defended_estimator,
            defense_signature=defense_signature,
            defense=defense,
            elapsed=end - start,
        )
        self._run_plugin_hook(
            "after_defense_dispatch",
            data=data,
            defense_type=defense_type,
            defense_subtype=defense_subtype,
            defense=defense,
            defended_estimator=defended_estimator,
            defense_application_time=self.defense_application_time,
        )
        model_cfg = self.model_config
        if model_cfg is not None:
            cast(Any, model_cfg).set_estimator(defended_estimator)
        return cast(BaseEstimator, defended_estimator)

    def get_art_class(
        self,
        data: DataConfig | None,
    ) -> tuple[ArtEsimtator, dict[str, DefenseInitParamValue]]:
        """Resolve the ART estimator wrapper class for the current model/data.

        Args:
                data: Runtime data payload used for wrapper shape metadata.

        Returns:
                ART wrapper class and initialization parameter mapping.

        Raises:
                ImportError: If optional torch/ART dependencies are unavailable.
                ValueError: If model type is required but missing.
                TypeError: If torch defenses receive non-torch base estimator.
        """
        if (
            _is_torch_model_instance(getattr(self, "_model", None))
            or (isinstance(self.name, str) and self.name.startswith("torch."))
            or _is_art_torch_wrapper(getattr(self, "_model", None))
        ):
            if data is None:
                raise ValueError(
                    "data must be provided before creating an ART defense estimator",
                )
            try:
                import torch
            except Exception as exc:  # pragma: no cover - optional dependency import may fail at runtime
                raise ImportError(
                    "Torch model defenses require optional dependency deckard[torch]",
                ) from exc

            X_train = getattr(data, "X_train")
            from torch.utils.data import DataLoader, Dataset, Subset

            if isinstance(X_train, (Dataset, Subset)):
                loader = DataLoader(X_train, batch_size=1, shuffle=False)
                batch = next(iter(loader))
                if isinstance(batch, (tuple, list)):
                    input_shape = batch[0].shape[1:]
                else:
                    input_shape = batch.shape[1:]
            else:
                input_shape = tuple(X_train.shape[1:])
            y_train = getattr(data, "y_train")
            if isinstance(y_train, torch.Tensor):
                nb_classes = int(torch.unique(y_train).numel())
            else:
                nb_classes = len(set(y_train))

            raw_model = self._model
            if _is_art_torch_wrapper(raw_model):
                wrapper_state = _get_wrapper_state(raw_model)
                state_base = (
                    None
                    if wrapper_state is None
                    else wrapper_state.get("base_estimator")
                )
                if _is_torch_model_instance(state_base):
                    raw_model = state_base
                else:
                    wrapped_model = getattr(raw_model, "model", None)
                    if _is_torch_model_instance(wrapped_model):
                        raw_model = wrapped_model

            if not _is_torch_model_instance(raw_model):
                raise TypeError(
                    "Torch defenses require a torch.nn.Module base estimator. "
                    f"Got {type(raw_model)} while constructing ART wrapper context.",
                )

            art_symbols = _get_art_symbols()

            if self.classifier:
                art_class = art_symbols["torch_classifier"]
                torch_model = cast(Any, raw_model)
                init_params = {
                    "loss": torch.nn.CrossEntropyLoss(),
                    "optimizer": torch.optim.SGD(
                        torch_model.parameters(),
                        lr=0.01,
                    ),
                    "input_shape": input_shape,
                    "nb_classes": nb_classes,
                    "clip_values": getattr(self, "clip_values", None) or (0.0, 1.0),
                    "device_type": ("gpu" if torch.cuda.is_available() else "cpu"),
                }
            else:
                art_class = art_symbols["torch_regressor"]
                torch_model = cast(Any, raw_model)
                init_params = {
                    "loss": torch.nn.MSELoss(),
                    "optimizer": torch.optim.SGD(
                        torch_model.parameters(),
                        lr=0.01,
                    ),
                    "input_shape": input_shape,
                    "nb_classes": None,
                    "clip_values": getattr(self, "clip_values", None) or (0.0, 1.0),
                    "device_type": ("gpu" if torch.cuda.is_available() else "cpu"),
                }
            if "preprocessing" not in init_params:
                init_params["preprocessing"] = None
            return art_class, init_params

        from ...utils import is_null_config_value

        resolve_name = getattr(self, "resolve_name", None)
        model_name = resolve_name(default=None) if callable(resolve_name) else None
        if is_null_config_value(model_name, allow_empty=True):
            raise ValueError(
                "name must be set before creating an ART defense estimator",
            )
        assert model_name is not None
        model_name = str(model_name)
        self.name = model_name
        try:
            art_symbols = _get_art_symbols()
        except Exception as exc:
            raise ImportError(
                "ART estimators are required for defense wrapping. Install optional dependencies that include ART.",
            ) from exc
        art_class = (
            art_symbols["classifier_dict"][model_name.split(".")[-1]]
            if self.classifier
            else art_symbols["regressor_dict"][model_name.split(".")[-1]]
        )
        if art_class in art_symbols["sklearn_dict"].values():
            init_params = {}
        else:
            if data is None:
                raise ValueError(
                    "data must be provided before creating an ART defense estimator",
                )
            init_params = {
                "input_shape": cast(Any, data).X_train.shape[1:],
                "nb_classes": (
                    len(set(cast(Any, data).y_train)) if self.classifier else None
                ),
            }
        if "preprocessing" not in init_params:
            init_params["preprocessing"] = None
        return art_class, init_params


@dataclass(eq=False, kw_only=True)
class DefenseConfig(
    DefensePipelineConfigBehaviorMixin,
    BaseConfig,
):
    """Runtime owner for applying an ordered chain of defense specs.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    defenses: list = field(default_factory=list, metadata={'help': 'Configuration field: defenses.'})
    name: StringifiedClass | None = None
    model_name: StringifiedClass | None = None
    classifier: Union[bool, str, None] = True
    model_params: dict = field(
        default_factory=dict,
        metadata={"help": "Parameters for the model."},
    )
    probability: bool = False
    clip_values: tuple | None = field(
        default=None,
        metadata={
            "help": "Tuple of the form (min, max) to clip input features.",
        },
    )
    plugins: list = field(default_factory=list, repr=True, metadata={'help': 'Configuration field: plugins.'})
    alias: str = field(default_factory=str, metadata={'help': 'Configuration field: alias.'})
    score_dict: ScoreDict = field(default_factory=ScoreDict, init=False, repr=False, metadata={'help': 'Configuration field: score_dict.'})
    defense_application_time: Union[float, None] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        metadata={"help": "Runtime-only timing: total defense application seconds."},
    )
    _target_: Union[str, None] = field(default='deckard.model.defense.base.DefenseConfig', init=True, repr=True, metadata={'help': 'Hydra target path used to rehydrate this defense config.'})
    _plugin_objects: Union[list, None] = field(default=None, init=False, repr=False, compare=False, metadata={'help': 'Configuration field: _plugin_objects.'})
    _model: Union[BaseEstimator, None] = field(default=None, init=False, repr=False, metadata={'help': 'Configuration field: _model.'})
    _model_config: Union[ModelConfig, None] = field(default=None, init=False, repr=False, compare=False, metadata={'help': 'Configuration field: _model_config.'})

    def __post_init__(self):
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = ScoreDict()
        else:
            self.score_dict = ScoreDict.from_payload(self.score_dict)
        if not hasattr(self, "_target_") or self._target_ in {None, ""}:
            self._target_ = "deckard.model.defense.base.DefenseConfig"
        self.defenses = self.normalize_defenses(getattr(self, "defenses", []))
        self.plugins = normalize_plugin_specs(getattr(self, "plugins", []))
        self.defense_application_time = None
        BaseConfig._after_post_init(self)

    def __hash__(self) -> int:
        return super().__hash__()




