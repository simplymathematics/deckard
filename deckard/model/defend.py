# A BaseConfig class for Configuration of Models using adversarial-robustness-toolbox (ART)
# https://adversarial-robustness-toolbox.readthedocs.io/en/latest

import time
import logging
import warnings
from sklearn.base import BaseEstimator
from dataclasses import dataclass, field
from typing import Any, cast, Union
from functools import lru_cache
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from ..data import DataConfig
from ..frameworks import ModelDefenseContractMixin
from .base import ModelConfig
from ..utils import (
    ConfigBase,
    coerce_config,
    resolve_class,
    coerce_to_list,
    is_null_config_value,
    normalize_plugin_specs,
    instantiate_plugin_spec,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_art_symbols() -> dict[str, Any]:
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
    from art.estimators.regression.scikitlearn import (
        ScikitlearnDecisionTreeRegressor,
        ScikitlearnRegressor,
    )
    from art.estimators.classification import PyTorchClassifier
    from art.estimators.regression import PyTorchRegressor

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


supported_defense_types = [
    "detector",
    "preprocessor",
    "postprocessor",
    "trainer",
    "regularizer",
    "transformer",
    None,
]


class _DefensePipelineConfigBehaviorMixin:
    """Reusable defense pipeline configuration behavior mixed into pipeline configs.

    Plugin hooks
    ------------
    resolve_defense_stage(self, *, default_stage, current_stage, **context)
        Override stage selection for pipeline application.
    before_apply_defense(self, *, estimator, data, stage, defense_chain)
        Runs once before the defense chain executes.
    before_apply_defense_step(self, *, estimator, data, stage, defense, applied_defenses)
        Runs before each defense step in the chain.
    after_apply_defense_step(self, *, estimator, data, stage, defense, applied_defenses, step_defense_time)
        Runs after each defense step; dict returns are merged into score_dict.
    after_apply_defense(self, *, estimator, data, stage, defense_chain, applied_defenses, applied_defense_types, defense_application_time)
        Runs once after the defense chain executes; dict returns are merged into score_dict.

    Additional DefenseConfig runtime hooks
    --------------------------------------
    resolve_defense_mixins(self, *, defense_type, defense_subtype, default_mixins)
        Return one mixin type, or a list/tuple of mixin types, to extend runtime
        handler resolution.
    resolve_defense_handler(self, *, defense_type, defense_subtype, default_handler, default_mixins)
        Return a callable handler (or handler type) to override default runtime
        handler resolution.
    before_defense_dispatch(self, *, data, defense_type, defense_subtype, art_class, init_params, base_estimator, existing_preprocessors, existing_postprocessors, handler)
        Runs immediately before runtime defense handler execution.
    after_defense_dispatch(self, *, data, defense_type, defense_subtype, defense, defended_estimator, defense_application_time)
        Runs immediately after runtime defense handler execution.
    """

    # Declared for static analyzers; concrete dataclass provides these fields.
    defenses: list
    plugins: list
    score_dict: dict
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
        if "defenses" in defense_spec:
            return False
        target = defense_spec.get("_target_")
        if target is not None and not cls._is_pipeline_target(target):
            return True
        legacy_keys = {
            "defense_name",
            "defense_params",
            "model_type",
            "classifier",
            "probability",
            "clip_values",
            "alias",
        }
        return any(key in defense_spec for key in legacy_keys)

    @classmethod
    def coerce(cls, defense_config: Any):
        if defense_config is None or isinstance(defense_config, cls):
            return defense_config

        # Keep concrete defense objects intact; converting them to dict can
        # capture runtime-only attrs that are not valid constructor kwargs.
        if hasattr(defense_config, "apply_to"):
            return DefensePipelineConfig(defenses=[defense_config])

        # Coerce config first (may return a list after converting DictConfig/YAML)
        defense_config = coerce_config(defense_config)

        # Handle lists (either provided directly or returned from coerce_config)
        if isinstance(defense_config, (list, ListConfig)):
            return DefensePipelineConfig(defenses=coerce_to_list(defense_config))

        if isinstance(defense_config, dict):
            defense_dict = cast(dict[str, Any], dict(defense_config))
            if cls._is_pipeline_target(defense_dict.get("_target_")):
                defense_dict.pop("_target_", None)
                return DefensePipelineConfig(**defense_dict)
            if "defenses" in defense_dict:
                return DefensePipelineConfig(**defense_dict)
            if cls._looks_like_single_defense_spec(defense_dict):
                return DefensePipelineConfig(defenses=[defense_dict])

        raise TypeError(
            "Defense config must be a DefensePipelineConfig, a single defense spec, or None",
        )

    def _instantiate_plugin(self, plugin_spec: Any):
        def _resolve_and_instantiate(path: str, **kwargs):
            return resolve_class(path)(**kwargs)

        return instantiate_plugin_spec(
            plugin_spec,
            loader=_resolve_and_instantiate,
        )

    def _get_plugins(self) -> list:
        if self._plugin_objects is None:
            plugin_specs = normalize_plugin_specs(self.plugins)
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs) -> list[Any]:
        """Execute one plugin hook across all instantiated plugins."""
        hook_outputs = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_outputs.append(hook(self, **kwargs))
        return hook_outputs

    def _merge_plugin_scores(self, hook_outputs):
        if self.score_dict is None:
            self.score_dict = {}
        for output in hook_outputs:
            if isinstance(output, dict):
                self.score_dict.update(output)

    def _coerce_single_defense(self, defense_obj):
        if hasattr(defense_obj, "apply_to"):
            return defense_obj

        defense_obj = coerce_config(defense_obj)

        if isinstance(defense_obj, dict):
            defense_dict = cast(dict[str, Any], dict(defense_obj))
            target = defense_dict.pop("_target_", None)
            if target is not None:
                return resolve_class(target)(**defense_dict)

            defense_name = defense_dict.get("defense_name")
            if isinstance(defense_name, str) and defense_name.startswith(
                "fairlearn.",
            ):
                try:
                    fair_cls = resolve_class(
                        "deckard.plugins.fairlearn.model.FairlearnDefenseConfig",
                    )
                    return fair_cls(**defense_dict)
                except Exception:
                    pass
            return DefenseConfig(**defense_dict)

        raise TypeError(
            f"Unsupported defense specification in pipeline: {type(defense_obj)}",
        )

    def _inherit_model_context(self, defense_obj, estimator) -> None:
        base_estimator = getattr(estimator, "model", estimator)

        blank_values = {None, "", "None", "null", "Null", "NULL"}

        if (
            hasattr(defense_obj, "model_type")
            and getattr(
                defense_obj,
                "model_type",
                None,
            )
            in blank_values
        ):
            defense_obj.model_type = (
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

    def normalize_defenses(self, defenses: Any) -> list:
        if defenses is None:
            return []
        if isinstance(defenses, (tuple, list)):
            defense_list = list(defenses)
        else:
            defense_list = [defenses]
        return [self._coerce_single_defense(item) for item in defense_list]

    def resolve_stage(
        self,
        default_stage: str = "post_fit_pre_predict",
        **context: Any,
    ) -> str:
        stage = default_stage
        # TODO: make this context aware since art defenses have _apply_fit and _apply_predict
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
        defense_name = getattr(defense_obj, "defense_name", None)
        if isinstance(defense_name, str) and defense_name.startswith("art."):
            return True
        # DefenseConfig instances without a fairlearn/art prefix are ART-style model wrappers
        try:
            from ..plugins.fairlearn.model import FairlearnDefenseConfig

            if isinstance(defense_obj, FairlearnDefenseConfig):
                return False
        except Exception:
            pass
        if isinstance(defense_obj, DefenseConfig):
            return True
        return False

    def _is_model_wrapper_defense(self, defense_obj) -> bool:
        """Return True for defenses that wrap/transform estimators rather than raw data."""
        if self._is_art_defense(defense_obj):
            return True
        try:
            from ..plugins.fairlearn.model import FairlearnDefenseConfig

            if isinstance(defense_obj, FairlearnDefenseConfig):
                return True
        except Exception:
            pass
        return False

    def _is_retraining_defense(self, defense_obj) -> bool:
        defense_name = getattr(defense_obj, "defense_name", None)
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
        data,
        stage: str = "post_fit_pre_predict",
    ) -> BaseEstimator:
        if estimator is None:
            raise ValueError(
                "estimator must be provided before applying defenses",
            )
        defense_chain = self.normalize_defenses(self.defenses)
        if len(defense_chain) == 0:
            return estimator

        # Enforce model-wrapper defenses last: ART/fairlearn estimator wrappers should run
        # after data-transform defenses, while preserving user order within each group.
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
                        getattr(d, "defense_name", type(d).__name__)
                        for d in data_defenses
                    ],
                    [
                        getattr(d, "defense_name", type(d).__name__)
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
        for defense_obj in defense_chain:
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
                defense=defense_obj,
                applied_defenses=applied_defenses,
            )
            started = time.process_time()
            defended_estimator = apply_to(
                estimator=defended_estimator,
                data=data,
            )
            elapsed = getattr(defense_obj, "defense_application_time", None)
            if elapsed is None:
                elapsed = time.process_time() - started
            total_defense_time += float(elapsed)
            applied_defenses.append(defense_obj)
            step_outputs = self._run_plugin_hook(
                "after_apply_defense_step",
                estimator=defended_estimator,
                data=data,
                stage=stage,
                defense=defense_obj,
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
            applied_defense_types=[type(d).__name__ for d in applied_defenses],
            defense_application_time=total_defense_time,
        )
        self._merge_plugin_scores(hook_outputs)
        return cast(BaseEstimator, defended_estimator)


def _is_torch_model_instance(model_obj) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return isinstance(model_obj, torch.nn.Module)


@dataclass(eq=True)
class _DefenseMixin:
    """Base callable defense handler used by runtime defense context resolution.

    Parameters
    ----------
    runtime : Any
        Runtime defense config object owned by defense orchestration.
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
        data: Any,
        defense_type: Union[str, None],
        defense_subtype: Union[str, None],
        defense_class: Any,
        art_class: Any,
        init_params: dict,
        base_estimator: Any,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[Any, Any]:
        """Execute one defense handler.

        Parameters
        ----------
        data : Any
            Data runtime containing train/test/val splits.
        defense_type : str | None
            Parsed defense family.
        defense_subtype : str | None
            Parsed defense subtype.
        defense_class : Any
            Concrete defense class resolved from ``defense_name``.
        art_class : Any
            ART estimator wrapper class selected for model type.
        init_params : dict
            Runtime ART estimator initialization kwargs resolved by
            ``DefenseConfig.get_art_class``. Handlers should treat this as
            library/class-specific defaults and merge with ``defense_params``
            when constructing wrapped estimators.
        base_estimator : Any
            Unwrapped model estimator used as defense target.
        existing_preprocessors : list
            Existing preprocessor defenses already attached to wrapper.
        existing_postprocessors : list
            Existing postprocessor defenses already attached to wrapper.
        """
        raise NotImplementedError("Defense handlers must implement __call__")


class _PassthroughDefenseMixin(_DefenseMixin):
    """Default handler for no-op/passthrough ART wrapping."""

    def __call__(
        self,
        *,
        data: Any,
        defense_type: Union[str, None],
        defense_subtype: Union[str, None],
        defense_class: Any,
        art_class: Any,
        init_params: dict,
        base_estimator: Any,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[Any, Any]:
        defended_estimator = self._build_art_wrapper(
            art_class=art_class,
            base_estimator=base_estimator,
            init_params={**self.defense_params, **init_params},
            preprocessing_defences=existing_preprocessors,
            postprocessing_defences=existing_postprocessors,
        )
        return None, defended_estimator


class _DefenseBehaviorMixin:
    """Reusable defense workflow behavior mixed into concrete config dataclasses."""

    # Declared for static analyzers; concrete dataclass provides these fields.
    model_type: Union[str, None]
    classifier: Union[bool, str, None]
    model_params: dict
    probability: bool
    alias: str
    defense_name: Union[str, None]
    defense_params: dict
    _model: Union[BaseEstimator, None]
    score_dict: dict
    _target_: Union[str, None]
    _model_config: Union[ModelConfig, None]
    plugins: list
    _plugin_objects: Union[list, None]

    def _instantiate_plugin(self, plugin_spec: Any):
        def _resolve_and_instantiate(path: str, **kwargs):
            return resolve_class(path)(**kwargs)

        return instantiate_plugin_spec(
            plugin_spec,
            loader=_resolve_and_instantiate,
        )

    def _get_plugins(self) -> list:
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

    def _resolve_runtime_defense_mixins(
        self,
        defense_type: Union[str, None],
        defense_subtype: Union[str, None],
    ) -> tuple[type, ...]:
        mixins: list[type] = []
        dtype = (defense_type or "").lower() if defense_type is not None else None
        if dtype is None:
            mixins.append(_PassthroughDefenseMixin)
        elif dtype == "detector":
            from .detector import _DetectorDefenseMixin

            mixins.append(_DetectorDefenseMixin)
        elif dtype == "preprocessor":
            from .preprocessor import _PreprocessorDefenseMixin

            mixins.append(_PreprocessorDefenseMixin)
        elif dtype == "postprocessor":
            from .postprocessor import _PostprocessorDefenseMixin

            mixins.append(_PostprocessorDefenseMixin)
        elif dtype == "trainer":
            from .trainer import _TrainerDefenseMixin

            mixins.append(_TrainerDefenseMixin)
        elif dtype == "transformer":
            from .transformer import _TransformerDefenseMixin

            mixins.append(_TransformerDefenseMixin)
        elif dtype == "regularizer":
            from .regularizer import _RegularizerDefenseMixin

            mixins.append(_RegularizerDefenseMixin)

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
        defense_type: Union[str, None],
        defense_subtype: Union[str, None],
    ):
        mixins = self._resolve_runtime_defense_mixins(defense_type, defense_subtype)
        default_handler = None
        for mixin in mixins:
            if isinstance(mixin, type) and issubclass(mixin, _DefenseMixin):
                default_handler = mixin(self)
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
            if isinstance(output, type) and issubclass(output, _DefenseMixin):
                return output(self)

        return default_handler

    def _get_model_config(self) -> ModelConfig:
        if getattr(self, "model_config", None) is None:
            self.model_config = ModelConfig(
                model_type=self.model_type,
                classifier=self.classifier,
                model_params=self.model_params,
                probability=self.probability,
                alias=self.alias,
            )
            self.model_config.defense = None
        assert self.model_config is not None
        return self.model_config

    def __post_init__(self):
        if not is_null_config_value(self.model_type, allow_empty=True):
            model_cfg = self._get_model_config()
            self.classifier = model_cfg.classifier
            self.model_params = model_cfg.model_params
            self.model = model_cfg.model
        elif not hasattr(self, "_model"):
            self.model = None

        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.DefenseConfig"

        # Initialize times, scores, and defended model
        self.defense_training_time = None
        self.defense_application_time = None
        self.defense_prediction_time = None
        self.defense_scoring_time = None
        self.defense_params = self.defense_params or {}
        self._apply_fit = True  # Whether to apply fit during defense application
        self.plugins = normalize_plugin_specs(getattr(self, "plugins", []))

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
        if hasattr(self.model, "model") and hasattr(
            self.model,
            "preprocessing_defences",
        ):
            base_estimator = getattr(self.model, "model")
            art_class = self.model.__class__
            existing_preprocessors = list(
                getattr(self.model, "preprocessing_defences", []) or [],
            )
            existing_postprocessors = list(
                getattr(self.model, "postprocessing_defences", []) or [],
            )
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
        return art_class(base_estimator, **art_params)


class ModelDefenseMixin(ModelDefenseContractMixin):
    """Reusable defense pipeline orchestration mixed into config shells."""

    @property
    def model(self) -> BaseEstimator | None:
        """Public accessor for the runtime estimator payload."""
        return getattr(self, "_model", None)

    @model.setter
    def model(self, value: BaseEstimator | None) -> None:
        """Set the runtime estimator payload."""
        self._model = value

    @property
    def model_config(self) -> ModelConfig | None:
        """Public accessor for the lazily built model config shell."""
        return getattr(self, "_model_config", None)

    @model_config.setter
    def model_config(self, value: ModelConfig | None) -> None:
        """Set the lazily built model config shell."""
        self._model_config = value

    def get_model(self) -> BaseEstimator:
        """Get the model's estimator.

        Returns
        -------
        BaseEstimator
            The model's estimator.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model

    def apply_to(
        self,
        estimator: Union["BaseEstimator", None],
        data: Any,
    ) -> "BaseEstimator":
        """Apply this defense to a pre-fitted estimator."""
        if estimator is None:
            raise ValueError(
                "estimator must be provided before applying defense",
            )
        self.model = estimator
        model_cfg = self.model_config
        if model_cfg is not None:
            model_cfg.model = estimator
        return self.apply_defense(data)

    def apply_defense(self, data: Any) -> "BaseEstimator":
        """Apply the specified defense to the model's estimator.

        Returns
        -------
        BaseEstimator
            The estimator wrapped with the specified defense.
        Raises
        ------
        ValueError
            If the model is not fitted before applying the defense.
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
            self._apply_fit = False
            self.defense_application_time = 0.0
            return self.model

        # Dynamically import the defense class with defense_params as kwargs
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
                check_is_fitted(base_estimator)
            except NotFittedError as e:
                raise ValueError(
                    "ModelConfig must have a fitted estimator before applying defense",
                ) from e
        start = time.process_time()
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

        defense, defended_estimator = handler(
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
        # Some defences can optionally be applied during training or prediction
        end = time.process_time()
        self._apply_fit = getattr(defense, "_apply_fit", True)
        self._mark_applied_defense_signature(
            defended_estimator,
            defense_signature,
        )

        self.defense_application_time = end - start
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
            model_cfg.model = defended_estimator
        return defended_estimator

    def parse_defense_name(self) -> tuple:
        if self.defense_name is not None and len(self.defense_name) > 0:
            module_name, class_name = self.defense_name.rsplit(".", 1)
        else:
            module_name = None
            class_name = None
        if module_name is None or class_name is None:
            defense_type = None
        else:
            try:
                defense_type = module_name.split(".")[2]  # e.g., 'preprocessor'
            except IndexError:
                raise ImportError(
                    f"Could not parse defense type from defense name {self.defense_name}",
                )
        if module_name is not None and len(module_name.split(".")) >= 4:
            defense_subtype = module_name.split(".")[3]  # e.g., 'FeatureSqueezing'
        else:
            defense_subtype = None
        if defense_type is not None:
            try:
                assert self.defense_name is not None
                defense_class = resolve_class(self.defense_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not import defense class {self.defense_name}",
                ) from e
        else:
            defense_class = None
        assert (
            defense_type in supported_defense_types
        ), f"Unsupported defense type: {defense_type}. Supported types are: {supported_defense_types}"

        return defense_type, defense_subtype, defense_class

    def get_art_class(self, data: Any):
        if (
            _is_torch_model_instance(getattr(self, "_model", None))
            or (
                isinstance(self.model_type, str)
                and self.model_type.startswith("torch.")
            )
            or _is_art_torch_wrapper(getattr(self, "_model", None))
        ):
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
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

            # Resolve the underlying nn.Module (unwrap if already an ART wrapper).
            raw_model = self._model
            if _is_art_torch_wrapper(raw_model):
                raw_model = getattr(raw_model, "model", raw_model)

            art_symbols = _get_art_symbols()

            if self.classifier:
                art_class = art_symbols["torch_classifier"]
                init_params = {
                    "loss": torch.nn.CrossEntropyLoss(),
                    "optimizer": torch.optim.SGD(
                        raw_model.parameters(),
                        lr=0.01,
                    ),
                    "input_shape": input_shape,
                    "nb_classes": nb_classes,
                    "clip_values": getattr(self, "clip_values", None) or (0.0, 1.0),
                    "device_type": ("gpu" if torch.cuda.is_available() else "cpu"),
                }
            else:
                art_class = art_symbols["torch_regressor"]
                init_params = {
                    "loss": torch.nn.MSELoss(),
                    "optimizer": torch.optim.SGD(
                        raw_model.parameters(),
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

        from ..utils import is_null_config_value

        if is_null_config_value(self.model_type, allow_empty=True):
            raise ValueError(
                "model_type must be set before creating an ART defense estimator",
            )
        assert self.model_type is not None
        try:
            art_symbols = _get_art_symbols()
        except Exception as exc:
            raise ImportError(
                "ART estimators are required for defense wrapping. Install optional dependencies that include ART.",
            ) from exc
        art_class = (
            art_symbols["classifier_dict"][self.model_type.split(".")[-1]]
            if self.classifier
            else art_symbols["regressor_dict"][self.model_type.split(".")[-1]]
        )
        if art_class in art_symbols["sklearn_dict"].values():
            init_params = {}
        else:
            init_params = {
                "input_shape": data.X_train.shape[1:],
                "nb_classes": (len(set(data.y_train)) if self.classifier else None),
            }
        if "preprocessing" not in init_params:
            init_params["preprocessing"] = None
        return art_class, init_params

    def __call__(
        self,
        data: DataConfig,
        model_file: Union[str, None] = None,
        test_predictions_file: Union[str, None] = None,
        train_predictions_file: Union[str, None] = None,
        training_probabilities_file: Union[str, None] = None,
        test_probabilities_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "DefenseConfig no longer owns model runtime orchestration. "
            "Use ModelConfig(defense=DefensePipelineConfig(...))(data=...) instead.",
        )


# Backward-compatible alias for internal imports.
_DefensePipelineMixin = ModelDefenseMixin


@dataclass(eq=False, kw_only=True)
class DefensePipelineConfig(_DefensePipelineConfigBehaviorMixin, ConfigBase):
    """Runtime owner for applying an ordered chain of defense specs."""

    defenses: list = field(default_factory=list)
    plugins: list = field(default_factory=list)
    alias: str = field(default_factory=str)
    score_dict: dict = field(default_factory=dict)
    defense_application_time: Union[float, None] = None
    _target_: Union[str, None] = None
    _plugin_objects: Union[list, None] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.model.DefensePipelineConfig"
        self.defenses = self.normalize_defenses(self.defenses)

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass(kw_only=True)
class DefenseConfig(_DefenseBehaviorMixin, ModelDefenseMixin, ConfigBase):
    """Concrete defense config dataclass that uses shared defense behavior mixin.

    Parameter layers
    ----------------
    model_params : dict
        Base model-constructor kwargs (owned by model runtime).
    defense_params : dict
        Defense-constructor or defense-call kwargs passed to the resolved
        defense class/callable.
    init_params : dict
        Declaration metadata for class/type/library docs. Runtime ART wrapper
        kwargs are resolved by ``get_art_class`` and passed to mixin handlers as
        ``init_params`` in ``_DefenseMixin.__call__``.

    Family-specific parameter semantics
    ----------------------------------
    sklearn ART wrappers
        Typically empty wrapper kwargs (plus ``preprocessing=None`` fallback).
    torch ART wrappers
        Include runtime wrapper kwargs such as ``input_shape`` and
        ``nb_classes`` (when classification is enabled), then merged with
        defense-specific params by mixin handlers.
    detector
        ``defense_params`` often configures detector constructor kwargs while
        runtime wrapper init kwargs come from resolved ART wrapper context.
    preprocessor/postprocessor
        ``defense_params`` configures transformation object behavior; runtime
        wrapper kwargs remain ART-estimator concerns.
    trainer/transformer/regularizer
        ``defense_params`` configures training-time behavior; runtime
        wrapper kwargs are still sourced from model/ART context.

    Plugin hook runtime params
    --------------------------
    Hooks are orchestrated by ``_run_plugin_hook(hook_name, **kwargs)``.
    Core hook names used by DefenseConfig runtime are:
    ``resolve_defense_mixins``, ``resolve_defense_handler``,
    ``before_defense_dispatch``, and ``after_defense_dispatch``.
    Hook kwargs are phase-specific runtime objects supplied by defense
    orchestration.
    """

    model_type: Union[str, None] = None
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
    defense_name: Union[str, None] = field(
        default=None,
        metadata={"help": "Name of the defense to apply."},
    )
    defense_params: dict = field(
        default_factory=dict,
        metadata={"help": "Parameters for the defense."},
    )
    init_params: dict = field(
        default_factory=dict,
        metadata={
            "help": "Initialization metadata for defense class/type/library declaration.",
        },
    )
    alias: str = field(default_factory=str)
    plugins: list = field(default_factory=list)
    _model: Union[BaseEstimator, None] = field(default=None, repr=False)
    score_dict: dict = field(default_factory=dict)
    _target_: Union[str, None] = field(default=None, repr=False)
    _plugin_objects: Union[list, None] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _model_config: Union[ModelConfig, None] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __hash__(self) -> int:
        return super().__hash__()
