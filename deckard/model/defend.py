# A BaseConfig class for Configuration of Models using adversarial-robustness-toolbox (ART)
# https://adversarial-robustness-toolbox.readthedocs.io/en/latest

import time
import logging
import warnings
from sklearn.base import BaseEstimator
from dataclasses import dataclass, field
from typing import Any, cast, Union
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

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
from ..data import DataConfig
from .base import ModelConfig
from ..utils import ConfigBase, coerce_config, resolve_class, coerce_to_list, is_null_config_value

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

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
sklearn_models = list(sklearn_dict.keys())

supported_defense_types = [
    "detector",
    "preprocessor",
    "postprocessor",
    "trainer",
    "regularizer",
    "transformer",
    None,
]


def _is_torch_model_instance(model_obj) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return isinstance(model_obj, torch.nn.Module)


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

    def _get_model_config(self) -> ModelConfig:
        if getattr(self, "_model_config", None) is None:
            self._model_config = ModelConfig(
                model_type=self.model_type,
                classifier=self.classifier,
                model_params=self.model_params,
                probability=self.probability,
                alias=self.alias,
            )
            self._model_config.defense = None
        assert self._model_config is not None
        return self._model_config

    def __post_init__(self):
        if not is_null_config_value(self.model_type, allow_empty=True):
            model_cfg = self._get_model_config()
            self.classifier = model_cfg.classifier
            self.model_params = model_cfg.model_params
            self._model = model_cfg._model
        elif not hasattr(self, "_model"):
            self._model = None

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
        base_estimator = self._model
        art_params = dict(init_params or {})
        existing_preprocessors = []
        existing_postprocessors = []
        if hasattr(self._model, "model") and hasattr(
            self._model,
            "preprocessing_defences",
        ):
            base_estimator = getattr(self._model, "model")
            art_class = self._model.__class__
            existing_preprocessors = list(
                getattr(self._model, "preprocessing_defences", []) or [],
            )
            existing_postprocessors = list(
                getattr(self._model, "postprocessing_defences", []) or [],
            )
            art_params["preprocessing"] = getattr(
                self._model,
                "preprocessing",
                art_params.get("preprocessing"),
            )
            clip_values = getattr(self._model, "clip_values", None)
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

    def get_model(self) -> BaseEstimator:
        """Get the model's estimator.

        Returns
        -------
        BaseEstimator
            The model's estimator.
        """
        if self._model is None:
            raise ValueError("Model is not fitted yet.")
        return self._model

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
        self._model = estimator
        model_cfg = getattr(self, "_model_config", None)
        if model_cfg is not None:
            model_cfg._model = estimator
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

        if self._model is None:
            raise ValueError(
                "ModelConfig must have a fitted estimator before applying defense",
            )
        elif (
            not isinstance(self._model, BaseEstimator)
            and not _is_torch_model_instance(self._model)
            and not hasattr(
                self._model,
                "model",
            )
        ):
            assert isinstance(
                self._model,
                BaseEstimator,
            ), "ModelConfig's _model must be a scikit-learn BaseEstimator"

        defense_signature = self._defense_signature()
        if defense_signature in self._get_applied_defense_signatures(
            self._model,
        ):
            self._apply_fit = False
            self.defense_application_time = 0.0
            return self._model

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
        defense = None
        defended_estimator = None
        match defense_type:  # Note: only one defense can be applied at a time
            case "preprocessor":
                assert defense_class is not None
                defense = defense_class(**(self.defense_params or {}))
                defended_estimator = self._build_art_wrapper(
                    art_class=art_class,
                    base_estimator=base_estimator,
                    init_params=init_params,
                    preprocessing_defences=existing_preprocessors + [defense],
                    postprocessing_defences=existing_postprocessors,
                )
            case "postprocessor":
                assert defense_class is not None
                defense = defense_class(**(self.defense_params or {}))
                defended_estimator = self._build_art_wrapper(
                    art_class=art_class,
                    base_estimator=base_estimator,
                    init_params=init_params,
                    preprocessing_defences=existing_preprocessors,
                    postprocessing_defences=existing_postprocessors + [defense],
                )
            case "detector":
                assert defense_class is not None
                match defense_subtype:
                    case "evasion":
                        # BinaryInputDetector expects a neural-network ART classifier.
                        if not _is_torch_model_instance(
                            base_estimator,
                        ) and not isinstance(
                            self._model,
                            (PyTorchClassifier, PyTorchRegressor),
                        ):
                            raise ValueError(
                                "Evasion detector defenses only support neural-network models. "
                                f"Got base estimator type {type(base_estimator)}.",
                            )

                        detector_classifier = self._build_art_wrapper(
                            art_class=art_class,
                            base_estimator=base_estimator,
                            init_params=init_params,
                            preprocessing_defences=existing_preprocessors,
                            postprocessing_defences=existing_postprocessors,
                        )

                        detector_params = dict(self.defense_params or {})
                        # ART detector constructors differ across versions; support
                        # both keyword and positional first-argument forms.
                        try:
                            defense = defense_class(
                                detector=detector_classifier,
                                **detector_params,
                            )
                        except TypeError:
                            defense = defense_class(
                                detector_classifier,
                                **detector_params,
                            )

                        # Keep estimator interface stable for normal model runtime.
                        setattr(
                            detector_classifier,
                            "_deckard_evasion_detector",
                            defense,
                        )
                        defended_estimator = detector_classifier
                    case "poison":
                        defense = defense_class(**(self.defense_params or {}))
                        defended_estimator = defense(
                            self.get_model(),
                            **init_params,
                        )
                    case _:
                        raise NotImplementedError(
                            f"Detector subtype '{defense_subtype}' is not implemented yet.",
                        )
                # Overwrite the _score method to handle each
            case "trainer":
                assert defense_class is not None
                trainer_params = dict(self.defense_params or {})

                # Adversarial retraining defenses currently require torch-backed
                # ART estimators (e.g., PyTorchClassifier).
                if not _is_torch_model_instance(base_estimator) and not isinstance(
                    self._model,
                    (PyTorchClassifier, PyTorchRegressor),
                ):
                    raise ValueError(
                        "Retraining trainer defenses only support neural-network models. "
                        f"Got base estimator type {type(base_estimator)}.",
                    )

                trainer_classifier = self._build_art_wrapper(
                    art_class=art_class,
                    base_estimator=base_estimator,
                    init_params=init_params,
                    preprocessing_defences=existing_preprocessors,
                    postprocessing_defences=existing_postprocessors,
                )

                # ART trainer constructors differ across versions; support both
                # classifier keyword and positional first argument.
                try:
                    defense = defense_class(
                        classifier=trainer_classifier,
                        **trainer_params,
                    )
                except TypeError:
                    defense = defense_class(trainer_classifier, **trainer_params)

                # Trainer defenses configure adversarial training on top of an ART
                # classifier wrapper; fitting remains owned by the model runtime.
                if hasattr(defense, "get_classifier"):
                    defended_estimator = defense.get_classifier()
                else:
                    defended_estimator = trainer_classifier
            case "transformer":
                assert defense_class is not None
                transformer_params = dict(self.defense_params or {})
                match defense_subtype:
                    case "evasion" | "poisoning":
                        # ART transformer defenses (e.g., DefensiveDistillation,
                        # NeuralCleanse) wrap/class-transform neural ART classifiers.
                        if not _is_torch_model_instance(
                            base_estimator,
                        ) and not isinstance(
                            self._model,
                            (PyTorchClassifier, PyTorchRegressor),
                        ):
                            raise ValueError(
                                "Transformer defenses only support neural-network models. "
                                f"Got base estimator type {type(base_estimator)}.",
                            )

                        transformer_classifier = self._build_art_wrapper(
                            art_class=art_class,
                            base_estimator=base_estimator,
                            init_params=init_params,
                            preprocessing_defences=existing_preprocessors,
                            postprocessing_defences=existing_postprocessors,
                        )

                        try:
                            defense = defense_class(
                                classifier=transformer_classifier,
                                **transformer_params,
                            )
                        except TypeError:
                            try:
                                defense = defense_class(
                                    transformer_classifier,
                                    **transformer_params,
                                )
                            except NotImplementedError as exc:
                                raise ValueError(
                                    "Transformer defense initialization failed for the current "
                                    "ART classifier backend. Ensure the estimator type is "
                                    "supported by the selected defense.",
                                ) from exc
                        except NotImplementedError as exc:
                            raise ValueError(
                                "Transformer defense initialization failed for the current "
                                "ART classifier backend. Ensure the estimator type is "
                                "supported by the selected defense.",
                            ) from exc

                        if hasattr(defense, "get_classifier"):
                            defended_estimator = defense.get_classifier()
                        else:
                            defended_estimator = transformer_classifier
                    case _:
                        raise ValueError(f"Unknown transformer subtype: {defense_subtype}")
            case "regularizer":
                raise NotImplementedError(
                    "Regularizer defenses are not implemented yet.",
                )
            case None:
                defense = None
                defended_estimator = self._build_art_wrapper(
                    art_class=art_class,
                    base_estimator=base_estimator,
                    init_params={**self.defense_params, **init_params},
                    preprocessing_defences=existing_preprocessors,
                    postprocessing_defences=existing_postprocessors,
                )
            case "_":
                raise NotImplementedError(
                    f"Defense type '{defense_type}' is not implemented yet.",
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
        model_cfg = getattr(self, "_model_config", None)
        if model_cfg is not None:
            model_cfg._model = defended_estimator
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
            or isinstance(
                getattr(self, "_model", None),
                (PyTorchClassifier, PyTorchRegressor),
            )
        ):
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Torch model defenses require optional dependency deckard[torch]",
                ) from exc

            input_shape = tuple(getattr(data, "X_train").shape[1:])
            y_train = getattr(data, "y_train")
            if isinstance(y_train, torch.Tensor):
                nb_classes = int(torch.unique(y_train).numel())
            else:
                nb_classes = len(set(y_train))

            # Resolve the underlying nn.Module (unwrap if already an ART wrapper).
            raw_model = self._model
            if isinstance(raw_model, (PyTorchClassifier, PyTorchRegressor)):
                raw_model = getattr(raw_model, "model", raw_model)

            if self.classifier:
                art_class = PyTorchClassifier
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
                art_class = PyTorchRegressor
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
        art_class = (
            classifier_dict[self.model_type.split(".")[-1]]
            if self.classifier
            else regressor_dict[self.model_type.split(".")[-1]]
        )
        if art_class in sklearn_dict.values():
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


@dataclass(eq=False)
class DefensePipelineConfig(ConfigBase):
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
    def coerce(cls, defense_config: Any) -> "DefensePipelineConfig":
        if defense_config is None or isinstance(defense_config, cls):
            return defense_config

        # Keep concrete defense objects intact; converting them to dict can
        # capture runtime-only attrs that are not valid constructor kwargs.
        if hasattr(defense_config, "apply_to"):
            return cls(defenses=[defense_config])

        # List of defense specs -> chain them all inside one pipeline
        if isinstance(defense_config, (list, ListConfig)):
            return cls(defenses=coerce_to_list(defense_config))

        defense_config = coerce_config(defense_config)

        # Re-check after coerce (coerce_config may return a list)
        if isinstance(defense_config, (list, ListConfig)):
            return cls(defenses=coerce_to_list(defense_config))

        if isinstance(defense_config, dict):
            defense_dict = cast(dict[str, Any], dict(defense_config))
            if cls._is_pipeline_target(defense_dict.get("_target_")):
                defense_dict.pop("_target_", None)
                return cls(**defense_dict)
            if "defenses" in defense_dict:
                return cls(**defense_dict)
            if cls._looks_like_single_defense_spec(defense_dict):
                return cls(defenses=[defense_dict])

        raise TypeError(
            "Defense config must be a DefensePipelineConfig, a single defense spec, or None",
        )

    def _instantiate_plugin(self, plugin_spec: Any):
        if isinstance(plugin_spec, dict):
            spec = dict(plugin_spec)
            class_path = spec.pop("name", spec.pop("_target_", None))
            if class_path is None:
                raise ValueError(
                    "Plugin dict must include 'name' or '_target_'",
                )
            return resolve_class(class_path)(**spec)

        if isinstance(plugin_spec, str):
            return resolve_class(plugin_spec)()

        if isinstance(plugin_spec, type):
            return plugin_spec()

        return plugin_spec

    def _get_plugins(self) -> list:
        if self._plugin_objects is None:
            plugin_specs = self.plugins if self.plugins is not None else []
            if not isinstance(plugin_specs, list):
                raise TypeError(
                    f"plugins must be a list, got {type(plugin_specs)}",
                )
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs):
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
                        "deckard.model.fairness.FairlearnDefenseConfig",
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
        from ..utils import is_null_config_value
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
            from .fairness import FairlearnDefenseConfig

            if isinstance(defense_obj, FairlearnDefenseConfig):
                return False
        except ImportError:
            pass
        if isinstance(defense_obj, DefenseConfig):
            return True
        return False

    def _is_model_wrapper_defense(self, defense_obj) -> bool:
        """Return True for defenses that wrap/transform estimators rather than raw data."""
        if self._is_art_defense(defense_obj):
            return True
        try:
            from .fairness import FairlearnDefenseConfig

            if isinstance(defense_obj, FairlearnDefenseConfig):
                return True
        except ImportError:
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
                    "Deckard will automatically move retraining defenses to the end."
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


@dataclass(kw_only=True)
class DefenseConfig(_DefenseBehaviorMixin, ConfigBase):
    """Concrete defense config dataclass that uses shared defense behavior mixin."""

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
    alias: str = field(default_factory=str)
    _model: Union[BaseEstimator, None] = field(default=None, repr=False)
    score_dict: dict = field(default_factory=dict)
    _target_: Union[str, None] = field(default=None, repr=False)
    _model_config: Union[ModelConfig, None] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __hash__(self) -> int:
        return super().__hash__()
