import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..data import DataConfig
from ..experiment import SurvivalExperimentConfig
from ..plugins.lifelines.plot import (
    SurvivalSeabornPlotConfigList,
)
from ..utils import create_parser_from_function, instantiate_config

__all__ = [
    "survival_main",
    "survival_parser",
]


def survival_main(cfg: dict = None) -> dict:
    """Run survival workflow from Hydra-parsed config.

    Routes to either plot-only rendering or full experiment based on config.
    All parameters come from Hydra instantiation, no runtime parameters.

    Args:
        cfg: DictConfig or dict containing survival experiment configuration.
             Should have a 'survival' section where values resolve via
             instantiation to:
             - data: required DataConfig
             - model: required survival fitter name (e.g. weibull, cox)
             - target: required event column name
             - duration_col: required duration column name
             - aux_model: optional, auxiliary model for attacks
             - attack: optional, attack config
             - plot or model_config with plot specs: optional, enables plot-only mode

    Returns:
        dict with aft_table, model_scores, and models (or plot results if plot mode)
    """
    if cfg is None:
        raise ValueError("survival_main requires a Hydra config (cfg)")

    cfg_dict = (
        OmegaConf.to_container(cfg) if isinstance(cfg, DictConfig) else dict(cfg)
    )

    # Extract survival section from Hydra config
    survival_cfg = cfg_dict.get("survival", cfg_dict)
    if not isinstance(survival_cfg, dict):
        raise ValueError("Hydra config must contain 'survival' section")

    _validate_raw_data_model_specs(survival_cfg)
    survival_cfg = _coerce_survival_model_spec(survival_cfg)

    # Check if this is plot-only mode (has plot specifications)
    has_plot_spec = _has_plot_specification(survival_cfg)

    if has_plot_spec:
        # Plot-only mode: instantiate plotter config and render
        return _run_plot_mode(survival_cfg)
    else:
        # Experiment mode: instantiate experiment config and run
        return _run_experiment_mode(survival_cfg, cfg_dict)


def _validate_raw_data_model_specs(survival_cfg: dict) -> None:
    data_spec = survival_cfg.get("data")
    model_spec = survival_cfg.get("model")

    if isinstance(data_spec, str):
        raise TypeError(
            "survival.data must be a DataConfig object or instantiable mapping, not a string",
        )
    if isinstance(model_spec, str):
        if model_spec.strip() == "":
            raise ValueError(
                "survival.model must be a non-empty survival model string",
            )
        return
    if isinstance(model_spec, dict):
        return
    raise TypeError(
        "survival.model must be a survival model string or model config mapping",
    )


def _coerce_survival_model_spec(survival_cfg: dict) -> dict:
    """Normalize `survival.model` to a survival-model string.

    Accepts either a direct model string (preferred) or a mapping from Hydra
    model group configs where the alias/model_type carries the fitter name.
    """

    def _resolve_model_alias_placeholders(value, model_name: str):
        if isinstance(value, str):
            return value.replace("${model.alias}", model_name)
        if isinstance(value, dict):
            return {
                k: _resolve_model_alias_placeholders(v, model_name)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [_resolve_model_alias_placeholders(v, model_name) for v in value]
        return value

    def _normalize_model_name(candidate: str) -> str:
        text = candidate.strip()
        if text == "":
            raise ValueError("survival.model value cannot be blank")
        short = text.rsplit(".", 1)[-1]
        lowered = short.lower()
        aliases = {
            "cox": "cox",
            "coxph": "cox",
            "coxphfitter": "cox",
            "weibull": "weibull",
            "weibullaft": "weibull",
            "weibullaftfitter": "weibull",
            "loglogistic": "log-logistic",
            "loglogisticaftfitter": "log-logistic",
            "lognormal": "log-normal",
            "lognormalaftfitter": "log-normal",
            "aalen": "aalen",
            "aalenadditive": "aalen",
            "aalenadditivefitter": "aalen",
            "gammaph": "gamma",
            "generalizedgamma": "gamma",
            "generalizedgammaregressionfitter": "gamma",
            "exponential": "exponential",
            "exponentialfitter": "exponential",
        }
        normalized = lowered.replace("-", "").replace("_", "")
        return aliases.get(normalized, lowered)

    model_spec = survival_cfg.get("model")
    if isinstance(model_spec, str):
        normalized = dict(survival_cfg)
        normalized["model"] = _normalize_model_name(model_spec)
        return normalized
    if not isinstance(model_spec, dict):
        return survival_cfg

    candidate = (
        model_spec.get("alias")
        or model_spec.get("survival_model")
        or model_spec.get("model_type")
    )
    if not isinstance(candidate, str) or candidate.strip() == "":
        raise ValueError(
            "Could not resolve survival model string from survival.model mapping. "
            "Provide 'alias', 'survival_model', or 'model_type'.",
        )
    normalized = dict(survival_cfg)
    model_name = _normalize_model_name(candidate)
    normalized["model"] = model_name
    if "plot" in normalized:
        normalized["plot"] = _resolve_model_alias_placeholders(
            normalized["plot"],
            model_name,
        )
    return normalized


def _validate_experiment_config_types(
    experiment_config: SurvivalExperimentConfig,
) -> None:
    if not isinstance(experiment_config.data, DataConfig):
        raise TypeError(
            "survival.data must resolve to a DataConfig instance via Hydra instantiation",
        )
    if not isinstance(experiment_config.model, str):
        raise TypeError(
            "survival.model must resolve to a survival model string",
        )


def _load_plot_dataframe(experiment_config: SurvivalExperimentConfig) -> pd.DataFrame:
    data_cfg = experiment_config.data
    if data_cfg.X is None:
        data_cfg._load_data()

    if data_cfg.X is None:
        raise ValueError("DataConfig did not load features for plot mode")

    frame = (
        data_cfg.X.to_frame().copy()
        if isinstance(data_cfg.X, pd.Series)
        else pd.DataFrame(data_cfg.X).copy()
    )

    if data_cfg.y is not None and experiment_config.target not in frame.columns:
        frame[experiment_config.target] = data_cfg.y.values

    if experiment_config.duration_col not in frame.columns:
        raise ValueError(
            f"duration_col {experiment_config.duration_col!r} not found in loaded data",
        )
    if experiment_config.target not in frame.columns:
        raise ValueError(
            f"target {experiment_config.target!r} not found in loaded data",
        )
    return frame


def _has_plot_specification(survival_cfg: dict) -> bool:
    """Check if config contains plot specifications."""
    # Plot-only mode is enabled when model_config contains explicit per-model
    # plot entries (or when an explicit flag is provided).
    if bool(survival_cfg.get("plot_only", False)):
        return True

    # Check for model_config with plot sub-configs.
    model_cfg = survival_cfg.get("model_config")
    if isinstance(model_cfg, dict):
        for model_type, model_spec in model_cfg.items():
            if isinstance(model_spec, dict) and "plot" in model_spec:
                return True

    return False


def _run_plot_mode(survival_cfg: dict) -> dict:
    """Run plot-only rendering mode."""
    # Create survival config first and enforce typed inputs.
    experiment_config = instantiate_config(
        survival_cfg,
        SurvivalExperimentConfig,
    )
    _validate_experiment_config_types(experiment_config)

    model_config = survival_cfg.get("model_config", {})

    if not isinstance(model_config, dict) or len(model_config) == 0:
        raise ValueError(
            "plot mode requires model_config dict with plot specifications",
        )

    # Instantiate plotter config list (handles single or multiple models)
    plotter_config = instantiate_config(
        {"model_config": model_config, **survival_cfg},
        SurvivalSeabornPlotConfigList,
    )

    plot_data = _load_plot_dataframe(experiment_config)

    # Call plotter with the config
    return plotter_config(
        model_config=model_config,
        data=plot_data,
        survival_config=experiment_config,
        dataset=survival_cfg.get("dataset"),
        test_size=survival_cfg.get("test_size", 0.25),
        folder=survival_cfg.get("plots_folder", "plots/survival"),
        dummy_dict=survival_cfg.get("dummies", {}),
    )


def _run_experiment_mode(survival_cfg: dict, full_cfg: dict) -> dict:
    """Run full experiment mode."""
    # Instantiate SurvivalExperimentConfig from Hydra config
    experiment_config = instantiate_config(
        survival_cfg,
        SurvivalExperimentConfig,
    )
    _validate_experiment_config_types(experiment_config)

    _ = full_cfg
    return experiment_config()


survival_parser = create_parser_from_function(survival_main)


if __name__ == "__main__":
    args = survival_parser.parse_args()
    survival_main(**vars(args))
