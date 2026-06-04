import argparse
import importlib
import logging
from pathlib import Path
from typing import Any, Dict

from hydra._internal.utils import get_args_parser
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..plot.canon import normalize_plot_backend

logger = logging.getLogger(__name__)


PLOT_MAIN_DEFAULTS = {
    "experiment_config": "",
    "data_file": "",
    "backend": "auto",
    "plot_type": "",
    "plots": "",
    "plot_params_file": "",
    "plot_file": "",
    "plot_folder": "",
    "features": "all",
    "classes": "all",
    "x": "",
    "y": "",
    "hue": "",
    "style": "",
    "title": "",
    "xlabel": "",
    "ylabel": "",
    "xscale": "",
    "yscale": "",
    "legend_title": "",
    "kwargs_file": "",
    "rc_config_file": "",
}


def _load_experiment_config(experiment_config: str) -> Dict[str, Any]:
    cfg_path = Path(experiment_config)
    assert cfg_path.exists(), f"Experiment config file not found: {cfg_path}"

    raw_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(raw_cfg, dict):
        raise TypeError(
            f"Experiment config must resolve to a dictionary: {cfg_path}",
        )
    return raw_cfg


def _parse_plots_arg(plots: str) -> list:
    if isinstance(plots, list):
        return [str(item).strip() for item in plots if str(item).strip()]
    return [item.strip() for item in str(plots).split(",") if item.strip()]


def _normalize_yellowbrick_plots(plots: Any):
    if isinstance(plots, str) and plots.strip().lower() == "all":
        return "all"
    if isinstance(plots, list):
        normalized = [str(item).strip() for item in plots if str(item).strip()]
        if len(normalized) == 1 and normalized[0].lower() == "all":
            return "all"
        return normalized
    return _parse_plots_arg(str(plots))


def _load_yaml(path: str):
    yaml_path = Path(path)
    assert yaml_path.exists(), f"YAML file not found: {yaml_path}"
    loaded = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    return loaded


def _instantiate_experiment_cfg(exp_cfg: Dict[str, Any]) -> Any:
    cfg = dict(exp_cfg)
    cfg["_target_"] = cfg.get("_target_", "deckard.ExperimentConfig")
    return instantiate(cfg)


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        resolved = OmegaConf.to_container(cfg, resolve=True)
        return resolved if isinstance(resolved, dict) else {}
    if isinstance(cfg, dict):
        return cfg
    try:
        resolved = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
        return resolved if isinstance(resolved, dict) else {}
    except Exception:
        return {}


def _resolve_plot_args_from_cfg(cfg: Any) -> Dict[str, Any]:
    """Resolve plot settings from Hydra cfg using `plot` block first, top-level as fallback."""
    cfg_dict = _cfg_to_dict(cfg)
    plot_block = (
        cfg_dict.get("plot", {}) if isinstance(cfg_dict.get("plot"), dict) else {}
    )

    resolved = dict(PLOT_MAIN_DEFAULTS)
    for key in PLOT_MAIN_DEFAULTS:
        if key in plot_block and plot_block[key] is not None:
            resolved[key] = plot_block[key]
        elif key in cfg_dict and cfg_dict[key] is not None:
            resolved[key] = cfg_dict[key]
    return resolved


def _extract_experiment_cfg_from_hydra_cfg(cfg: Any) -> Dict[str, Any]:
    """Return experiment-like config from Hydra cfg for Yellowbrick backend inference."""
    cfg_dict = _cfg_to_dict(cfg)
    if not cfg_dict:
        return {}

    # If cfg contains a dedicated plot block with experiment inputs, prefer it.
    if isinstance(cfg_dict.get("plot"), dict):
        plot_block = cfg_dict["plot"]
        if isinstance(plot_block.get("experiment"), dict):
            return plot_block["experiment"]

    # Common optimize/default Hydra config shape: experiment fields are top-level.
    if "data" in cfg_dict and ("model" in cfg_dict or "files" in cfg_dict):
        return cfg_dict

    # Alternate shape where experiment is nested explicitly.
    if isinstance(cfg_dict.get("experiment"), dict):
        return cfg_dict["experiment"]

    return {}


def _resolve_experiment_config_path(cfg: Any) -> str:
    cfg_dict = _cfg_to_dict(cfg)
    plot_block = (
        cfg_dict.get("plot", {}) if isinstance(cfg_dict.get("plot"), dict) else {}
    )
    if isinstance(plot_block.get("experiment_config"), str):
        return plot_block["experiment_config"]
    if isinstance(cfg_dict.get("experiment_config"), str):
        return cfg_dict["experiment_config"]
    return ""


def _resolve_data_file(cfg: Any) -> str:
    cfg_dict = _cfg_to_dict(cfg)
    plot_block = (
        cfg_dict.get("plot", {}) if isinstance(cfg_dict.get("plot"), dict) else {}
    )
    if isinstance(plot_block.get("data_file"), str):
        return plot_block["data_file"]
    if isinstance(cfg_dict.get("data_file"), str):
        return cfg_dict["data_file"]
    if isinstance(cfg_dict.get("compile_results"), dict):
        candidate = cfg_dict["compile_results"].get("output_file", "")
        if isinstance(candidate, str):
            return candidate
    return ""


def _extract_backend(
    cfg: Any,
    data_file: str,
    experiment_cfg: Dict[str, Any],
    experiment_config: str,
) -> str:
    def _normalize_backend_value(value: Any) -> str:
        token = str(value).strip().lower()
        if token == "auto":
            return "auto"
        try:
            return normalize_plot_backend(value)
        except Exception as exc:
            raise ValueError(
                "backend must be one of: auto, yellowbrick, seaborn",
            ) from exc

    cfg_dict = _cfg_to_dict(cfg)
    plot_block = (
        cfg_dict.get("plot", {}) if isinstance(cfg_dict.get("plot"), dict) else {}
    )
    backend = plot_block.get("backend", cfg_dict.get("backend", None))
    if backend is not None:
        backend = _normalize_backend_value(backend)
    else:
        backend = "auto"
    if backend not in {"auto", "yellowbrick", "seaborn"}:
        raise ValueError("backend must be one of: auto, yellowbrick, seaborn")
    if backend == "auto":
        if experiment_config or experiment_cfg:
            return "yellowbrick"
        if data_file:
            return "seaborn"
        raise ValueError(
            "Could not infer backend: provide plot.data_file or plot.experiment_config/experiment cfg",
        )
    return backend


def plot_main(cfg: Any) -> dict:
    """Execute plotting from experiment config or tabular results.

    Args:
        cfg: Plot configuration payload containing backend selection and
            backend-specific parameters.

    Returns:
        A dictionary describing plotting backend, mode, output locations, and
        score payload returned by the plotting runtime.

    Raises:
        ValueError: If backend routing inputs are missing or inconsistent.
        TypeError: If YAML-backed plotting parameter files do not resolve to
            dictionaries.
    """
    extracted_experiment_cfg = _extract_experiment_cfg_from_hydra_cfg(cfg)
    resolved = _resolve_plot_args_from_cfg(cfg)

    experiment_config = _resolve_experiment_config_path(cfg)
    data_file = _resolve_data_file(cfg)
    backend = _extract_backend(
        cfg,
        data_file=data_file,
        experiment_cfg=extracted_experiment_cfg,
        experiment_config=experiment_config,
    )

    plot_type = resolved["plot_type"]
    plots = resolved["plots"]
    plot_params_file = resolved["plot_params_file"]
    plot_file = resolved["plot_file"]
    plot_folder = resolved["plot_folder"]
    features = resolved["features"]
    classes = resolved["classes"]
    x = resolved["x"]
    y = resolved["y"]
    hue = resolved["hue"]
    style = resolved["style"]
    title = resolved["title"]
    xlabel = resolved["xlabel"]
    ylabel = resolved["ylabel"]
    xscale = resolved["xscale"]
    yscale = resolved["yscale"]
    legend_title = resolved["legend_title"]
    kwargs_file = resolved["kwargs_file"]
    rc_config_file = resolved["rc_config_file"]

    _validate_backend_inputs(
        backend=backend,
        experiment_config=experiment_config,
        extracted_experiment_cfg=extracted_experiment_cfg,
        data_file=data_file,
    )
    if backend == "yellowbrick":
        return _run_yellowbrick_backend(
            extracted_experiment_cfg=extracted_experiment_cfg,
            experiment_config=experiment_config,
            plot_type=plot_type,
            plots=plots,
            plot_params_file=plot_params_file,
            plot_file=plot_file,
            plot_folder=plot_folder,
            features=features,
            classes=classes,
            title=title,
        )
    return _run_seaborn_backend(
        data_file=data_file,
        plot_type=plot_type,
        plots=plots,
        plot_params_file=plot_params_file,
        plot_file=plot_file,
        x=x,
        y=y,
        hue=hue,
        style=style,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xscale=xscale,
        yscale=yscale,
        legend_title=legend_title,
        kwargs_file=kwargs_file,
        rc_config_file=rc_config_file,
    )


def _validate_backend_inputs(
    *,
    backend: str,
    experiment_config: str,
    extracted_experiment_cfg: Dict[str, Any],
    data_file: str,
) -> None:
    if (
        backend == "yellowbrick"
        and not experiment_config
        and not extracted_experiment_cfg
    ):
        raise ValueError(
            "yellowbrick backend requires plot.experiment_config or a Hydra experiment cfg",
        )
    if backend == "seaborn" and not data_file:
        raise ValueError("seaborn backend requires plot.data_file")


def _run_yellowbrick_backend(
    *,
    extracted_experiment_cfg: Dict[str, Any],
    experiment_config: str,
    plot_type: str,
    plots: Any,
    plot_params_file: str,
    plot_file: str,
    plot_folder: str,
    features: Any,
    classes: Any,
    title: str,
) -> dict:
    exp_cfg = (
        extracted_experiment_cfg
        if extracted_experiment_cfg
        else _load_experiment_config(experiment_config)
    )
    exp_obj = _instantiate_experiment_cfg(exp_cfg)
    resolved_plots = plots
    if not plot_type and not resolved_plots:
        resolved_plots = "all"

    plot_params = {}
    if plot_params_file:
        loaded = _load_yaml(plot_params_file)
        if not isinstance(loaded, dict):
            raise TypeError("plot_params_file must contain a dictionary.")
        plot_params = loaded

    yellowbrick_module = importlib.import_module("deckard.plugins.yellowbrick.plot")
    YellowbrickConfigList = yellowbrick_module.YellowbrickConfigList
    YellowbrickPlotConfig = yellowbrick_module.YellowbrickPlotConfig

    if plot_type:
        default_folder = Path(plot_folder) if plot_folder else Path.cwd()
        output_path = (
            Path(plot_file) if plot_file else default_folder / f"{plot_type}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        single_title = title if title else plot_type.replace("_", " ").title()
        single_cfg = YellowbrickPlotConfig(
            experiment=exp_obj,
            plot_type=plot_type,
            features=features,
            classes=classes,
            title=single_title,
            save_path=output_path.as_posix(),
            plot_params=plot_params,
        )
        scores = single_cfg()
        return {
            "backend": "yellowbrick",
            "mode": "single",
            "plot_type": plot_type,
            "plot_file": output_path.as_posix(),
            "scores": scores,
        }

    plot_list = _normalize_yellowbrick_plots(resolved_plots)
    if isinstance(plot_list, list) and len(plot_list) == 0:
        raise ValueError(
            "--plots must contain at least one plot type for yellowbrick backend.",
        )

    output_dir = Path(plot_folder) if plot_folder else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    list_cfg = YellowbrickConfigList(
        experiment=exp_obj,
        plots=plot_list,
        plot_folder=output_dir.as_posix(),
    )
    scores = list_cfg()
    return {
        "backend": "yellowbrick",
        "mode": "multi",
        "plots": plot_list,
        "plot_folder": output_dir.as_posix(),
        "scores": scores,
    }


def _run_seaborn_backend(
    *,
    data_file: str,
    plot_type: str,
    plots: Any,
    plot_params_file: str,
    plot_file: str,
    x: str,
    y: str,
    hue: str,
    style: str,
    title: str,
    xlabel: str,
    ylabel: str,
    xscale: str,
    yscale: str,
    legend_title: str,
    kwargs_file: str,
    rc_config_file: str,
) -> dict:
    _ = plots
    seaborn_module = importlib.import_module("deckard.plugins.seaborn.plot")
    SeabornPlotConfig = seaborn_module.SeabornPlotConfig
    SeabornPlotConfigList = seaborn_module.SeabornPlotConfigList

    kwargs = {}
    if kwargs_file:
        loaded = _load_yaml(kwargs_file)
        if not isinstance(loaded, dict):
            raise TypeError("kwargs_file must contain a dictionary.")
        kwargs = loaded

    rc_config = {}
    if rc_config_file:
        loaded = _load_yaml(rc_config_file)
        if not isinstance(loaded, dict):
            raise TypeError("rc_config_file must contain a dictionary.")
        rc_config = loaded

    if plot_type and plots:
        raise ValueError("Provide only one of plot.plot_type or plot.plots")
    if not plot_type and not plot_params_file:
        raise ValueError(
            "Provide one of plot.plot_type or plot.plot_params_file for seaborn backend.",
        )
    if plot_type and plot_params_file:
        raise ValueError(
            "Provide only one of plot.plot_type or plot.plot_params_file for seaborn backend.",
        )

    if plot_type:
        if not x or not y:
            raise ValueError("seaborn single-plot mode requires plot.x and plot.y")
        single_cfg = SeabornPlotConfig(
            data_file=data_file,
            plot_type=plot_type,
            x=x,
            y=y,
            hue=hue or None,
            style=style or None,
            title=title or None,
            xlabel=xlabel or None,
            ylabel=ylabel or None,
            xscale=xscale or None,
            yscale=yscale or None,
            legend_title=legend_title or None,
            kwargs=kwargs,
            rc_config=rc_config,
            plot_file=plot_file or None,
        )
        single_cfg()
        return {
            "backend": "seaborn",
            "mode": "single",
            "plot_type": plot_type,
            "plot_file": plot_file or None,
        }

    loaded = _load_yaml(plot_params_file)
    if isinstance(loaded, dict) and "plots" in loaded:
        plot_specs = loaded["plots"]
    elif isinstance(loaded, list):
        plot_specs = loaded
    else:
        raise TypeError(
            "plot_params_file must contain a list or a dict with key 'plots'.",
        )

    plot_cfgs = []
    for spec in plot_specs:
        if not isinstance(spec, dict):
            raise TypeError("Each item in plot_params_file must be a dictionary.")
        merged = dict(spec)
        merged.setdefault("data_file", data_file)
        if "kwargs" not in merged and kwargs:
            merged["kwargs"] = kwargs
        if "rc_config" not in merged and rc_config:
            merged["rc_config"] = rc_config
        plot_cfgs.append(SeabornPlotConfig(**merged))

    list_cfg = SeabornPlotConfigList(plots=plot_cfgs, data_file=data_file)
    list_cfg.file = plot_file or None
    list_cfg()
    return {
        "backend": "seaborn",
        "mode": "multi",
        "plot_params_file": str(Path(plot_params_file).resolve()),
        "plot_file": plot_file or None,
        "num_plots": len(plot_cfgs),
    }


hydra_parser = argparse.ArgumentParser(
    parents=[get_args_parser()],
    add_help=False,
    usage="deckard plot --config-dir=conf --config-name=default.yaml plot.plot_type=pairplot",
)

plot_parser = hydra_parser


if __name__ == "__main__":
    raise SystemExit(
        "Run this layer via deckard entrypoint with Hydra, e.g. `deckard plot ...`",
    )
