import logging
import argparse
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from hydra._internal.utils import get_args_parser
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


PLOT_MAIN_DEFAULTS = {
    "experiment_config": "",
    "data_file": "",
    "backend": "auto",
    "plot_type": "",
    "plots": "",
    "plots_file": "",
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
    "plot_params_file": "",
    "kwargs_file": "",
    "rc_config_file": "",
}


def _load_experiment_config(experiment_config: str) -> Dict[str, Any]:
    cfg_path = Path(experiment_config)
    assert cfg_path.exists(), f"Experiment config file not found: {cfg_path}"

    raw_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(raw_cfg, dict):
        raise TypeError(f"Experiment config must resolve to a dictionary: {cfg_path}")
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
    cfg: Any, data_file: str, experiment_cfg: Dict[str, Any], experiment_config: str
) -> str:
    cfg_dict = _cfg_to_dict(cfg)
    plot_block = (
        cfg_dict.get("plot", {}) if isinstance(cfg_dict.get("plot"), dict) else {}
    )
    backend = plot_block.get("backend", cfg_dict.get("backend", "auto"))
    if backend not in {"auto", "yellowbrick", "seaborn"}:
        raise ValueError("backend must be one of: auto, yellowbrick, seaborn")
    if backend == "auto":
        if data_file:
            return "seaborn"
        if experiment_config or experiment_cfg:
            return "yellowbrick"
        raise ValueError(
            "Could not infer backend: provide plot.data_file or plot.experiment_config/experiment cfg"
        )
    return backend


def plot_main(cfg: Any) -> dict:
    """Execute plotting from either experiment config (Yellowbrick) or tabular results (Seaborn).

    Parameters
    ----------
    experiment_config:
            Path to an experiment YAML file used by Yellowbrick mode.
    data_file:
            Path to an aggregated data file (CSV/Parquet/etc.) used by Seaborn mode.
    backend:
            `auto`, `yellowbrick`, or `seaborn`.
    plot_type:
            Plot type for single-plot mode.
    plots:
            Comma-separated plot types for multi-plot mode.
    plots_file:
            YAML file containing a list of Seaborn plot configurations under `plots` or as a top-level list.
    plot_file:
            Output path for single-plot mode or optional combined figure in Seaborn list mode.
    plot_folder:
            Output directory for Yellowbrick multi-plot mode.
    features:
            Feature selection for Yellowbrick plot config.
    classes:
            Class selection for Yellowbrick plot config.
    x, y, hue, style:
            Seaborn axis/channel columns for single-plot mode.
    title:
            Optional custom plot title.
    xlabel, ylabel, xscale, yscale, legend_title:
            Optional Seaborn axis/legend formatting.
    plot_params_file:
            Optional YAML file containing Yellowbrick `plot_params`.
    kwargs_file:
            Optional YAML file containing Seaborn `kwargs`.
    rc_config_file:
            Optional YAML file containing matplotlib rcParams for Seaborn.
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
    plots_file = resolved["plots_file"]
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
    plot_params_file = resolved["plot_params_file"]
    kwargs_file = resolved["kwargs_file"]
    rc_config_file = resolved["rc_config_file"]

    if backend == "yellowbrick" and not experiment_config:
        if not extracted_experiment_cfg:
            raise ValueError(
                "yellowbrick backend requires plot.experiment_config or a Hydra experiment cfg"
            )
    if backend == "seaborn" and not data_file:
        raise ValueError("seaborn backend requires plot.data_file")

    if not plot_type and not plots and not plots_file:
        if backend == "yellowbrick":
            plots = "all"
        else:
            raise ValueError(
                "Provide one of plot.plot_type, plot.plots, or plot.plots_file."
            )
    if sum(bool(x) for x in [plot_type, plots, plots_file]) > 1:
        raise ValueError(
            "Provide only one of plot.plot_type, plot.plots, or plot.plots_file."
        )

    if backend == "yellowbrick" and plots_file:
        raise ValueError("plot.plots_file is only supported for seaborn backend.")

    if backend == "yellowbrick":
        exp_cfg = (
            extracted_experiment_cfg
            if extracted_experiment_cfg
            else _load_experiment_config(experiment_config)
        )
        exp_obj = _instantiate_experiment_cfg(exp_cfg)

        plot_params = {}
        if plot_params_file:
            loaded = _load_yaml(plot_params_file)
            if not isinstance(loaded, dict):
                raise TypeError("plot_params_file must contain a dictionary.")
            plot_params = loaded

        # Import lazily so this layer can be listed even when optional plotting deps are missing.
        from ..plot.yellowbrick_plots import (
            YellowbrickConfigList,
            YellowbrickPlotConfig,
        )

        if plot_type:
            default_folder = Path(plot_folder) if plot_folder else Path.cwd()
            output_path = (
                Path(plot_file) if plot_file else default_folder / f"{plot_type}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            single_title = title if title else plot_type.replace("_", " ").title()
            cfg = YellowbrickPlotConfig(
                experiment=exp_obj,
                plot_type=plot_type,
                features=features,
                classes=classes,
                title=single_title,
                save_path=output_path.as_posix(),
                plot_params=plot_params,
            )
            scores = cfg()
            return {
                "backend": "yellowbrick",
                "mode": "single",
                "plot_type": plot_type,
                "plot_file": output_path.as_posix(),
                "scores": scores,
            }

        plot_list = _normalize_yellowbrick_plots(plots)
        if isinstance(plot_list, list) and len(plot_list) == 0:
            raise ValueError(
                "--plots must contain at least one plot type for yellowbrick backend."
            )

        output_dir = Path(plot_folder) if plot_folder else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = YellowbrickConfigList(
            experiment=exp_obj,
            plots=plot_list,
            plot_folder=output_dir.as_posix(),
        )
        scores = cfg()
        return {
            "backend": "yellowbrick",
            "mode": "multi",
            "plots": plot_list,
            "plot_folder": output_dir.as_posix(),
            "scores": scores,
        }

    # Seaborn backend: designed for aggregated/tabular outputs from many experiments.
    from ..plot.seaborn_plots import SeabornPlotConfig, SeabornPlotConfigList

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

    if plot_type:
        if not x or not y:
            raise ValueError("seaborn single-plot mode requires plot.x and plot.y")
        cfg = SeabornPlotConfig(
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
        cfg()
        return {
            "backend": "seaborn",
            "mode": "single",
            "plot_type": plot_type,
            "plot_file": plot_file or None,
        }

    if not plots_file:
        raise ValueError("seaborn multi-plot mode requires plot.plots_file")

    loaded = _load_yaml(plots_file)
    if isinstance(loaded, dict) and "plots" in loaded:
        plot_specs = loaded["plots"]
    elif isinstance(loaded, list):
        plot_specs = loaded
    else:
        raise TypeError("plots_file must contain a list or a dict with key 'plots'.")

    plot_cfgs = []
    for spec in plot_specs:
        if not isinstance(spec, dict):
            raise TypeError("Each item in plots_file must be a dictionary.")
        merged = dict(spec)
        merged.setdefault("data_file", data_file)
        if "kwargs" not in merged and kwargs:
            merged["kwargs"] = kwargs
        if "rc_config" not in merged and rc_config:
            merged["rc_config"] = rc_config
        plot_cfgs.append(SeabornPlotConfig(**merged))

    list_cfg = SeabornPlotConfigList(plots=plot_cfgs, data_file=data_file)
    # SeabornPlotConfigList.__call__ checks `self.file`; set explicitly for stability.
    list_cfg.file = plot_file or None
    list_cfg()
    return {
        "backend": "seaborn",
        "mode": "multi",
        "plots_file": str(Path(plots_file).resolve()),
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
        "Run this layer via deckard entrypoint with Hydra, e.g. `deckard plot ...`"
    )
