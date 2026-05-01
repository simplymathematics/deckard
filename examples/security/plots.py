import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
LOGS_DIR = ROOT / "outputs" / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_dict(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}
    flat = {}

    def _walk(prefix: str, value):
        if isinstance(value, dict):
            for key, val in value.items():
                next_key = f"{prefix}.{key}" if prefix else str(key)
                _walk(next_key, val)
        else:
            flat[prefix] = value

    _walk("", data)
    return flat


def load_trial_frame(logs_dir: Path) -> pd.DataFrame:
    rows = []
    for trial_dir in logs_dir.glob("*/*"):
        params_file = trial_dir / "params.yaml"
        scores_file = trial_dir / "scores.json"
        if not params_file.exists() or not scores_file.exists():
            continue
        with params_file.open("r") as f:
            params = yaml.safe_load(f) or {}
        with scores_file.open("r") as f:
            scores = json.load(f) or {}
        row = {}
        row.update(_flatten_dict(params))
        row.update(_flatten_dict(scores))
        row["trial_dir"] = str(trial_dir)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No trial artifacts found under {logs_dir}")
    return pd.DataFrame(rows)


def _safe_lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    style: str,
    output_name: str,
    xlabel: str,
    ylabel: str,
    xlog: bool = False,
    ylog: bool = False,
) -> None:
    if any(col not in df.columns for col in [x, y, style]):
        logger.warning("Skipping %s due to missing columns", output_name)
        return
    ax = sns.lineplot(
        x=x,
        y=y,
        data=df,
        style=style,
        err_style="bars",
        errorbar=("ci", 99),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, title=style)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(PLOTS_DIR / output_name)
    plt.gcf().clear()


def main() -> None:
    df = load_trial_frame(LOGS_DIR)

    # Normalize current config keys into plotting aliases.
    rename_map = {
        "model.model_params.kernel": "Kernel",
        "data.data_params.n_features": "Features",
        "data.train_size": "Samples",
        "attack.attack_params.eps": "eps",
        "attack.attack_params.eps_step": "eps_step",
        "attack.attack_params.max_iter": "max_iter",
        "attack.attack_params.batch_size": "batch_size",
        "attack_generation_time": "attack_time",
        "training_time": "train_time",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "Kernel" in df.columns:
        df = df[df["Kernel"] != "sigmoid"]

    _safe_lineplot(
        df,
        x="Samples",
        y="accuracy",
        style="Kernel",
        output_name="accuracy_vs_samples.eps",
        xlabel="Number of Samples",
        ylabel="Accuracy",
        xlog=True,
    )
    _safe_lineplot(
        df,
        x="Features",
        y="accuracy",
        style="Kernel",
        output_name="accuracy_vs_features.eps",
        xlabel="Number of Features",
        ylabel="Accuracy",
        xlog=True,
    )
    _safe_lineplot(
        df,
        x="Features",
        y="train_time",
        style="Kernel",
        output_name="train_time_vs_features.eps",
        xlabel="Number of Features",
        ylabel="Training Time",
        xlog=True,
        ylog=True,
    )
    _safe_lineplot(
        df,
        x="Samples",
        y="train_time",
        style="Kernel",
        output_name="train_time_vs_samples.eps",
        xlabel="Number of Samples",
        ylabel="Training Time",
        xlog=True,
        ylog=True,
    )

    # Attack parameter plots.
    for x_col, out_name, x_label in [
        ("eps", "accuracy_vs_attack_eps.eps", "Perturbation Distance"),
        ("eps_step", "accuracy_vs_attack_eps_step.eps", "Perturbation Step"),
        ("max_iter", "accuracy_vs_attack_max_iter.eps", "Maximum Iterations"),
        ("batch_size", "accuracy_vs_attack_batch_size.eps", "Batch Size"),
    ]:
        _safe_lineplot(
            df,
            x=x_col,
            y="accuracy",
            style="Kernel",
            output_name=out_name,
            xlabel=x_label,
            ylabel="Accuracy",
            xlog=True,
        )

    for x_col, out_name, x_label in [
        ("eps", "attack_time_vs_eps.eps", "Perturbation Distance"),
        ("eps_step", "attack_time_vs_eps_step.eps", "Perturbation Step"),
        ("max_iter", "attack_time_vs_max_iter.eps", "Maximum Iterations"),
        ("batch_size", "attack_time_vs_batch_size.eps", "Batch Size"),
    ]:
        _safe_lineplot(
            df,
            x=x_col,
            y="attack_time",
            style="Kernel",
            output_name=out_name,
            xlabel=x_label,
            ylabel="Attack Time",
            xlog=True,
            ylog=True,
        )


if __name__ == "__main__":
    main()
