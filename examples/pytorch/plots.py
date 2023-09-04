# %% [markdown]
# # Dependencies

# %%

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)


def cat_plot(
    data,
    x,
    y,
    hue,
    kind,
    titles,
    xlabels,
    ylabels,
    legend_title,
    file,
    folder,
    set={},
    **kwargs,
):
    plt.gcf().clear()
    graph = sns.catplot(data=data, x=x, y=y, hue=hue, kind=kind, **kwargs)
    graph.set_xlabels(xlabels)
    graph.set_ylabels(ylabels)
    graph.set_titles(titles)
    graph.legend.set_title(title=legend_title)
    graph.set(**set)
    graph.savefig(folder / file)
    plt.gcf().clear()
    logger.info(f"Saved graph to {folder / file}")


def line_plot(
    data,
    x,
    y,
    hue,
    xlabel,
    ylabel,
    title,
    file,
    folder,
    y_scale=None,
    x_scale=None,
    legend={},
    control=None,
    control_color=None,
):
    plt.gcf().clear()
    graph = sns.lineplot(data=data, x=x, y=y, hue=hue, style=control)
    if control is not None:
        assert control_color is not None, "Please specify a control color"
        graph.add_line(plt.axhline(y=control, color=control_color, linestyle="-"))
    graph.set_xlabel(xlabel)
    graph.set_ylabel(ylabel)
    graph.set_title(title)
    graph.legend(**legend)
    if y_scale is not None:
        graph.set_yscale(y_scale)
    if x_scale is not None:
        graph.set_xscale(x_scale)
    graph.get_figure().tight_layout()
    graph.get_figure().savefig(folder / file)
    plt.gcf().clear()
    return graph

def scatter_plot(
    data,
    x,
    y,
    hue,
    xlabel,
    ylabel,
    title,
    file,
    folder,
    y_scale=None,
    x_scale=None,
    legend={},
):
    graph = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
    )
    graph.set_yscale(y_scale)
    graph.set_xscale(x_scale)
    graph.set_xlabel(xlabel)
    graph.set_ylabel(ylabel)
    graph.legend(**legend)
    graph.set_title(title)
    graph.get_figure().tight_layout()
    graph.get_figure().savefig(Path(folder) / file)
    logger.info(f"Rendering graph {i+1}")
    logger.info(f"Saved graph to {Path(folder) / file}")
    plt.gcf().clear()
    return graph

def calculate_failure_rate(data):
    data = data[data.columns.drop(list(data.filter(regex="\.1$")))]
    data.columns.str.replace(" ", "")
    data.dropna(axis=0, subset=['accuracy', 'adv_accuracy', 'train_time_per_sample', 'adv_fit_time_per_sample', 'predict_time_per_sample'], inplace=True)
    data["failure_rate"] = (1 - data["accuracy"]) / data["predict_time_per_sample"]
    data["adv_failure_rate"] = (1 - data["adv_accuracy"]) / data[
        "adv_fit_time_per_sample"
    ]
    data["training_time_per_failure"] = (
        data["train_time_per_sample"] / data["failure_rate"]
    )
    data["training_time_per_adv_failure"] = (
        data["train_time_per_sample"] / data["adv_failure_rate"]
    )
    data["adv_training_time_per_failure"] = (
        data["adv_fit_time_per_sample"] / data["adv_failure_rate"]
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to the plot folder",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the plot folder",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--plotfiletype",
        type=str,
        help="Filetype of the plots",
        default=".pdf",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default="INFO",
        help="Increase output verbosity",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file",
        default="conf/plots.yaml",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    # %%
    assert Path(
        args.file,
    ).exists(), f"File {args.file} does not exist. Please specify a valid file using the -f flag."
    csv_file = args.file
    data = pd.read_csv(csv_file)
    data = calculate_failure_rate(data)
    if "Unnamed: 0" in data.columns:
        data.drop("Unnamed: 0", axis=1, inplace=True)

    FOLDER = Path(Path(), args.path)
    IMAGE_FILETYPE = (
        args.plotfiletype
        if args.plotfiletype.startswith(".")
        else f".{args.plotfiletype}"
    )
    if Path(FOLDER).exists():
        pass
    else:
        logger.info(f"Creating folder {FOLDER}")
        FOLDER.mkdir(parents=True, exist_ok=True)

    # Reads Config file
    with open(Path(args.config), "r") as f:
        big_dict = yaml.load(f, Loader=yaml.FullLoader)
    # %%
    cat_plot_list = big_dict["cat_plot"]
    i = 0
    for dict_ in cat_plot_list:
        i += 1
        logger.info(f"Rendering graph {i}")
        locals()[f"graph{i}"] = cat_plot(data, **dict_, folder=FOLDER)
    # %%
    line_plot_list = big_dict["line_plot"]
    for dict_ in line_plot_list:
        i += 1
        logger.info(f"Rendering graph {i}")
        locals()[f"graph{i}"] = line_plot(data, **dict_, folder=FOLDER)

        # %%

    scatter_plot_dict = big_dict["scatter_plot"]

    
    graph = scatter_plot(data=data, **scatter_plot_dict, folder = FOLDER)
