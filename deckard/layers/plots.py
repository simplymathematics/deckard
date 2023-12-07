import argparse
import logging
from pathlib import Path
from paretoset import paretoset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from math import isnan
import numpy as np
from .utils import deckard_nones as nones
from tqdm import tqdm

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.8, font="times new roman")


def cat_plot(
    data,
    x,
    y,
    hue,
    kind,
    titles,
    xlabels,
    ylabels,
    file,
    folder,
    legend_title=None,
    hue_order=None,
    rotation=0,
    set={},
    filetype = ".pdf",
    **kwargs,
):
    plt.gcf().clear()
    file = Path(file).with_suffix(filetype)
    logger.info(f"Rendering graph {file}")
    data = data.sort_values(by=[hue, x, y])
    graph = sns.catplot(
        data=data, x=x, y=y, hue=hue, kind=kind, hue_order=hue_order, **kwargs
    )
    graph.set_xlabels(xlabels)
    graph.set_ylabels(ylabels)
    graph.set_titles(titles)
    if legend_title is not None:
        graph.legend.set_title(title=legend_title)
    else:
        graph.legend.remove()
    graph.set_xticklabels(graph.axes.flat[-1].get_xticklabels(), rotation=rotation)
    graph.set(**set)
    graph.tight_layout()
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
    hue_order=None,
    filetype = ".pdf",
    **kwargs,
):
    plt.gcf().clear()
    file = Path(file).with_suffix(filetype)
    logger.info(f"Rendering graph {file}")
    data = data.sort_values(by=[hue, x, y])
    graph = sns.lineplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, **kwargs)
    graph.legend(**legend)
    graph.set_xlabel(xlabel)
    graph.set_ylabel(ylabel)
    graph.set_title(title)
    if y_scale is not None:
        graph.set_yscale(y_scale)
    if x_scale is not None:
        graph.set_xscale(x_scale)
    graph.get_figure().tight_layout()
    graph.get_figure().savefig(folder / file)
    logger.info(f"Saved graph to {folder/file}")
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
    hue_order=None,
    filetype = ".pdf",
    **kwargs,
):
    plt.gcf().clear()
    file = Path(file).with_suffix(filetype)
    logger.info(f"Rendering graph {file}")
    data = data.sort_values(by=[hue, x, y])
    graph = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        **kwargs,
    )
    graph.set_yscale(y_scale)
    graph.set_xscale(x_scale)
    graph.set_xlabel(xlabel)
    graph.set_ylabel(ylabel)
    graph.legend(**legend)
    graph.set_title(title)
    graph.get_figure().tight_layout()
    graph.get_figure().savefig(Path(folder) / file)

    logger.info(f"Saved graph to {Path(folder) / file}")
    plt.gcf().clear()
    return graph




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
        help="Data file to read from",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file name",
        default="data.csv",
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
    assert Path(
        args.file,
    ).exists(), f"File {args.file} does not exist. Please specify a valid file using the -f flag."
    data = pd.read_csv(args.file)
     # Reads Config file
    with open(Path(args.config), "r") as f:
        big_dict = yaml.load(f, Loader=yaml.FullLoader)
    cat_plot_list = big_dict.get("cat_plot", [])
    if Path(args.path).absolute() == Path(args.path):
        logger.info("Absolute path specified")
        FOLDER = Path(args.path).absolute()
    else:
        logger.info("Relative path specified")
        FOLDER = Path(Path(), args.path)
    logger.info(f"Creating folder {FOLDER}")
    FOLDER.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving data to {FOLDER / args.output}")
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

    i = 0
    for dict_ in cat_plot_list:
        i += 1
        cat_plot(data, **dict_, folder=FOLDER, filetype=IMAGE_FILETYPE)
        
    line_plot_list = big_dict.get("line_plot", [])
    for dict_ in line_plot_list:
        i += 1
        line_plot(data, **dict_, folder=FOLDER, filetype=IMAGE_FILETYPE)

    scatter_plot_list = big_dict.get("scatter_plot", [])
    for dict_ in scatter_plot_list:
        i += 1
        scatter_plot(data, **dict_, folder=FOLDER, filetype=IMAGE_FILETYPE)
