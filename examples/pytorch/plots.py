# %% [markdown]
# # Dependencies

# %%

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    graph.savefig(FOLDER / file)
    plt.gcf().clear()
    logger.info(f"Saved graph to {FOLDER / file}")


def line_plot(
    data,
    x,
    y,
    hue,
    xlabel,
    ylabel,
    title,
    file,
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
    graph.get_figure().savefig(FOLDER / file)
    plt.gcf().clear()
    return graph


def format_control_parameter(data, control_dict, min_max=True):
    data.def_gen.fillna("Control", inplace=True)
    new_data = pd.DataFrame()
    for _, row in data.iterrows():
        if row.defence in ["Control", None, "None", "none", "null", np.nan]:
            row["def_param"] = np.nan
            row["def_value"] = np.nan
        else:
            param = control_dict[row.defence]
            row["def_param"] = param.split(".")[-1]
            value = row[param]
            row["def_value"] = value
        if row.attack in ["Control", None, "None", "none", "null", np.nan]:
            row["atk_param"] = np.nan
            row["atk_value"] = np.nan
        else:
            param = control_dict[row.attack]
            row["atk_param"] = param.split(".")[-1]
            value = row[param]
            row["atk_value"] = value
        new_data = pd.concat([new_data, row], axis=1)
    data = new_data.T
    data.def_value.fillna(0, inplace=True)
    del new_data

    if min_max is True:
        defs = data.def_gen.unique()
        atks = data.atk_gen.unique()
        # Min-max scaling of control parameters
        for def_ in defs:
            max_ = data[data.def_gen == def_].def_value.max()
            min_ = data[data.def_gen == def_].def_value.min()
            scaled_value = (data[data.def_gen == def_].def_value - min_) / (max_ - min_)
            data.loc[data.def_gen == def_, "def_value"] = scaled_value

        for atk in atks:
            max_ = data[data.atk_gen == atk].atk_value.max()
            min_ = data[data.atk_gen == atk].atk_value.min()
            scaled_value = (data[data.atk_gen == atk].atk_value - min_) / (max_ - min_)
            data.loc[data.atk_gen == atk, "atk_value"] = scaled_value
    return data


def clean_data_for_plotting(data, def_gen_dict, atk_gen_dict, control_dict):
    def_gen = data.def_gen.map(def_gen_dict)
    data.def_gen = def_gen
    atk_gen = data.atk_gen.map(atk_gen_dict)
    data.atk_gen = atk_gen
    # Drops poorly merged columns
    data = data[data.columns.drop(list(data.filter(regex=".1")))]
    data = data[data.columns.drop(list(data.filter(regex=".1")))]
    # Replaces model names with short names
    model_names = data["model.init.name"]
    # model_names = [x.get_text() for x in model_names]
    model_names = [x.split(".")[-1] for x in model_names]
    data["model_name"] = model_names
    # %%
    # Replace data.sample.random_state with random_state
    data["random_state"] = data["data.sample.random_state"].copy()
    del data["data.sample.random_state"]
    data = format_control_parameter(data, control_dict, min_max=True)
    # Calculates various failure rates
    data["adv_failures_per_training_time"] = (
        data["train_time_per_sample"] / (1 - data["adv_accuracy"]) * 100
    )
    data["adv_failure_rate"] = (1 - data["adv_accuracy"]) * data[
        "adv_fit_time_per_sample"
    ]
    data["failure_rate"] = (1 - data["accuracy"]) * data["predict_time_per_sample"]
    data["failures_per_training_time"] = (
        data["train_time_per_sample"] / (1 - data["accuracy"]) * 100
    )
    logger.info(f"Saving data to {FOLDER / 'data.csv'}")
    data.to_csv(FOLDER / "data.csv")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to the plot folder",
        default="output/plots",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the plot folder",
        default="output/reports/results.csv",
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
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    # %%
    assert Path(
        args.file,
    ).exists(), f"File {args.file} does not exist. Please specify a valid file using the -f flag."
    csv_file = args.file
    data = pd.read_csv(csv_file)
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
    with open(FOLDER / "config/default.yaml", "r") as f:
        big_dict = yaml.load(f, Loader=yaml.FullLoader)
    def_gen_dict = big_dict["defences"]
    atk_gen_dict = big_dict["attacks"]
    control_dict = big_dict["params"]

    data = clean_data_for_plotting(data, def_gen_dict, atk_gen_dict, control_dict)
    # %%
    cat_plot_list = big_dict["cat_plot"]
    i = 0
    for dict_ in cat_plot_list:
        i += 1
        logger.info(f"Rendering graph {i}")
        locals()[f"graph{i}"] = cat_plot(data, **dict_)
    # %%
    line_plot_list = big_dict["line_plot"]
    for dict_ in line_plot_list:
        i += 1
        logger.info(f"Rendering graph {i}")
        locals()[f"graph{i}"] = line_plot(data, **dict_)

    # %%
    graph14 = sns.scatterplot(
        data=data,
        x="train_time_per_sample",
        y="adv_failure_rate",
        hue="model_name",
    )
    graph14.set_yscale("log")
    graph14.set_xscale("log")
    graph14.set_xlabel("Training Time")
    graph14.set_ylabel("Adversarial Failure Rate")
    graph14.legend(title="Model Name")
    # graph6.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    # graph6.legend(labels=["ResNet18", "Resnet34", "Resnet50"])
    graph14.set_title("Adversarial Failure Rate vs Training Time")
    # graph6.get_figure().tight_layout()
    file = f"adv_failure_rate_vs_train_time{IMAGE_FILETYPE}"
    graph14.get_figure().savefig(FOLDER / file)
    logger.info(f"Rendering graph {i+1}")
    logger.info(f"Saved graph to {FOLDER / file}")
    plt.gcf().clear()
    conf_path = Path("output/plots/config")
    conf_path.mkdir(parents=True, exist_ok=True)
    conf_dict = {
        **vars(args),
        "cat_plot": cat_plot_list,
        "line_plot": line_plot_list,
        "params": control_dict,
        "attacks": atk_gen_dict,
        "defences": def_gen_dict,
    }
    with open(conf_path / "default.yaml", "w") as f:
        yaml.dump(conf_dict, f)
