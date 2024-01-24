import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from lifelines import (
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter,
    CoxPHFitter,
)
from .clean_data import drop_frames_without_results
from .clean_data import drop_frames_without_results
import matplotlib
import logging
import yaml
import argparse

logger = logging.getLogger(__name__)


def plot_aft(
    df,
    file,
    event_col,
    duration_col,
    title,
    mtype,
    xlabel=None,
    ylabel=None,
    replacement_dict={},
    filetype=".pdf",
    folder=".",
    **kwargs,
):
    file = Path(folder, file).with_suffix(filetype)
    if mtype == "weibull":
        aft = WeibullAFTFitter(**kwargs)
    elif mtype == "log_normal":
        aft = LogNormalAFTFitter(**kwargs)
    elif mtype == "log_logistic":
        aft = LogLogisticAFTFitter(**kwargs)
    elif mtype == "cox":
        aft = CoxPHFitter(**kwargs)
    assert (
        duration_col in df.columns
    ), f"Column {duration_col} not in dataframe with columns {df.columns}"
    if event_col is not None:
        assert (
            event_col in df.columns
        ), f"Column {event_col} not in dataframe with columns {df.columns}"
    plt.gcf().clear()
    aft.fit(df, duration_col=duration_col, event_col=event_col)
    ax = aft.plot()
    labels = ax.get_yticklabels()
    labels = [label.get_text() for label in labels]
    for k, v in replacement_dict.items():
        labels = [label.replace(k, v) for label in labels]
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(file)
    logger.info(f"Saved graph to {file}")
    plt.show()
    plt.gcf().clear()
    return ax, aft


def plot_partial_effects(
    aft,
    covariate_array,
    values_array,
    title=None,
    file="partial_effects.pdf",
    xlabel="Covariate",
    ylabel="Failure rate",
    legend_kwargs={"loc": "upper left"},
    replacement_dict={},
    cmap="coolwarm",
    folder=".",
    filetype=".pdf",
    **kwargs,
):
    plt.gcf().clear()
    file = Path(folder, file).with_suffix(filetype)
    pareto = aft.plot_partial_effects_on_outcome(
        covariate_array, values_array, cmap=cmap, **kwargs
    )
    labels = pareto.get_yticklabels()
    labels = [label.get_text() for label in labels]
    for k, v in replacement_dict.items():
        labels = [label.replace(k, v) for label in labels]
    pareto.set_yticklabels(labels)
    pareto.legend(**legend_kwargs)
    pareto.set_ylabel(ylabel)
    pareto.set_xlabel(xlabel)
    pareto.set_title(title)
    pareto.get_figure().tight_layout()
    pareto.get_figure().savefig(file)
    logger.info(f"Saved graph to {file}")
    plt.gcf().clear()
    return pareto

def plot_partial_effects(
    aft,
    covariate_array,
    values_array,
    title=None,
    file="partial_effects.eps",
    xlabel="Covariate",
    ylabel="Failure rate",
    legend_kwargs={"loc": "upper left"},
    replacement_dict={},
    cmap="coolwarm",
    folder=".",
    filetype=".eps",
    **kwargs,
):
    plt.gcf().clear()
    file = Path(folder, file).with_suffix(filetype)
    pareto = aft.plot_partial_effects_on_outcome(
        covariate_array, values_array, cmap=cmap, **kwargs
    )
    labels = pareto.get_yticklabels()
    labels = [label.get_text() for label in labels]
    for k, v in replacement_dict.items():
        labels = [label.replace(k, v) for label in labels]
    pareto.set_yticklabels(labels)
    pareto.legend(**legend_kwargs)
    pareto.set_ylabel(ylabel)
    pareto.set_xlabel(xlabel)
    pareto.set_title(title)
    pareto.get_figure().tight_layout()
    pareto.get_figure().savefig(file)
    logger.info(f"Saved graph to {file}")
    plt.gcf().clear()
    return pareto


def score_model(aft, train, test):
    train_score = aft.score(train)
    test_score = aft.score(test)
    scores = {"train_score": train_score, "test_score": test_score}
    plt.show()
    return scores


def make_afr_table(score_list, aft_dict, dataset, X_train, folder="."):
    assert len(score_list) == len(
        aft_dict,
    ), "Length of score list and aft dict must be equal"
    folder = Path(folder)
    aft_data = pd.DataFrame()
    
    aft_data["AIC"] = [
        x.AIC_ if not isinstance(x, CoxPHFitter) else np.nan for x in aft_dict.values()
    ]
    aft_data["Concordance"] = [x.concordance_index_ for x in aft_dict.values()]
    aft_data["BIC"] = [
        x.AIC_ if not isinstance(x, CoxPHFitter) else np.nan for x in aft_dict.values()
    ]
    # aft_data["Train LL"] = [x["train_score"] for x in score_list]
    # aft_data["Test LL"] = [x["test_score"] for x in score_list]
    aft_data["Mean $S(t;\\theta)$"] = [
        x.predict_expectation(X_train).mean() for x in aft_dict.values()
    ]
    aft_data["Median $S(t;\\theta)$"] = [
        x.predict_median(X_train).median() for x in aft_dict.values()
    ]
    aft_data.index.name = "Distribution"
    aft_data.index = [str(x).replace("_", " ").title() for x in aft_dict.keys()]
    label = f"tab:{dataset}"
    upper = dataset.upper()
    logger.info(f"Saved AFT comparison to {folder / 'aft_comparison.csv'}")
    aft_data.to_latex(
        buf = folder / "aft_comparison.tex",
        float_format="%.3g",
        na_rep="--",
        label=label,
        caption=f"Comparison of AFR Models on the {upper} dataset.",
    )
    aft_data.to_csv(Path(folder / "aft_comparison.csv"), index_label="Distribution", na_rep="--")
    
    
    return aft_data


def clean_data_for_aft(
    data,
    covariate_list,
    target="adv_accuracy",
):
    subset = data.copy()
    assert (
        target in subset
    ), f"Target {target} not in dataframe with columns {subset.columns}"
    logger.info(f"Shape of dirty data: {subset.shape}")
    cleaned = pd.DataFrame()
    covariate_list.append(target)
    logger.info(f"Covariates : {covariate_list}")
    for kwarg in covariate_list:
        assert kwarg in subset.columns, f"{kwarg} not in data.columns"
        cleaned = pd.concat([cleaned, subset[kwarg]], axis=1)
    cols = cleaned.columns
    cleaned = pd.DataFrame(subset, columns=cols)
    cleaned.index = subset.index
    for col in cols:
        cleaned = cleaned[cleaned[col] != -1e10]
        cleaned = cleaned[cleaned[col] != 1e10]
    
    cleaned.dropna(inplace=True, how="any", axis=0)
    cleaned = pd.get_dummies(cleaned)
    # de-duplicate index
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="first")]
    assert (
        target in cleaned
    ), f"Target {target} not in dataframe with columns {cleaned.columns}"
    logger.info(f"Shape of cleaned data: {cleaned.shape}")

    return cleaned


def split_data_for_aft(
    data,
    target,
    duration_col,
    covariate_list,
    test_size=0.2,
    random_state=42,
):
    cleaned = clean_data_for_aft(data, covariate_list, target=target)
    X_train, X_test = train_test_split(
        cleaned,
        train_size=(1 - test_size),
        test_size=test_size,
        random_state=random_state,
    )
    assert (
        target in cleaned
    ), f"Target {target} not in dataframe with columns {cleaned.columns}"
    assert (
        duration_col in cleaned
    ), f"Duration {duration_col} not in dataframe with columns {cleaned.columns}"
    X_train = X_train.dropna(axis=0, how="any")
    X_test = X_test.dropna(axis=0, how="any")
    return X_train, X_test


def render_afr_plot(mtype, config, X_train, X_test, target, duration_col, folder="."):
    if len(config.keys()) > 0:
        plots = []
        plot_dict = config.get("plot", {})
        label_dict = config.get("labels", [])
        partial_effect_list = config.get("partial_effect", [])
        afr_plot, aft = plot_aft(
            X_train,
            event_col=target,
            duration_col=duration_col,
            mtype=mtype,
            folder=folder,
            **plot_dict,
            replacement_dict=label_dict,
        )
        plots.append(afr_plot)
        score = score_model(aft, X_train, X_test)
        for partial_effect_dict in partial_effect_list:
            partial_effect_plot = plot_partial_effects(
                aft=aft, **partial_effect_dict, folder=folder
            )
            plots.append(partial_effect_plot)
    return aft, plots, score


def render_all_afr_plots(
    config,
    duration_col,
    target,
    data,
    dataset,
    test_size=0.8,
    folder=".",
):
    covariate_list = config.pop("covariates", [])
    X_train, X_test = split_data_for_aft(
        data,
        target,
        duration_col,
        covariate_list,
        test_size=test_size,
        random_state=42,
    )
    plots = {}
    scores = {}
    models = {}
    mtypes = list(config.keys())
    for mtype in mtypes:
        sub_config = config.get(mtype, {})
        models[mtype], plots[mtype], scores[mtype] = render_afr_plot(
            mtype=mtype,
            config=sub_config,
            X_train=X_train,
            X_test=X_test,
            target=target,
            duration_col=duration_col,
            folder=folder,
        )
        models[mtype].print_summary()
        if isinstance(models[mtype], CoxPHFitter):
            models[mtype].check_assumptions(X_train, show_plots=True)
    score_list = list(scores.values())
    aft_data = make_afr_table(score_list, models, dataset, X_train, folder=folder)
    print("*" * 80)
    print("*" * 34 + "  RESULTS   " + "*" * 34)
    print("*" * 80)
    print(f"{aft_data}")
    print("*" * 80)


if "__main__" == __name__:
    afr_parser = argparse.ArgumentParser()
    afr_parser.add_argument("--target", type=str, default="adv_failures")
    afr_parser.add_argument("--duration_col", type=str, default="adv_fit_time")
    afr_parser.add_argument("--dataset", type=str, default="mnist")
    afr_parser.add_argument("--data_file", type=str, default="data.csv")
    afr_parser.add_argument("--config_file", type=str, default="afr.yaml")
    afr_parser.add_argument("--plots_folder", type=str, default="plots")
    afr_parser.add_argument("--attack_size", type=int, default=100)
    args = afr_parser.parse_args()
    target = args.target
    duration_col = args.duration_col
    dataset = args.dataset
    logging.basicConfig(level=logging.INFO)
    font = {
        "family": "Times New Roman",
        "weight": "bold",
        "size": 22,
    }

    matplotlib.rc("font", **font)

    csv_file = args.data_file
    FOLDER = args.plots_folder
    Path(FOLDER).mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(csv_file, index_col=0)
    logger.info(f"Shape of data: {data.shape}")
    data.columns = data.columns.str.strip()
    with Path(args.config_file).open("r") as f:
        config = yaml.safe_load(f)
    fillna = config.pop("fillna", {})
    for k, v in fillna.items():
        assert k in data.columns, f"{k} not in data"
        data[k] = data[k].fillna(v)
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    assert Path(args.config_file).exists(), f"{args.config_file} does not exist."
    covariates = config.get("covariates", [])
    assert len(covariates) > 0, "No covariates specified in config file"
    logger.info(f"Shape of data before data before dropping na: {data.shape}")
    data = drop_frames_without_results(data, covariates)
    logger.info(f"Shape of data before data before dropping na: {data.shape}")
    data.loc[:, "adv_failures"] = (1 - data.loc[:, "adv_accuracy"]) * args.attack_size
    data.loc[:, "ben_failures"] = (1 - data.loc[:, "accuracy"]) * args.attack_size
    render_all_afr_plots(
        config,
        duration_col,
        target,
        data,
        dataset,
        test_size=0.8,
        folder=FOLDER,
    )
