import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from lifelines import (
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter,
    CoxPHFitter,
    WeibullFitter,
    LogNormalFitter,
    LogLogisticFitter,
    plotting,
)

from .clean_data import drop_frames_without_results
import matplotlib
import logging
import yaml
import argparse

logger = logging.getLogger(__name__)


def fit_aft(
    df,
    event_col,
    duration_col,
    mtype,
    summary_file=None,
    summary_plot=None,
    folder=None,
    replacement_dict={},
    **kwargs,
):

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
    aft.fit(df, event_col=event_col, duration_col=duration_col)
    if summary_file is not None:
        summary = pd.DataFrame(aft.summary)
        suffix = Path(summary_file).suffix
        if folder is not None:
            summary_file = Path(folder, summary_file)
        if suffix == "":
            Path(summary_file).parent.mkdir(exist_ok=True, parents=True)
            summary.to_csv(summary_file)
            logger.info(f"Saved summary to {summary_file}")
        elif suffix == ".csv":
            summary.to_csv(summary_file)
            logger.info(f"Saved summary to {summary_file}")
        elif suffix == ".tex":
            summary.to_latex(summary_file, float_format="%.2f")
            logger.info(f"Saved summary to {summary_file}")
        elif suffix == ".json":
            summary.to_json(summary_file)
            logger.info(f"Saved summary to {summary_file}")
        elif suffix == ".html":
            summary.to_html(summary_file)
            logger.info(f"Saved summary to {summary_file}")
        else:
            logger.warning(f"suffix {suffix} not recognized. Saving to csv")
            summary.to_csv(summary_file)
            logger.info(f"Saved summary to {summary_file}")
    if summary_plot is not None:
        plot_summary(
            aft=aft,
            title=kwargs.get(
                "title",
                f"{mtype} AFT Summary".replace("_", " ").replace("-", "").title(),
            ),
            file=summary_plot,
            xlabel=kwargs.get("xlabel", "Covariate"),
            ylabel=kwargs.get("ylabel", "p-value"),
            replacement_dict=replacement_dict,
            folder=folder,
            filetype=".pdf",
        )
    return aft


def plot_aft(
    aft,
    title,
    file,
    xlabel,
    ylabel,
    replacement_dict={},
    folder=None,
    filetype=".pdf",
):
    suffix = Path(file).suffix
    if suffix == "":
        file = Path(file).with_suffix(filetype)
    else:
        file = Path(file)
    if folder is not None:
        file = Path(folder, file)
    plt.gcf().clear()
    # Only plot the covariates, skipping the intercept and dummy variables
    # Dummy variables can be examined using the plot_partial_effects function
    try:
        columns = list(aft.summary.index.get_level_values(1))
    except IndexError:
        columns = list(aft.summary.index)
    for col in columns:
        while "Intercept" in columns:
            columns.remove("Intercept")
        if col.startswith("dummy_"):
            columns.remove(col)
    ax = aft.plot(columns=columns)
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
    plt.gcf().clear()
    logger.info(f"Saved graph to {file}")
    return ax


def plot_summary(
    aft,
    title,
    file,
    xlabel,
    ylabel,
    replacement_dict={},
    folder=None,
    filetype=".pdf",
):
    suffix = Path(file).suffix
    if suffix == "":
        file = Path(file).with_suffix(filetype)
    else:
        file = Path(file)
    if folder is not None:
        file = Path(folder, file)
    plt.gcf().clear()
    summary = aft.summary.copy()
    summary = pd.DataFrame(summary)
    try:
        cov = summary.index.get_level_values(1)
        # print(list(set(cov)))
        # input("List of covariates. Press Enter to continue...")
        par = summary.index.get_level_values(0)
        covariates = [f"{c}: {p}" for p, c in zip(par, cov)]
    except IndexError:
        covariates = summary.index
    ax = sns.barplot(data=summary, x=covariates, y="p")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    labels = ax.get_xticklabels()
    labels = [label.get_text() for label in labels]
    for k, v in replacement_dict.items():
        labels = [label.replace(k, v) for label in labels]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yscale("log")
    plt.tight_layout()
    ax.get_figure().savefig(file)
    plt.gcf().clear()
    return ax


def plot_qq(
    aft,
    title,
    file,
    xlabel=None,
    ylabel=None,
    replacement_dict={},
    folder=None,
    filetype=".pdf",
    ax=None,
):
    suffix = Path(file).suffix
    if suffix == "":
        file = Path(file).with_suffix(filetype)
    else:
        file = Path(file)
    if folder is not None:
        file = Path(folder, file)
    plt.gcf().clear()
    ax = plotting.qq_plot(aft, ax=ax)
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
    return ax


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
        covariate_array,
        values_array,
        cmap=cmap,
        **kwargs,
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
    train_score = aft.score(train, scoring_method="concordance_index")
    test_score = aft.score(test, scoring_method="concordance_index")
    scores = {"train_score": train_score, "test_score": test_score}
    plt.show()
    return scores


def make_afr_table(
    aft_dict,
    dataset,
    X_train,
    X_test,
    folder=".",
    span_columns=True,
):
    folder = Path(folder)
    aft_data = pd.DataFrame()
    aft_data.index.name = "Model"
    model_names = [x.replace("-", " ").replace("_", " ").title() for x in aft_dict.keys()]
    aft_data.index = model_names
    aft_data["AIC"] = [
        x.AIC_ if not isinstance(x, CoxPHFitter) else np.nan for x in aft_dict.values()
    ]
    aft_data["Concordance"] = [
        x.concordance_index_ for x in aft_dict.values()
    ]
    aft_data["BIC"] = [
        x.AIC_ if not isinstance(x, CoxPHFitter) else np.nan for x in aft_dict.values()
    ]
    aft_data["Mean $S(t;\\theta)$"] = [
        x.predict_expectation(X_train).mean() for x in aft_dict.values()
    ]
    aft_data["Median $S(t; \\theta)$"] = [
        x.predict_expectation(X_train).median() for x in aft_dict.values()
    ]
    pretty_dataset = dataset.upper() if dataset != "combined" else "combined"
    aft_data = aft_data.round(2)
    aft_data.to_csv(folder / "aft_comparison.csv")
    logger.info(f"Saved AFT comparison to {folder / 'aft_comparison.csv'}")
    aft_data = aft_data.round(2)
    aft_data.fillna("--", inplace=True)
    aft_data.to_latex(
        folder / "aft_comparison.tex",
        float_format="%.2f", # Two decimal places, since we have 100 adversarial examples
        label=f"tab:{dataset}", # Label for cross-referencing
        caption=f"Comparison of AFR Models on the {pretty_dataset} dataset.",
    )
    # Change to table* if span_columns is True
    if span_columns is True:
        with open(folder / "aft_comparison.tex", "r") as f:
            tex_data = f.read()
        tex_data = tex_data.replace(r"\begin{table}", r"\begin{table*}"+"\n"+r"\centering")
        tex_data = tex_data.replace(r"\end{table}", r"\end{table*}")
        with open(folder / "aft_comparison.tex", "w") as f:
            f.write(tex_data)
    return aft_data


def clean_data_for_aft(
    data,
    covariate_list,
    target="adv_failure_rate",
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
        cleaned[kwarg] = subset[kwarg]
    cols = cleaned.columns
    cleaned = pd.DataFrame(subset, columns=cols)
    for col in cols:
        cleaned = cleaned[cleaned[col] != -1e10]
        cleaned = cleaned[cleaned[col] != 1e10]
    cleaned.dropna(inplace=True, how="any", axis=0)
    assert (
        target in cleaned.columns
    ), f"Target {target} not in dataframe with columns {cleaned.columns}"
    logger.info(f"Shape of cleaned data: {cleaned.shape}")
    return cleaned


def split_data_for_aft(
    data,
    target,
    duration_col,
    test_size=0.2,
    random_state=42,
):
    cleaned = pd.get_dummies(data, prefix="dummy", prefix_sep="_")
    X_train, X_test = train_test_split(
        cleaned,
        train_size=(1 - test_size),
        test_size=test_size,
        random_state=random_state,
    )
    assert (
        target in cleaned.columns
    ), f"Target {target} not in dataframe with columns {cleaned.columns}"
    assert (
        duration_col in cleaned.columns
    ), f"Duration {duration_col} not in dataframe with columns {cleaned.columns}"
    X_train = X_train.dropna(axis=0, how="any")
    X_test = X_test.dropna(axis=0, how="any")
    return X_train, X_test


def render_afr_plot(
    mtype,
    config,
    X_train,
    target,
    duration_col,
    X_test=None,
    folder=".",
):
    if len(config.keys()) > 0:
        plots = []
        plot_dict = config.pop("plot", {})
        label_dict = plot_dict.pop("labels", plot_dict.get("labels", {}))
        partial_effect_list = config.pop("partial_effect", [])
        model_config = config.pop("model", {})
        aft = fit_aft(
            summary_file=config.get("summary_file", f"{mtype}_summary.csv"),
            summary_plot=config.get("summary_plot", f"{mtype}_summary.pdf"),
            folder=folder,
            df=X_train,
            event_col=target,
            duration_col=duration_col,
            replacement_dict=label_dict,
            mtype=mtype,
            **model_config,
        )
        afr_plot = plot_aft(
            aft=aft,
            title=config.get(
                "title",
                f"{mtype} AFT".replace("_", " ").replace("-", " ").title(),
            ),
            file=config.get("file", f"{mtype}_aft.pdf"),
            xlabel=label_dict.get("xlabel", "Coefficient"),
            ylabel=label_dict.get("ylabel", r"$\mathbb{P}~(T>t)$"),  # noqa W605
            replacement_dict=label_dict,
            folder=folder,
        )
        plots.append(afr_plot)
        if mtype == "cox":
            logger.warning("Cox model does not have a CDF plot")
        else:
            if mtype == "weibull":
                univariate_aft = WeibullFitter()
            elif mtype == "log_normal":
                univariate_aft = LogNormalFitter()
            elif mtype == "log_logistic":
                univariate_aft = LogLogisticFitter()
            else:
                raise ValueError(f"Model {mtype} not recognized")
            if X_test is not None:
                data = X_test
            else:
                data = X_train
            univariate_aft.fit(data[duration_col], data[target])
            cdf_plot = plot_qq(
                aft=univariate_aft,
                title=config.get(
                    "title",
                    f"{mtype}".replace("_", " ").replace("-", " ").title()
                    + " AFT QQ Plot",
                ),
                file=config.get("file", f"{mtype}_qq.pdf"),
                xlabel=label_dict.get("xlabel", "Theoretical Quantiles"),
                ylabel=label_dict.get("ylabel", "Empirical Quantiles"),
                replacement_dict=label_dict,
                folder=folder,
            )
            plots.append(cdf_plot)
        for partial_effect_dict in partial_effect_list:
            file = partial_effect_dict.pop("file", "partial_effects.pdf")
            partial_effect_plot = plot_partial_effects(
                aft=aft,
                file=file,
                **partial_effect_dict,
                folder=folder,
            )
            plots.append(partial_effect_plot)
    return aft, plots


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
    cleaned = clean_data_for_aft(data, covariate_list, target=target)
    assert target in cleaned.columns, f"{target} not in data.columns"
    assert duration_col in cleaned.columns, f"{duration_col} not in data.columns"
    X_train, X_test = split_data_for_aft(
        cleaned,
        target,
        duration_col,
        test_size=test_size,
        random_state=42,
    )
    plots = {}
    models = {}
    mtypes = list(config.keys())
    for mtype in mtypes:
        sub_config = config.get(mtype, {})
        models[mtype], plots[mtype] = render_afr_plot(
            mtype=mtype,
            config=sub_config,
            X_train=X_train,
            X_test=X_test,
            target=target,
            duration_col=duration_col,
            folder=folder,
        )
    aft_data = make_afr_table(models, dataset, X_train, X_test, folder=folder)
    print("*" * 80)
    print("*" * 34 + "  RESULTS   " + "*" * 34)
    print("*" * 80)
    print(f"{aft_data}")
    print("*" * 80)


def prepare_aft_data(args, data, config):
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    assert Path(args.config_file).exists(), f"{args.config_file} does not exist."
    covariates = config.get("covariates", [])
    assert len(covariates) > 0, "No covariates specified in config file"
    logger.info(f"Shape of data before data before dropping na: {data.shape}")
    logger.info(f"Shape of data before data before dropping na: {data.shape}")
    if "adv_failures" in covariates and "adv_failures" not in data.columns:
        data.loc[:, "adv_failures"] = (1 - data.loc[:, "adv_accuracy"]) * data.loc[
            :,
            "attack.attack_size",
        ]
    if "ben_failures" in covariates and "ben_failures" not in data.columns:
        if "predict_time" in data.columns:
            data["n_samples"] = data["predict_time"] / data[
                "predict_time_per_sample"
            ].astype(int)
        elif "predict_proba_time" in data.columns:
            data["n_samples"] = data["predict_proba_time"] / data[
                "predict_time_per_sample"
            ].astype(int)
        elif "predict_loss_time" in data.columns:
            data["n_samples"] = data["predict_loss_time"] / data[
                "predict_time_per_sample"
            ].astype(int)
        else:
            assert "n_samples" in data.columns, "n_samples not in data"
        data.loc[:, "ben_failures"] = (1 - data.loc[:, "accuracy"]) * data.loc[
            :,
            "n_samples",
        ]
    data = drop_frames_without_results(data, covariates)
    return data


def main(args):
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
    data = prepare_aft_data(args, data, config)
    assert target in data.columns, f"{target} not in data.columns"
    assert duration_col in data.columns, f"{duration_col} not in data.columns"
    render_all_afr_plots(
        config,
        duration_col,
        target,
        data,
        dataset,
        test_size=0.8,
        folder=FOLDER,
    )


if "__main__" == __name__:
    afr_parser = argparse.ArgumentParser()
    afr_parser.add_argument("--target", type=str, default="adv_failures")
    afr_parser.add_argument("--duration_col", type=str, default="adv_fit_time")
    afr_parser.add_argument("--dataset", type=str, default="mnist")
    afr_parser.add_argument("--data_file", type=str, default="data.csv")
    afr_parser.add_argument("--config_file", type=str, default="afr.yaml")
    afr_parser.add_argument("--plots_folder", type=str, default="plots")
    args = afr_parser.parse_args()
    main(args)
