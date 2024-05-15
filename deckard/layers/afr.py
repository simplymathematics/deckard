# -*- coding: utf-8 -*-
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import logging
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lifelines import (
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter,
    CoxPHFitter,
    CRCSplineFitter,
    AalenAdditiveFitter,
    GeneralizedGammaRegressionFitter,
    PiecewiseExponentialRegressionFitter,
)
from lifelines.utils import CensoringType
from lifelines.fitters import RegressionFitter
from lifelines.exceptions import ConvergenceError
from .clean_data import drop_rows_without_results
from .compile import load_results, save_results
logger = logging.getLogger(__name__)


# Modified from https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/calibration.py
def survival_probability_calibration(
    model: RegressionFitter,
    df: pd.DataFrame,
    t0: float,
    ax=None,
    color="red",
):
    r"""
    Smoothed calibration curves for time-to-event models. This is analogous to
    calibration curves for classification models, extended to handle survival probabilities
    and censoring. Produces a matplotlib figure and some metrics.

    We want to calibrate our model's prediction of :math:`P(T < \text{t0})` against the observed frequencies.

    Parameters
    -------------

    model:
        a fitted lifelines regression model to be evaluated
    df: DataFrame
        a DataFrame - if equal to the training data, then this is an in-sample calibration. Could also be an out-of-sample
        dataset.
    t0: float
        the time to evaluate the probability of event occurring prior at.

    Returns
    ----------
    ax:
        mpl axes
    ICI:
        mean absolute difference between predicted and observed
    E50:
        median absolute difference between predicted and observed

    https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8570

    """

    def ccl(p):
        return np.log(-np.log(1 - p))

    if ax is None:
        ax = plt.gca()
    T = model.duration_col
    E = model.event_col

    predictions_at_t0 = np.clip(
        1 - model.predict_survival_function(df, times=[t0]).T.squeeze(),
        1e-10,
        1 - 1e-10,
    )

    # create new dataset with the predictions
    prediction_df = pd.DataFrame(
        {"ccl_at_%d" % t0: ccl(predictions_at_t0), T: df[T], E: df[E]},
    )

    # fit new dataset to flexible spline model
    # this new model connects prediction probabilities and actual survival. It should be very flexible, almost to the point of overfitting. It's goal is just to smooth out the data!
    n_knots = 3
    regressors = {
        "beta_": ["ccl_at_%d" % t0],
        "gamma0_": "1",
        "gamma1_": "1",
        "gamma2_": "1",
    }

    # this model is from examples/royson_crowther_clements_splines.py
    crc = CRCSplineFitter(n_baseline_knots=n_knots, penalizer=0.000001)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            if CensoringType.is_right_censoring(model):
                crc.fit_right_censoring(prediction_df, T, E, regressors=regressors)
            elif CensoringType.is_left_censoring(model):
                crc.fit_left_censoring(prediction_df, T, E, regressors=regressors)
            elif CensoringType.is_interval_censoring(model):
                crc.fit_interval_censoring(prediction_df, T, E, regressors=regressors)
        except ConvergenceError:
            crc._scipy_fit_method = "SLSQP"
            try:
                if CensoringType.is_right_censoring(model):
                    crc.fit_right_censoring(prediction_df, T, E, regressors=regressors)
                elif CensoringType.is_left_censoring(model):
                    crc.fit_left_censoring(prediction_df, T, E, regressors=regressors)
                elif CensoringType.is_interval_censoring(model):
                    crc.fit_interval_censoring(prediction_df, T, E, regressors=regressors)
            except ConvergenceError as e:
                return ax, np.nan, np.nan

    # predict new model at values 0 to 1, but remember to ccl it!
    x = np.linspace(
        np.clip(predictions_at_t0.min() - 0.01, 0, 1),
        np.clip(predictions_at_t0.max() + 0.01, 0, 1),
        100,
    )
    y = (
        1
        - crc.predict_survival_function(
            pd.DataFrame({"ccl_at_%d" % t0: ccl(x)}),
            times=[t0],
        ).T.squeeze()
    )

    # plot our results

    ax.plot(x, y, label="Calibration Curve", color=color)
    ax.set_xlabel("Predicted P(t ≤ %.2f )" % round(t0, 3))
    ax.set_ylabel("Observed P(t ≤ %.2f )" % round(t0, 3))
    ax.tick_params(axis="y")

    # plot x=y line
    ax.plot(x, x, c="k", ls="--")
    ax.legend()

    plt.tight_layout()

    deltas = (
        (1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze()
        - predictions_at_t0
    ).abs()
    ICI = deltas.mean()
    E50 = np.percentile(deltas, 50)
    # print("ICI = ", ICI)
    # print("E50 = ", E50)

    return ax, ICI, E50


def fit_aft(
    df,
    event_col,
    duration_col,
    mtype,
    summary_file=None,
    folder=None,
    **kwargs,
):
    logger.info(f"Trying to initialize model {mtype} with {kwargs}")
    if mtype == "weibull":
        aft = WeibullAFTFitter(penalizer=kwargs.pop("penalizer",0.1), **kwargs)
    elif mtype == "log_normal":
        aft = LogNormalAFTFitter(penalizer=kwargs.pop("penalizer",0.1), **kwargs)
    elif mtype == "log_logistic":
        aft = LogLogisticAFTFitter(penalizer=kwargs.pop("penalizer",0.1), **kwargs)
    elif mtype == "cox":
        aft = CoxPHFitter(penalizer=kwargs.pop("penalizer",0.1), **kwargs)
    elif mtype == "aalen":
        aft = AalenAdditiveFitter(alpha=kwargs.pop("alpha",0.1), **kwargs)
    elif mtype == "gamma":
        aft = GeneralizedGammaRegressionFitter(
            penalizer=kwargs.pop("penalizer",0.1), **kwargs
        )
    elif mtype == "exponential":
        aft = PiecewiseExponentialRegressionFitter(
            penalizer=kwargs.pop("penalizer",0.1), **kwargs
        )
    else:
        raise ValueError(f"Model type {mtype} not recognized")
    assert (
        duration_col in df.columns
    ), f"Column {duration_col} not in dataframe with columns {df.columns}"
    if event_col is not None:
        assert (
            event_col in df.columns
        ), f"Column {event_col} not in dataframe with columns {df.columns}"
    start = df[duration_col].min() 
    end = df[duration_col].max()
    start = start - 0.01 * (end - start)
    timeline = np.linspace(start, end, 1000)
    try:
        aft.fit(df, event_col=event_col, duration_col=duration_col, timeline=timeline)
    except TypeError as e:
        if "AalenAdditiveFitter" in str(e):
            logger.debug("AalenAdditiveFitter does not support timeline")
            aft.fit(df, event_col=event_col, duration_col=duration_col)
    except AttributeError as e:
        logger.error(f"Could not fit {mtype} model")
        raise e
    except ConvergenceError as e:
        logger.info("Trying to fit with SLSQP")
        aft._scipy_fit_method = "SLSQP"
        try:
            aft.fit(df, event_col=event_col, duration_col=duration_col, timeline=timeline)
        except AttributeError as e:
            raise ConvergenceError(f"Could not fit {mtype} model with SLSQP")
        except ConvergenceError as e:
            logger.error(f"Could not fit {mtype} model")
            raise ConvergenceError(f"Could not fit {mtype} model")
        
    else:
        logger.info(f"Fitted {mtype} model")
    if summary_file is not None:
        summary = pd.DataFrame(aft.summary).copy()
        if folder is None:
            folder = "."
        save_results(summary, results_file=summary_file, results_folder=folder)
    return aft


def plot_aft(
    aft,
    title,
    file,
    xlabel,
    ylabel,
    replacement_dict={},
    dummy_dict={},
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
        columns = list(aft.summary.index.get_level_values(1)).copy()
    except IndexError:
        columns = list(aft.summary.index).copy()
    clean_cols = []
    dummy_cols = []
    dummy_prefixes = tuple(dummy_dict.values())
    for col in columns:
        if str(col).startswith(dummy_prefixes) or str(col).startswith("dummy_"):
            dummy_cols.append(col)
        elif col.startswith("Intercept"):
            continue
        else:
            clean_cols.append(col)
    columns = clean_cols
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
    if len(dummy_cols) > 0:
        plt.gcf().clear()
        logger.info(f"Dummy variables: {dummy_cols}")
        ax2 = aft.plot(columns=dummy_cols)
        labels = ax2.get_yticklabels()
        labels = [label.get_text() for label in labels]
        i = 0 
        
        labels = [x.replace("dummy_", "") for x in labels]
        for k,v in dummy_dict.items():
            labels = [label.replace(k, v) for label in labels]
        for k, v in replacement_dict.items():
            labels = [label.replace(k, v) for label in labels]
        for label in labels:
            if label.startswith("Data:"):
                dataset = label.split(":")[1].upper()
                new_label = f"Data: {dataset}"
            else:
                new_label = label
            labels[i] = new_label
            i += 1
        ax2.set_yticklabels(labels)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(title)
        ax2.get_figure().tight_layout()
        ax2.get_figure().savefig(file.with_name(file.stem + "_dummies" + file.suffix))
        plt.gcf().clear()
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
    if isinstance(summary.index, pd.MultiIndex):
        covariates = list(summary.index.get_level_values(1))
        summary["covariate"] = covariates
        params = list(summary.index.get_level_values(0))
        fullnames = [f"{cov}: {param}" for cov, param in zip(covariates, params)]
    else:
        covariates = list(summary.index)
        summary["covariate"] = covariates
        fullnames = covariates
    summary["covariate"] = covariates
    summary["fullnames"] = fullnames
    summary = summary[summary["covariate"] != "Intercept"]
    summary = summary[
        summary["covariate"].str.startswith("dummy_") != True  # noqa E712
    ]
    ax = sns.barplot(data=summary, x="covariate", y="p")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    labels = fullnames
    for k, v in replacement_dict.items():
        labels = [label.replace(k, v) for label in labels]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yscale("log")
    plt.tight_layout()
    ax.get_figure().savefig(file)
    plt.gcf().clear()
    return ax


def plot_qq(
    X_train,
    aft,
    title,
    file,
    X_test=None,
    xlabel=None,
    ylabel=None,
    folder=None,
    ax=None,
    filetype=".pdf",
    t0=0.35,
):
    suffix = Path(file).suffix
    if suffix == "":
        file = Path(file).with_suffix(filetype)
    else:
        file = Path(file)
    if folder is not None:
        file = Path(folder, file)
    plt.gcf().clear()
    
    if X_test is not None:
        ax, _, _ = survival_probability_calibration(aft, X_train, t0=t0, ax=ax, color="red")
        ax, _, _ = survival_probability_calibration(aft, X_test, t0=t0, ax=ax, color="blue")
    else:
        ax, _, _ = survival_probability_calibration(aft, X_train, t0=t0, ax=ax, color="red")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend().remove()
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(file)
    logger.info(f"Saved graph to {file}")
    plt.gcf().clear()
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
    train_ll = aft.score(train, scoring_method="log_likelihood")
    test_ll = aft.score(test, scoring_method="log_likelihood")
    scores = {
        "train_score": train_score,
        "test_score": test_score,
        "train_ll": train_ll,
        "test_ll": test_ll,
    }
    return scores


def make_afr_table(
    aft_dict,
    dataset,
    X_train,
    X_test,
    t0s,
    folder=".",
    span_columns=True,
):
    folder = Path(folder)
    aft_data = pd.DataFrame()
    aft_data.index.name = "Model"
    model_names = [
        x.replace("-", " ").replace("_", " ").title() for x in aft_dict.keys()
    ]
    aft_data.index = model_names
    aft_data["AIC"] = [
        x.AIC_ if not isinstance(x, CoxPHFitter) else np.nan for x in aft_dict.values()
    ]
    aft_data["BIC"] = [
        x.AIC_ if not isinstance(x, CoxPHFitter) else np.nan for x in aft_dict.values()
    ]
    aft_data["Concordance"] = [
        x.score(X_train, scoring_method="concordance_index") for x in aft_dict.values()
    ]
    aft_data["Test Concordance"] = [
        x.score(X_test, scoring_method="concordance_index") for x in aft_dict.values()
    ]
    icis = []
    e50s = []
    for key in aft_dict.keys():
        model = aft_dict[key]
        t0 = t0s[key]
        _, ici, e50 = survival_probability_calibration(model, X_train, t0=t0)
        icis.append(ici)
        e50s.append(e50)
    
    t_icis = []
    t_e50s = []
    for model in aft_dict.values():
        try:
            _, ici, e50 = survival_probability_calibration(model, X_test, t0=t0)
        except ConvergenceError as e:
            ici = np.nan
            e50 = np.nan
        t_icis.append(ici)
        t_e50s.append(e50)
    aft_data['ICI'] = icis
    aft_data['Test ICI'] = t_icis
    aft_data['E50'] = e50s
    aft_data['Test E50'] = t_e50s
    if dataset in ["combined", "Combined", "COMBINED"]:
        pretty_dataset = "combined"
    elif dataset is None:
        pretty_dataset = None
    else:
        pretty_dataset = dataset.upper()
    aft_data = aft_data.round(2)
    aft_data.to_csv(folder / "aft_comparison.csv")
    logger.info(f"Saved AFR comparison to {folder / 'aft_comparison.csv'}")
    aft_data = aft_data.round(2)
    aft_data.fillna("--", inplace=True)
    aft_data.to_latex(
        folder / "aft_comparison.tex",
        float_format="%.3g",  # Two decimal places, since we have 100 adversarial examples
        label=f"tab:{dataset.lower()}" if dataset is not None else "tab:afr_models",  # Label for cross-referencing
        caption=f"Comparison of AFR Models on the {pretty_dataset} dataset." if pretty_dataset is not None else None,
    )
    # Change to table* if span_columns is True
    if span_columns is True:
        with open(folder / "aft_comparison.tex", "r") as f:
            tex_data = f.read()
        tex_data = tex_data.replace(
            r"\begin{table}",
            r"\begin{table*}" + "\n" + r"\centering",
        )
        tex_data = tex_data.replace(r"\end{table}", r"\end{table*}")
        with open(folder / "aft_comparison.tex", "w") as f:
            f.write(tex_data)
    return aft_data


def clean_data_for_aft(
    data,
    covariate_list,
    target="adv_failure_rate",
    dummy_dict={},
):
    # Find categorical vars
    
    assert (
        target in data
    ), f"Target {target} not in dataframe with columns {data.columns}"
    logger.info(f"Shape of dirty data: {data.shape}")
    covariate_list.append(target)
    covariate_list = list(set(covariate_list))
    logger.info(f"Covariates : {covariate_list}")
    subset = data[covariate_list]
    for col in subset.columns:
        subset = subset[subset[col] != -1e10]
        subset = subset[subset[col] != 1e10]
    if len(dummy_dict) > 0:
        dummy_subset=subset[list(dummy_dict.keys())]
        dummies = pd.get_dummies(dummy_subset, prefix=dummy_dict, prefix_sep=" ", columns=list(dummy_dict.keys()))
        subset = subset.drop(columns=list(dummy_dict.keys()))
        cleaned = pd.concat([subset, dummies], axis=1)
    else:
        cleaned = pd.get_dummies(subset, prefix="dummy", prefix_sep="_")
    assert (
        target in cleaned.columns
    ), f"Target {target} not in dataframe with columns {cleaned.columns}"
    logger.info(f"Shape of data data: {cleaned.shape}")
    return cleaned


def split_data_for_aft(
    data,
    target,
    duration_col,
    test_size=0.25,
    random_state=42,
):
    
    X_train, X_test = train_test_split(
        data,
        train_size=(1 - test_size),
        test_size=test_size,
        random_state=random_state,
    )
    assert (
        target in data.columns
    ), f"Target {target} not in dataframe with columns {data.columns}"
    assert (
        duration_col in data.columns
    ), f"Duration {duration_col} not in dataframe with columns {data.columns}"
    X_train = X_train.dropna(axis=0, how="any")
    X_test = X_test.dropna(axis=0, how="any")
    return X_train, X_test


def run_afr_experiment(
    mtype,
    config,
    X_train,
    target,
    duration_col,
    t0,
    X_test=None,
    dummy_dict={},
    folder=".",
):  
    if len(config.keys()) > 0:
        plots = []
        plot_dict = config.pop("plot", {})
        label_dict = config.pop("labels", {})
        partial_effect_list = config.pop("partial_effect", [])
        model_config = config.pop("model", {})
        model_config.update(**config)
        aft = fit_aft(
            summary_file=plot_dict.get("summary_file", f"{mtype}_summary.csv"),
            folder=folder,
            df=X_train,
            event_col=target,
            duration_col=duration_col,
            mtype=mtype,
            **model_config,
        )
        afr_plot = plot_aft(
            aft=aft,
            title=plot_dict.get(
                "qq_title",
                f"{mtype}".replace("_", " ").replace("-", " ").title(),
            ),
            file=plot_dict.get("plot", f"{mtype}_aft.pdf"),
            xlabel=label_dict.pop("xlabel", "log$(\phi)$"),
            ylabel=label_dict.pop("ylabel", ""),  # noqa W605
            replacement_dict=label_dict,
            dummy_dict=dummy_dict,
            folder=folder,
        )
        plots.append(afr_plot)
        qq_plot = plot_qq(
            X_train=X_train,
            X_test=X_test,
            aft=aft,
            title=plot_dict.get(
                "title",
                f"{mtype}".replace("_", " ").replace("-", " ").title() + " AFR QQ Plot",
            ),
            t0=t0,
            file=plot_dict.get("qq_file", f"{mtype}_qq.pdf"),
            xlabel=label_dict.pop("xlabel", None),
            ylabel=label_dict.pop("ylabel", None),
            folder=folder,
        )
        plots.append(qq_plot)
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
    test_size=0.25,
    folder=".",
    dummy_dict={},
):
    
    assert target in data.columns, f"{target} not in data.columns"
    assert duration_col in data.columns, f"{duration_col} not in data.columns"
    X_train, X_test = split_data_for_aft(
        data,
        target,
        duration_col,
        test_size=test_size,
        random_state=42,
    )
    plots = {}
    models = {}
    mtypes = list(config.keys())
    t0s = {}
    for mtype in mtypes:
        sub_config = config.get(mtype, {})
        t0 = sub_config.pop("t0", 0.35)
        models[mtype], plots[mtype] = run_afr_experiment(
            t0=t0,
            mtype=mtype,
            config=sub_config,
            X_train=X_train,
            X_test=X_test,
            target=target,
            dummy_dict=dummy_dict,
            duration_col=duration_col,
            folder=folder,
        )
        t0s[mtype] = t0
    aft_data = make_afr_table(
        models,
        dataset,
        X_train,
        X_test,
        folder=folder,
        t0s=t0s
    )
    print("*" * 80)
    print("*" * 34 + "  RESULTS   " + "*" * 34)
    print("*" * 80)
    print(f"{aft_data}")
    print("*" * 80)


def calculate_raw_failures(args, data, config):
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    assert Path(args.config_file).exists(), f"{args.config_file} does not exist."
    covariates = config.get("covariates", [])
    assert len(covariates) > 0, "No covariates specified in config file"
    if "adv_failures" in covariates and "adv_failures" not in data.columns:
        data.loc[:, "adv_failures"] = (1 - data.loc[:, "adv_accuracy"]) * data.loc[
            :,
            "attack.attack_size",
        ]
        del data['adv_accuracy']
        covariates.remove("adv_accuracy")
    if "ben_failures" in covariates and "ben_failures" not in data.columns:
        data.loc[:, "ben_failures"] = (1 - data.loc[:, "accuracy"]) * data.loc[
            :,
            "attack.attack_size",
        ]
        del data['accuracy']
    data = drop_rows_without_results(data, covariates)
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
    data = load_results(results_file = Path(csv_file).name, results_folder = Path(csv_file).parent)
    
    logger.info(f"Shape of data: {data.shape}")
    data.columns = data.columns.str.strip()
    if len(str(args.config_file).split(":")) > 1:
        subdict = str(args.config_file).split(":")[1]
        config_file = str(args.config_file).split(":")[0]
        with Path(config_file).open("r") as f:
            config = yaml.safe_load(f)
            config = config[subdict]
    else:
        with Path(args.config_file).open("r") as f:
            config = yaml.safe_load(f)
    fillna = config.pop("fillna", {})
    for k, v in fillna.items():
        assert k in data.columns, f"{k} not in data"
        data[k] = data[k].fillna(v)
    dummies  = config.pop("dummies", {"atk_gen" : "Atk:", "def_gen" : "Def:", "id" : "Data:"})
    data = calculate_raw_failures(args, data, config)
    covariate_list = config.pop("covariates", [])
    data = clean_data_for_aft(data, covariate_list, target=target, dummy_dict=dummies)
    assert target in data.columns, f"{target} not in data.columns"
    assert duration_col in data.columns, f"{duration_col} not in data.columns"
    render_all_afr_plots(
        config,
        duration_col,
        target,
        data,
        dataset,
        test_size=0.25,
        folder=FOLDER,
        dummy_dict=dummies
    )


if "__main__" == __name__:
    afr_parser = argparse.ArgumentParser()
    afr_parser.add_argument("--target", type=str, default="adv_failures")
    afr_parser.add_argument("--duration_col", type=str, default="adv_fit_time")
    afr_parser.add_argument("--dataset", type=str, default=None)
    afr_parser.add_argument("--data_file", type=str, default="data.csv")
    afr_parser.add_argument("--config_file", type=str, default="afr.yaml")
    afr_parser.add_argument("--plots_folder", type=str, default="plots")
    args = afr_parser.parse_args()
    main(args)
