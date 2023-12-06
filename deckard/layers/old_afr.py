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
from .plots import calculate_failure_rate, drop_frames_without_results, min_max_scaling
import matplotlib
import logging
import yaml
import argparse

logger = logging.getLogger(__name__)

if "__main__" == __name__:
    afr_parser = argparse.ArgumentParser()
    afr_parser.add_argument("--target", type=str, default="adv_failures")
    afr_parser.add_argument("--duration_col", type=str, default="adv_fit_time")
    afr_parser.add_argument("--dataset", type=str, default="mnist")
    afr_parser.add_argument("--data_file", type=str, default="data.csv")
    afr_args = afr_parser.parse_args()
    target = afr_args.target
    duration_col = afr_args.duration_col
    dataset = afr_args.dataset

    font = {
        "family": "Times New Roman",
        "weight": "bold",
        "size": 22,
    }

    matplotlib.rc("font", **font)

    csv_file = afr_args.data_file
    FOLDER = Path(csv_file).parent
    data = pd.read_csv(csv_file, index_col=0)
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    data.def_value.replace("", 0, inplace=True)
    data.atk_value.replace("", 0, inplace=True)
    data = drop_frames_without_results(data)
    data = calculate_failure_rate(data)
    data = min_max_scaling(data)
    data.dropna(axis=0, subset=["atk_value", "atk_param"], inplace=True)
    data.dropna(axis=0, subset=["def_value", "def_param"], inplace=True)
    data.loc[:, "adv_failures"] = (1 - data.loc[:, "adv_accuracy"]) * data.loc[
        :, "attack.attack_size"
    ]
    data.loc[:, "ben_failures"] = (1 - data.loc[:, "accuracy"]) * data.loc[
        :, "attack.attack_size"
    ]

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
        ax.get_figure().savefig(FOLDER / file)
        logger.info(f"Saved graph to {FOLDER / file}")
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
        **kwargs,
    ):
        plt.gcf().clear()
        # kwargs.pop("replacement_dict")
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
        pareto.get_figure().savefig(FOLDER / file)
        logger.info(f"Saved graph to {FOLDER / file}")
        plt.gcf().clear()
        return pareto

    def score_model(aft, train, test):
        train_score = aft.score(train)
        test_score = aft.score(test)
        scores = {"train_score": train_score, "test_score": test_score}
        plt.show()
        return scores

    def make_afr_table(score_list, aft_dict, dataset):
        assert len(score_list) == len(
            aft_dict,
        ), "Length of score list and aft dict must be equal"
        models = list(aft_dict.keys())
        aft_data = pd.DataFrame(models, columns=["Distribution"])
        aft_data.index.name = "Model"
        aft_data["AIC"] = [
            x.AIC_ if not isinstance(x, CoxPHFitter) else np.nan
            for x in aft_dict.values()
        ]
        aft_data["Concordance"] = [x.concordance_index_ for x in aft_dict.values()]
        aft_data["BIC"] = [
            x.AIC_ if not isinstance(x, CoxPHFitter) else np.nan
            for x in aft_dict.values()
        ]
        # aft_data["Train LL"] = [x["train_score"] for x in score_list]
        # aft_data["Test LL"] = [x["test_score"] for x in score_list]
        aft_data["Mean $S(t;\\theta)$"] = [
            x.predict_expectation(X_train).mean() for x in aft_dict.values()
        ]
        aft_data["Median $S(t;\\theta)$"] = [
            x.predict_median(X_train).median() for x in aft_dict.values()
        ]
        aft_data = aft_data.round(2)
        aft_data.to_csv(FOLDER / "aft_comparison.csv")
        logger.info(f"Saved AFT comparison to {FOLDER / 'aft_comparison.csv'}")
        aft_data = aft_data.round(2)
        aft_data.fillna("--", inplace=True)
        aft_data = pd.read_csv(FOLDER / "aft_comparison.csv", index_col=0)
        # aft_data.fillna("--", inplace=True)
        aft_data.to_latex(
            FOLDER / "aft_comparison.tex",
            float_format="%.2f",
            label=f"tab:{dataset}",
            caption=f"Comparison of AFR Models on the {dataset.upper()} dataset.",
        )
        return aft_data

    def clean_data_for_aft(
        data,
        kwarg_list,
        target="adv_failure_rate",
    ):
        subset = data.copy()
        assert (
            target in subset
        ), f"Target {target} not in dataframe with columns {subset.columns}"

        cleaned = pd.DataFrame()
        kwarg_list.append(target)
        for kwarg in kwarg_list:
            cleaned = pd.concat([cleaned, subset[kwarg]], axis=1)
        cols = cleaned.columns
        cleaned = pd.DataFrame(subset, columns=cols)
        if "accuracy" in cleaned.columns:
            cleaned = cleaned[cleaned["accuracy"] != 1e10]
            cleaned = cleaned[cleaned["accuracy"] != -1e10]
        if "adv_accuracy" in cleaned.columns:
            cleaned = cleaned[cleaned["adv_accuracy"] != 1e10]
            cleaned = cleaned[cleaned["adv_accuracy"] != -1e10]
        cleaned.dropna(inplace=True, how="any", axis=0)
        y = cleaned[target]
        assert (
            target in cleaned
        ), f"Target {target} not in dataframe with columns {cleaned.columns}"
        return cleaned, y, data

    def split_data_for_aft(
        data,
        target,
        duration_col,
        kwarg_list,
        test_size=0.2,
        random_state=42,
    ):
        cleaned, y, data = clean_data_for_aft(data, kwarg_list, target=target)
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        assert (
            target in cleaned
        ), f"Target {target} not in dataframe with columns {cleaned.columns}"
        assert (
            duration_col in cleaned
        ), f"Duration {duration_col} not in dataframe with columns {cleaned.columns}"
        return X_train, X_test, y_train, y_test

    kwarg_list = [
        "accuracy",
        "train_time",
        "predict_time",
        "atk_value",
        "def_value",
        "data.sample.random_state",
        "adv_failure_rate",
        "model_layers",
        "adv_fit_time",
        # "model.art.pipeline.initialize.kwargs.optimizer.lr",
    ]

    X_train, X_test, y_train, y_test = split_data_for_aft(
        data,
        target,
        duration_col,
        kwarg_list,
        test_size=0.2,
        random_state=42,
    )

    weibull_dict = {  # noqa w605
        "Intercept: rho_": "$\\rho$",
        "Intercept: lambda_": "$\lambda$",  # noqa w605
        "data.sample.random_state: lambda_": "Random State",  # noqa w605
        "def_value: lambda_": "Defence Strength",
        "atk_value: lambda_": "Attack Strength",
        "train_time: lambda_": "$t_{train}$",
        "predict_time: lambda_": "$t_{predict}$",
        "adv_accuracy: lambda_": "$\lambda_{adv.}$",  # noqa w605
        "accuracy: lambda_": "$\lambda_{ben.}$",  # noqa w605
        "adv_fit_time: lambda_": "$t_{attack}$",
        "adv_log_loss: lambda_": "Adv. Log Loss",
        "adv_failure_rate: lambda_": "$h_{adv.}(t,;\\theta)$",
        "failure_rate: lambda_": "$h_{ben.}(t,;\\theta)$",
        "model_layers: lambda_": "No. of Layers",
        "model.art.pipeline.initialize.kwargs.optimizer.lr: lambda_": "Learning Rate",
        "def_gen": "Defence",
    }  # noqa w605

    weibull_plot_dict = {
        "file": "weibull_aft.pdf",
        "title": "Weibull AFR Model",
        "mtype": "weibull",
    }

    weibull_afr, wft = plot_aft(
        X_train,
        event_col=target,
        duration_col=duration_col,
        **weibull_plot_dict,
        replacement_dict=weibull_dict,
    )

    weibull_partial_dict_layers = {
        "file": "weibull_partial_effects.pdf",
        "covariate_array": "model_layers",
        "values_array": [18, 34, 50, 101, 152],
        "title": "$S(t)$ for Weibull AFR",
        "ylabel": "Expectation of $S(t)$",
        "xlabel": "Time $T$ (seconds)",
        "legend_kwargs": {
            "title": "No. of Layers",
            "labels": ["18", "34", "50", "101", "152"],
        },
    }

    weibull_layers = plot_partial_effects(aft=wft, **weibull_partial_dict_layers)
    wft_scores = score_model(wft, X_train, X_test)

    cox_replacement_dict = {
        "adv_failure_rate": "$h_{adv}(t,;\\theta)$",
        "def_value": "Defence Strength",
        "data.sample.random_state": "Random State",
        "train_time": "$t_{train}$",
        "model_layers": "No. of Layers",
        "model.art.pipeline.initialize.kwargs.optimizer.lr": "Learning Rate",
        "adv_accuracy": "$\lambda_{adv.}$",  # noqa w605
        "adv_fit_time": "$t_{attack}$",
        "adv_log_loss": "Adv. Log Loss",
        "predict_time": "$t_{predict}$",
        "accuracy": "$\lambda_{ben.}$",  # noqa w605
        "failure_rate": "$h_{ben.}(t,;\\theta)$",
        "atk_value": "Attack Strength",
    }  # noqa w605
    cox_partial_dict = {
        "file": "cox_partial_effects.pdf",
        "covariate_array": "model_layers",
        "values_array": [18, 34, 50, 101, 152],
        "replacement_dict": cox_replacement_dict,
        "title": "$S(t)$ for  Cox AFR",
        "ylabel": "Expectation of $S(t)$",
        "xlabel": "Time $T$ (seconds)",
        "legend_kwargs": {
            "title": "No. of Layers",
            "labels": ["18", "34", "50", "101", "152"],
        },
    }
    cox_plot_dict = {
        "file": "cox_aft.pdf",
        "duration_col": duration_col,
        "title": "Cox AFR Model",
        "mtype": "cox",
        "replacement_dict": cox_replacement_dict,
    }
    cox_afr, cft = plot_aft(df=X_train, event_col=target, **cox_plot_dict)
    cox_scores = score_model(cft, X_train, X_test)
    cox_partial = plot_partial_effects(aft=cft, **cox_partial_dict)

    log_normal_dict = {
        "Intercept: sigma_": "$\sigma$",  # noqa w605
        "Intercept: mu_": "$\mu$",  # noqa w605
        "def_value: mu_": "Defence Strength",
        "atk_value: mu_": "Attack Strength",
        "train_time: mu_": "$t_{train}$",
        "predict_time: mu_": "$t_{predict}$",
        "adv_fit_time: mu_": "$t_{attack}$",
        "model_layers: mu_": "No. of Layers",
        "model.art.pipeline.initialize.kwargs.optimizer.lr: mu_": "Learning Rate",
        "data.sample.random_state: mu_": "Random State",
        "adv_log_loss: mu_": "Adv. Log Loss",
        "adv_accuracy: mu_": "$\lambda_{adv.}$",  # noqa w605
        "accuracy: mu_": "$\lambda_{ben.}$",  # noqa w605
        "adv_failure_rate: mu_": "$h_{adv}(t,;\\theta)$",
        "def_gen": "Defence",
        "learning_rate: mu_": "Learning Rate",
    }  # noqa w605

    log_normal_graph, lnt = plot_aft(
        X_train,
        "log_normal_aft.pdf",
        target,
        duration_col,
        "Log Normal AFR Model",
        "log_normal",
        replacement_dict=log_normal_dict,
    )
    lnt_scores = score_model(lnt, X_train, X_test)
    lnt_partial = plot_partial_effects(
        file="log_normal_partial_effects.pdf",
        aft=lnt,
        covariate_array="model_layers",
        values_array=[18, 34, 50, 101, 152],
        replacement_dict=log_normal_dict,
        title="$S(t)$ for Log-Normal AFR",
        ylabel="Expectation of $S(t)$",
        xlabel="Time $T$ (seconds)",
        legend_kwargs={
            "title": "No. of Layers",
            "labels": ["18", "34", "50", "101", "152"],
        },
    )
    log_logistic_dict = {  # noqa w605
        "Intercept: beta_": "$\\beta$",  # noqa w605
        "Intercept: alpha_": "$\\alpha$",
        "data.sample.random_state: alpha_": "Random State",
        "def_value: alpha_": "Defence Strength",
        "atk_value: alpha_": "Attack Strength",
        "train_time: alpha_": "$t_{train}$",
        "predict_time: alpha_": "$t_{predict}$",
        "adv_accuracy: alpha_": "$\lambda_{adv.}$",  # noqa w605
        "accuracy: alpha_": "$\lambda_{ben.}$",  # noqa w605
        "adv_fit_time: alpha_": "$t_{attack}$",
        "model_layers: alpha_": "No. of Layers",
        "model.art.pipeline.initialize.kwargs.optimizer.lr": "Learning Rate",
        "adv_failure_rate: alpha_": "$h_{adv.}(t,\\theta)$",
        "alpha_": "",
    }

    log_logistic_graph, llt = plot_aft(
        X_train,
        "log_logistic_aft.pdf",
        target,
        duration_col,
        "Log Logistic AFR Model",
        "log_logistic",
        replacement_dict=log_logistic_dict,
    )
    llt_scores = score_model(llt, X_train, X_test)
    llt_partial = plot_partial_effects(
        file="log_logistic_partial_effects.pdf",
        aft=llt,
        covariate_array="model_layers",
        values_array=[18, 34, 50, 101, 152],
        replacement_dict=log_logistic_dict,
        title="$S(t)$  for Log-Logistic AFR",
        ylabel="Expectation of $S(t)$",
        xlabel="Time $T$ (seconds)",
        legend_kwargs={
            "title": "No. of Layers",
            "labels": ["18", "34", "50", "101", "152"],
        },
    )
    aft_dict = {
        "Weibull": wft,
        "LogNormal": lnt,
        "LogLogistic": llt,
        "Cox": cft,
    }
    score_list = [
        wft_scores,
        lnt_scores,
        llt_scores,
        cox_scores,
    ]
    aft_data = make_afr_table(score_list, aft_dict, dataset)
