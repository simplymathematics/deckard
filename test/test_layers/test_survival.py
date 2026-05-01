import matplotlib
import numpy as np
import optuna
import pandas as pd
import pytest
import yaml

pytest.importorskip("lifelines")

from deckard.attack import AttackConfig  # NOQA E402
from deckard.data import DataConfig  # NOQA E402
from deckard.layers.survival import (  # NOQA E402
    _build_runtime_survival_data_config,
    calculate_failures_under_attack,
    fit_aft,
    survival_main,
    survival_probability_calibration,
)  # NOQA E402

matplotlib.use("Agg")


def _make_survival_dataframe(n=80, seed=7):
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=n)
    group = rng.choice(["a", "b"], size=n)
    risk = np.exp(0.4 * feature + (group == "b") * 0.25)
    duration = rng.exponential(scale=np.clip(risk, 0.2, 10.0), size=n)
    event = rng.binomial(1, 0.8, size=n)
    return pd.DataFrame(
        {
            "feature": feature,
            "group": group,
            "duration": duration,
            "event": event,
        },
    )


def test_fit_aft_and_calibration_curve(tmp_path):
    df = _make_survival_dataframe(n=60)
    df = pd.get_dummies(df, columns=["group"], drop_first=True)

    summary_file = tmp_path / "weibull_summary.csv"
    aft = fit_aft(
        df=df,
        event_col="event",
        duration_col="duration",
        mtype="weibull",
        summary_file=summary_file.as_posix(),
    )

    assert summary_file.exists()
    assert hasattr(aft, "summary")

    _, ici, e50, curve = survival_probability_calibration(
        aft,
        df,
        t0=0.5,
        return_curve=True,
        plot=False,
    )
    assert np.isfinite(ici)
    assert np.isfinite(e50)
    assert {"predicted", "observed"}.issubset(curve.columns)


def test_survival_main_end_to_end(tmp_path):
    data = _make_survival_dataframe()
    data_file = tmp_path / "survival_data.csv"
    data.to_csv(data_file, index=False)

    plots_folder = tmp_path / "plots"
    config_file = tmp_path / "survival_config.yaml"
    config = {
        "test_size": 0.25,
        "random_state": 42,
        "covariates": ["feature", "group", "duration", "event"],
        "dummies": {"group": "Group"},
        "weibull": {
            "t0": 0.5,
            "model": {"penalizer": 0.01},
            "plot": {
                "plot": "weibull_aft.pdf",
                "qq_file": "weibull_qq.pdf",
                "summary_file": "weibull_summary.csv",
                "summary_plot": "weibull_summary_plot.pdf",
            },
        },
    }
    config_file.write_text(yaml.safe_dump(config))

    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=plots_folder.as_posix(),
        config_file=config_file.as_posix(),
        target="event",
        duration_col="duration",
        dataset="toy",
    )

    assert "aft_table" in result
    assert not result["aft_table"].empty
    assert result["model_scores"] is None
    assert (plots_folder / "weibull_summary.csv").exists()
    assert (plots_folder / "weibull_aft.pdf").exists()
    assert (plots_folder / "weibull_qq.pdf").exists()
    assert (plots_folder / "weibull_summary_plot.pdf").exists()
    assert (plots_folder / "aft_comparison.csv").exists()


def test_survival_main_with_attack_uses_separate_survival_model(tmp_path):
    data = _make_survival_dataframe()
    data_file = tmp_path / "survival_data.csv"
    data.to_csv(data_file, index=False)

    plots_folder = tmp_path / "plots"
    config_file = tmp_path / "survival_attack_config.yaml"
    config = {
        "test_size": 0.25,
        "random_state": 42,
        "covariates": ["feature", "group", "duration", "event"],
        "dummies": {"group": "Group"},
        "cox": {
            "t0": 0.5,
            "model": {"penalizer": 0.01},
            "plot": {
                "plot": "cox_aft.pdf",
                "qq_file": "cox_qq.pdf",
                "summary_file": "cox_summary.csv",
                "summary_plot": "cox_summary_plot.pdf",
            },
        },
    }
    config_file.write_text(yaml.safe_dump(config))

    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=plots_folder.as_posix(),
        config_file=config_file.as_posix(),
        target="event",
        duration_col="duration",
        dataset="toy",
        model={
            "model_type": "sklearn.linear_model.LinearRegression",
            "classifier": False,
            "model_params": {},
            "alias": "attack-model",
        },
        survival_model="cox",
        attack={"attack_type": "art.attacks.evasion.HopSkipJump"},
    )

    assert "aft_table" in result
    assert not result["aft_table"].empty
    assert result["model_scores"] is not None
    assert (plots_folder / "cox_summary.csv").exists()


def test_runtime_survival_data_split_uses_dataconfig():
    data = _make_survival_dataframe(n=40)
    runtime_data = _build_runtime_survival_data_config(
        data=data,
        target="event",
        test_size=0.25,
        random_state=42,
    )

    assert isinstance(runtime_data, DataConfig)
    assert runtime_data.X_train is not None
    assert runtime_data.X_test is not None
    assert runtime_data.y_train is not None
    assert runtime_data.y_test is not None
    assert "event" not in runtime_data.X_train.columns
    assert "event" not in runtime_data.X_test.columns
    assert len(runtime_data.X_train) + len(runtime_data.X_test) == len(data)


def test_calculate_failures_under_attack_evasion():
    df = pd.DataFrame(
        {
            "accuracy": [0.9, 0.8],
            "evasion_accuracy": [0.6, 0.5],
        },
    )
    attack = AttackConfig(attack_type="art.attacks.evasion.HopSkipJump")
    output = calculate_failures_under_attack(df, attack)
    assert "ben_failures" in output.columns
    assert "adv_failures" in output.columns
    assert np.allclose(output["ben_failures"].values, [0.1, 0.2])
    assert np.allclose(output["adv_failures"].values, [0.4, 0.5])


def test_calculate_failures_under_attack_mixed_attack_rows():
    df = pd.DataFrame(
        {
            "accuracy": [1.0, 1.0, 1.0],
            "evasion_accuracy": [0.0, np.nan, np.nan],
            "membership_inference_accuracy": [np.nan, 1.0, np.nan],
            "sex_inference_accuracy": [np.nan, np.nan, 0.0],
            "attack name": ["hsj", "membership", "attribute-bb"],
        },
    )

    output = calculate_failures_under_attack(df)

    assert np.allclose(output["ben_failures"].values, [0.0, 0.0, 0.0])
    assert np.allclose(output["adv_failures"].values, [1.0, 0.0, 1.0])


def test_survival_main_from_optuna_attack_db(tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'attack_results.sqlite3').as_posix()}"
    schema = {
        "optimization": 0,
        "dataset": 1,
        "model type": 2,
        "defense_name": 3,
        "attack name": 4,
        "sep": "_",
    }

    trials = [
        (
            "survival_toy_rf_baseline_hsj",
            {
                "accuracy": 1.0,
                "evasion_accuracy": 0.0,
                "attack_generation_time": 1.0,
            },
        ),
        (
            "survival_toy_rf_baseline_membership",
            {
                "accuracy": 1.0,
                "membership_inference_accuracy": 1.0,
                "attack_generation_time": 2.0,
            },
        ),
        (
            "survival_toy_rf_baseline_attribute-bb",
            {
                "accuracy": 1.0,
                "sex_inference_accuracy": 0.0,
                "attack_generation_time": 3.0,
            },
        ),
    ]

    for study_name, attrs in trials:
        study = optuna.create_study(
            study_name=study_name,
            storage=db_url,
            load_if_exists=True,
        )

        def objective(trial, trial_attrs=attrs):
            enriched_attrs = dict(trial_attrs)
            enriched_attrs["attack_generation_time"] = (
                trial_attrs["attack_generation_time"] + trial.number * 0.25
            )
            for key, value in enriched_attrs.items():
                trial.set_user_attr(key, value)
            return 0.5

        study.optimize(objective, n_trials=3)

    plots_folder = tmp_path / "plots"
    config_file = tmp_path / "survival_optuna_config.yaml"
    config = {
        "test_size": 0.34,
        "random_state": 42,
        "covariates": ["accuracy", "attack name", "attack_generation_time"],
        "weibull": {
            "t0": 1.5,
            "model": {"penalizer": 0.01},
            "plot": {
                "plot": "weibull_aft.pdf",
                "qq_file": "weibull_qq.pdf",
                "summary_file": "weibull_summary.csv",
            },
        },
    }
    config_file.write_text(yaml.safe_dump(config))

    result = survival_main(
        attack_optuna_db=db_url,
        attack_schema=schema,
        plots_folder=plots_folder.as_posix(),
        config_file=config_file.as_posix(),
        target="adv_failures",
        duration_col="attack_generation_time",
        dataset="compiled-attacks",
        model="weibull",
        calculate_attack_failures=True,
    )

    assert "aft_table" in result
    assert not result["aft_table"].empty
    assert (plots_folder / "weibull_summary.csv").exists()
