import matplotlib
import numpy as np
import optuna
import pandas as pd
import pytest
import yaml

try:
    import lifelines  # noqa: F401
    from deckard.attack import AttackConfig
    from deckard.data import DataConfig
    from deckard.layers.survival import (
        calculate_failures_under_attack,
        fit_aft,
        survival_main,
        survival_probability_calibration,
    )
except Exception:
    pytest.skip("lifelines is required for survival layer tests", allow_module_level=True)

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
    runtime_data = DataConfig(
        dataset_name="make_regression",
        target="event",
        classifier=False,
        stratify=False,
        train_size=0.75,
        test_size=0.25,
        random_state=42,
    )
    runtime_data._X = data.drop(columns=["event"]).reset_index(drop=True)
    runtime_data._y = data["event"].reset_index(drop=True)
    runtime_data.data_load_time = 0.0
    runtime_data._sample()

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
    attack = AttackConfig(
        attack_type="art.attacks.evasion.HopSkipJump",
        attack_size=10,
    )
    output = calculate_failures_under_attack(df, attack)
    assert "ben_failures" in output.columns
    assert "adv_failures" in output.columns
    assert np.allclose(output["ben_failures"].values, [1.0, 2.0])
    assert np.allclose(output["adv_failures"].values, [4.0, 5.0])


def test_calculate_failures_under_attack_mixed_attack_rows():
    df = pd.DataFrame(
        {
            "accuracy": [1.0, 1.0, 1.0],
            "attack_size": [10, 20, 5],
            "evasion_accuracy": [0.0, np.nan, np.nan],
            "membership_inference_accuracy": [np.nan, 1.0, np.nan],
            "sex_inference_accuracy": [np.nan, np.nan, 0.0],
            "attack name": ["hsj", "membership", "attribute-bb"],
        },
    )

    output = calculate_failures_under_attack(df)

    assert np.allclose(output["ben_failures"].values, [0.0, 0.0, 0.0])
    assert np.allclose(output["adv_failures"].values, [10.0, 0.0, 5.0])


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


def test_survival_main_raises_on_missing_duration_col(tmp_path):
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    with pytest.raises(ValueError, match="duration_col 'time' not found"):
        survival_main(
            data_file=data_file.as_posix(),
            plots_folder=(tmp_path / "plots").as_posix(),
            target="event",
            duration_col="time",
            survival_model="weibull",
        )


def test_survival_main_raises_on_missing_target_col(tmp_path):
    # Use the optuna path so data loading doesn't validate the target column;
    # survival_main's own check should fire after data is loaded.
    import optuna as _optuna

    db_url = f"sqlite:///{(tmp_path / 'attack.sqlite3').as_posix()}"
    schema = {"optimization": 0, "dataset": 1, "model type": 2, "defense_name": 3, "attack name": 4, "sep": "_"}
    study = _optuna.create_study(study_name="run_toy_rf_baseline_hsj", storage=db_url, load_if_exists=True)

    def _obj(trial):
        trial.set_user_attr("accuracy", 0.9)
        trial.set_user_attr("evasion_accuracy", 0.5)
        trial.set_user_attr("attack_generation_time", 1.0)
        return 0.5

    study.optimize(_obj, n_trials=2)

    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(yaml.safe_dump({
        "weibull": {"t0": 1.0, "model": {}, "plot": {"summary_file": "s.csv"}, "labels": {}},
    }))
    with pytest.raises(ValueError, match="target 'mystery' not found"):
        survival_main(
            attack_optuna_db=db_url,
            attack_schema=schema,
            plots_folder=(tmp_path / "plots").as_posix(),
            config_file=config_file.as_posix(),
            target="mystery",
            duration_col="attack_generation_time",
            survival_model="weibull",
        )


def test_survival_main_raises_on_unresolvable_survival_model(tmp_path):
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    # model is a Mapping with no recognized model-name key; survival_model is None
    with pytest.raises(ValueError, match="Could not resolve survival model name"):
        survival_main(
            data_file=data_file.as_posix(),
            plots_folder=(tmp_path / "plots").as_posix(),
            target="event",
            duration_col="duration",
            model={"penalizer": 0.01},
            survival_model=None,
        )


def test_survival_main_raises_on_unsupported_data_spec_type(tmp_path):
    with pytest.raises(TypeError, match="Unsupported data_spec type"):
        survival_main(
            data=12345,
            plots_folder=(tmp_path / "plots").as_posix(),
            target="event",
            duration_col="duration",
            survival_model="weibull",
        )


def test_survival_main_cfg_mapping_overrides(tmp_path):
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    plots_folder = tmp_path / "plots"
    cfg = {
        "survival": {
            "test_size": 0.3,
            "random_state": 1,
            "weibull": {
                "t0": 0.5,
                "model": {},
                "plot": {"summary_file": "cfg_summary.csv"},
                "labels": {},
            },
        }
    }
    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=plots_folder.as_posix(),
        target="event",
        duration_col="duration",
        survival_model="weibull",
        cfg=cfg,
    )
    assert "aft_table" in result


def test_survival_main_data_as_dataconfig(tmp_path):
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    data_cfg = DataConfig(
        dataset_name=data_file.as_posix(),
        target="event",
        classifier=False,
        stratify=False,
        test_size=0.25,
        random_state=42,
    )
    plots_folder = tmp_path / "plots"
    result = survival_main(
        data=data_cfg,
        plots_folder=plots_folder.as_posix(),
        target="event",
        duration_col="duration",
        survival_model="weibull",
    )
    assert "aft_table" in result


def test_survival_main_data_as_mapping(tmp_path):
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    plots_folder = tmp_path / "plots"
    result = survival_main(
        data={"dataset_name": data_file.as_posix(), "target": "event", "classifier": False, "stratify": False, "test_size": 0.25, "random_state": 42},
        plots_folder=plots_folder.as_posix(),
        target="event",
        duration_col="duration",
        survival_model="weibull",
    )
    assert "aft_table" in result


def test_survival_main_model_resolved_from_mapping_alias(tmp_path):
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    plots_folder = tmp_path / "plots"
    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=plots_folder.as_posix(),
        target="event",
        duration_col="duration",
        model={"alias": "weibull"},
    )
    assert "aft_table" in result


def test_survival_main_model_resolved_from_cfg_mapping(tmp_path):
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    plots_folder = tmp_path / "plots"
    cfg = {"model": "weibull"}
    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=plots_folder.as_posix(),
        target="event",
        duration_col="duration",
        cfg=cfg,
    )
    assert "aft_table" in result


def test_survival_main_model_resolved_from_cfg_survival_nested(tmp_path):
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    plots_folder = tmp_path / "plots"
    cfg = {"survival": {"model": "weibull"}}
    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=plots_folder.as_posix(),
        target="event",
        duration_col="duration",
        cfg=cfg,
    )
    assert "aft_table" in result


def test_survival_main_fillna_raises_on_missing_col(tmp_path):
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    plots_folder = tmp_path / "plots"
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(yaml.safe_dump({"fillna": {"nonexistent_col": 0}}))
    with pytest.raises(ValueError, match="nonexistent_col not found in input data"):
        survival_main(
            data_file=data_file.as_posix(),
            plots_folder=plots_folder.as_posix(),
            config_file=config_file.as_posix(),
            target="event",
            duration_col="duration",
            survival_model="weibull",
        )


def test_survival_main_partial_effects_in_run_experiment(tmp_path):
    df = _make_survival_dataframe(n=60)
    df = pd.get_dummies(df, columns=["group"], drop_first=True)
    from deckard.layers.survival import run_survival_model_experiment
    _, plots = run_survival_model_experiment(
        mtype="weibull",
        config={
            "model": {},
            "plot": {"summary_plot": "summary.pdf"},
            "labels": {},
            "partial_effect": [{"covariate_array": ["feature"], "values": [-1, 0, 1], "title": "PE", "xlabel": "x", "ylabel": "y"}],
        },
        X_train=df,
        target="event",
        duration_col="duration",
        t0=0.5,
        folder=tmp_path.as_posix(),
    )
    assert len(plots) >= 3


def test_survival_main_config_file_overrides_all_defaults(tmp_path):
    """Cover config-file override paths for parameters at their defaults."""
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    plots_output = tmp_path / "custom_plots"
    config = {
        "data_file": data_file.as_posix(),
        "plots_folder": plots_output.as_posix(),
        "target": "event",
        "duration_col": "duration",
        "dataset": "override-ds",
        "survival_model": "weibull",
        "test_size": 0.3,
        "random_state": 7,
        "covariates": ["feature", "event", "duration"],
        "fillna": {"feature": 0.0},
        "weibull": {"t0": 0.5, "model": {}, "plot": {"summary_file": "s.csv"}, "labels": {}},
    }
    config_file = tmp_path / "full_cfg.yaml"
    config_file.write_text(yaml.safe_dump(config))
    # Call with all parameters at their defaults so config-file overrides fire
    result = survival_main(config_file=config_file.as_posix())
    assert "aft_table" in result
    assert plots_output.exists()


def test_survival_main_cfg_model_resolution_via_cfg_mapping(tmp_path):
    """Lines 424-426: resolved via cfg['model'] string when model arg is a Mapping without name keys."""
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=(tmp_path / "plots").as_posix(),
        target="event",
        duration_col="duration",
        model={"penalizer": 0.01},    # Mapping without recognised name key
        cfg={"model": "weibull"},      # resolved from cfg['model']
    )
    assert "aft_table" in result


def test_survival_main_cfg_model_resolution_via_nested_survival(tmp_path):
    """Lines 427-430: resolved via cfg['survival']['model'] when model is a Mapping."""
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=(tmp_path / "plots").as_posix(),
        target="event",
        duration_col="duration",
        model={"penalizer": 0.01},
        cfg={"survival": {"model": "weibull"}},
    )
    assert "aft_table" in result


def test_survival_main_lifelines_dataset_as_dataconfig(tmp_path):
    """Lines 382-386: DataConfig path where dataset is a lifelines name → target set to None."""
    from deckard.data import DataConfig as _DC
    result = survival_main(
        data=_DC(dataset_name="lifelines.rossi", target="week", classifier=False, stratify=False),
        plots_folder=(tmp_path / "plots").as_posix(),
        target="arrest",
        duration_col="week",
        survival_model="weibull",
    )
    assert "aft_table" in result


def test_survival_main_lifelines_dataset_as_mapping(tmp_path):
    """Line 400: Mapping data_spec where dataset_name is a lifelines name → target set to None."""
    result = survival_main(
        data={"dataset_name": "lifelines.rossi", "classifier": False, "stratify": False},
        plots_folder=(tmp_path / "plots").as_posix(),
        target="arrest",
        duration_col="week",
        survival_model="weibull",
    )
    assert "aft_table" in result


def test_survival_main_fillna_applies_to_column(tmp_path):
    """Line 491: fillna actually fills values in the loaded data."""
    data = _make_survival_dataframe(n=40)
    data.loc[0, "feature"] = float("nan")
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(yaml.safe_dump({
        "fillna": {"feature": 0.0},
        "weibull": {"t0": 0.5, "model": {}, "plot": {"summary_file": "s.csv"}, "labels": {}},
    }))
    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=(tmp_path / "plots").as_posix(),
        config_file=config_file.as_posix(),
        target="event",
        duration_col="duration",
        survival_model="weibull",
    )
    assert "aft_table" in result


def test_survival_main_covariates_missing_duration_gets_appended(tmp_path):
    """Line 509: duration_col appended to covariates when not present."""
    data = _make_survival_dataframe(n=40)
    data_file = tmp_path / "data.csv"
    data.to_csv(data_file, index=False)
    config_file = tmp_path / "cfg.yaml"
    # covariates list has event but not duration
    config_file.write_text(yaml.safe_dump({
        "covariates": ["feature", "event"],
        "weibull": {"t0": 0.5, "model": {}, "plot": {"summary_file": "s.csv"}, "labels": {}},
    }))
    result = survival_main(
        data_file=data_file.as_posix(),
        plots_folder=(tmp_path / "plots").as_posix(),
        config_file=config_file.as_posix(),
        target="event",
        duration_col="duration",
        survival_model="weibull",
    )
    assert "aft_table" in result


def test_load_optuna_frame_with_query_filters_results(tmp_path):
    """Lines 246/248: _load_optuna_frame with a query param."""
    from deckard.layers.survival import _load_optuna_frame
    db_url = f"sqlite:///{(tmp_path / 'q.sqlite3').as_posix()}"
    schema = {"optimization": 0, "dataset": 1, "model type": 2, "defense_name": 3, "attack name": 4, "sep": "_"}
    study = optuna.create_study(study_name="q_toy_rf_baseline_hsj", storage=db_url, load_if_exists=True)

    def _obj(trial):
        trial.set_user_attr("accuracy", 0.9)
        return 0.5

    study.optimize(_obj, n_trials=2)

    # Query that keeps all results
    frame = _load_optuna_frame(db_url, schema=schema, query="accuracy > 0.5")
    assert not frame.empty

    # Query that removes all results → should raise
    with pytest.raises(ValueError, match="No attack results found"):
        _load_optuna_frame(db_url, schema=schema, query="accuracy > 99")
