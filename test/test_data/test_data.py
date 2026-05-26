import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from deckard.data import DataConfig

try:
    import lifelines  # noqa: F401

    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


class TestDataConfigListMerge:
    """DataConfig should accept a list of step-dicts and merge them."""

    steps_a = {
        "imputer": {"name": "sklearn.impute.SimpleImputer", "strategy": "mean"},
    }
    steps_b = {"scaler": {"name": "sklearn.preprocessing.StandardScaler"}}
    steps_override = {
        "imputer": {
            "name": "sklearn.impute.SimpleImputer",
            "strategy": "median",
        },
    }

    def test_list_of_two_dicts_merges_steps(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            pipeline=[self.steps_a, self.steps_b],
        )
        assert isinstance(cfg.pipeline, dict)
        assert "imputer" in cfg.pipeline
        assert "scaler" in cfg.pipeline

    def test_list_later_entry_wins_on_key_conflict(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            pipeline=[self.steps_a, self.steps_override],
        )
        assert cfg.pipeline["imputer"]["strategy"] == "median"

    def test_single_dict_still_works(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            pipeline={"imputer": {"name": "sklearn.impute.SimpleImputer"}},
        )
        assert "imputer" in cfg.pipeline


class TestDataConfig:
    def setup_method(self):
        self.pipeline_config_dict = {
            "imputer": {
                "name": "sklearn.impute.SimpleImputer",
                "strategy": "mean",
            },
            "scaler": {"name": "sklearn.preprocessing.StandardScaler"},
        }
        self.X_train = pd.DataFrame(
            {
                "feature1": [
                    1.0,
                    2.0,
                    np.nan,
                    4.0,
                    5.0,
                    1.0,
                    2.0,
                    np.nan,
                    4.0,
                    5.0,
                ],
                "feature2": [
                    np.nan,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    np.nan,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
            },
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        self.X_test = pd.DataFrame(
            {
                "feature1": [5.0, 6.0],
                "feature2": [4.0, np.nan],
            },
        )
        self.y_test = pd.Series([1, 0])
        self.pipeline_selector_dict = {
            "imputer": {
                "name": "sklearn.impute.SimpleImputer",
                "strategy": "mean",
                "dtype": "num",
            },
            "scaler": {
                "name": "sklearn.preprocessing.StandardScaler",
                "dtype": "num",
            },
        }

    def test_pipelineconfig_initialization(self):
        config = DataConfig(pipeline=self.pipeline_config_dict)
        assert isinstance(config.pipeline, dict)
        assert "imputer" in config.pipeline
        assert "scaler" in config.pipeline

    def test_pipeline_initialization(self):
        config = DataConfig(pipeline=self.pipeline_config_dict)
        (
            pipeline,
            _,
        ) = config._init_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "imputer"
        assert pipeline.steps[1][0] == "scaler"

    def test_pipeline_fit_and_transform(self):
        config = DataConfig(pipeline=self.pipeline_config_dict)
        config._y = self.y_train
        config.data_load_time = 3
        pipeline, _ = config._init_pipeline()
        config.X_train, config.X_test, _, _ = config._fit_transform_X(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            pipeline,
        )
        assert config.X_train.shape == (10, 2)
        assert config.X_test.shape == (2, 2)
        assert not self.X_train.equals(config.X_train)
        assert not self.X_test.equals(config.X_test)

    def test_pipeline_fit_time(self):
        config = DataConfig(pipeline=self.pipeline_config_dict)
        config()
        assert config.pipeline_fit_time is not None
        assert config.pipeline_fit_time > 0

    def test_pipeline_transform_time(self):
        config = DataConfig(pipeline=self.pipeline_config_dict)
        config()
        assert config.pipeline_transform_time is not None
        assert config.pipeline_transform_time > 0

    def test_pipeline_selector_initialization(self):
        config = DataConfig(pipeline=self.pipeline_selector_dict)
        (
            pipeline,
            _,
        ) = config._init_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.steps[0][1], ColumnTransformer)
        pipeline.fit(self.X_train, self.y_train)
        assert pipeline.steps[0][0] == "preprocess"


class TestDataConfig:
    def test_invalid_score_mode_raises(self):
        with pytest.raises(ValueError):
            DataConfig(
                dataset_name="make_classification",
                data_params={"n_samples": 10, "n_features": 2},
                score_mode="invalid",
                scorer=lambda y_true, y_pred: {"dummy": 1},
            )

    def basic_config(self):
        # Minimal config for DataConfig
        return DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 100,
                "n_features": 5,
                "n_informative": 1,
                "n_redundant": 0,
                "random_state": 42,
                "n_clusters_per_class": 1,
            },
            classifier=True,
        )

    def test_make_classification_data_loading_and_sampling(self):
        cfg = self.basic_config()
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        total = len(X_train) + len(X_test)
        assert total == 100

    def test_private_max_samples_caps_loaded_dataset(self):
        with patch.dict("os.environ", {"DECKARD_TEST_MAX_SAMPLES": "40"}):
            cfg = DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 300,
                    "n_features": 5,
                    "n_informative": 2,
                    "n_redundant": 0,
                    "random_state": 42,
                    "n_clusters_per_class": 1,
                },
                classifier=True,
            )

            cfg()

        assert len(cfg._X) == 40
        assert len(cfg._y) == 40
        assert len(cfg.X_train) + len(cfg.X_test) == 40

    def test_scorer_none_skips_data_scoring(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 30,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
                "random_state": 42,
            },
            scorer=None,
        )
        scores = cfg()
        assert "data_load_time" in scores
        assert "data_sample_time" in scores
        assert "class_counts" not in scores

    def test_presample_stage_does_not_override_score_mode(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 30,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
                "random_state": 42,
            },
            score_mode="test",
            scorer={
                "n_samples": {
                    "score_name": "n_samples",
                    "score_function": lambda y_true, y_pred: len(y_true),
                    "stage": "post-sample",
                },
            },
        )
        scores = cfg()
        assert scores["test"]["n_samples"] == len(cfg.y_test)
        assert scores["test"]["n_samples"] != len(cfg._y)

    def test_score_mode_test_uses_test_split(self):
        captured = {}

        class _CaptureScorer:
            def __call__(self, *args, **kwargs):
                _ = args
                captured.update(kwargs)
                return 1.0

        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 30,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
                "random_state": 42,
            },
            score_mode="test",
            scorer={
                "capture": {
                    "score_function": _CaptureScorer(),
                    "stage": "post-pipeline",
                },
            },
        )
        cfg()
        assert captured.get("mode") == "test"
        assert "stage" not in captured

    def test_make_regression_data_loading_and_sampling(self):
        cfg = DataConfig(
            dataset_name="make_regression",
            data_params={
                "n_samples": 50,
                "n_features": 4,
                "n_informative": 2,
                "random_state": 1,
            },
            classifier=False,
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == 50

    def test_diabetes_data_loading_and_sampling(self):
        cfg = DataConfig(
            dataset_name="diabetes",
            data_params={},
            classifier=False,
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(cfg._X)

    def test_digits_data_loading_and_sampling(self):
        cfg = DataConfig(
            dataset_name="digits",
            data_params={},
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(cfg._X)

    def test_hash_method_is_consistent(self):
        cfg = self.basic_config()
        h1 = hash(cfg)
        h2 = hash(cfg)
        assert h1 == h2

    def test_split_data_loads_when_data_missing(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 6,
                "n_informative": 4,
                "random_state": 7,
                "n_redundant": 0,
                "n_repeated": 0,
            },
        )
        cfg._X = None
        cfg._y = None
        cfg.fit()
        assert cfg.X_train is not None
        assert cfg.y_train is not None
        assert cfg.X_test is not None
        assert cfg.y_test is not None

    def test_load_data_raises_not_implemented_for_unknown_dataset(self):
        cfg = DataConfig(dataset_name="unknown_dataset", data_params={})
        with pytest.raises(NotImplementedError):
            cfg.load_dataset()

    @patch("deckard.data.base._load_optuna_studies_dataframe")
    def test_load_dataset_from_optuna_storage(self, mock_optuna_loader):
        mock_optuna_loader.return_value = pd.DataFrame(
            {
                "value": [0.1, 0.2, 0.3],
                "params_alpha": [1, 2, 3],
                "params_beta": [4, 5, 6],
            },
        )
        cfg = DataConfig(
            dataset_name="optuna",
            target="value",
            data_params={
                "optuna_storage": "sqlite:///optuna.db",
                "study_name": "demo",
            },
        )
        cfg.load_dataset()
        assert isinstance(cfg._X, pd.DataFrame)
        assert isinstance(cfg._y, pd.Series)
        assert list(cfg._X.columns) == ["params_alpha", "params_beta"]
        assert len(cfg._y) == 3

    @patch("deckard.data.base._load_optuna_studies_dataframe")
    def test_load_dataset_from_optuna_storage_forwards_query_options(
        self,
        mock_optuna_loader,
    ):
        mock_optuna_loader.return_value = pd.DataFrame(
            {
                "value": [0.1, 0.2],
                "params_alpha": [1, 2],
            },
        )
        cfg = DataConfig(
            dataset_name="optuna",
            target="value",
            data_params={
                "optuna_storage": "sqlite:///optuna.db",
                "study_names": ["a", "b"],
                "trial_number_range": [0, 10],
                "columns": ["value", "params_alpha"],
                "limit": 5,
            },
        )
        cfg.load_dataset()

        kwargs = mock_optuna_loader.call_args.kwargs
        assert kwargs["study_names"] == ["a", "b"]
        assert kwargs["trial_number_range"] == [0, 10]
        assert kwargs["columns"] == ["value", "params_alpha"]
        assert kwargs["limit"] == 5

    def test_load_data_raises_value_error_for_csv_without_target(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdirname:
            csv_path = Path(tmpdirname) / "test.csv"
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(
                csv_path,
                index=False,
            )
            cfg = DataConfig(dataset_name=str(csv_path), data_params={})
            with pytest.raises(ValueError):
                cfg.load_dataset()

    def test_call_returns_expected_shapes_for_make_classification(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 6,
                "n_informative": 4,
                "random_state": 7,
                "n_redundant": 0,
            },
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert X_train.shape[0] == 48
        assert X_test.shape[0] == 12
        assert X_train.shape[1] == 6
        assert X_test.shape[1] == 6
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == 60

    def test_save_self(self):
        import tempfile

        cfg = self.basic_config()
        cfg()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            score_path = Path(tmpdirname) / "scores.json"
            results = cfg(
                files={
                    "data_file": str(data_path),
                    "score_file": str(score_path),
                },
            )
            assert data_path.exists()
            assert score_path.exists()
            assert "data_load_time" in results
            assert "data_sample_time" in results

    def test_load_self(self):
        import tempfile

        cfg = self.basic_config()
        cfg()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            cfg(files={"data_file": str(data_path)})
            assert cfg._X is not None

    def test_save_score_dict(self):
        cfg = self.basic_config()
        cfg()
        cfg.score_dict = {"mutual_info": 0.95, "chisquare": 0.9}
        with tempfile.TemporaryDirectory() as tmpdirname:
            score_path = Path(tmpdirname) / "scores.json"
            # save scores
            cfg.save_scores(cfg.score_dict, score_path)
            loaded_scores = cfg.load_scores(score_path)
            cfg(files={"score_file": str(score_path)})
            assert score_path.exists()
            assert "mutual_info" in loaded_scores
            assert "chisquare" in loaded_scores
            assert round(abs(loaded_scores["mutual_info"] - 0.95), 7) == 0
            assert round(abs(loaded_scores["chisquare"] - 0.9), 7) == 0

    def test_save_data_file(self):
        import tempfile

        cfg = self.basic_config()
        cfg()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            cfg(files={"data_file": str(data_path)})
            assert data_path.exists()
            # Load the data back and verify
            cfg = cfg.load(filepath=str(data_path))
            assert cfg._X is not None
            assert cfg._y is not None

    @pytest.mark.skipif(
        not HAS_LIFELINES, reason="lifelines is required for this test"
    )
    def test_load_lifelines_lung_dataset(self):
        cfg = DataConfig(
            dataset_name="lung",
            classifier=False,
            target="status",
            sampler={"name": "deckard.data.sample.SplitSampler", "stratify": False},
        )
        cfg.load_dataset()
        assert "time" in cfg.X.columns
        assert len(cfg.X) == len(cfg.y)

    @pytest.mark.skipif(
        not HAS_LIFELINES, reason="lifelines is required for this test"
    )
    def test_load_lifelines_leukemia_dataset(self):
        cfg = DataConfig(
            dataset_name="leukemia",
            classifier=False,
            target="status",
            sampler={"name": "deckard.data.sample.SplitSampler", "stratify": False},
        )
        cfg.load_dataset()
        assert len(cfg.X) > 0
        assert len(cfg.X) == len(cfg.y)

    @pytest.mark.skipif(
        not HAS_LIFELINES, reason="lifelines is required for this test"
    )
    def test_load_lifelines_diabetes_dataset_with_prefix(self):
        cfg = DataConfig(
            dataset_name="lifelines_diabetes",
            classifier=False,
            target="gender",
            sampler={"name": "deckard.data.sample.SplitSampler", "stratify": False},
        )
        cfg.load_dataset()
        assert len(cfg.X) > 0
        assert len(cfg.X) == len(cfg.y)

    def test_hash_stable_after_call_for_data_config(self):
        cfg = self.basic_config()
        original_hash = hash(cfg)
        cfg()
        assert original_hash == hash(cfg), "Hash changed after call for DataConfig"


try:
    import fairlearn  # noqa: F401
except Exception:
    pytest.skip(
        "fairlearn is required for fairness data tests",
        allow_module_level=True,
    )


class HookRecorderPlugin:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def before_load_data(self, data_config):
        self.events.append(f"{self.name}:before_load_data")

    def after_load_data(self, data_config):
        self.events.append(f"{self.name}:after_load_data")

    def before_sample(self, data_config):
        self.events.append(f"{self.name}:before_sample")

    def after_sample(self, data_config):
        self.events.append(f"{self.name}:after_sample")

    def before_score(self, data_config, **kwargs):
        _ = kwargs
        self.events.append(f"{self.name}:before_score")


class ScorePlugin:
    def after_score(self, data_config, scores, **kwargs):
        _ = kwargs
        return {"plugin_metric": float(len(scores))}


class TestFairlearnDataConfig:
    def setup_method(self):
        self.pipeline_config_dict = {
            "imputer": {
                "name": "sklearn.impute.SimpleImputer",
                "strategy": "mean",
            },
            "scaler": {"name": "sklearn.preprocessing.StandardScaler"},
        }
        self.X_train = pd.DataFrame(
            {
                "feature1": [
                    1.0,
                    2.0,
                    np.nan,
                    4.0,
                    5.0,
                    1.0,
                    2.0,
                    np.nan,
                    4.0,
                    5.0,
                ],
                "feature2": [
                    np.nan,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    np.nan,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
            },
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        self.X_test = pd.DataFrame(
            {
                "feature1": [5.0, 6.0],
                "feature2": [4.0, np.nan],
            },
        )
        self.y_test = pd.Series([1, 0])
        self.pipeline_selector_dict = {
            "imputer": {
                "name": "sklearn.impute.SimpleImputer",
                "strategy": "mean",
                "dtype": "num",
            },
            "scaler": {
                "name": "sklearn.preprocessing.StandardScaler",
                "dtype": "num",
            },
        }

    def test_pipelineconfig_initialization(self):
        config = DataConfig(pipeline=self.pipeline_config_dict)
        assert isinstance(config.pipeline, dict)
        assert "imputer" in config.pipeline
        assert "scaler" in config.pipeline

    def test_pipeline_initialization(self):
        config = DataConfig(pipeline=self.pipeline_config_dict)
        pipeline, _ = config._init_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "imputer"
        assert pipeline.steps[1][0] == "scaler"

    def test_pipeline_fit_and_transform(self):
        config = DataConfig(
            pipeline=self.pipeline_config_dict,
            score_mode="train",
        )
        config._X = self.X_train
        config._y = self.y_train
        config.data_load_time = 3
        config()
        assert config.X_train.shape == (8, 2)
        assert config.X_test.shape == (2, 2)
        assert not self.X_train.equals(config.X_train)
        assert not self.X_test.equals(config.X_test)

    def test_pipeline_fit_time(self):
        config = DataConfig(pipeline=self.pipeline_config_dict)
        config()
        assert config.pipeline_fit_time is not None
        assert config.pipeline_fit_time > 0

    def test_pipeline_transform_time(self):
        config = DataConfig(pipeline=self.pipeline_config_dict)
        config()
        assert config.pipeline_transform_time is not None
        assert config.pipeline_transform_time > 0

    def test_pipeline_selector_initialization(self):
        config = DataConfig(pipeline=self.pipeline_selector_dict)
        (
            pipeline,
            _,
        ) = config._init_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.steps[0][1], ColumnTransformer)
        pipeline.fit(self.X_train, self.y_train)
        assert pipeline.steps[0][0] == "preprocess"


class TestDataConfigAdditional:

    def basic_config(self):
        # Minimal config for DataConfig
        return DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 100,
                "n_features": 5,
                "n_informative": 1,
                "n_redundant": 0,
                "random_state": 42,
                "n_clusters_per_class": 1,
            },
            classifier=True,
        )

    def test_make_classification_data_loading_and_sampling(self):
        cfg = self.basic_config()
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        total = len(X_train) + len(X_test)
        assert total == 100

    def test_make_regression_data_loading_and_sampling(self):
        cfg = DataConfig(
            dataset_name="make_regression",
            data_params={
                "n_samples": 50,
                "n_features": 4,
                "n_informative": 2,
                "random_state": 1,
            },
            classifier=False,
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == 50

    def test_diabetes_data_loading_and_sampling(self):
        cfg = DataConfig(
            dataset_name="diabetes",
            data_params={},
            classifier=False,
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(cfg._X)

    def test_digits_data_loading_and_sampling(self):
        cfg = DataConfig(
            dataset_name="digits",
            data_params={},
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(cfg._X)

    def test_hash_method_is_consistent(self):
        cfg = self.basic_config()
        h1 = hash(cfg)
        h2 = hash(cfg)
        assert h1 == h2

    def test_split_data_loads_when_data_missing(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 6,
                "n_informative": 4,
                "random_state": 7,
                "n_redundant": 0,
                "n_repeated": 0,
            },
        )
        cfg._X = None
        cfg._y = None
        cfg.fit()
        assert cfg.X_train is not None
        assert cfg.y_train is not None
        assert cfg.X_test is not None
        assert cfg.y_test is not None

    def test_load_data_raises_not_implemented_for_unknown_dataset(self):
        cfg = DataConfig(dataset_name="unknown_dataset", data_params={})
        with pytest.raises(NotImplementedError):
            cfg.load_dataset()

    def test_load_data_raises_value_error_for_csv_without_target(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdirname:
            csv_path = Path(tmpdirname) / "test.csv"
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(
                csv_path,
                index=False,
            )
            cfg = DataConfig(dataset_name=str(csv_path), data_params={})
            with pytest.raises(ValueError):
                cfg.load_dataset()

    def test_call_returns_expected_shapes_for_make_classification(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 6,
                "n_informative": 4,
                "random_state": 7,
                "n_redundant": 0,
            },
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        assert X_train.shape[0] == 48
        assert X_test.shape[0] == 12
        assert X_train.shape[1] == 6
        assert X_test.shape[1] == 6
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == 60

    def test_save_self(self):
        import tempfile

        cfg = self.basic_config()
        cfg()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            score_path = Path(tmpdirname) / "scores.json"
            results = cfg(
                files={
                    "data_file": str(data_path),
                    "score_file": str(score_path),
                },
            )
            assert data_path.exists()
            assert score_path.exists()
            assert "data_load_time" in results
            assert "data_sample_time" in results

    def test_load_self(self):
        import tempfile

        cfg = self.basic_config()
        cfg()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            cfg(files={"data_file": str(data_path)})
            assert cfg._X is not None

    def test_save_score_dict(self):
        cfg = self.basic_config()
        cfg()
        cfg.score_dict = {"mutual_info": 0.95, "chisquare": 0.9}
        with tempfile.TemporaryDirectory() as tmpdirname:
            score_path = Path(tmpdirname) / "scores.json"
            # save scores
            cfg.save_scores(cfg.score_dict, score_path)
            loaded_scores = cfg.load_scores(score_path)
            cfg(files={"score_file": str(score_path)})
            assert score_path.exists()
            assert "mutual_info" in loaded_scores
            assert "chisquare" in loaded_scores
            assert round(abs(loaded_scores["mutual_info"] - 0.95), 7) == 0
            assert round(abs(loaded_scores["chisquare"] - 0.9), 7) == 0

    def test_save_data_file(self):
        import tempfile

        cfg = self.basic_config()
        cfg()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            cfg(files={"data_file": str(data_path)})
            assert data_path.exists()
            # Load the data back and verify
            cfg = cfg.load(filepath=str(data_path))
            assert cfg._X is not None
            assert cfg._y is not None

    def test_multiple_plugins_run_in_order(self):
        events = []
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
                "random_state": 1,
            },
            plugins=[
                HookRecorderPlugin(events=events, name="p1"),
                HookRecorderPlugin(events=events, name="p2"),
            ],
        )

        cfg()

        expected = [
            "p1:before_load_data",
            "p2:before_load_data",
            "p1:after_load_data",
            "p2:after_load_data",
            "p1:before_sample",
            "p2:before_sample",
            "p1:after_sample",
            "p2:after_sample",
            "p1:before_score",
            "p2:before_score",
        ]
        assert events == expected

    def test_plugin_can_augment_scores(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
                "random_state": 1,
            },
            plugins=[ScorePlugin()],
        )

        scores = cfg()
        assert "plugin_metric" in scores
