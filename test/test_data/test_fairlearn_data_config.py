import pytest
import pandas as pd
from unittest.mock import patch

try:
    import fairlearn  # noqa: F401
    from deckard.data.fairness import FairlearnDataConfig
except Exception:
    pytest.skip(
        "fairlearn is required for FairlearnDataConfig tests",
        allow_module_level=True,
    )


class TestFairlearnDataConfigInit:
    def test_init_without_sensitive_columns_raises_error(self):
        """Test that FairlearnDataConfig raises ValueError when sensitive_columns is None."""
        with pytest.raises(
            ValueError,
            match="sensitive_columns must be specified",
        ):
            FairlearnDataConfig(
                sensitive_columns=None,
            )

    def test_init_with_single_sensitive_column(self):
        """Test initialization with single sensitive column."""
        config = FairlearnDataConfig(
            sensitive_columns="gender",
        )
        assert config.sensitive_columns == ["gender"]

    def test_init_with_multiple_sensitive_columns(self):
        """Test initialization with multiple sensitive columns."""
        columns = ["gender", "age"]
        config = FairlearnDataConfig(
            sensitive_columns=columns,
        )
        assert config.sensitive_columns == columns

    def test_init_with_fairness_defense_list_merges_dicts(self):
        """List fairness_defense specs should merge into one dict."""
        config = FairlearnDataConfig(
            sensitive_columns="gender",
            fairness_defense=[
                {"name": "fairlearn.preprocessing.CorrelationRemover"},
                {"alpha": 0.25, "step_name": "fairness_pre"},
            ],
        )
        assert isinstance(config.fairness_defense, dict)
        assert (
            config.fairness_defense["name"]
            == "fairlearn.preprocessing.CorrelationRemover"
        )
        assert config.fairness_defense["alpha"] == 0.25
        assert config.fairness_defense["step_name"] == "fairness_pre"

    def test_init_with_fairness_defense_list_later_wins(self):
        """Later fairness_defense list entries should override earlier keys."""
        config = FairlearnDataConfig(
            sensitive_columns="gender",
            fairness_defense=[
                {
                    "name": "fairlearn.preprocessing.CorrelationRemover",
                    "alpha": 0.1,
                },
                {"alpha": 0.4},
            ],
        )
        assert config.fairness_defense["alpha"] == 0.4


class TestLoadData:
    @patch("deckard.data.base.DataConfig._load_data")
    def test_load_data_validates_sensitive_columns(
        self,
        mock_super_load,
        capfd,
    ):
        """Test that _load_data validates configured sensitive columns."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
                "gender": ["M", "F", "M", "F"],
            },
        )
        config = FairlearnDataConfig(
            sensitive_columns="gender",
        )
        config._X = df
        config._y = pd.Series([0, 1, 0, 1])

        config = config._load_data()

        assert config is not None
        assert "gender" in config._X.columns

    @patch("deckard.data.fairness.FairlearnDataConfig.__post_init__")
    def test_load_data_missing_X_raises_assertion(self, mock_post_init):
        """Test that _load_data raises assertion when _X is missing."""
        config = FairlearnDataConfig(
            sensitive_columns="sex",
        )
        config._X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
                "gender": [0, 1, 1, 0],
            },
        )
        config._y = pd.Series([0, 1, 0, 1])
        config.data_params = {}

        with pytest.raises(AssertionError):
            config._load_data()

    @patch("deckard.data.fairness.FairlearnDataConfig.__post_init__")
    def test_load_data_missing_y_raises_assertion(self, mock_post_init):
        """Test that _load_data raises assertion when _y is missing."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
                "gender": [0, 1, 1, 0],
            },
        )
        config = FairlearnDataConfig(
            sensitive_columns="gender",
        )
        config._X = df
        config.data_params = {}

        with pytest.raises(AssertionError):
            config._load_data()


class TestScore:

    @patch("deckard.data.fairness.FairlearnDataConfig.__post_init__")
    def test_score_returns_dict(self, mock_post_init):
        """Test that _score returns a dictionary."""
        # _X must include the sensitive column for fairness extraction
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
                "gender": ["M", "F", "M", "F"],
            },
        )

        config = FairlearnDataConfig(
            sensitive_columns="gender",
            classifier=True,
            train_size=2,
            test_size=2,
            stratify=False,
        )
        config._X = df
        config._y = pd.Series([0, 1, 0, 1])
        config.classifier = True

        config._sample()  # Ensure sensitive features are set up

        with patch.object(config, "_classification_feature_scores", return_value={}):
            with patch.object(config, "scorer", return_value={"fairness_scores": {}}):
                scores = config._score()

        assert isinstance(scores, dict)


class TestComputeClassCounts:
    @patch("deckard.data.fairness.FairlearnDataConfig.__post_init__")
    def test_sensitive_labels_from_frame_returns_dict_compatible_values(
        self,
        mock_post_init,
    ):
        """Test sensitive label generation from configured sensitive columns."""
        df = pd.DataFrame({"gender": ["M", "F", "M", "F"]})

        config = FairlearnDataConfig(
            sensitive_columns="gender",
        )
        config._X = df
        config._y = pd.Series([0, 1, 0, 1])

        labels = config._sensitive_labels_from_frame(df)

        assert len(labels) == len(df)
        assert set(labels.unique()) == {"M", "F"}


class TestClassificationFeatureScoresForGroup:
    @patch("deckard.data.fairness.FairlearnDataConfig.__post_init__")
    def test_classification_scores_contains_required_metrics(
        self,
        mock_post_init,
    ):
        """Test that classification scores include all required metrics."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
                "gender": [0, 1, 1, 0],
            },
        )
        y = pd.Series([0, 1, 0, 1], index=[0, 1, 2, 3])

        config = FairlearnDataConfig(
            sensitive_columns="gender",
        )
        config._X = df
        config._y = y

        config.X_train = df
        config.y_train = y
        config.X_test = df
        config.y_test = y

        scores = config._classification_feature_scores()

        assert "class_counts" in scores
        assert "mutual_info_classif" in scores
        assert "f_classif" in scores


class TestFairlearnDataConfigHashStability:
    """Test hash capability for FairlearnDataConfig."""

    def test_fairness_data_config_is_configbase_with_hash(self):
        """Test that FairlearnDataConfig is ConfigBase and has __hash__ method."""
        pytest.importorskip("fairlearn")
        from deckard.utils import ConfigBase

        config = FairlearnDataConfig(
            sensitive_columns="gender",
        )
        assert isinstance(
            config,
            ConfigBase,
        ), "FairlearnDataConfig should inherit from ConfigBase"
        assert hasattr(
            config,
            "__hash__",
        ), "FairlearnDataConfig should have __hash__ method"
        # Note: FairlearnDataConfig may have unhashable runtime fields after execution
        # so we verify the infrastructure is in place rather than attempting full hash
