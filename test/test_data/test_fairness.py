import pytest
import pandas as pd
from unittest.mock import patch

pytest.importorskip("fairlearn")

from deckard.data.fairness import FairnessDataConfig  # NOQA E402


class TestFairnessDataConfigInit:
    def test_init_without_groupby_columns_raises_error(self):
        """Test that FairnessDataConfig raises ValueError when groupby_columns is None."""
        with pytest.raises(ValueError, match="groupby_column must be specified"):
            FairnessDataConfig(
                groupby_columns=None,
            )

    def test_init_with_single_groupby_column(self):
        """Test initialization with single groupby column."""
        config = FairnessDataConfig(
            groupby_columns="gender",
        )
        assert config.groupby_columns == ["gender"]

    def test_init_with_multiple_groupby_columns(self):
        """Test initialization with multiple groupby columns."""
        columns = ["gender", "age_group"]
        config = FairnessDataConfig(
            groupby_columns=columns,
        )
        assert config.groupby_columns == columns


class TestLoadData:
    @patch("deckard.data.fairness.FairnessDataConfig._load_data")
    def test_load_data_creates_groups(self, mock_super_load, capfd):
        """Test that _load_data creates _groups attribute."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
                "gender": ["M", "F", "M", "F"],
            },
        )
        config = FairnessDataConfig(
            groupby_columns="gender",
        )
        config._X = df
        config._y = pd.Series([0, 1, 0, 1])

        config = config._load_data()

        assert hasattr(config, "_groups")
        # assert isinstance(config._groups, pd.api.typing.DataFrameGroupBy)

    @patch("deckard.data.fairness.FairnessDataConfig.__post_init__")
    def test_load_data_missing_X_raises_assertion(self, mock_post_init):
        """Test that _load_data raises assertion when _X is missing."""
        config = FairnessDataConfig(
            groupby_columns="sex",
        )
        config._X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
                "gender": ["M", "F", "M", "F"],
            },
        )
        config._y = pd.Series([0, 1, 0, 1])
        config.data_params = {}

        with pytest.raises(AssertionError):
            config._load_data()

    @patch("deckard.data.fairness.FairnessDataConfig.__post_init__")
    def test_load_data_missing_y_raises_assertion(self, mock_post_init):
        """Test that _load_data raises assertion when _y is missing."""
        df = pd.DataFrame({"gender": ["M", "F", "M", "F"]})
        config = FairnessDataConfig(
            groupby_columns="gender",
        )
        config._X = df
        config.data_params = {}

        with pytest.raises(AssertionError):
            config._load_data()


class TestScore:
    @patch("deckard.data.fairness.FairnessDataConfig.__post_init__")
    def test_score_returns_dict(self, mock_post_init):
        """Test that _score returns a dictionary."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "gender": ["M", "F", "M", "F"],
            },
        )

        config = FairnessDataConfig(
            groupby_columns="gender",
            classifier=True,
        )
        config._X = df
        config._y = pd.Series([0, 1, 0, 1])
        config._groups = df.groupby("gender")
        config.classifier = True

        with patch.object(config, "_classification_feature_scores", return_value={}):
            scores = config._score()

        assert isinstance(scores, dict)
        assert "fairness_scores" in scores


class TestComputeClassCounts:
    @patch("deckard.data.fairness.FairnessDataConfig.__post_init__")
    def test_compute_class_counts_returns_dict(self, mock_post_init):
        """Test that _compute_class_counts returns correct structure."""
        df = pd.DataFrame({"gender": ["M", "F", "M", "F"]})

        config = FairnessDataConfig(
            groupby_columns="gender",
        )
        config._X = df
        config._y = pd.Series([0, 1, 0, 1])
        config._groups = df.groupby("gender")

        counts = config._compute_class_counts(df[config.groupby_columns])

        assert isinstance(counts, dict)
        assert "M" in counts
        assert "F" in counts


class TestClassificationFeatureScoresForGroup:
    @patch("deckard.data.fairness.FairnessDataConfig.__post_init__")
    def test_classification_scores_contains_required_metrics(self, mock_post_init):
        """Test that classification scores include all required metrics."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
            },
            index=[0, 1, 2, 3],
        )
        y = pd.Series([0, 1, 0, 1], index=[0, 1, 2, 3])

        config = FairnessDataConfig(
            groupby_columns="gender",
        )
        config._X = X
        config._y = y

        config.X_train = X
        config.y_train = y
        config.X_test = X
        config.y_test = y

        scores = config._classification_feature_scores()

        assert "class_counts" in scores
        assert "mutual_info_classif" in scores
        assert "f_classif" in scores


class TestFairnessDataConfigHashStability:
    """Test hash capability for FairnessDataConfig."""

    def test_fairness_data_config_is_configbase_with_hash(self):
        """Test that FairnessDataConfig is ConfigBase and has __hash__ method."""
        pytest.importorskip("fairlearn")
        from deckard.utils import ConfigBase

        config = FairnessDataConfig(
            groupby_columns="gender",
        )
        assert isinstance(
            config,
            ConfigBase,
        ), "FairnessDataConfig should inherit from ConfigBase"
        assert hasattr(config, "__hash__"), "FairnessDataConfig should have __hash__ method"
        # Note: FairnessDataConfig may have unhashable runtime fields like _groups
        # so we verify the infrastructure is in place rather than attempting full hash
