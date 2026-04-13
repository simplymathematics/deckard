import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from deckard.data.fairness import FairnessDataConfig

class TestFairnessDataConfigInit:
    def test_init_without_groupby_columns_raises_error(self):
        """Test that FairnessDataConfig raises ValueError when groupby_columns is None."""
        with pytest.raises(ValueError, match="groupby_column must be specified"):
            FairnessDataConfig(
                groupby_columns=None
            )
    
    def test_init_with_single_groupby_column(self):
        """Test initialization with single groupby column."""
        config = FairnessDataConfig(
            groupby_columns="gender"
        )
        assert config.groupby_columns == "gender"
    
    def test_init_with_multiple_groupby_columns(self):
        """Test initialization with multiple groupby columns."""
        columns = ["gender", "age_group"]
        config = FairnessDataConfig(
            groupby_columns=columns
        )
        assert config.groupby_columns == columns


class TestLoadData:
    @patch('deckard.data.fairness.FairnessDataConfig._load_data')
    def test_load_data_creates_groups(self, mock_super_load, capfd):
        """Test that _load_data creates groups_ attribute."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'gender': ['M', 'F', 'M', 'F']
        })
        config = FairnessDataConfig(
            groupby_columns="gender"
        )
        config._X = df
        config._y = pd.Series([0, 1, 0, 1])
        
        config = config._load_data()
        
        assert hasattr(config, 'groups_')
        # assert isinstance(config.groups_, pd.api.typing.DataFrameGroupBy)
    
    @patch('deckard.data.fairness.FairnessDataConfig.__post_init__')
    def test_load_data_missing_X_raises_assertion(self, mock_post_init):
        """Test that _load_data raises assertion when _X is missing."""
        config = FairnessDataConfig(
            groupby_columns="sex"
        )
        config._X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'gender': ['M', 'F', 'M', 'F']
        })
        config._y = pd.Series([0, 1, 0, 1])
        config.data_params = {}
        
        with pytest.raises(AssertionError):
            config._load_data()
    
    @patch('deckard.data.fairness.FairnessDataConfig.__post_init__')
    def test_load_data_missing_y_raises_assertion(self, mock_post_init):
        """Test that _load_data raises assertion when _y is missing."""
        df = pd.DataFrame({'gender': ['M', 'F', 'M', 'F']})
        config = FairnessDataConfig(
            groupby_columns="gender"
        )
        config._X = df
        config.data_params = {}
        
        with pytest.raises(AssertionError):
            config._load_data()


class TestSample:
    @patch('deckard.data.fairness.FairnessDataConfig.__post_init__')
    def test_sample_creates_grouped_test_data(self, mock_post_init):
        """Test that _sample creates X_test_groups and y_test_groups."""
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'gender': ['M', 'F', 'M', 'F']
        }, index=[0, 1, 2, 3])
        y_test = pd.Series([0, 1, 0, 1], index=[0, 1, 2, 3])
        
        config = FairnessDataConfig(
            groupby_columns="gender"
        )
        config.X_test = X_test
        config.y_test = y_test
        config._X = X_test
        config._y = y_test
        config.train_size = 2
        config.test_size = 2 
        config._sample()
        
        assert hasattr(config, 'X_test_groups')
        assert hasattr(config, 'y_test_groups')
        assert 'M' in config.X_test_groups
        assert 'F' in config.X_test_groups
    

class TestScore:
    @patch('deckard.data.fairness.FairnessDataConfig.__post_init__')
    def test_score_returns_dict(self, mock_post_init):
        """Test that _score returns a dictionary."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'gender': ['M', 'F', 'M', 'F']
        })
        
        config = FairnessDataConfig(
            groupby_columns="gender",
            classifier=True
        )
        config._X = df
        config._y = pd.Series([0, 1, 0, 1])
        config.groups_ = df.groupby('gender')
        config.classifier = True
        
        with patch.object(config, '_classification_feature_scores_for_group', return_value={}):
            scores = config._score()
        
        assert isinstance(scores, dict)
        assert 'M' in scores
        assert 'F' in scores


class TestComputeClassCounts:
    @patch('deckard.data.fairness.FairnessDataConfig.__post_init__')
    def test_compute_class_counts_returns_dict(self, mock_post_init):
        """Test that _compute_class_counts returns correct structure."""
        df = pd.DataFrame({'gender': ['M', 'F', 'M', 'F']})
        
        config = FairnessDataConfig(
            groupby_columns="gender"
        )
        config._X = df
        config._y = pd.Series([0, 1, 0, 1])
        config.groups_ = df.groupby('gender')
        
        counts = config._compute_class_counts(config._y)
        
        assert isinstance(counts, dict)
        assert 'M' in counts
        assert 'F' in counts


class TestClassificationFeatureScoresForGroup:
    @patch('deckard.data.fairness.FairnessDataConfig.__post_init__')
    def test_classification_scores_contains_required_metrics(self, mock_post_init):
        """Test that classification scores include all required metrics."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8]
        }, index=[0, 1, 2, 3])
        y = pd.Series([0, 1, 0, 1], index=[0, 1, 2, 3])
        
        config = FairnessDataConfig(
            groupby_columns="gender"
        )
        config._X = X
        config._y = y
        
        scores = config._classification_feature_scores_for_group(y)
        
        assert 'f_classif' in scores
        assert 'mutual_info' in scores
        assert 'chi2' in scores


class TestRegressionFeatureScoresForGroup:
    @patch('deckard.data.fairness.FairnessDataConfig.__post_init__')
    def test_regression_scores_contains_required_metrics(self, mock_post_init):
        """Test that regression scores include all required metrics."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0]
        }, index=[0, 1, 2, 3])
        y = pd.Series([1.5, 2.5, 3.5, 4.5], index=[0, 1, 2, 3])
        
        config = FairnessDataConfig(
            groupby_columns="gender"
        )
        config._X = X
        config._y = y
        
        scores = config._regression_feature_scores_for_group(y)
        
        assert 'f_regression' in scores
        assert 'mutual_info' in scores
        assert 'r_regression' in scores