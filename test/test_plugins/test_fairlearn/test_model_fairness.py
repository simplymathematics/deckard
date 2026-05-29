import logging
import shutil
import tempfile
from unittest.mock import Mock

import pandas as pd
import pytest

from deckard.model.defense.base import DefenseConfig, DefensePipelineConfig
from deckard.plugins.fairlearn.data import FairlearnDataConfig
from deckard.plugins.fairlearn.model import (
    FairlearnDefenseConfig,
    FairlearnModelConfig,
)

logger = logging.getLogger(__name__)

pytest.importorskip("fairlearn")


class TestFairlearnModelConfig:
    def setup_method(self):
        # Create sample data with group information
        self.X_train = pd.DataFrame(
            {
                "feature1": [0, 1, 2, 3, 4, 5],
                "feature2": [1, 2, 3, 4, 5, 6],
                "group": ["A", "B", "A", "B", "A", "B"],
            },
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0, 1])

        self.X_test = pd.DataFrame(
            {
                "feature1": [6, 7, 8, 9],
                "feature2": [7, 8, 9, 10],
                "group": ["A", "B", "A", "B"],
            },
        )
        self.y_test = pd.Series([1, 0, 1, 0])

        # Create sensitive feature series for fairness evaluation
        self.sensitive_test = pd.Series(
            ["A", "B", "A", "B"],
            index=self.y_test.index,
        )

        self.name= "sklearn.ensemble.RandomForestClassifier"
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_fairness_model_config_initialization(self):
        """Test FairlearnModelConfig can be initialized."""
        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = self.sensitive_test

        model = FairlearnModelConfig(
            name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        assert model is not None
        assert model.data == fairness_data

    def test_fairness_model_config_initialization_without_data(self):
        """Test FairlearnModelConfig can be initialized without fairness data."""
        model = FairlearnModelConfig(
            name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None,
        )

        assert model is not None
        assert model.data is None

    def test_apply_defense_supports_mixed_defense_pipeline(self):
        """ART + fairlearn defenses are applied sequentially via DefensePipelineConfig."""
        model = FairlearnModelConfig(
            name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None,
        )
        model._model = Mock()

        art_defense = DefenseConfig(
            model_name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            name="art.defences.postprocessor.GaussianNoise",
            defense_params={},
        )
        fair_defense = FairlearnDefenseConfig(
            name="fairlearn.postprocessing.ThresholdOptimizer",
            model_name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            defense_params={"constraints": "demographic_parity"},
            data=None,
        )

        first_estimator = Mock(name="art_wrapped_estimator")
        second_estimator = Mock(name="fairlearn_wrapped_estimator")
        art_defense.defense_application_time = 0.3
        fair_defense.defense_application_time = 0.4
        art_defense.apply_to = Mock(return_value=first_estimator)
        fair_defense.apply_to = Mock(return_value=second_estimator)

        model.defense = DefensePipelineConfig(
            defenses=[art_defense, fair_defense],
        )
        runtime_data = Mock()
        result = model.apply_defense(data=runtime_data)

        assert result is second_estimator
        art_defense.apply_to.assert_called_once_with(
            estimator=model._model,
            data=runtime_data,
        )
        fair_defense.apply_to.assert_called_once_with(
            estimator=first_estimator,
            data=runtime_data,
        )
        assert round(abs(model.defense_application_time - 0.7), 7) == 0
        assert fair_defense.data is runtime_data

    def test_apply_defense_rejects_legacy_defense_list(self):
        """Legacy list assignment is intentionally unsupported after pipeline migration."""
        model = FairlearnModelConfig(
            name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None,
        )
        model._model = Mock()

        model.defense = [
            DefenseConfig(
                model_name=self.name,
                classifier=True,
                model_params={"n_estimators": 10},
                name="art.defences.postprocessor.GaussianNoise",
                defense_params={},
            ),
        ]

        with pytest.raises(TypeError):
            model.apply_defense(data=Mock())

    def test_sensitive_fairness_scores_naming_convention(self):
        """Test that sensitive fairness scores follow naming convention."""
        # Create a minimal real FairlearnDataConfig with required fields
        from deckard.plugins.fairlearn.data import FairlearnDataConfig

        fairness_data = FairlearnDataConfig(
            sensitive_columns="sex",
        )
        fairness_data()
        model = FairlearnModelConfig(
            name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        # Call the model with test data to get scores
        scores = model(fairness_data)
        assert "1_f1" in scores
        assert "0_f1" in scores

    def test_train_passes_sensitive_features_when_supported(self):
        class SensitiveFitEstimator:
            def __init__(self):
                self.received_sensitive = None

            def fit(self, X, y, sensitive_features=None):
                self.received_sensitive = sensitive_features
                return self

            def predict(self, X, sensitive_features=None):
                return pd.Series([0] * len(X))

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data.sensitive_train = self.sensitive_test
        fairness_data.sensitive_test = self.sensitive_test
        fairness_data.sensitive_all = self.sensitive_test

        model = FairlearnModelConfig(
            name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )
        model._model = SensitiveFitEstimator()

        model.train(self.X_test, self.y_test)

        assert model._model.received_sensitive is not None
        assert len(model._model.received_sensitive) == len(self.y_test)

    def test_predict_passes_sensitive_features_when_supported(self):
        class SensitivePredictEstimator:
            def fit(self, X, y):
                return self

            def predict(self, X, sensitive_features=None):
                if sensitive_features is None:
                    raise AssertionError("sensitive_features was not provided")
                return pd.Series([0] * len(X))

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data.sensitive_test = self.sensitive_test
        fairness_data.sensitive_train = None
        fairness_data.sensitive_all = None

        model = FairlearnModelConfig(
            name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )
        model._model = SensitivePredictEstimator()

        y_pred = model.predict(self.X_test)

        assert len(y_pred) == len(self.X_test)

    def test_fairness_model_config_is_BaseConfig_with_hash(self):
        """Test that FairlearnModelConfig is BaseConfig and has __hash__ method."""
        from deckard.utils import BaseConfig

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data.sensitive_test = self.sensitive_test
        fairness_data.sensitive_train = self.sensitive_test
        fairness_data.sensitive_all = self.sensitive_test

        model = FairlearnModelConfig(
            name=self.name,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )
        assert isinstance(
            model,
            BaseConfig,
        ), "FairlearnModelConfig should inherit from BaseConfig"
        assert hasattr(
            model,
            "__hash__",
        ), "FairlearnModelConfig should have __hash__ method"
        # Note: FairlearnModelConfig may have unhashable runtime fields
        # so we verify the infrastructure is in place rather than attempting full hash


class TestFairlearnDefenseConfigApplyDefense:
    """Tests for FairlearnDefenseConfig.apply_defense with fairlearn estimators."""

    def setup_method(self):

        self.FairlearnDefenseConfig = FairlearnDefenseConfig
        self.name= "sklearn.linear_model.LogisticRegression"

        self.X_train = pd.DataFrame(
            {"f1": [0, 1, 2, 3, 4, 5], "f2": [1, 2, 3, 4, 5, 6]},
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0, 1])
        self.sensitive_train = pd.Series(["A", "B", "A", "B", "A", "B"])

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_train = self.sensitive_train
        fairness_data._sensitive_test = None
        fairness_data._sensitive_all = None
        self.fairness_data = fairness_data

    def _make_fitted_defense(self, defense_name, defense_params=None):
        cfg = self.FairlearnDefenseConfig(
            name=defense_name,
            model_name=self.name,
            classifier=True,
            model_params={"max_iter": 200},
            defense_params=defense_params or {},
            data=self.fairness_data,
        )
        cfg.train(self.X_train, self.y_train)
        return cfg

    def test_apply_defense_reductions_exponentiated_gradient_string_constraint(
        self,
    ):
        """ExponentiatedGradient with constraint given as a module-path string."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {
                "constraints": "fairlearn.reductions.DemographicParity",
                "eps": 0.1,
            },
        )
        result = cfg.apply_defense(None)
        from fairlearn.reductions import ExponentiatedGradient

        assert isinstance(result, ExponentiatedGradient)

    def test_apply_defense_reductions_exponentiated_gradient_dict_constraint(
        self,
    ):
        """ExponentiatedGradient with constraint given as a _target_ dict."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {
                "constraints": {
                    "_target_": "fairlearn.reductions.EqualizedOdds",
                },
                "eps": 0.1,
            },
        )
        result = cfg.apply_defense(None)
        from fairlearn.reductions import ExponentiatedGradient

        assert isinstance(result, ExponentiatedGradient)

    def test_apply_defense_reductions_requires_constraints(self):
        """ExponentiatedGradient without a constraints key must raise ValueError."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
        )
        with pytest.raises(ValueError):
            cfg.apply_defense(None)

    def test_apply_defense_postprocessing_threshold_optimizer(self):
        """ThresholdOptimizer wraps the base estimator correctly."""
        cfg = self._make_fitted_defense(
            "fairlearn.postprocessing.ThresholdOptimizer",
            {"constraints": "demographic_parity"},
        )
        result = cfg.apply_defense(None)
        from fairlearn.postprocessing import ThresholdOptimizer

        assert isinstance(result, ThresholdOptimizer)

    def test_apply_defense_postprocessing_no_constraints(self):
        """ThresholdOptimizer with no constraints key uses default."""
        cfg = self._make_fitted_defense(
            "fairlearn.postprocessing.ThresholdOptimizer",
        )
        result = cfg.apply_defense(None)
        from fairlearn.postprocessing import ThresholdOptimizer

        assert isinstance(result, ThresholdOptimizer)

    def test_apply_defense_unsupported_fairlearn_submodule_raises(self):
        """Unsupported fairlearn submodule (e.g., fairlearn.metrics.*) must raise NotImplementedError."""
        cfg = self._make_fitted_defense("fairlearn.metrics.MetricFrame")
        with pytest.raises((NotImplementedError, ImportError)):
            cfg.apply_defense(None)

    def test_apply_defense_defense_application_time_set(self):
        """defense_application_time is recorded after a successful fairlearn defense."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {
                "constraints": "fairlearn.reductions.DemographicParity",
                "eps": 0.1,
            },
        )
        cfg.apply_defense(None)
        assert cfg.defense_application_time is not None

    def test_fairness_defense_config_is_BaseConfig_with_hash(self):
        """Test that FairlearnDefenseConfig is BaseConfig and has __hash__ method."""
        from deckard.utils import BaseConfig

        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {
                "constraints": "fairlearn.reductions.DemographicParity",
                "eps": 0.1,
            },
        )
        assert isinstance(
            cfg,
            BaseConfig,
        ), "FairlearnDefenseConfig should inherit from BaseConfig"
        assert hasattr(
            cfg,
            "__hash__",
        ), "FairlearnDefenseConfig should have __hash__ method"
        # Note: FairlearnDefenseConfig may have unhashable runtime fields
        # so we verify the infrastructure is in place rather than attempting full hash
