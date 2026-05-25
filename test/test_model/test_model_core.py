import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from deckard.data import DataConfig
from deckard.model import ModelConfig
import pytest


class TestModelConfig:

    def setup_method(self):
        # Simple binary classification data
        self.X_train = pd.DataFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, 4]})
        self.y_train = pd.Series([0, 1, 0, 1])
        self.X_test = pd.DataFrame({"a": [4, 5], "b": [5, 6]})
        self.y_test = pd.Series([1, 0])
        self.model_params = {"probability": True}
        self.model_type = "sklearn.ensemble.RandomForestClassifier"
        self.model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        self.tmpdir = tempfile.mkdtemp()
        self.model_file = os.path.join(self.tmpdir, "model.pkl")
        self.pred_file = os.path.join(self.tmpdir, "preds.pkl")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_post_init(self):
        assert hasattr(self.model._model, "fit")
        assert hasattr(self.model._model, "predict")

    def test_train_and_predict(self):
        self.model.train(self.X_train, self.y_train)
        preds = self.model.predict(self.X_train)
        assert len(preds) == len(self.y_train)

    def test_predict_proba(self):
        self.model.train(self.X_train, self.y_train)
        self.model.probability = True
        proba = self.model.predict_proba(self.X_train)
        assert len(proba) == len(self.y_train)

    def test_score(self):
        self.model.train(self.X_train, self.y_train)
        proba = self.model.predict_proba(self.X_train)
        scores = self.model.score(self.y_train, proba, y_proba=proba)
        assert isinstance(scores, dict)
        assert "accuracy" in scores

    def test_call_training_and_prediction(self):
        data = DataConfig()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        data()
        score_dict = model(data=data, model_file=self.model_file)
        scores = model.score_dict
        assert isinstance(scores, dict)
        assert "training_time" in scores and "prediction_time" in scores
        assert "accuracy" in scores["test"]
        assert "training_time" in scores
        assert "prediction_time" in scores
        assert hasattr(model, "score_dict")
        assert hasattr(model, "training_time")
        assert hasattr(model, "training_score_time")
        assert hasattr(model, "prediction_score_time")
        assert hasattr(model, "prediction_time")
        assert hasattr(model, "training_predictions")
        assert hasattr(model, "predictions")
        # Assert that the keys in score dict are also in the model.score_dit
        for key in score_dict:
            assert key in scores
        for key in scores:
            assert key in score_dict

    def test_call_skips_scoring_when_scorer_none(self):
        data = DataConfig(scorer=None)
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            scorer=None,
        )
        scores = model(data=data, model_file=self.model_file)
        assert isinstance(scores, dict)
        assert "training_time" in scores
        assert "prediction_time" in scores
        assert "accuracy" not in scores
        assert "training_accuracy" not in scores

    def test_call_saves_test_predictions_when_file_requested(self):
        data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
            },
        )
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        test_pred_file = os.path.join(self.tmpdir, "test_predictions.pkl")
        model(
            data=data,
            model_file=self.model_file,
            test_predictions_file=test_pred_file,
        )
        assert os.path.exists(test_pred_file)

    def test_call_saves_train_predictions_when_file_requested(self):
        data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
            },
        )
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        train_pred_file = os.path.join(self.tmpdir, "train_predictions.pkl")
        model(
            data=data,
            model_file=self.model_file,
            train_predictions_file=train_pred_file,
        )
        assert os.path.exists(train_pred_file)

    def test_call_saves_train_and_test_predictions_when_requested(self):
        data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
            },
        )
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        train_pred_file = os.path.join(self.tmpdir, "train_predictions.pkl")
        test_pred_file = os.path.join(self.tmpdir, "test_predictions.pkl")
        model(
            data=data,
            model_file=self.model_file,
            train_predictions_file=train_pred_file,
            test_predictions_file=test_pred_file,
        )
        assert os.path.exists(train_pred_file)
        assert os.path.exists(test_pred_file)

    def test_call_saves_test_probabilities_when_file_requested(self):
        data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
            },
        )
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        test_prob_file = os.path.join(self.tmpdir, "test_probabilities.pkl")
        model(
            data=data,
            model_file=self.model_file,
            test_probabilities_file=test_prob_file,
        )
        assert os.path.exists(test_prob_file)

    def test_call_saves_train_probabilities_when_file_requested(self):
        data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
            },
        )
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        train_prob_file = os.path.join(self.tmpdir, "train_probabilities.pkl")
        model(
            data=data,
            model_file=self.model_file,
            training_probabilities_file=train_prob_file,
        )
        assert os.path.exists(train_prob_file)

    def test_load_predictions(self):
        preds = np.array([0, 1, 1, 0])
        pred_file = os.path.join(self.tmpdir, "preds.npy")
        np.save(pred_file, preds)
        # Patch load_data to use np.load for this test
        orig_load_data = self.model.load_data
        self.model.load_data = lambda fp: np.load(fp)
        loaded = self.model._load_predictions(pred_file)
        assert np.array_equal(loaded, preds)
        self.model.load_data = orig_load_data

    def test_load_or_train_model_trains_when_not_fitted_even_if_training_time_set(
        self,
    ):
        data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
            },
        )
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 5},
        )
        model.training_time = 1.23
        model._model = RandomForestClassifier(n_estimators=5)
        times = {}
        times = model._load_or_train_model(data, model_file=None, times=times)
        assert "training_time" in times
        assert "training_n" in times
        assert times["training_n"] == len(data.y_train)

    def test_predict_falls_back_when_wrapped_output_matrix_is_invalid(self):
        class BasePredictor:
            def predict(self, X):
                return np.array([0, 1])

        class WrappedPredictor:
            def __init__(self):
                self.model = BasePredictor()

            def predict(self, X):
                return np.array([[1.0, 1.0], [1.0, 1.0]])

        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        model._model = WrappedPredictor()

        preds = model.predict(self.X_test)
        assert np.array_equal(preds, np.array([0, 1]))

    def test_decode_predictions_for_persistence_converts_2d_classifier_output(
        self,
    ):
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([[1.0, 1.0], [0.0, 2.0], [2.0, 0.0], [1.0, 1.0]])
        decoded = model._decode_predictions_for_persistence(
            y_pred,
            y_true=y_true,
        )
        assert decoded.ndim == 1
        assert len(decoded) == len(y_true)

    def test_hash_stable_after_call_for_model_config(self):
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        original_hash = hash(model)
        data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 0,
            },
        )
        data()
        model(data)
        assert original_hash == \
            hash(model), \
            "Hash changed after call for ModelConfig"


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_data(n_train=60, n_test=20, n_features=4, classifier=True):
    """Return a loaded DataConfig (data on object after calling it)."""
    data_cfg = DataConfig(
        dataset_name="make_classification" if classifier else "make_regression",
        data_params={
            "n_samples": n_train + n_test,
            "n_features": n_features,
            "n_informative": n_features - 1,
            "n_redundant": 1 if classifier else 0,
            "random_state": 0,
        },
        train_size=n_train,
        test_size=n_test,
        random_state=42,
        classifier=classifier,
    )
    data_cfg()  # populates X_train, y_train, X_test, y_test on the object
    return data_cfg


def _make_fitted_model(n_train=60, n_test=20, n_features=4, classifier=True):
    """Return a (loaded_data, ModelConfig) pair."""
    data = _make_data(n_train, n_test, n_features, classifier)
    model_type = (
        "sklearn.tree.DecisionTreeClassifier"
        if classifier
        else "sklearn.tree.DecisionTreeRegressor"
    )
    model_cfg = ModelConfig(
        model_type=model_type,
        classifier=classifier,
        model_params={"max_depth": 2},
    )
    return data, model_cfg


# ── __post_init__ scorer branches ──────────────────────────────────────────


class TestModelPostInitScorerBranches:
    def test_null_scorer_becomes_none(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        assert model.scorer is None

    def test_default_scorer_loads_classifier_scorer(self):
        # AUTO_SCORER sentinel is "auto" (defined in model/base.py)
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer="auto",
        )
        # scorer should be resolved to an actual scorer instance, not the sentinel
        assert model.scorer is not None
        assert model.scorer != "auto"

    def test_default_scorer_regressor_resolves(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeRegressor",
            classifier=False,
            model_params={"max_depth": 2},
            scorer="auto",
        )
        assert model.scorer is not None
        assert model.scorer != "auto"

    def test_scorer_as_dict_becomes_scorer_dict_config(self):
        scorer_dict = {
            "scorers": {
                "acc": {
                    "score_name": "accuracy",
                    "score_function": "sklearn.metrics.accuracy_score",
                    "score_params": {},
                    "greater_is_better": True,
                    "needs_labels": True,
                },
            },
        }
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=scorer_dict,
        )
        # Should not raise; scorer transformed from raw dict to ScorerDictConfig
        assert model.scorer is not None

    def test_defense_with_null_defense_name_sets_defense_none(self):
        defense = SimpleNamespace(defense_name=None)
        model = ModelConfig.__new__(ModelConfig)
        object.__setattr__(model, "defense", defense)
        object.__setattr__(model, "model_type", "sklearn.tree.DecisionTreeClassifier")
        object.__setattr__(model, "classifier", True)
        object.__setattr__(model, "model_params", {"max_depth": 2})
        object.__setattr__(model, "plugins", None)
        object.__setattr__(model, "scorer", None)
        object.__setattr__(model, "_model", None)
        object.__setattr__(model, "probability", False)
        object.__setattr__(model, "alias", None)
        object.__setattr__(model, "_plugin_objects", None)
        object.__setattr__(model, "_defense_pipeline", None)
        object.__setattr__(model, "score_dict", None)
        model.__post_init__()
        assert model.defense is None

    def test_classifier_string_to_bool(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier="classifier",
            model_params={"max_depth": 2},
            scorer=None,
        )
        assert model.classifier

    def test_regressor_string_to_bool(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeRegressor",
            classifier="regressor",
            model_params={"max_depth": 2},
            scorer=None,
        )
        assert not model.classifier

    def test_other_classifier_value_raises_valueerror(self):
        model = ModelConfig.__new__(ModelConfig)
        object.__setattr__(model, "model_type", "sklearn.tree.DecisionTreeClassifier")
        object.__setattr__(model, "classifier", "unknown_value")
        object.__setattr__(model, "model_params", {"max_depth": 2})
        object.__setattr__(model, "plugins", None)
        object.__setattr__(model, "scorer", None)
        object.__setattr__(model, "_model", None)
        object.__setattr__(model, "probability", False)
        object.__setattr__(model, "alias", None)
        object.__setattr__(model, "_plugin_objects", None)
        object.__setattr__(model, "_defense_pipeline", None)
        object.__setattr__(model, "score_dict", None)
        object.__setattr__(model, "defense", None)
        with pytest.raises(ValueError):
            model.__post_init__()


# ── Plugin system ──────────────────────────────────────────────────────────


class TestPluginSystem:
    def _model(self):
        return ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )

    def test_instantiate_plugin_from_string(self):
        model = self._model()
        # Reuse sklearn's DecisionTreeClassifier as a loadable class
        instance = model._instantiate_plugin(
            "sklearn.tree.DecisionTreeClassifier",
        )
        from sklearn.tree import DecisionTreeClassifier

        assert isinstance(instance, DecisionTreeClassifier)

    def test_instantiate_plugin_from_type(self):
        model = self._model()

        class Dummy:
            pass

        instance = model._instantiate_plugin(Dummy)
        assert isinstance(instance, Dummy)

    def test_instantiate_plugin_from_object(self):
        model = self._model()

        class Obj:
            pass

        obj = Obj()
        result = model._instantiate_plugin(obj)
        assert result is obj

    def test_instantiate_plugin_dict_with_name_key(self):
        model = self._model()
        result = model._instantiate_plugin(
            {
                "name": "sklearn.tree.DecisionTreeClassifier",
                "max_depth": 2,
            },
        )
        from sklearn.tree import DecisionTreeClassifier

        assert isinstance(result, DecisionTreeClassifier)

    def test_instantiate_plugin_dict_missing_name_raises(self):
        model = self._model()
        with pytest.raises(ValueError):
            model._instantiate_plugin({"no_name_key": "value"})

    def test_get_plugins_non_list_raises(self):
        model = self._model()
        model.plugins = "not_a_list"
        model._plugin_objects = None
        with pytest.raises(TypeError):
            model._get_plugins()

    def test_run_plugin_hook_calls_callable_hook(self):
        model = self._model()

        class Spy:
            called = False

            def my_hook(self, caller, **kwargs):
                Spy.called = True
                return {"spy": True}

        model._plugin_objects = [Spy()]
        outputs = model._run_plugin_hook("my_hook", extra=1)
        assert Spy.called
        assert outputs == [{"spy": True}]

    def test_merge_plugin_scores_updates_score_dict(self):
        model = self._model()
        model.score_dict = {"existing": 1}
        model._merge_plugin_scores([{"new_metric": 42}])
        assert "new_metric" in model.score_dict
        assert "existing" in model.score_dict

    def test_merge_plugin_scores_with_none_score_dict(self):
        model = self._model()
        model.score_dict = None
        model._merge_plugin_scores([{"m": 99}])
        assert model.score_dict["m"] == 99


# ── _predict TypeError branches ────────────────────────────────────────────


class TestPredictTypeErrorHandling:
    def _model_with_mock(self, side_effect_first, return_value_second):
        """Make a ModelConfig whose _model.predict raises then returns on second call."""
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        call_count = [0]

        def _predict(X):
            call_count[0] += 1
            if call_count[0] == 1:
                raise side_effect_first
            return return_value_second

        model._model = SimpleNamespace(predict=_predict)
        return model

    def test_ufunc_type_error_falls_back_to_array(self):
        arr = np.array([0, 1, 0, 1])
        model = self._model_with_mock(
            TypeError("loop of ufunc does not support argument"),
            arr,
        )
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = model.predict(X)
        np.testing.assert_array_equal(result, arr)

    def test_cant_convert_type_error_falls_back_to_array(self):
        arr = np.array([1, 0, 1, 0])
        model = self._model_with_mock(
            TypeError("can't convert"),
            arr,
        )
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = model.predict(X)
        np.testing.assert_array_equal(result, arr)

    def test_other_type_error_reraises(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = SimpleNamespace(
            predict=MagicMock(side_effect=TypeError("something else entirely")),
        )
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError):
            model.predict(X)

    def test_predict_without_model_raises_value_error(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = None
        with pytest.raises(ValueError):
            model.predict(pd.DataFrame({"a": [1]}))


# ── _predict_proba branches ────────────────────────────────────────────────


class TestPredictProbaBranches:
    def test_predict_proba_not_probability_raises(self):
        from sklearn.tree import DecisionTreeClassifier

        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            probability=False,
            scorer=None,
        )
        # Assign a real sklearn model to _model
        model._model = DecisionTreeClassifier(max_depth=2)
        model.probability = False
        data = _make_data()
        with pytest.raises(ValueError):
            model.predict_proba(data.X_test)

    def test_predict_proba_no_model_raises(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = None
        model.probability = True
        with pytest.raises(ValueError):
            model.predict_proba(pd.DataFrame({"a": [1]}))


# ── get_art_class and get_art_model ────────────────────────────────────────


class TestGetArtClassAndModel:
    def test_get_art_class_sklearn_no_input_shape(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        data, _ = _make_fitted_model()
        art_class, init_params = model.get_art_class(data)
        # sklearn ART wrappers don't need input_shape
        assert "input_shape" not in init_params
        assert "preprocessing" in init_params

    def test_get_art_model_no_defense_returns_art_wrapper(self):
        data, model = _make_fitted_model()
        model(data)  # fit model first
        art_model = model.get_art_model(data)
        assert art_model is not None

    def test_get_model_raises_when_no_model(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = None
        with pytest.raises(ValueError):
            model.get_model()


# ── _load_score_file ───────────────────────────────────────────────────────


class TestLoadScoreFile:
    def test_load_score_file_existing_merges_scores(self):
        import json

        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        with tempfile.TemporaryDirectory() as td:
            score_path = Path(td) / "scores.json"
            score_path.write_text(
                json.dumps({"accuracy": 0.9, "training_time": 1.2, "training_n": 50}),
            )
            times = model._load_score_file(str(score_path))
        assert "training_time" in times
        assert "training_n" in times

    def test_load_score_file_nonexistent_returns_empty(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        times = model._load_score_file("/nonexistent/path.json")
        assert times == {}


# ── _load_or_train_model ───────────────────────────────────────────────────


class TestLoadOrTrainModel:
    def test_model_file_present_loads_from_file(self):
        data, model = _make_fitted_model()
        with tempfile.TemporaryDirectory() as td:
            model_path = str(Path(td) / "model.pkl")
            model(data, model_file=model_path)
            # Now create a fresh model and load from file
            model2 = ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
                scorer=None,
            )
            times = {}
            model2._load_or_train_model(data, model_path, times)
            assert model2._model is not None
            assert model2._is_model_fitted(model2._model, X_sample=data.X_train)

    def test_model_file_present_unwraps_loaded_model_config(self):
        data, model = _make_fitted_model()
        model.train(data.X_train, data.y_train)

        with tempfile.TemporaryDirectory() as td:
            model_path = str(Path(td) / "model.pkl")
            Path(model_path).write_text("placeholder")

            loaded_obj = ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
                scorer=None,
            )
            loaded_obj._model = model._model

            model2 = ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
                scorer=None,
            )
            model2.load = lambda _fp: loaded_obj

            times = {}
            model2._load_or_train_model(data, model_path, times)

            assert not isinstance(model2._model, ModelConfig)
            assert hasattr(model2._model, "predict")
            assert model2.get_model() is model2._model
            assert model2._model is loaded_obj._model

    def test_model_file_present_loaded_estimator_syncs_model_signature(self):
        from sklearn.tree import DecisionTreeClassifier

        data, _ = _make_fitted_model()
        loaded_estimator = DecisionTreeClassifier(max_depth=3, random_state=0)
        loaded_estimator.fit(data.X_train, data.y_train)

        with tempfile.TemporaryDirectory() as td:
            model_path = str(Path(td) / "model.pkl")
            Path(model_path).write_text("placeholder")

            model2 = ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
                scorer=None,
            )
            model2.load = lambda _fp: loaded_estimator

            times = {}
            model2._load_or_train_model(data, model_path, times)

            assert model2._model is loaded_estimator
            assert model2.model_type == \
                "sklearn.tree._classes.DecisionTreeClassifier"
            assert model2.model_params.get("max_depth") == 3

    def test_no_model_no_file_raises_value_error(self):
        data, _ = _make_fitted_model()
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = None
        with pytest.raises(ValueError):
            model._load_or_train_model(data, None, {})


# ── decode/load/score branch coverage ───────────────────────────────────────


class TestDecodePredictionsForPersistence:
    def test_regressor_passthrough(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeRegressor",
            classifier=False,
            model_params={"max_depth": 2},
            scorer=None,
        )
        y_pred = np.array([1.2, 3.4])
        out = model._decode_predictions_for_persistence(y_pred)
        np.testing.assert_array_equal(out, y_pred)

    def test_binary_single_column_numeric_labels(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        y_pred = np.array([[0.2], [0.8], [1.2]])
        y_true = np.array([10, 20, 10])
        out = model._decode_predictions_for_persistence(y_pred, y_true=y_true)
        assert list(out) == [20.0, 20.0, 20.0]

    def test_multiclass_argmax(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        y_pred = np.array([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1]])
        out = model._decode_predictions_for_persistence(y_pred)
        np.testing.assert_array_equal(out, np.array([1, 0]))


class TestLoadAllPredictionsBranches:
    def _model(self):
        return ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )

    def test_train_predictions_loaded_without_time_asserts(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train.csv"
            p.write_text("1\n2\n")
            model._load_predictions = lambda _f: [1, 2]
            model.training_prediction_time = None
            with pytest.raises(AssertionError):
                model._load_all_predictions(str(p), None, {})

    def test_test_predictions_loaded_without_time_asserts(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.csv"
            p.write_text("1\n2\n")
            model._load_predictions = lambda _f: [1, 2]
            model.prediction_time = None
            with pytest.raises(AssertionError):
                model._load_all_predictions(None, str(p), {})


class TestScoreValidationBranches:
    def test_score_with_non_callable_scorer_raises(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model.scorer = "not-callable"
        with pytest.raises(TypeError):
            model.score(np.array([0, 1]), np.array([0, 1]), mode="test")

    def test_load_predictions_invalid_type_raises(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model.load_data = lambda _f: 12345
        with pytest.raises(ValueError):
            model._load_predictions("dummy")


# ── _copy_runtime_state_to ─────────────────────────────────────────────────


class TestCopyRuntimeStateTo:
    def test_copies_all_runtime_fields(self):
        data, model = _make_fitted_model()
        model(data)
        target = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._copy_runtime_state_to(target)
        assert target._model is not None
        # at minimum the score_dict should be copied
        assert target.score_dict is not None


# ── _require_defense_pipeline ─────────────────────────────────────────────


class TestRequireDefensePipeline:
    def test_no_defense_returns_none(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        result = model._require_defense_pipeline()
        assert result is None


class TestModelEvaluateAndScoreBranches:
    def _model(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        return model

    def test_evaluate_training_loss_curve_key_removed(self):
        model = self._model()
        data = _make_data()

        class _PredictModel:
            def predict_proba(self, X):
                n = len(X)
                return np.column_stack([np.ones(n) * 0.4, np.ones(n) * 0.6])

        model._model = _PredictModel()
        model.predict = lambda X: np.ones(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {
            "loss_curve": [1.0, 0.5],
            "accuracy": 0.5,
        }
        model.score_dict = {}

        model._evaluate_and_score(
            data,
            times={},
            persist_training_probabilities=True,
            persist_test_probabilities=True,
        )

        assert "training_loss_curve" not in model.score_dict
        assert "accuracy" in model.score_dict
        assert "training_accuracy" not in model.score_dict

    def test_evaluate_with_no_test_data_raises(self):
        model = self._model()
        data = _make_data()
        data.X_test = None
        model._model = SimpleNamespace(predict_proba=lambda X: np.zeros((len(X), 2)))
        model.predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with pytest.raises(ValueError):
            model._evaluate_and_score(data, times={})

    def test_evaluate_with_no_test_labels_raises(self):
        model = self._model()
        data = _make_data()
        data.y_test = None
        model._model = SimpleNamespace(predict_proba=lambda X: np.zeros((len(X), 2)))
        model.predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with pytest.raises(ValueError):
            model._evaluate_and_score(data, times={})

    def test_evaluate_predict_proba_typeerror_fallback_uses_numpy_cast(self):
        model = self._model()
        data = _make_data()

        def _predict_proba(X):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                raise TypeError("can't convert")
            n = len(X)
            return np.column_stack([np.ones(n) * 0.3, np.ones(n) * 0.7])

        model._model = SimpleNamespace(predict_proba=_predict_proba)
        model.predict = lambda X: np.ones(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = {}

        model._evaluate_and_score(
            data,
            times={},
            persist_training_probabilities=True,
            persist_test_probabilities=True,
        )

        assert model.training_probabilities is not None
        assert model.probabilities is not None

    def test_evaluate_uses_preexisting_predictions(self):
        model = self._model()
        data = _make_data()
        model.training_predictions = np.zeros(len(data.y_train), dtype=int)
        model.predictions = np.zeros(len(data.y_test), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = {}

        times = {}
        model._evaluate_and_score(data, times=times)

        assert times["prediction_n"] == len(data.y_test)
        assert "training_n" not in times

    def test_evaluate_applies_defense_before_predict_and_merges_scores(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace()
        model.defense = object()
        model.score_dict = None
        model.training_predictions = np.zeros(len(data.y_train), dtype=int)
        model.predictions = np.zeros(len(data.y_test), dtype=int)
        model.scorer = None

        defense_pipeline = SimpleNamespace(
            resolve_stage=lambda **_kwargs: "before_predict",
            apply=lambda estimator, data, stage: estimator,
            defense_application_time=0.25,
            score_dict={"defense_metric": 1.0},
        )
        model._require_defense_pipeline = lambda: defense_pipeline

        times = {}
        model._evaluate_and_score(data, times=times)

        assert model.score_dict["defense_metric"] == 1.0
        assert model.score_dict["defense_application_time"] == 0.25

    def test_evaluate_probability_persistence_valueerror_sets_none(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace(
            predict_proba=lambda _X: (_ for _ in ()).throw(ValueError("bad probs")),
        )
        model.predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = {}

        model._evaluate_and_score(
            data,
            times={},
            persist_training_probabilities=True,
            persist_test_probabilities=True,
        )

        assert model.training_probabilities is None
        assert model.probabilities is None

    def test_evaluate_training_probability_typeerror_reraises(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace(
            predict_proba=lambda _X: (_ for _ in ()).throw(TypeError("unexpected")),
        )
        model.predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with pytest.raises(TypeError):
            model._evaluate_and_score(
                data,
                times={},
                persist_training_probabilities=True,
                persist_test_probabilities=False,
            )

    def test_evaluate_test_probability_typeerror_reraises(self):
        model = self._model()
        data = _make_data()
        model.training_predictions = np.zeros(len(data.y_train), dtype=int)
        model._model = SimpleNamespace(
            predict_proba=lambda _X: (_ for _ in ()).throw(TypeError("unexpected")),
        )
        model.predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with pytest.raises(TypeError):
            model._evaluate_and_score(
                data,
                times={},
                persist_training_probabilities=False,
                persist_test_probabilities=True,
            )

    def test_evaluate_probability_persistence_without_predict_proba_uses_predict(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace()
        model.predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = {}

        model._evaluate_and_score(
            data,
            times={},
            persist_training_probabilities=True,
            persist_test_probabilities=True,
        )

        assert model.training_probabilities is not None
        assert model.probabilities is not None

    def test_val_mode_resamples_when_validation_split_missing(self):
        data = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 120,
                "n_features": 6,
                "n_informative": 4,
                "n_redundant": 2,
                "random_state": 0,
            },
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            sampler="split",
            random_state=42,
            classifier=True,
        )
        data()
        data.X_val = None
        data.y_val = None
        data.val_n = None

        model = self._model()
        model.score_mode = "val"
        model.train(data.X_train, data.y_train)
        model.scorer = lambda y_true, y_pred, mode="val", **kwargs: {
            "validation_accuracy": 1.0,
        }

        times = {}
        model._evaluate_and_score(data, times=times)

        assert data.X_val is not None
        assert data.y_val is not None
        assert "validation_n" in times

    def test_evaluate_initializes_score_dict_when_none(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace(predict_proba=lambda X: np.zeros((len(X), 2)))
        model.predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = None

        model._evaluate_and_score(data, times={})
        assert isinstance(model.score_dict, dict)

    def test_evaluate_raises_when_train_predictions_none(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace()
        model.predict = lambda _X: None
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with pytest.raises(TypeError):
            model._evaluate_and_score(data, times={})


class TestModelLoadOrTrainBranches:
    def test_load_or_train_trains_when_model_file_exists_but_not_fitted(self):
        data = _make_data()
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )

        class _Bare:
            pass

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.pkl"
            p.write_text("x")

            loaded_obj = ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
                scorer=None,
            )
            loaded_obj._model = _Bare()
            loaded_obj.defense = None

            model.load = lambda _fp: loaded_obj
            model.save_object = lambda *_args, **_kwargs: None
            loaded_obj.train = (
                lambda X, y: setattr(loaded_obj, "training_time", 0.01)
                or setattr(loaded_obj, "training_n", len(y))
                or setattr(
                    loaded_obj,
                    "_model",
                    SimpleNamespace(predict=lambda s: np.zeros(len(s), dtype=int)),
                )
            )
            model._model = SimpleNamespace(
                predict=lambda s: np.zeros(len(s), dtype=int),
            )
            out = model._load_or_train_model(data, str(p), {})

            assert "training_time" in out
            assert "training_n" in out

    def test_load_or_train_with_none_model_and_missing_file_trains(self):
        data = _make_data()
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model.train(data.X_train, data.y_train)

        with tempfile.TemporaryDirectory() as td:
            out = model._load_or_train_model(data, str(Path(td) / "missing.pkl"), {})

        assert "training_time" in out
        assert "training_n" in out

    def test_load_or_train_raises_notfitted_when_model_missing_after_train(self):
        data = _make_data()
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = SimpleNamespace()

        def train(_X, y):
            model.training_time = 0.01
            model.training_n = len(y)
            model._model = None

        model.train = train
        with pytest.raises(Exception):
            model._load_or_train_model(data, None, {})

    def test_load_or_train_existing_file_not_fitted_applies_defense(self):
        data = _make_data()
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )

        class _Unfitted:
            pass

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.pkl"
            p.write_text("x")

            loaded_obj = ModelConfig(
                model_type="sklearn.tree.DecisionTreeClassifier",
                classifier=True,
                model_params={"max_depth": 2},
                scorer=None,
            )
            loaded_obj._model = _Unfitted()
            loaded_obj.defense = object()
            loaded_obj.train = (
                lambda X, y: setattr(loaded_obj, "training_time", 0.01)
                or setattr(loaded_obj, "training_n", len(y))
                or setattr(
                    loaded_obj,
                    "_model",
                    SimpleNamespace(predict=lambda s: np.zeros(len(s), dtype=int)),
                )
            )
            loaded_obj.apply_defense = lambda _data, stage="post_fit_pre_predict": SimpleNamespace(
                predict=lambda s: np.zeros(len(s), dtype=int),
            )
            loaded_obj._require_defense_pipeline = lambda: SimpleNamespace(
                resolve_stage=lambda **_kwargs: "skip",
            )

            model.load = lambda _fp: loaded_obj
            model.save_object = lambda *_args, **_kwargs: None
            out = model._load_or_train_model(data, str(p), {})

            assert "training_time" in out
            assert "training_n" in out

