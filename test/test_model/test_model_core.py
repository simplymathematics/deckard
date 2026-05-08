import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from deckard.model import ModelConfig
from deckard.data import DataConfig
from sklearn.ensemble import RandomForestClassifier


class TestModelConfig(unittest.TestCase):
    def setUp(self):
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

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_post_init(self):
        self.assertTrue(hasattr(self.model._model, "fit"))
        self.assertTrue(hasattr(self.model._model, "predict"))

    def test_train_and_predict(self):
        self.model._train(self.X_train, self.y_train)
        preds = self.model._predict(self.X_train)
        self.assertEqual(len(preds), len(self.y_train))

    def test_predict_proba(self):
        self.model._train(self.X_train, self.y_train)
        self.model.probability = True
        proba = self.model._predict_proba(self.X_train)
        self.assertEqual(len(proba), len(self.y_train))

    def test_classification_scores(self):
        scores = self.model._classification_scores(self.y_train, self.y_train)
        self.assertIn("accuracy", scores)
        self.assertIn("precision", scores)
        self.assertIn("recall", scores)
        self.assertIn("f1-score", scores)

    def test_classification_scores_handles_single_column_binary_outputs(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred_single_col = np.array([[0.1], [0.9], [0.2], [0.8]])
        scores = self.model._classification_scores(y_true, y_pred_single_col)
        self.assertIn("accuracy", scores)
        self.assertGreaterEqual(scores["accuracy"], 0.99)

    def test_regression_scores(self):
        # Use regression scores with float values
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.1, 1.9, 3.2])
        scores = self.model._regression_scores(y_true, y_pred)
        self.assertIn("mse", scores)
        self.assertIn("rmse", scores)
        self.assertIn("mae", scores)

    def test_score(self):
        self.model._train(self.X_train, self.y_train)
        proba = self.model._predict_proba(self.X_train)
        scores = self.model._score(self.y_train, proba, y_proba=proba)
        self.assertIsInstance(scores, dict)
        self.assertIn("accuracy", scores)

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
        self.assertIsInstance(scores, dict)
        self.assertTrue(
            "training_time" in scores and "prediction_time" in scores,
        )
        self.assertTrue("accuracy" in scores)
        self.assertTrue("training_time" in scores)
        self.assertTrue("prediction_time" in scores)
        self.assertTrue(hasattr(model, "score_dict"))
        self.assertTrue(hasattr(model, "training_time"))
        self.assertTrue(hasattr(model, "training_score_time"))
        self.assertTrue(hasattr(model, "prediction_score_time"))
        self.assertTrue(hasattr(model, "prediction_time"))
        self.assertTrue(hasattr(model, "training_predictions"))
        self.assertTrue(hasattr(model, "predictions"))
        # Assert that the keys in score dict are also in the model.score_dit
        for key in score_dict:
            self.assertIn(key, scores)
        for key in scores:
            self.assertIn(key, score_dict)

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
        self.assertIsInstance(scores, dict)
        self.assertIn("training_time", scores)
        self.assertIn("prediction_time", scores)
        self.assertNotIn("accuracy", scores)
        self.assertNotIn("training_accuracy", scores)

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
        self.assertTrue(os.path.exists(test_pred_file))

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
        self.assertTrue(os.path.exists(train_pred_file))

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
        self.assertTrue(os.path.exists(train_pred_file))
        self.assertTrue(os.path.exists(test_pred_file))

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
        self.assertTrue(os.path.exists(test_prob_file))

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
        self.assertTrue(os.path.exists(train_prob_file))

    def test_load_predictions(self):
        preds = np.array([0, 1, 1, 0])
        pred_file = os.path.join(self.tmpdir, "preds.npy")
        np.save(pred_file, preds)
        # Patch load_data to use np.load for this test
        orig_load_data = self.model.load_data
        self.model.load_data = lambda fp: np.load(fp)
        loaded = self.model._load_predictions(pred_file)
        self.assertTrue(np.array_equal(loaded, preds))
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
        self.assertIn("training_time", times)
        self.assertIn("training_n", times)
        self.assertEqual(times["training_n"], len(data.y_train))

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

        preds = model._predict(self.X_test)
        self.assertTrue(np.array_equal(preds, np.array([0, 1])))

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
        self.assertEqual(decoded.ndim, 1)
        self.assertEqual(len(decoded), len(y_true))

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
        self.assertEqual(
            original_hash,
            hash(model),
            msg="Hash changed after call for ModelConfig",
        )


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


class TestModelPostInitScorerBranches(unittest.TestCase):
    def test_null_scorer_becomes_none(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        self.assertIsNone(model.scorer)

    def test_default_scorer_loads_classifier_scorer(self):
        # AUTO_SCORER sentinel is "auto" (defined in model/base.py)
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer="auto",
        )
        # scorer should be resolved to an actual scorer instance, not the sentinel
        self.assertIsNotNone(model.scorer)
        self.assertNotEqual(model.scorer, "auto")

    def test_default_scorer_regressor_resolves(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeRegressor",
            classifier=False,
            model_params={"max_depth": 2},
            scorer="auto",
        )
        self.assertIsNotNone(model.scorer)
        self.assertNotEqual(model.scorer, "auto")

    def test_scorer_as_dict_becomes_scorer_dict_config(self):
        scorer_dict = {
            "scorers": {
                "acc": {
                    "score_name": "accuracy",
                    "score_function": "sklearn.metrics.accuracy_score",
                    "score_params": {},
                    "greater_is_better": True,
                    "needs_proba": False,
                }
            }
        }
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=scorer_dict,
        )
        # Should not raise; scorer transformed from raw dict to ScorerDictConfig
        self.assertIsNotNone(model.scorer)

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
        self.assertIsNone(model.defense)

    def test_classifier_string_to_bool(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier="classifier",
            model_params={"max_depth": 2},
            scorer=None,
        )
        self.assertTrue(model.classifier)

    def test_regressor_string_to_bool(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeRegressor",
            classifier="regressor",
            model_params={"max_depth": 2},
            scorer=None,
        )
        self.assertFalse(model.classifier)

    def test_other_classifier_value_becomes_none(self):
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
        model.__post_init__()
        self.assertIsNone(model.classifier)


# ── Plugin system ──────────────────────────────────────────────────────────


class TestPluginSystem(unittest.TestCase):
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
            "sklearn.tree.DecisionTreeClassifier"
        )
        from sklearn.tree import DecisionTreeClassifier

        self.assertIsInstance(instance, DecisionTreeClassifier)

    def test_instantiate_plugin_from_type(self):
        model = self._model()

        class Dummy:
            pass

        instance = model._instantiate_plugin(Dummy)
        self.assertIsInstance(instance, Dummy)

    def test_instantiate_plugin_from_object(self):
        model = self._model()

        class Obj:
            pass

        obj = Obj()
        result = model._instantiate_plugin(obj)
        self.assertIs(result, obj)

    def test_instantiate_plugin_dict_with_name_key(self):
        model = self._model()
        result = model._instantiate_plugin(
            {
                "name": "sklearn.tree.DecisionTreeClassifier",
                "max_depth": 2,
            }
        )
        from sklearn.tree import DecisionTreeClassifier

        self.assertIsInstance(result, DecisionTreeClassifier)

    def test_instantiate_plugin_dict_missing_name_raises(self):
        model = self._model()
        with self.assertRaises(ValueError):
            model._instantiate_plugin({"no_name_key": "value"})

    def test_get_plugins_non_list_raises(self):
        model = self._model()
        model.plugins = "not_a_list"
        model._plugin_objects = None
        with self.assertRaises(TypeError):
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
        self.assertTrue(Spy.called)
        self.assertEqual(outputs, [{"spy": True}])

    def test_merge_plugin_scores_updates_score_dict(self):
        model = self._model()
        model.score_dict = {"existing": 1}
        model._merge_plugin_scores([{"new_metric": 42}])
        self.assertIn("new_metric", model.score_dict)
        self.assertIn("existing", model.score_dict)

    def test_merge_plugin_scores_with_none_score_dict(self):
        model = self._model()
        model.score_dict = None
        model._merge_plugin_scores([{"m": 99}])
        self.assertEqual(model.score_dict["m"], 99)


# ── _predict TypeError branches ────────────────────────────────────────────


class TestPredictTypeErrorHandling(unittest.TestCase):
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
        result = model._predict(X)
        np.testing.assert_array_equal(result, arr)

    def test_cant_convert_type_error_falls_back_to_array(self):
        arr = np.array([1, 0, 1, 0])
        model = self._model_with_mock(
            TypeError("can't convert"),
            arr,
        )
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = model._predict(X)
        np.testing.assert_array_equal(result, arr)

    def test_other_type_error_reraises(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = SimpleNamespace(
            predict=MagicMock(side_effect=TypeError("something else entirely"))
        )
        X = pd.DataFrame({"a": [1, 2, 3]})
        with self.assertRaises(TypeError):
            model._predict(X)

    def test_predict_without_model_raises_value_error(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = None
        with self.assertRaises(ValueError):
            model._predict(pd.DataFrame({"a": [1]}))


# ── _predict_proba branches ────────────────────────────────────────────────


class TestPredictProbaBranches(unittest.TestCase):
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
        with self.assertRaises(ValueError):
            model._predict_proba(data.X_test)

    def test_predict_proba_no_model_raises(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = None
        model.probability = True
        with self.assertRaises(ValueError):
            model._predict_proba(pd.DataFrame({"a": [1]}))


# ── get_art_class and get_art_model ────────────────────────────────────────


class TestGetArtClassAndModel(unittest.TestCase):
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
        self.assertNotIn("input_shape", init_params)
        self.assertIn("preprocessing", init_params)

    def test_get_art_model_no_defense_returns_art_wrapper(self):
        data, model = _make_fitted_model()
        model(data)  # fit model first
        art_model = model.get_art_model(data)
        self.assertIsNotNone(art_model)

    def test_get_model_raises_when_no_model(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = None
        with self.assertRaises(ValueError):
            model.get_model()


# ── _load_score_file ───────────────────────────────────────────────────────


class TestLoadScoreFile(unittest.TestCase):
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
                json.dumps({"accuracy": 0.9, "training_time": 1.2, "training_n": 50})
            )
            times = model._load_score_file(str(score_path))
        self.assertIn("training_time", times)
        self.assertIn("training_n", times)

    def test_load_score_file_nonexistent_returns_empty(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        times = model._load_score_file("/nonexistent/path.json")
        self.assertEqual(times, {})


# ── _load_or_train_model ───────────────────────────────────────────────────


class TestLoadOrTrainModel(unittest.TestCase):
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
            self.assertIsNotNone(model2._model)

    def test_no_model_no_file_raises_value_error(self):
        data, _ = _make_fitted_model()
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = None
        with self.assertRaises(ValueError):
            model._load_or_train_model(data, None, {})


# ── decode/load/score branch coverage ───────────────────────────────────────


class TestDecodePredictionsForPersistence(unittest.TestCase):
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
        self.assertEqual(list(out), [20.0, 20.0, 20.0])

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


class TestLoadAllPredictionsBranches(unittest.TestCase):
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
            with self.assertRaises(AssertionError):
                model._load_all_predictions(str(p), None, {})

    def test_test_predictions_loaded_without_time_asserts(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.csv"
            p.write_text("1\n2\n")
            model._load_predictions = lambda _f: [1, 2]
            model.prediction_time = None
            with self.assertRaises(AssertionError):
                model._load_all_predictions(None, str(p), {})


class TestScoreValidationBranches(unittest.TestCase):
    def test_score_with_non_callable_scorer_raises(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model.scorer = "not-callable"
        with self.assertRaises(TypeError):
            model._score(np.array([0, 1]), np.array([0, 1]), mode="test")

    def test_load_predictions_invalid_type_raises(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model.load_data = lambda _f: 12345
        with self.assertRaises(ValueError):
            model._load_predictions("dummy")


# ── _copy_runtime_state_to ─────────────────────────────────────────────────


class TestCopyRuntimeStateTo(unittest.TestCase):
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
        self.assertIsNotNone(target._model)
        # at minimum the score_dict should be copied
        self.assertIsNotNone(target.score_dict)


# ── _require_defense_pipeline ─────────────────────────────────────────────


class TestRequireDefensePipeline(unittest.TestCase):
    def test_no_defense_returns_none(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        result = model._require_defense_pipeline()
        self.assertIsNone(result)


class TestModelEvaluateAndScoreBranches(unittest.TestCase):
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
        model._predict = lambda X: np.ones(len(X), dtype=int)
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

        self.assertNotIn("training_loss_curve", model.score_dict)
        self.assertIn("accuracy", model.score_dict)
        self.assertNotIn("training_accuracy", model.score_dict)

    def test_evaluate_with_no_test_data_raises(self):
        model = self._model()
        data = _make_data()
        data.X_test = None
        model._model = SimpleNamespace(predict_proba=lambda X: np.zeros((len(X), 2)))
        model._predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with self.assertRaises(ValueError):
            model._evaluate_and_score(data, times={})

    def test_evaluate_with_no_test_labels_raises(self):
        model = self._model()
        data = _make_data()
        data.y_test = None
        model._model = SimpleNamespace(predict_proba=lambda X: np.zeros((len(X), 2)))
        model._predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with self.assertRaises(ValueError):
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
        model._predict = lambda X: np.ones(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = {}

        model._evaluate_and_score(
            data,
            times={},
            persist_training_probabilities=True,
            persist_test_probabilities=True,
        )

        self.assertIsNotNone(model.training_probabilities)
        self.assertIsNotNone(model.probabilities)

    def test_evaluate_uses_preexisting_predictions(self):
        model = self._model()
        data = _make_data()
        model.training_predictions = np.zeros(len(data.y_train), dtype=int)
        model.predictions = np.zeros(len(data.y_test), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = {}

        times = {}
        model._evaluate_and_score(data, times=times)

        self.assertEqual(times["prediction_n"], len(data.y_test))
        self.assertNotIn("training_n", times)

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

        self.assertEqual(model.score_dict["defense_metric"], 1.0)
        self.assertEqual(model.score_dict["defense_application_time"], 0.25)

    def test_evaluate_probability_persistence_valueerror_sets_none(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace(
            predict_proba=lambda _X: (_ for _ in ()).throw(ValueError("bad probs")),
        )
        model._predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = {}

        model._evaluate_and_score(
            data,
            times={},
            persist_training_probabilities=True,
            persist_test_probabilities=True,
        )

        self.assertIsNone(model.training_probabilities)
        self.assertIsNone(model.probabilities)

    def test_evaluate_training_probability_typeerror_reraises(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace(
            predict_proba=lambda _X: (_ for _ in ()).throw(TypeError("unexpected")),
        )
        model._predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with self.assertRaises(TypeError):
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
        model._predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with self.assertRaises(TypeError):
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
        model._predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = {}

        model._evaluate_and_score(
            data,
            times={},
            persist_training_probabilities=True,
            persist_test_probabilities=True,
        )

        self.assertIsNotNone(model.training_probabilities)
        self.assertIsNotNone(model.probabilities)

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
            sample="split",
            random_state=42,
            classifier=True,
        )
        data()
        data.X_val = None
        data.y_val = None
        data.val_n = None

        model = self._model()
        model.score_mode = "val"
        model._train(data.X_train, data.y_train)
        model.scorer = lambda y_true, y_pred, mode="val", **kwargs: {"validation_accuracy": 1.0}

        times = {}
        model._evaluate_and_score(data, times=times)

        self.assertIsNotNone(data.X_val)
        self.assertIsNotNone(data.y_val)
        self.assertIn("validation_n", times)

    def test_evaluate_initializes_score_dict_when_none(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace(predict_proba=lambda X: np.zeros((len(X), 2)))
        model._predict = lambda X: np.zeros(len(X), dtype=int)
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}
        model.score_dict = None

        model._evaluate_and_score(data, times={})
        self.assertIsInstance(model.score_dict, dict)

    def test_evaluate_raises_when_train_predictions_none(self):
        model = self._model()
        data = _make_data()
        model._model = SimpleNamespace()
        model._predict = lambda _X: None
        model.scorer = lambda y_true, y_pred, mode="test", **kwargs: {"accuracy": 1.0}

        with self.assertRaises(TypeError):
            model._evaluate_and_score(data, times={})


class TestModelLoadOrTrainBranches(unittest.TestCase):
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
            loaded_obj._train = lambda X, y: setattr(loaded_obj, "training_time", 0.01) or setattr(loaded_obj, "training_n", len(y)) or setattr(loaded_obj, "_model", SimpleNamespace(predict=lambda s: np.zeros(len(s), dtype=int)))
            model._model = SimpleNamespace(predict=lambda s: np.zeros(len(s), dtype=int))
            out = model._load_or_train_model(data, str(p), {})

            self.assertIn("training_time", out)
            self.assertIn("training_n", out)

    def test_regression_scores_rethrows_unexpected_logloss_error(self):
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeRegressor",
            classifier=False,
            model_params={"max_depth": 2},
            scorer=None,
        )
        with patch("deckard.model.base.log_loss", side_effect=ValueError("other logloss error")):
            with self.assertRaises(ValueError):
                model._regression_scores(np.array([1.0, 2.0]), np.array([1.5, 2.5]))

    def test_load_or_train_with_none_model_and_missing_file_trains(self):
        data = _make_data()
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = None

        def _train(_X, y):
            model.training_time = 0.01
            model.training_n = len(y)
            model._model = SimpleNamespace(predict=lambda s: np.zeros(len(s), dtype=int))

        model._train = _train
        model.save = lambda filepath: None
        with tempfile.TemporaryDirectory() as td:
            out = model._load_or_train_model(data, str(Path(td) / "missing.pkl"), {})

        self.assertIn("training_time", out)
        self.assertIn("training_n", out)

    def test_load_or_train_raises_notfitted_when_model_missing_after_train(self):
        data = _make_data()
        model = ModelConfig(
            model_type="sklearn.tree.DecisionTreeClassifier",
            classifier=True,
            model_params={"max_depth": 2},
            scorer=None,
        )
        model._model = SimpleNamespace()

        def _train(_X, y):
            model.training_time = 0.01
            model.training_n = len(y)
            model._model = None

        model._train = _train
        with self.assertRaises(Exception):
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
            loaded_obj._train = lambda X, y: setattr(loaded_obj, "training_time", 0.01) or setattr(loaded_obj, "training_n", len(y)) or setattr(loaded_obj, "_model", SimpleNamespace(predict=lambda s: np.zeros(len(s), dtype=int)))
            loaded_obj._apply_defense = lambda _data: SimpleNamespace(predict=lambda s: np.zeros(len(s), dtype=int))
            loaded_obj._require_defense_pipeline = lambda: SimpleNamespace(
                resolve_stage=lambda **_kwargs: "skip",
            )

            model.load = lambda _fp: loaded_obj
            out = model._load_or_train_model(data, str(p), {})

            self.assertIn("training_time", out)
            self.assertIn("training_n", out)


if __name__ == "__main__":
    unittest.main()
