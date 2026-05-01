import unittest
from deckard.score import (
    ScorerConfig,
    ScorerDictConfig,
    DefaultClassifierDict,
    DefaultRegressorDict,
    survival_concordance_score,
    survival_aic_score,
    survival_bic_score,
)
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score


class TestScorerConfig(unittest.TestCase):
    def test_scorer_config_initialization(self):
        config = ScorerConfig(
            score_name="accuracy",
            score_function=accuracy_score,
            score_params={"normalize": True},
        )
        self.assertEqual(config.score_name, "accuracy")
        self.assertTrue(callable(config.score_function))

    def test_scorer_config_callable(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        config = ScorerConfig(
            score_name="accuracy",
            score_function=accuracy_score,
            score_params={},
        )
        score = config(y_true=y_true, y_pred=y_pred)
        self.assertEqual(score, accuracy_score(y_true, y_pred))

    def test_scorer_config_swap(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        config = ScorerConfig(
            score_name="accuracy",
            score_function=accuracy_score,
            score_params={},
        )
        score_swap = config(y_true=y_true, y_pred=y_pred, swap=True)
        score_normal = config(y_true=y_pred, y_pred=y_true)
        self.assertEqual(score_swap, score_normal)

    def test_scorer_config_with_additional_params(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        config = ScorerConfig(
            score_name="precision",
            score_function=precision_score,
            score_params={"average": "binary", "zero_division": 0},
        )
        score = config(y_true=y_true, y_pred=y_pred)
        self.assertEqual(
            score,
            precision_score(y_true, y_pred, average="binary", zero_division=0),
        )


class TestScorerDictConfig(unittest.TestCase):
    def test_scorer_dict_config_initialization_and_call(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        scorer_dict = ScorerDictConfig(
            scorers={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function=accuracy_score,
                    score_params={},
                ),
                "mse": ScorerConfig(
                    score_name="mse",
                    score_function=mean_squared_error,
                    score_params={},
                ),
            },
        )
        scores = scorer_dict(y_true=y_true, y_pred=y_pred)
        self.assertIn("accuracy", scores)
        self.assertIn("mse", scores)
        self.assertEqual(scores["accuracy"], accuracy_score(y_true, y_pred))
        self.assertEqual(scores["mse"], mean_squared_error(y_true, y_pred))

    def test_scorer_dict_config_get_callables(self):
        scorer_dict = ScorerDictConfig(
            scorers={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function=accuracy_score,
                    score_params={},
                ),
            },
        )
        callables = scorer_dict.get_callables()
        self.assertIn("accuracy", callables)
        self.assertTrue(callable(callables["accuracy"]))


class TestDefaultScorerDicts(unittest.TestCase):
    def test_default_classifier_dict(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        scores = DefaultClassifierDict.scorers(y_true=y_true, y_pred=y_pred)
        self.assertIn("accuracy", scores)
        self.assertIn("precision", scores)
        self.assertIn("recall", scores)
        self.assertIn("f1", scores)
        self.assertIn("roc_auc", scores)
        self.assertIn("log_loss", scores)

    def test_default_regressor_dict(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.2, 3.8]
        scores = DefaultRegressorDict.scorers(y_true=y_true, y_pred=y_pred)
        self.assertIn("mse", scores)
        self.assertIn("mae", scores)
        self.assertIn("r2", scores)

    def test_default_classifier_dict_with_empty_predictions(self):
        y_true = []
        y_pred = []
        with self.assertRaises(ValueError):
            DefaultClassifierDict.scorers(y_true=y_true, y_pred=y_pred)

    def test_default_regressor_dict_with_empty_predictions(self):
        y_true = []
        y_pred = []
        with self.assertRaises(ValueError):
            DefaultRegressorDict.scorers(y_true=y_true, y_pred=y_pred)


class TestSurvivalScorers(unittest.TestCase):
    class _MockFitter:
        def __init__(self):
            self.concordance_index_ = 0.71
            self.AIC_ = 123.4
            self.log_likelihood_ = -50.0
            self.params_ = [1.0, 2.0, 3.0]

    def test_survival_concordance_score(self):
        fitter = self._MockFitter()
        score = survival_concordance_score(y_true=[1, 2, 3], y_pred=fitter)
        self.assertEqual(score, fitter.concordance_index_)

    def test_survival_aic_score(self):
        fitter = self._MockFitter()
        score = survival_aic_score(y_true=[1, 2, 3], y_pred=fitter)
        self.assertEqual(score, fitter.AIC_)

    def test_survival_bic_score_computed(self):
        fitter = self._MockFitter()
        score = survival_bic_score(y_true=[1, 2, 3, 4, 5], y_pred=fitter)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)


if __name__ == "__main__":
    unittest.main()
