import unittest
import pandas as pd
from omegaconf import ListConfig
from deckard.impute import ResultImputerConfig

class TestResultImputerConfig(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "index" : [0, 1, 2, 3],
            "metric1": [1.0, 2.0, None, 4.0],
            "metric2": [None, 2.5, 3.5, 4.5],
            "metric3": [1.1, None, 3.3, 4.4]
        })

    def test_knn_imputer(self):
        config = ResultImputerConfig(
            metric_columns=ListConfig(["metric1", "metric2", "metric3"]),
            sig_figs=ListConfig([1,1,1]),
            imputer_type="knn",
            imputer_params={"n_neighbors": 2, "metric" : "nan_euclidean", "weights": "distance"}
        )
        result = config(self.df[config.metric_columns])
        self.assertFalse(result.isnull().any().any())

    def test_iterative_imputer(self):
        config = ResultImputerConfig(
            metric_columns=ListConfig(["metric1", "metric2", "metric3"]),
            sig_figs=ListConfig([1,1,1]),
            imputer_type="knn",
            imputer_params={"n_neighbors": 2, "metric" : "nan_euclidean", "weights": "distance"}
        )
        result = config(self.df)
        self.assertFalse(result.isnull().any().any())

    def test_invalid_imputer_type(self):
        config = ResultImputerConfig(
            metric_columns=ListConfig(["metric1"]),
            sig_figs=ListConfig([2]),
            imputer_type="unknown"
        )
        with self.assertRaises(ValueError):
            config(self.df)

    def test_mismatched_columns_and_sig_figs(self):
        with self.assertRaises(AssertionError):
            ResultImputerConfig(
                metric_columns=ListConfig(["metric1", "metric2"]),
                sig_figs=ListConfig([2]),
                imputer_type="knn"
            )

if __name__ == "__main__":
    unittest.main()