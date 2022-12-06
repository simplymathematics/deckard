import logging
import unittest
import warnings
from pathlib import Path

from sklearn.base import BaseEstimator

from deckard.base.utils import factory, load_from_tup, parse_config_for_libraries

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


class testUtils(unittest.TestCase):
    def setUp(self):
        here = Path(__file__).parent
        self.path = here / "configs"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.factory = {
            "module_class_string": "sklearn.linear_model.LogisticRegression",
            "super_cls": BaseEstimator,
            "penalty": "l2",
        }
        self.obj_gen = ("sklearn.linear_model.LogisticRegression", {"penalty": "l2"})
        self.regex = "params.yaml"
        self.file = Path(self.path) / self.regex
        self.output = "requirements.txt"
        assert (
            self.file.exists()
        ), f"File {self.file} does not exist in {self.path}. Found {list(Path(self.path).iterdir())}"

    def test_load_from_tuple(self):
        obj = load_from_tup(self.obj_gen)
        self.assertIsInstance(obj, BaseEstimator)

    def test_factory(self):
        obj = factory(**self.factory)
        self.assertIsInstance(obj, BaseEstimator)

    def test_parse_config_for_libraries(self):
        (libraries, path) = parse_config_for_libraries(
            path=self.path,
            regex=self.regex,
            output=self.output,
        )
        for lib in libraries:
            test_list = ["sklearn", "art"]
            self.assertTrue(lib in test_list)
        with open(path, "r") as f:
            for count, _ in enumerate(f):
                pass
        self.assertEqual(count + 1, len(libraries))
