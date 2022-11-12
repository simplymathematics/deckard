import os
import subprocess
import tempfile
import unittest
import warnings

import yaml
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class testSklearnPreprocess(unittest.TestCase):
    def setUp(self):
        self.path = os.path.abspath(tempfile.mkdtemp())
        self.file = "test_filename"
        self.here = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.join(self.here, "..", "..", "examples", "iris"))
        self.here = os.path.realpath(
            os.path.join(self.here, "..", "..", "examples", "iris"),
        )

    def test_cli(self):
        with open(os.path.join(self.here, "dvc.yaml")) as f:
            dictionary = yaml.load(f, Loader=yaml.FullLoader)
        command = dictionary["stages"]["preprocess"]["cmd"]
        subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def tearDown(self) -> None:
        from shutil import rmtree

        rmtree(self.path)


if __name__ == "__main__":
    unittest.main()
