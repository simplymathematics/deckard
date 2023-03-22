import logging
import unittest
import yaml
from pathlib import Path
import subprocess
from deckard.base.data import Data, config


names = ["regression"]
# TODO other names

logger = logging.basicConfig(level=logging.DEBUG)


class testExamples(unittest.TestCase):
    def setUp(self, config=config):
        yaml.add_constructor("!Data:", Data)
        self.data_document = "!Data:\n" + config
        self.data = yaml.load(self.data_document, Loader=yaml.Loader)
        self.path = Path(__file__).parent.parent.parent / "examples"
        self.examples = [Path(self.path, name) for name in names]
        self.data_file = Path(self.path, "data.pickle")
        self.model_file = Path(self.path, "model.pickle")

    def test_examples(self):
        cmd = "dvc repro --force"
        for example in self.examples:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                bufsize=1,
                cwd=example,
                shell=True,
            )
            for line in iter(p.stdout.readline, b""):
                print(line)
            p.stdout.close()
            exit_code = p.wait()
            self.assertEqual(exit_code, 0)

    def tearDown(self) -> None:
        from shutil import rmtree

        if Path("model").is_dir():
            rmtree("model")
        if Path("data").is_dir():
            rmtree("data")
        if Path("reports").is_dir():
            rmtree("reports")
        del self.path
