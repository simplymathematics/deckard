import unittest
from tempfile import TemporaryDirectory
import os
import sys
from deckard import ExperimentConfig, DataConfig, ModelConfig

from deckard.__main__ import (
    optimize,
    initialize_config,
    parse_optional_args,
    parse_files_from_optional_args,
    handle_default_module,
    validate_module_and_files,
    validate_files,
    handle_other_modules,
    module_file_dict,
    main,
)

class TestMain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_cfg = ExperimentConfig(
            data=DataConfig(),
            model=ModelConfig(),
            attack = None,
            optimizers = ["accuracy"],
        )
        cls.mock_dict = cls.mock_cfg.to_dict()

    def test_optimize(self):
        result = optimize(self.mock_cfg, target="deckard.experiment.ExperimentConfig", return_runner=False)
        self.assertIsInstance(result, list)
        for entry in result:
            self.assertIsInstance(entry, (float, int))
            
    def test_initialize_config(self):
        runner = initialize_config(self.mock_dict, target="deckard.experiment.ExperimentConfig")
        self.assertIsNotNone(runner)
        self.assertEqual(runner.__class__.__name__, "ExperimentConfig")

    def test_parse_optional_args(self):
        sys.argv = ["script_name", "arg1=data", "data"]
        optional_args, modules = parse_optional_args()
        self.assertIn("data", modules)
        self.assertIn("arg1=data", optional_args)

    def test_parse_files_from_optional_args(self):
        optional_args = ["data_config_file=config.yaml", "model_config_file=model.yaml"]
        module = "data"
        files = parse_files_from_optional_args(optional_args, module)
        self.assertIn("data_config_file", files)
        self.assertEqual(files["data_config_file"], "config.yaml")

    def test_handle_default_module(self):
        os.environ["DECKARD_CONFIG_DIR"] = "/tmp"
        with self.assertRaises(SystemExit):
            handle_default_module("/invalid/path")

    def test_validate_module_and_files(self):
        files = {"data_config_file": "config.yaml"}
        module = "data"
        result = validate_module_and_files(module, files)
        self.assertEqual(result, "config.yaml")

    def test_validate_files(self):
        files = ["data_config_file"]
        supported_files = ["data_config_file", "model_config_file"]
        try:
            validate_files(files, supported_files, "data")
        except SystemExit:
            self.fail("validate_files raised SystemExit unexpectedly!")

            
if __name__ == "__main__":
    unittest.main()