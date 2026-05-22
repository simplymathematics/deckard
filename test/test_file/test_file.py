import shutil
import tempfile
import time
import unittest
from pathlib import Path

from deckard.file import CanonFileHandler, FileConfig, FileConfigError


class TestFileConfig(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for testing
        self.temp_dirs = {}
        for d in [
            "result_directory",
            "model_directory",
            "data_directory",
            "log_directory",
        ]:
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs[d] = temp_dir

        self.config = FileConfig(
            log_file="{experiment_name}.log",
            model_file="{experiment_name}.pkl",
            data_file="{experiment_name}.csv",
            score_file="{experiment_name}_score.txt",
            attack_file="{hash}.pkl",
            replace={
                "{hash}": 10,
                "{timestamp}": str(time.time()),
                "{experiment_name}": "foo",
            },
        )

    def tearDown(self):
        # Remove temporary directories
        for d in self.temp_dirs.values():
            shutil.rmtree(d, ignore_errors=True)
        # Remove any additional test_models2 directory if created
        if Path("test_models2").exists():
            shutil.rmtree("test_models2", ignore_errors=True)

    def test_file_paths_contain_experiment_name(self):
        exp_name = self.config.replace["{experiment_name}"]
        self.assertIn(exp_name, self.config.model_file)
        self.assertIn(exp_name, self.config.data_file)
        self.assertIn(exp_name, self.config.log_file)

    def test_file_dict(self):
        self.assertIn("model_file", self.config)
        self.assertIn("data_file", self.config)
        self.assertIn("log_file", self.config)
        self.assertIn("score_file", self.config)
        self.assertTrue(self.config["model_file"].endswith(".pkl"))
        self.assertTrue(self.config["data_file"].endswith(".csv"))
        self.assertTrue(self.config["log_file"].endswith(".log"))
        self.assertTrue(self.config["score_file"].endswith("_score.txt"))

    def test_hash_placeholder(self):
        attack_file = self.config["attack_file"]
        self.assertNotEqual(attack_file, "{hash}")

    def test_timestamp_placeholder(self):
        cfg = FileConfig(
            replace=self.config.replace,
            attack_file="{timestamp}",
        )
        self.assertNotEqual(cfg.attack_file, "{timestamp}")

    def test_unused_directory_removed(self):
        config = FileConfig()
        with self.assertRaises(AttributeError):
            getattr(config, "foo")

    def test_iter_and_len_reflect_active_file_fields(self):
        config = FileConfig(model_file="m.pkl", score_file="s.json")
        keys = list(iter(config))

        self.assertIn("model_file", keys)
        self.assertIn("score_file", keys)
        self.assertGreaterEqual(len(config), 2)

    def test_canon_handler_parses_and_replaces_placeholders(self):
        handler = CanonFileHandler()
        template = "run-{num}-{hash}-{timestamp}.json"
        parsed = handler.parse_placeholders(template)
        self.assertIn("{num}", parsed)
        self.assertIn("{hash}", parsed)
        rendered = handler.replace_placeholders(
            template,
            {"{num}": "7", "{hash}": "abc", "{timestamp}": "now"},
        )
        self.assertEqual(rendered, "run-7-abc-now.json")

    def test_handler_validate_keys_rejects_unknown(self):
        handler = CanonFileHandler()
        with self.assertRaises(FileConfigError):
            handler.validate_keys({"unknown_file": "x.txt"})

    def test_file_config_disk_status_reports_existing_paths(self):
        temp_file = Path(self.temp_dirs["data_directory"]) / "d.csv"
        temp_file.write_text("x,y\n1,2\n", encoding="utf-8")

        config = FileConfig(data_file=str(temp_file), model_file="/tmp/missing.pkl")
        status = config.disk_status()
        self.assertTrue(status["data_file"])
        self.assertFalse(status["model_file"])
