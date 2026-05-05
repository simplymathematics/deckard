import unittest
from pathlib import Path
from deckard.file import FileConfig
import tempfile
import shutil
import time
import os


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
            replace={
                "{hash}": "null",
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
        files_dict = self.config._file_dict
        self.assertIn("model_file", files_dict)
        self.assertIn("data_file", files_dict)
        self.assertIn("log_file", files_dict)
        self.assertIn("score_file", files_dict)
        self.assertTrue(files_dict["model_file"].endswith(".pkl"))
        self.assertTrue(files_dict["data_file"].endswith(".csv"))
        self.assertTrue(files_dict["log_file"].endswith(".log"))
        self.assertTrue(files_dict["score_file"].endswith("_score.txt"))

    def test_hash_placeholder(self):
        config = FileConfig(replace=self.config.replace, attack_file="{hash}")
        file_dict = config._file_dict
        attack_file = file_dict.get("attack_file", KeyError)
        self.assertNotEqual(attack_file, "{hash}")
        self.assertIsInstance(int(attack_file), int)

    def test_timestamp_placeholder(self):
        config = FileConfig(
            replace=self.config.replace,
            attack_file="{timestamp}",
        )
        file_dict = config._file_dict
        attack_file = file_dict.get("attack_file", KeyError)
        self.assertNotEqual(attack_file, "{timestamp}")

    def test_unused_directory_removed(self):
        config = FileConfig()
        with self.assertRaises(AttributeError):
            getattr(config, "foo")

    def test_generate_file_hash_returns_md5_hex(self):
        tmp_file = Path(self.temp_dirs["data_directory"]) / "payload.bin"
        tmp_file.write_bytes(b"deckard-hash-test")

        digest = self.config.generate_file_hash(str(tmp_file))

        self.assertEqual(len(digest), 32)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in digest))

    def test_get_hydra_job_num_reads_env_and_defaults(self):
        old = os.environ.get("HYDRA_JOB_NUM")
        try:
            os.environ["HYDRA_JOB_NUM"] = "7"
            self.assertEqual(self.config.get_hydra_job_num(), "7")
            del os.environ["HYDRA_JOB_NUM"]
            self.assertEqual(self.config.get_hydra_job_num(), "0")
        finally:
            if old is None:
                os.environ.pop("HYDRA_JOB_NUM", None)
            else:
                os.environ["HYDRA_JOB_NUM"] = old

    def test_replace_placeholders_handles_empty_and_custom_mapping(self):
        config = FileConfig(
            data_file="",
            replace=[("{exp}", "demo")],
            attack_file="out/{exp}/#/*.json",
        )

        self.assertIsNone(config._replace_placeholders(""))
        self.assertIn("demo", config.attack_file)
        self.assertIn("0", config.attack_file)

    def test_iter_and_len_reflect_active_file_fields(self):
        config = FileConfig(model_file="m.pkl", score_file="s.json")
        keys = list(iter(config))

        self.assertIn("model_file", keys)
        self.assertIn("score_file", keys)
        self.assertGreaterEqual(len(config), 2)
