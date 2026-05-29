import shutil
import tempfile
import time
from pathlib import Path

from deckard.file import CanonFileHandler, FileConfig, FileConfigError
import pytest


class TestFileConfig:
    def setup_method(self):
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

    def teardown_method(self):
        # Remove temporary directories
        for d in self.temp_dirs.values():
            shutil.rmtree(d, ignore_errors=True)
        # Remove any additional test_models2 directory if created
        if Path("test_models2").exists():
            shutil.rmtree("test_models2", ignore_errors=True)

    def test_file_paths_contain_experiment_name(self):
        exp_name = self.config.replace["{experiment_name}"]
        assert exp_name in self.config.model_file
        assert exp_name in self.config.data_file
        assert exp_name in self.config.log_file

    def test_file_dict(self):
        assert "model_file" in self.config
        assert "data_file" in self.config
        assert "log_file" in self.config
        assert "score_file" in self.config
        assert self.config["model_file"].endswith(".pkl")
        assert self.config["data_file"].endswith(".csv")
        assert self.config["log_file"].endswith(".log")
        assert self.config["score_file"].endswith("_score.txt")

    def test_hash_placeholder(self):
        attack_file = self.config["attack_file"]
        assert attack_file != "{hash}"

    def test_timestamp_placeholder(self):
        cfg = FileConfig(
            replace=self.config.replace,
            attack_file="{timestamp}",
        )
        assert cfg.attack_file != "{timestamp}"

    def test_unused_directory_removed(self):
        config = FileConfig()
        with pytest.raises(AttributeError):
            getattr(config, "foo")

    def test_iter_and_len_reflect_active_file_fields(self):
        config = FileConfig(model_file="m.pkl", score_file="s.json")
        keys = list(iter(config))

        assert "model_file" in keys
        assert "score_file" in keys
        assert len(config) >= 2

    def test_canon_handler_parses_and_replaces_placeholders(self):
        handler = CanonFileHandler()
        template = "run-{num}-{hash}-{timestamp}.json"
        parsed = handler.parse_placeholders(template)
        assert "{num}" in parsed
        assert "{hash}" in parsed
        rendered = handler.replace_placeholders(
            template,
            {"{num}": "7", "{hash}": "abc", "{timestamp}": "now"},
        )
        assert rendered == "run-7-abc-now.json"

    def test_handler_validate_keys_rejects_unknown(self):
        handler = CanonFileHandler()
        with pytest.raises(FileConfigError):
            handler.validate_keys({"unknown_file": "x.txt"})

    def test_file_config_disk_status_reports_existing_paths(self):
        temp_file = Path(self.temp_dirs["data_directory"]) / "d.csv"
        temp_file.write_text("x,y\n1,2\n", encoding="utf-8")

        config = FileConfig(data_file=str(temp_file), model_file="/tmp/missing.pkl")
        status = config.disk_status()
        assert status["data_file"]
        assert not status["model_file"]

    def test_to_init_dict_preserves_templates_and_omits_handler(self):
        payload = self.config.to_init_dict()

        assert payload["model_file"] == "{experiment_name}.pkl"
        assert payload["attack_file"] == "{hash}.pkl"
        assert payload["replace"]["{experiment_name}"] == "foo"
        assert "handler" not in payload

    def test_hash_artifact_paths_only_updates_runtime_values(self):
        config = FileConfig(
            data_file="outputs/raw.csv",
            score_file="outputs/scores.json",
        )

        config.hash_artifact_paths("abc123", exclude={"score_file"})

        assert config.data_file == "outputs/abc123.csv"
        assert config.score_file == "outputs/scores.json"
        assert config.to_init_dict()["data_file"] == "outputs/raw.csv"
