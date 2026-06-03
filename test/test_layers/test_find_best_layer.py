import tempfile
from pathlib import Path

import optuna
import pytest
import yaml

from deckard.layers.find_best import find_best_main


def _write_config(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "default.yaml").write_text(
        """
optimizers:
  - accuracy
direction:
  - maximize
model:
  lr: 0.0
  depth: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_multi_config(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "default.yaml").write_text(
        """
optimizers:
  - accuracy
  - loss
directions:
  - maximize
  - minimize
model:
  lr: 0.0
""".strip()
        + "\n",
        encoding="utf-8",
    )


class TestFindBestMain:
    def test_single_objective_uses_configured_optimizers(self):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name="single_best",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        trial_0 = study.ask()
        trial_0.suggest_float("model.lr", 0.1, 0.1)
        study.tell(trial_0, 0.8)

        trial_1 = study.ask()
        trial_1.suggest_float("model.lr", 0.2, 0.2)
        study.tell(trial_1, 0.95)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            _write_config(config_dir)
            output_file = Path(tmpdir) / "best.yaml"

            result = find_best_main(
                output_file=output_file.as_posix(),
                optuna_db=storage,
                study_name=study.study_name,
                config_dir=config_dir.as_posix(),
                config_name="default.yaml",
            )

            assert result["trial_number"] == 1
            payload = yaml.safe_load(output_file.read_text(encoding="utf-8"))
            assert payload["model"]["lr"] == pytest.approx(0.2)

    def test_multi_objective_subset_filter(self):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name="multi_best",
            storage=storage,
            directions=["maximize", "minimize"],
            load_if_exists=True,
        )

        with pytest.warns(optuna.exceptions.ExperimentalWarning):
            study.set_metric_names(["accuracy", "loss"])

        t0 = study.ask()
        t0.suggest_categorical("data", ["mnist", "cifar"])
        t0.suggest_float("model.lr", 0.01, 0.01)
        study.tell(t0, [0.8, 0.3])

        t1 = study.ask()
        t1.suggest_categorical("data", ["mnist", "cifar"])
        t1.suggest_float("model.lr", 0.02, 0.02)
        study.tell(t1, [0.9, 0.4])

        t2 = study.ask()
        t2.suggest_categorical("data", ["mnist", "cifar"])
        t2.suggest_float("model.lr", 0.03, 0.03)
        study.tell(t2, [0.99, 0.9])

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            _write_multi_config(config_dir)
            output_file = Path(tmpdir) / "best_multi.yaml"

            result = find_best_main(
                output_file=output_file.as_posix(),
                optuna_db=storage,
                study_name=study.study_name,
                config_dir=config_dir.as_posix(),
                config_name="default.yaml",
                subset="data=mnist",
            )

            assert result["trial_number"] in {0, 1, 2}
            payload = yaml.safe_load(output_file.read_text(encoding="utf-8"))
            assert payload["data"] == "mnist"

    def test_exclude_parameter_not_applied(self):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(
            study_name="exclude_param",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        trial = study.ask()
        trial.suggest_float("model.lr", 0.2, 0.2)
        trial.suggest_int("model.depth", 5, 5)
        study.tell(trial, 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            _write_config(config_dir)
            output_file = Path(tmpdir) / "best_exclude.yaml"

            find_best_main(
                output_file=output_file.as_posix(),
                optuna_db=storage,
                study_name=study.study_name,
                config_dir=config_dir.as_posix(),
                config_name="default.yaml",
                exclude="model.depth",
            )

            payload = yaml.safe_load(output_file.read_text(encoding="utf-8"))
            assert payload["model"]["lr"] == pytest.approx(0.2)
            # depth should stay from base config because excluded override was omitted
            assert payload["model"]["depth"] == 1
