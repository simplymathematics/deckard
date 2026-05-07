"""Named score-profile declarations and ConfigStore registrations."""

from pathlib import Path
from omegaconf import OmegaConf
from .base import DefaultModelScoreConfig, safe_store
from .data import DefaultDataScoreConfig


class DefaultClassifierDict:
    scorers = DefaultModelScoreConfig(classifier=True)


class DefaultRegressorDict:
    scorers = DefaultModelScoreConfig(classifier=False)


class DefaultDataClassificationDict:
    scorers = DefaultDataScoreConfig(classifier=True)


class DefaultDataRegressionDict:
    scorers = DefaultDataScoreConfig(classifier=False)


def _load_example_score_configs():
    """Load score configs from examples/sklearn/config/score and register with ConfigStore."""
    examples_dir = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "sklearn"
        / "config"
        / "score"
    )

    if not examples_dir.exists():
        return

    for yaml_file in sorted(examples_dir.glob("*.yaml")):
        try:
            config_name = yaml_file.stem
            cfg = OmegaConf.load(yaml_file)
            safe_store(group="score", name=config_name, node=cfg)
        except Exception:
            pass  # Silently skip any problematic configs


_load_example_score_configs()
