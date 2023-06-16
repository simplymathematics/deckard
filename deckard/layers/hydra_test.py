from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
import os

working_dir = os.getcwd()
config_path = Path(working_dir, "conf").as_posix()


@hydra.main(version_base=None, config_path=config_path, config_name="default")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    return 0


if __name__ == "__main__":
    my_app()
