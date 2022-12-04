import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
from pathlib import Path

filename = "params.yaml"
config_path = "conf"

@hydra.main(config_path=config_path, config_name="config", version_base="1.2") 
def parse(cfg: DictConfig, queue_path = "queue", filename = "params.yaml") -> Path:
    with open(Path(filename), "w") as f:
        yaml.dump(OmegaConf.to_object(cfg), f)
    return Path(filename)


if __name__ == "__main__":
    cfg = parse()
    with open(Path("params.yaml"), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    print(yaml.dump(params))
    input("Press enter to continue")
    # dump(yaml_, Path('params.yaml'))
    # input("Press Enter to continue...")
