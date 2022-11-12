import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

PATH = "params.yaml"


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def parse(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    input("YAML above, press enter to continue")
    print(OmegaConf.to_container(cfg))
    input("Container above, press enter to continue")
    print(OmegaConf.to_object(cfg))
    input("Object above, press enter to continue")
    with open(PATH, "w") as f:
        yaml.dump(OmegaConf.to_object(cfg), f)


if __name__ == "__main__":
    cfg = parse()
    input("Press enter to continue")
    # dump(yaml_, Path('params.yaml'))
    # input("Press Enter to continue...")
