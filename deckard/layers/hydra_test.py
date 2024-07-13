from omegaconf import DictConfig, OmegaConf
from pathlib import Path


working_dir = Path().cwd()
config_dir = "conf"
config_path = Path(working_dir, config_dir).as_posix()
config_file = "default"


def main():
    # Use sys calls to look for --working_dir, --config_dir, and --config_file
    args = sys.argv
    if "--working_dir" in args:
        working_dir = args[args.index("--working_dir") + 1]
        # remove working_dir from args
        args.pop(args.index("--working_dir"))
        args.pop(args.index(working_dir))
    else:
        working_dir = Path().cwd()
    if "--config_dir" in args:
        config_dir = args[args.index("--config_dir") + 1]
        # remove config_dir from args
        args.pop(args.index("--config_dir"))
        args.pop(args.index(config_dir))
    else:
        config_dir = "conf"
    if "--config_file" in args:
        config_file = args[args.index("--config_file") + 1]
        # remove config_file from args
        args.pop(args.index("--config_file"))
        args.pop(args.index(config_file))
    else:
        config_file = "default"
    if "--version_base" in args:
        version_base = args[args.index("--version_base") + 1]
        # remove version_base from args
        args.pop(args.index("--version_base"))
        args.pop(args.index(version_base))
    else:
        version_base = "1.3"

    @hydra.main(
        version_base=version_base,
        config_path=config_path,
        config_name=config_file,
    )
    def hydra_main(cfg: DictConfig) -> None:
        print(OmegaConf.to_yaml(cfg))
        return 0


if __name__ == "__main__":
    my_app()
