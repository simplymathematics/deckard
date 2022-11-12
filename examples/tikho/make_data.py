from pathlib import Path
import yaml
from dvc.api import params_show

# from dvc.api import open

from data import Data
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--stage", default="data")
    args = parser.parse_args()
    data = params_show(args.config, stages=[args.stage])[args.stage]
    plots = params_show(args.config, stages=["plots"])["plots"]
    metrics = params_show(args.config, stages=["scorers"])["scorers"]
    files = params_show(args.config, stages=["files"])["files"]
    model = params_show(args.config, stages=["model"])["model"]
    yaml.add_constructor("!Data:", Data)
    data = yaml.load("!Data:\n" + str(data), Loader=yaml.FullLoader)
    namespace = data.load()
    path = Path(files["path"])
    path.mkdir(parents=True, exist_ok=True)
    plot_paths = data.visualize(data=namespace, files=files, plots=plots)
    Path(files["data_file"]).parent.mkdir(parents=True, exist_ok=True)
    data_paths = data.save(filename=files["data_file"], data=namespace)
