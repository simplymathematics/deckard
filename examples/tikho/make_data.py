from pathlib import Path
import numpy as np
import yaml
from dvc.api import params_show
# from dvc.api import open

from data import Data
import argparse





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    args = parser.parse_args()
    params = params_show(args.config)
    if "plots" in params:
        plots = params.pop('plots')
    if "scorers" in params:
        metrics = params.pop('scorers')
    if "files" in params:
        files = params.pop('files')
    if "model" in params:
        _ = params.pop('model')
    if "data" in params:
        data = params.pop("data")
    else:
        raise ValueError("No data specified in params.yaml")
    yaml.add_constructor('!Data:', Data)
    data = yaml.load("!Data:\n" + str(data), Loader=yaml.FullLoader)
    namespace = data.load()
    path = Path(files['path'])
    path.mkdir(parents=True, exist_ok=True)
    plot_paths = data.visualize(data = namespace, files = files, plots = plots)
    data_paths = data.save(filename = path/files['data_file'], data =  namespace)

        
    
    
