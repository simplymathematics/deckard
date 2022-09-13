import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable, Union
import dvc.api
import numpy as np
import yaml
from deckard.base import Data, Model
from tqdm import tqdm
from yellowbrick.exceptions import YellowbrickValueError

logger = logging.getLogger(__name__)

def visualise_classifier_experiment(args:argparse.Namespace, path:[str,Path] = Path("."), type:str = 'ROC_AUC') -> None:    
    """
    Visualise the results of a single experiment.
    :param args: a dictionary read at run-time
    :param path: the path to the experiment
    :param type: the type of visualisation to perform
    """
    assert isinstance(args, argparse.Namespace), "args must be a dictionary-like object"
    assert isinstance(path, (str, Path)), "path must be a string or a Path object"
    assert isinstance(type, str), "type must be a string"
    assert Path(args.model_folder).exists(), "Problem finding model folder: {}".format(args.model_folder)
    assert Path(args.data_folder).exists(), "Problem finding data folder: {}".format(args.data_folder)
    assert Path(args.model_folder, args.model_file).exists(), "Problem finding model file: {}".format(Path(args.model_folder, args.model_file))
    assert Path(args.data_folder, args.data_file).exists(), "Problem finding data file: {}".format(Path(args.data_folder, args.data_file))
    data = Data(Path(args.data_folder, args.data_file))
    model = Model(Path(args.model_folder, args.model_file), model_type = args.model_type, art = args.art)
    try:
        classes = list(set(data.y_train))
        y_train = data.y_train
        y_test = data.y_test
    except:
        y_train = [np.argmax(y) for y in data.y_train]
        y_test = [np.argmax(y) for y in data.y_test]
        classes = list(set(y_train))
    if args.art == True:
        logger.info("Using ART model")
        viz_mod = model.model
    else:
        viz_mod = model
    assert isinstance(viz_mod, (Callable, Model)), "model must be a callable object. It is type {}".format(type(viz_mod))
    if type == 'ROC_AUC':
        from yellowbrick.classifier import ROCAUC
        func = ROCAUC(viz_mod.model, classes=classes)
        outpath = Path(args.root_folder, args.plot_folder, "ROC_AUC.pdf")
        outpath = outpath.resolve()
        outpath.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise NotImplementedError("Visualisation type {} not implemented".format(type))
    func.fit(data.X_train, y_train)
    func.score(data.X_test, y_test)
    func.show(outpath = outpath) 
    print("Saving visualisation to {}".format(outpath))
    input("Press enter to continue")
    return outpath.resolve()
        
if __name__ == '__main__':
    # Initialize logger
    # command line arguments
    parser = argparse.ArgumentParser(description='Run a preprocessor on a dataset')
    parser.add_argument('--layer_name','-l', type=str, default = "visualise", help='Name of layer, e.g. "attack"')
    parser.add_argument('--config','-c', type=str, default = None, help='Path to the attack config file')
    args = parser.parse_args()
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    logger.info(f"Running {cli_args.layer_name} with args: {args}")
    # assert Path(args.config).exists(), f"Config file {args.config} does not exist"
    assert Path(args.root_folder).exists(), f"Root folder {args.root_folder} does not exist"
    assert Path(args.data_folder, args.data_file).exists(), f"Data file {args.data_file} does not exist in {args.data_folder}"
    assert Path(args.model_folder, args.model_file).exists(), f"Model file {args.model_file} does not exist in {args.model_folder}"
    output = visualise_classifier_experiment(args, path = args.input_folder)
    assert Path(args.plot_folder).exists(), f"Plot folder {args.plot_folder} does not exist"
    assert Path(output).exists(), f"Output file {output} does not exist"

  

