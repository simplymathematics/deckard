import logging
from sklearn.base import is_classifier, is_regressor, BaseEstimator
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pickle
import os
from deckard.base.model import Model
from deckard.base.data import Data
from deckard.base.experiment import Experiment

def return_result(scorer:str, filename = 'results.json')-> float:
    import json
    assert os.path.isfile(filename), "{} does not exist".format(filename)
    # load json
    with open(filename) as f:
        results = json.load(f)
    # return the result
    return results[scorer.upper()]

def load_model(model_file:str = None) -> Pipeline:
    from deckard.base.model import Model
    logging.debug("Loading model")
    # load the model
    model = pickle.load(open(model_file, 'rb'))
    logging.info("Loaded model")
    return model

def load_data(data_file:str = None) -> np.ndarray:
    from deckard.base.data import Data
    logging.debug("Loading data")
    # load the data
    data = pickle.load(open(data_file, 'rb'))
    assert isinstance(data, Data), "Data is not an instance of Data. It is type: {}".format(type(data))
    logging.info("Loaded model")
    return data

def load_experiment(experiment_file:str = None) -> Experiment:
    from deckard.base.experiment import Experiment
    logging.debug("Loading experiment")
    # load the experiment
    experiment = pickle.load(open(experiment_file, 'rb'))
    logging.info("Loaded experiment")
    return experiment

def push_json(json_file:str, remote_folder:str, remote_host:str, remote_user:str, remote_password:str) -> None:
    assert isinstance(json_file, str), "json_file must be specified"
    assert isinstance(remote_folder, str), "remote_folder must be specified"
    assert isinstance(remote_host, str), "remote_host must be specified"
    assert isinstance(remote_user, str), "remote_user must be specified"
    assert isinstance(remote_password, str), "remote_password must be specified"
    logging.debug("Pushing json to remote server")
    logging.debug("json_file: " + json_file)
    logging.debug("remote_folder: " + remote_folder)
    logging.debug("remote_host: " + remote_host)
    logging.debug("remote_user: " + remote_user)
    logging.debug("remote_password: " + "*************************")
    cmd = 'scp ' + json_file + ' ' + remote_user + '@' + remote_host + ':' + remote_folder
    logging.debug("cmd: " + cmd)
    os.system(cmd)
    logging.debug("Pushed json to remote server")
    return None
