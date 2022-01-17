import logging
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pickle
import os
from deckard.base.model import Model
from deckard.base.data import Data
from deckard.base.experiment import Experiment

def return_result(scorer:str, filename = 'results.json')-> float:
    """
    Return the result of the experiment.
    scorer: the scorer to use
    filename: the filename to load the results from
    """
    import json
    assert os.path.isfile(filename), "{} does not exist".format(filename)
    # load json
    with open(filename) as f:
        results = json.load(f)
    # return the result
    return results[scorer.upper()]

def load_model(model_file:str = None) -> Pipeline:
    """
    Load a model from a pickle file.
    model_file: the pickle file to load the model from
    """
    from deckard.base.model import Model
    logging.debug("Loading model")
    # load the model
    model = pickle.load(open(model_file, 'rb'))
    logging.info("Loaded model")
    return model

def load_data(data_file:str = None) -> np.ndarray:
    """
    Load a data file.
    data_file: the data file to load
    """
    from deckard.base.data import Data
    logging.debug("Loading data")
    # load the data
    data = pickle.load(open(data_file, 'rb'))
    assert isinstance(data, Data), "Data is not an instance of Data. It is type: {}".format(type(data))
    logging.info("Loaded model")
    return data

def load_experiment(experiment_file:str = None) -> Experiment:
    """
    load an experiment from a file
    experiment_file: the file to load the experiment from
    """
    from deckard.base.experiment import Experiment
    logging.debug("Loading experiment")
    # load the experiment
    experiment = pickle.load(open(experiment_file, 'rb'))
    logging.info("Loaded experiment")
    return experiment

def push_json(json_file:str, remote_folder:str, remote_host:str, remote_user:str, remote_password:str) -> None:
    """
    Push a json file to a remote host.
    json_file: the json file to push
    remote_folder: the remote folder to push the json file to
    remote_host: the remote host to push the json file to
    remote_user: the remote user to push the json file to
    remote_password: the remote password to push the json file to
    """
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

def save_best_only(folder:str, exp_list:list, scorer:str, bigger_is_better:bool):
        """
        Save the best experiment only.
        folder: the folder to save the experiment
        exp_list: the list of experiments
        scorer: the scorer to use
        bigger_is_better: if True, the best experiment is the one with the highest score, otherwise the best is the one with the lowest score
        """
        folder = os.path.join(folder, 'best_preprocess')
        flag = False
        for exp in exp_list:
            exp.run()
            exp.save_results(folder)
            if flag == False:
                best = exp
                flag = True
            elif exp.scores[scorer] >= best.scores[scorer] and bigger_is_better:
                best = exp
        best.save_experiment(folder)

def save_all(folder:str, exp_list:list, scorer:str, bigger_is_better:bool):
        """
        Save all experiments.
        folder: the folder to save the experiments
        exp_list: the list of experiments
        scorer: the scorer to use
        bigger_is_better: if True, the best experiment is the one with the highest score, otherwise the best is the one with the lowest score
        """
        folder = os.path.join(folder, 'best_preprocess')
        flag = False
        for exp in exp_list:
            exp.run()
            exp.save_results(os.path.join(folder, exp.filename))
            exp.save_experiment(os.path.join(folder, exp.filename))
            if flag == False:
                best = exp
                flag = True
            elif exp.scores[scorer] >= best.scores[scorer] and bigger_is_better:
                best = exp
        best.save_experiment(folder)