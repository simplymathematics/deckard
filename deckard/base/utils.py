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
from copy import deepcopy


def check_if_supervised_or_unsupervised(model)-> bool:
    """
    Returns true if supervised, false if unsupervised. 
    """
    if hasattr(model, 'fit_predict'):
        model_type = False
        logging.info("Model is unsupervised")
    elif hasattr(model, 'fit') and not hasattr(model, 'fit_predict'):
        model_type = True
        logging.info("Model is supervised")
    else:
        model_type = None
        logging.error("Model is not regressor or classifier")
        raise ValueError("Model is not regressor or classifier")
        # exit
        sys.exit(1)  
    return model_type

def check_estimator(estimator) -> None:
    logging.debug("Checking if model is estimator")
    # check if model is estimator
    result = isinstance(estimator, BaseEstimator)
    if result == True:
        logging.debug("Model is estimator: %s" % result)
    else:
        logging.error("Model is not estimator: %s" % result)
        raise ValueError("Model is not estimator")
        # exit
        sys.exit(1) 
    return result

def check_if_regressor_or_classifier(model)-> bool:
    """
    Returns true if regressor, false if classifier.
    """
    if is_regressor(model):
        model_type = True
        logging.info("Model is regressor")
    elif is_classifier(model):
        model_type = False
        logging.info("Model is classifier")
    else:
        model_type = None
        logging.info("Model was not detected as either a classifier or regressor.")
        raise ValueError("Model was not detected as either a classifier or regressor.")
    return model_type



def load_csv(dataset:str=None) -> np.ndarray:
    """ Loads csv and assumes last column is the label vector."""
    import os
    if not os.path.isfile(dataset) and dataset.endswith('.csv'):
        raise ValueError("File does not exist")
    else:
        new_data = {'data' : None, 'target' : None}
        logging.warning("Assuming that the target is the last column in the csv.")
        new_data.data = pd.read_csv(dataset)
        new_data.target = new_data.iloc[:, -1]
        new_data = new_data.data.iloc[:, :-1]
    return new_data

def print_results_to_terminal(scores:dict) -> None:
    # print results to terminal
    if len(scores) == 0:
        logging.error("No scores to print")
        raise ValueError("No scores to print")
        sys.exit(1)
    elif len(scores.keys()) == 1:
        logging.info("Score %s: %s" % (list(scores.keys())[0], scores[list(scores.keys())[0]]))
    else:
        for scorer in scores:
            logging.info("Score %s: %s" % (scorer, scores[scorer]))
    return None

def save_model(model:BaseEstimator, model_file:str, folder:str = None) -> None:
    assert os.path.exists(folder), "Folder does not exist"
    assert isinstance(model, BaseEstimator), "Model is not an estimator"
    logging.debug("Saving model")
    model_file = os.path.join(folder, model_file)
    # save the model
    pickle.dump(model, open(model_file, 'wb'))
    assert os.path.exists(model_file), "Saving model unsuccessful"
    logging.info("Saved model to {}".format(model_file))

def save_data(data:np.ndarray, data_file:str, folder:str = None) -> None:
    assert os.path.exists(folder), "Folder does not exist"
    logging.debug("Saving data")
    data_file = os.path.join(folder, data_file)
    # save the data
    pickle.dump(data, open(data_file, 'wb'))
    assert os.path.exists(data_file), "Saving data unsuccessful"
    logging.info("Saved data to {}.".format(data_file))

def return_result(scorer:str, filename = 'results.json')-> float:
    import json
    assert os.path.isfile(filename), "{} does not exist".format(filename)
    # load json
    with open(filename) as f:
        results = json.load(f)
    # return the result
    return results[scorer.upper()]

def save_results(results:dict, experiment:Experiment, identifier: str = None, folder:str = None) -> None:
    import json
    assert os.path.exists(folder), "Folder does not exist"
    assert identifier is not None, "Identifier must be specified."
    logging.debug("Saving results")
    score_file = os.path.join(folder, "results.json")
    data_file = os.path.join(folder,"data_params.json")
    model_file = os.path.join(folder, "model_params.json")

    results = pd.Series(results.values(), name =  identifier, index = results.keys())
    data_params = pd.Series(experiment.data.params.update({'id_' : experiment.filename}), name = identifier)
    model_params = pd.Series(experiment.model.params.update({'id_': experiment.filename}), name = identifier)
    if hasattr(experiment.data, "attack_params"):
        attack_file = os.path.join(folder, "attack_params.json")
        attack_params = pd.Series(experiment.data.attack_params.update({'id_': experiment.filename}), name = identifier)
        attack_params.to_json(attack_file)
    if hasattr(experiment.model.model, "cv_results_"):
        cv_file = os.path.join(folder, "cv_results.json")
        cv_results = pd.Series(experiment.model.model.cv_results_.update({'id_':experiment.filename}), name = identifier)
        cv_results.to_json(cv_file)
    results.to_json(score_file)
    data_params.to_json(data_file)
    model_params.to_json(model_file)
    assert os.path.exists(score_file), "Saving results unsuccessful"
    assert os.path.exists(data_file), "Saving data_params unsuccessful"
    assert os.path.exists(model_file), "Saving model_params unsuccessful"
    logging.info("Saved results")
    return None




def load_model(model_file:str = None) -> BaseEstimator:
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

def checkpoint( filename:str, result_folder:str, experiment:Experiment) -> None:
    data = experiment.data
    pipeline = experiment.model.model
    if result_folder is None:
        result_folder = os.path.dirname(filename)
    assert isinstance(filename, str), "filename must be specified"
    assert isinstance(pipeline, BaseEstimator), "pipeline must be a BaseEstimator"
    assert isinstance(experiment.scores, dict), "scores must be a dictionary"
    logging.debug("Filename is: " + filename)
    result_folder = os.path.join(result_folder, filename)
    logging.debug("Folder is: " + result_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_model(pipeline, model_file ='model.pkl', folder = result_folder)
    # save the data
    save_data(data, data_file='data.pkl', folder = result_folder)
    # dump results to json file
    save_results(experiment.scores, experiment, identifier=filename, folder = result_folder)
    return None

def load_checkpoint(folder:str, model:str, data:str) -> tuple:
    assert isinstance(folder, str), "results_folder must be specified"
    logging.debug("Folder is: " + folder)
    # load the model
    model = load_model(os.path.join(folder, model))
    # load the data
    data = load_data(os.path.join(folder, data))
    return data, model

# push json to remote server
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
    logging.debug("remote_password: " + remote_password)
    # push json to remote server
    cmd = 'scp ' + json_file + ' ' + remote_user + '@' + remote_host + ':' + remote_folder
    logging.debug("cmd: " + cmd)
    os.system(cmd)
    logging.debug("Pushed json to remote server")
    return None


if __name__ == '__main__':
    # setting logging to debup
    logging.basicConfig(level=logging.DEBUG)
    import sys
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    assert check_if_regressor_or_classifier(model) == False
    assert check_estimator(model) == True
    logging.info("All tests passed")
    sys.exit(0)
