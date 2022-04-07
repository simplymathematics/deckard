import logging
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pickle
import os
from deckard.base.model import Model
from deckard.base.data import Data

logger = logging.getLogger(__name__)

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

def load_model(filename:str = None) -> Model:
    """
    Load a model from a pickle file.
    filename: the pickle file to load the model from
    """
    from deckard.base.model import Model
    logger.debug("Loading model")
    # load the model
    if filename.endswith('.pkl'):
        model = pickle.load(open(filename, 'rb'))
    elif filename.endswith('.h5'):
        from tensorflow.keras.models import load_model as keras_load_model, clone_model
        # TODO add support for tf, pytorch
        cloned_model = keras_load_model(filename)
        model = Model(cloned_model)
    else:
        raise ValueError("filename must be a pickle file or a keras model")
    logger.info("Loaded model")
    return model

def load_data(data_file:str = None) -> Data:
    """
    Load a data file.
    data_file: the data file to load
    """
    from deckard.base.data import Data
    logger.debug("Loading data")
    # load the data
    data = pickle.load(open(data_file, 'rb'))
    assert isinstance(data, Data), "Data is not an instance of Data. It is type: {}".format(type(data))
    logger.info("Loaded model")
    return data

def save_best_only(folder:str, exp_list:list, scorer:str, bigger_is_better:bool, name:str):
        """
        Save the best experiment only.
        folder: the folder to save the experiment
        exp_list: the list of experiments
        scorer: the scorer to use
        bigger_is_better: if True, the best experiment is the one with the highest score, otherwise the best is the one with the lowest score
        name: the name of the experiment
        """
        new_folder = os.path.join(folder, 'best_'+name)
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)
        flag = False
        for exp in exp_list:
            exp.run()
            exp.save_results(folder = new_folder)
            if flag == False:
                best = exp
                flag = True
            elif exp.scores[scorer] >= best.scores[scorer] and bigger_is_better:
                best = exp
        
        best.save_model(folder = new_folder)
        best.save_results(folder = new_folder)
        logger.info("Saved best experiment to {}".format(new_folder))
        logger.info("Best score: {}".format(best.scores[scorer]))

def save_all(folder:str, exp_list:list, scorer:str, bigger_is_better:bool, name:str):
        """
        Save all experiments.
        folder: the folder to save the experiments
        exp_list: the list of experiments
        scorer: the scorer to use
        bigger_is_better: if True, the best experiment is the one with the highest score, otherwise the best is the one with the lowest score
        """
        new_folder = os.path.join(folder, 'best_'+name)
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)
            logger.info("Created folder: " + new_folder)
        flag = False
        for exp in exp_list:
            exp.run()
            if not os.path.isdir(os.path.join(new_folder, exp.filename)):
                os.mkdir(os.path.join(new_folder, exp.filename))
                logger.info("Created folder: " + os.path.join(new_folder, exp.filename))
            exp.save_results(folder = os.path.join(new_folder, exp.filename))
            exp.save_model(folder = os.path.join(new_folder, exp.filename))
            if flag == False:
                best = exp
                flag = True
            elif exp.scores[scorer] >= best.scores[scorer] and bigger_is_better:
                best = exp
        best.filename = 'best_' + name
        best.save_model(folder = new_folder)
        best.save_results(folder = new_folder)
        logger.info("Saved best experiment to {}".format(new_folder))
        logger.info("Best score: {}".format(best.scores[scorer]))