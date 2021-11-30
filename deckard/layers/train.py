from deckard.base import Data
from deckard.base import Model
from deckard.base import Experiment
import logging
import os
from time import process_time_ns
from sklearn.base import BaseEstimator
from deckard.base import checkpoint, load_data
import sys
import numpy as np
import yaml
import importlib
# import is_regressor from sklearn
import logging
from sklearn.model_selection import GridSearchCV, ParameterGrid

from deckard.base import return_result

def parse_gridsearch_from_yml(filename:str = None, obj_type = BaseEstimator, cv = 10 ) -> dict:
    search_list = list()
    CROSS_VALIDATION = 5
    LOADER = yaml.FullLoader
    # check if the file exists
    params = dict()
    if not os.path.isfile(filename):
        raise ValueError("File, {},  does not exist".format(filename))
    # read the yml file
    with open(filename, 'r') as stream:
        try:
            models = yaml.load(stream, Loader=LOADER)
        except yaml.YAMLError as exc:
            logging.error("Error parsing yml file {}".format(filename))
            logging.error(exc)
            sys.exit(1)
    # check that models is a list
    if not isinstance(models, list):
        logging.error("Error parsing yml file {}".format(filename))
        logging.error("models must be a list of dictionaries")
        sys.exit(1)
    for model in models:
        if not isinstance(model, dict):
            logging.error("Error parsing yml file {}".format(filename))
            logging.error("models must be a list of dictionaries")
            sys.exit(1)
        library_name = model['name'].split('.')[:-1]
        library_name = str('.'.join(library_name))
        logging.debug("Library name: "+ library_name)
        object_name = str(model['name'].split('.')[-1])
        logging.debug("Object name: "+ object_name)
        global dependency
        dependency = importlib.import_module(library_name)
        logging.debug(dependency)
        params = model['params']
        logging.debug("Params: " + str(params))
        logging.debug("Params type: "+ str(type(params)))
        global object_instance
        object_instance = None
        exec("object_instance = dependency." + object_name + "()", globals())
        assert isinstance(object_instance, obj_type)
        logging.debug("Model Type: " + str(type(object_instance)))
        for param, values in params.items():
            logging.debug("Param: " + param)
            logging.debug("Values: " + str(values))
        assert isinstance(params, dict)
        assert isinstance(object_instance, obj_type)
        search = GridSearchCV(object_instance, params, cv=CROSS_VALIDATION, refit=True)
        search_list.append(search)
    return search_list





if __name__ == '__main__':
    # command line arguments
    import argparse
    import os
    import logging
    import uuid
    parser = argparse.ArgumentParser(description='Run a model on a dataset')
    parser.add_argument('-m', '--model', default = 'configs/model.yml',type=str, help='Model file to use')
    parser.add_argument('-f', '--folder', type=str, help='Experiment folder to use', default = './')
    parser.add_argument('-d', '--dataset', type=str, help='Data file to use', default = "data.pkl")
    parser.add_argument('-v', '--verbosity', type = str, default='DEBUG', help='set python verbosity level')
    parser.add_argument('-s', '--scorer', default = 'f1', type = str, help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    args = parser.parse_args()
    # initialize logging
    logging.basicConfig(level=args.verbosity)
    model_file = args.model
    if os.path.isfile(os.path.join(args.folder, 'best_train', 'results.json')):
        result_file = os.path.join(args.folder, 'best_train',"results.json")
        best_score = return_result(filename = result_file, scorer = args.scorer)
    else:
        result_file = os.path.join('best_train', "results.json")
        best_score = 0
    better_than_disk = False
    try:
         data = load_data(os.path.join(args.folder, args.dataset))
    except:
        raise ValueError("Unable to load dataset {}".format(os.path.join(args.folder, args.dataset)))
    assert isinstance(data, Data)
    models = parse_gridsearch_from_yml(model_file)
    logging.info("Best score: {}".format(best_score))
   
    for model in models:
        assert isinstance(model, BaseEstimator)
        model_obj = Model(model)
        experiment = Experiment(data= data, model = model_obj)
        experiment.run()
        score = experiment.scores[args.scorer.upper()]
        checkpoint(filename = os.path.join("all_train", ), experiment = experiment, result_folder = args.folder)
        if score > best_score:
            best_score = score
            checkpoint(filename = 'best_train', result_folder= args.folder, experiment = experiment)
            better_than_disk = True
    if not better_than_disk:
        logging.warning("No better model found")
    logging.info("Best Score for trainer: {}".format(best_score))
   