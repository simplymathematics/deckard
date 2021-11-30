from data import Data
from model import Model
from experiment import Experiment
from utils import checkpoint, load_checkpoint, return_result

from sklearn.base import BaseEstimator
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.model_selection import ParameterGrid
import numpy as np
import yaml, os, importlib, logging

def parse_preprocessor_from_yml(data:Data, filename:str, obj_type:BaseEstimator) -> dict:
    assert isinstance(data, Data)
    assert isinstance(filename, str)
    data_list = list()
    LOADER = yaml.FullLoader
    # check if the file exists
    params = dict()
    new_preprocessors = dict()
    if not os.path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, 'r') as stream:
        try:
            preprocessors = yaml.load(stream, Loader=LOADER)
        except yaml.YAMLError as exc:
            logging.error("Error parsing yml file {}".format(filename))
            raise ValueError("Error parsing yml file {}".format(filename))
    # check that preprocessors is a list
    if not isinstance(preprocessors, list):
        logging.error("Error parsing yml file {}".format(filename))
        logging.error("preprocessors must be a list of dictionaries")
        raise ValueError("Error parsing yml file {}".format(filename))
    for preprocessor in preprocessors:
        grid_list = list(ParameterGrid(preprocessor['params']))
        for combination in grid_list:
            if not isinstance(preprocessor, dict):
                logging.error("Error parsing yml file {}".format(filename))
                logging.error("preprocessors must be a list of dictionaries")
                raise ValueError("Error parsing yml file {}".format(filename))
            library_name = preprocessor['name'].split('.')[:-1]
            library_name = str('.'.join(library_name))
            logging.debug("Library name: "+ library_name)
            preprocessor_name = str(preprocessor['name'].split('.')[-1])
            logging.debug("preprocessor name: "+ preprocessor_name)
            global dependency
            dependency = importlib.import_module(library_name)
            logging.debug(dependency)
            global preprocessor_instance
            preprocessor_instance = None
            exec("preprocessor_instance = dependency." + preprocessor_name , globals())
            preprocessor_instance = preprocessor_instance(**combination)
            assert isinstance(preprocessor_instance, obj_type)
            data.X_train = preprocessor_instance.fit_transform(data.X_train, data.y_train)
            data.X_test = preprocessor_instance.transform(data.X_test)
            data.params.update({preprocessor_name : dict(preprocessor_instance.get_params())})
            logging.debug("Preprocessor params: {}".format(data.params[preprocessor_name]))
            assert isinstance(data.X_train, np.ndarray)
            assert isinstance(data.X_test, np.ndarray)
            assert isinstance(data, Data)
            assert hasattr(data, 'params')
            data_list.append(data)
    return data_list



if __name__ == '__main__':
    # command line arguments
    import argparse
    import os
    import logging
    import uuid
    parser = argparse.ArgumentParser(description='Run a preprocessor on a dataset')
    parser.add_argument('-p', '--preprocess', default = 'configs/preprocess.yml',type=str, help='preprocessor file to use')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset file to use', default = "data.pkl")
    parser.add_argument('-o,', '--output', type=str, help='Output file to use', default = 'data')
    parser.add_argument('-f', '--folder', type=str, default = 'data', help='Folder to use', required=False)
    # parse argument for verbosity
    parser.add_argument('-v', '--verbosity', type = str, default='DEBUG', help='set python verbosity level')
    parser.add_argument('-s', '--scorer', type = str, default='f1', help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    parser.add_argument('-m', '--model', type = str, default='model.pkl', help='model file to use')
    args = parser.parse_args()
    # initialize logging
    logging.basicConfig(level=args.verbosity)
    # TODO: add support for multiple folders
    # find all folders in folder
    # folders = [f for f in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, f))]
    # logging.info("Found {} folder(s) in {} folder.".format(len(folders), args.folder))
    logging.debug(args.folder)
    folder = os.path.join(args.folder, 'best_train')
    assert os.path.isdir(folder), "Folder {} does not exist".format(folder)
    assert os.path.isfile(os.path.join(folder, args.model)), "Model file {} does not exist".format(os.path.join(folder, args.model))
    if os.path.isdir(os.path.join(args.output, 'best_preprocess')):
        result_file = os.path.join(args.output, 'best_preprocess',"results.json")
    best_score = 0
    data, model = load_checkpoint(folder = folder, data = args.dataset, model = args.model)
    # set sklearn is_fitted flag to false
    pres = parse_preprocessor_from_yml(data, args.preprocess, obj_type=BaseEstimator)
    for data in pres:
        model_obj = Model(model)
        experiment = Experiment(data= data, model = model_obj)
        experiment.run()
        score = experiment.scores[args.scorer.upper()]
        checkpoint(filename = os.path.join('all_preprocess', experiment.filename), experiment = experiment, result_folder = args.output)
        if score >= best_score:
            best_score = score
            checkpoint(filename = 'best_preprocess', experiment = experiment, result_folder= args.output)
        else:
            logging.info("Score {} is lower than best score {}".format(score, best_score))
    logging.info("Best score for preprocessor: {}".format(best_score))

        