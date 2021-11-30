from base.data import Data
from base.model import Model
from base.experiment import Experiment
from base.utils import checkpoint, load_checkpoint, return_result
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.model_selection import ParameterGrid
import numpy as np
import yaml, os, importlib, logging


def parse_featurizer_from_yml(data:Data, filename:str, obj_type:BaseEstimator) -> dict:
    assert isinstance(data, Data)
    assert isinstance(filename, str)
    data_list = list()
    hashes = list()
    CROSS_VALIDATION = 5
    LOADER = yaml.FullLoader
    # check if the file exists
    params = dict()
    new_featurizers = dict()
    if not os.path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, 'r') as stream:
        try:
            featurizers = yaml.load(stream, Loader=LOADER)
        except yaml.YAMLError as exc:
            logging.error("Error parsing yml file {}".format(filename))
            raise ValueError("Error parsing yml file {}".format(filename))
    # check that featurizers is a list
    if not isinstance(featurizers, list):
        logging.error("Error parsing yml file {}".format(filename))
        logging.error("featurizers must be a list of dictionaries")
        raise ValueError("Error parsing yml file {}".format(filename))
    for featurizer in featurizers:
        grid_list = list(ParameterGrid(featurizer['params']))
        for combination in grid_list:
            if not isinstance(featurizer, dict):
                logging.error("Error parsing yml file {}".format(filename))
                logging.error("featurizers must be a list of dictionaries")
                raise ValueError("Error parsing yml file {}".format(filename))
            library_name = featurizer['name'].split('.')[:-1]
            library_name = str('.'.join(library_name))
            logging.debug("Library name: "+ library_name)
            featurizer_name = str(featurizer['name'].split('.')[-1])
            logging.debug("featurizer name: "+ featurizer_name)
            global dependency
            dependency = importlib.import_module(library_name)
            logging.debug(dependency)
            global featurizer_instance
            featurizer_instance = None
            exec("featurizer_instance = dependency." + featurizer_name , globals())
            featurizer_instance = featurizer_instance(**combination)
            assert isinstance(featurizer_instance, obj_type)
            data.X_train = featurizer_instance.fit_transform(data.X_train, data.y_train)
            data.X_test = featurizer_instance.transform(data.X_test)
            data.params.update({"Featurizer" : featurizer_name, "Featurizer Params" : combination})
            logging.debug("Featurizer params: {}".format(data.params["Featurizer Params"]))
            assert isinstance(data.X_train, np.ndarray)
            assert isinstance(data.X_test, np.ndarray)
            assert isinstance(data, Data)
            assert hasattr(data, 'params')
            data_list.append(data)
            # hashes.append(hash(str(featurizer_instance.get_params())))
    #         logging.debug("Hash is " + str(hash(data)))
    # assert len(hashes) == len(data_list)
    # assert len(set(hashes)) == len(data_list)
    return data_list



if __name__ == '__main__':
    # command line arguments
    import argparse
    import os
    import logging
    import uuid
    parser = argparse.ArgumentParser(description='Run a featurizer on a dataset')
    parser.add_argument('-p', '--featurize', default = 'configs/featurize.yml',type=str, help='featurizer file to use')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset file to use', default = "data.pkl")
    parser.add_argument('-f', '--folder', type=str, default = './', help='Folder to use', required=False)
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
    folder = os.path.join(args.folder, 'best_preprocess')
    if os.path.isdir(os.path.join(args.folder, 'best_features')):
        result_file = os.path.join(args.folder, 'best_features',"results.json")
        best_score = return_result(filename = result_file, scorer=args.scorer)
    else:
        best_score = 0
    assert os.path.isdir(folder), "Folder {} does not exist".format(folder)
    assert os.path.isfile(os.path.join(folder, args.model)), "Model file {} does not exist".format(os.path.join(folder, args.model))
    (data, model) = load_checkpoint(folder = folder, data = args.dataset, model = args.model)
    # set sklearn is_fitted flag to false
    pres = parse_featurizer_from_yml(data, args.featurize, obj_type=BaseEstimator)
    for data in pres:
        model_obj = Model(model)
        experiment = Experiment(data= data, model = model_obj)
        experiment.run()
        score = experiment.scores[args.scorer.upper()]
        checkpoint(filename = os.path.join('all_features', experiment.filename), experiment = experiment, result_folder = args.folder)
        if score > best_score:
            best_score = score
            checkpoint(filename = 'best_features', experiment = experiment, result_folder = args.folder)
        else:
            logging.info("Score {} is lower than best score {}".format(score, best_score))
            cmd = "cp -r {} {}".format(os.path.join(args.folder, 'best_preprocess'), os.path.join(args.folder, 'best_retrain'))
    logging.info("Best Score of featurizer: {}".format(best_score))
