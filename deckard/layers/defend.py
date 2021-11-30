
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import BaseEstimator
import numpy as np
# import is_regressor from sklearn
from art.defences.preprocessor import Preprocessor
from art.defences.postprocessor import Postprocessor

from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import yaml, sys, os, importlib, logging, copy
from deckard.base.data import Data
from deckard.base.model import Model
from deckard.base.experiment import Experiment
from deckard.base.utils import return_result, load_checkpoint, checkpoint, load_model


def parse_defense_from_yml(data:Data, filename:str) -> dict:
    assert isinstance(data, Data)
    assert isinstance(filename, str)
    data_list = list()
    hashes = list()
    CROSS_VALIDATION = 5
    LOADER = yaml.FullLoader
    # check if the file exists
    params = dict()
    new_defenses = dict()
    if not os.path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, 'r') as stream:
        try:
            defenses = yaml.load(stream, Loader=LOADER)
        except yaml.YAMLError as exc:
            logging.error("Error parsing yml file {}".format(filename))
            raise ValueError("Error parsing yml file {}".format(filename))
    # check that defenses is a list
    if not isinstance(defenses, list):
        logging.error("Error parsing yml file {}".format(filename))
        logging.error("defenses must be a list of dictionaries")
        raise ValueError("Error parsing yml file {}".format(filename))
    for defense in defenses:
        grid_list = list(ParameterGrid(defense['params']))
        for combination in grid_list:
            if not isinstance(defense, dict):
                logging.error("Error parsing yml file {}".format(filename))
                logging.error("defenses must be a list of dictionaries")
                raise ValueError("Error parsing yml file {}".format(filename))
            library_name = defense['name'].split('.')[:-1]
            library_name = str('.'.join(library_name))
            logging.debug("Library name: "+ library_name)
            defense_name = str(defense['name'].split('.')[-1])
            logging.debug("defense name: "+ defense_name)
            global dependency
            dependency = importlib.import_module(library_name)
            logging.debug(dependency)
            global defense_instance
            defense_instance = None
            exec("defense_instance = dependency." + defense_name , globals())
            if 'art' in library_name:
                new_data = copy.copy(data)
                if defense_instance.__dict__['_apply_fit'] == True:
                    logging.info("Applying fit")
                    (new_data.X_train, new_data.y_train) = defense_instance(data.X_train, data.y_train)
                    (new_data.X_test, new_data.y_test) = defense_instance(data.X_test, data.y_test)
                if defense_instance.__dict__['_apply_predict'] == True:
                    new_data.post_processor = defense_instance
            else:
                raise NotImplementedError("Defense not implemented")
            data.params.update({defense_name : combination})
            logging.debug("defense params: {}".format(data.params[defense_name]))
            assert isinstance(data.X_train, np.ndarray)
            assert isinstance(data.X_test, np.ndarray)
            assert isinstance(data, Data)
            assert hasattr(data, 'params')
            data_list.append(data)
            # hashes.append(hash(str(defense_instance.get_params())))
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
    parser = argparse.ArgumentParser(description='Run a defense on a dataset')
    parser.add_argument('-p', '--defend', default = 'configs/defend.yml',type=str, help='defense file to use')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset file to use', default = "iris")
    parser.add_argument('-o,', '--output', type=str, help='Output file to use', default = './')
    parser.add_argument('-f', '--folder', type=str, default = './', help='Folder to use', required=False)
    # parse argument for verbosity
    parser.add_argument('-v', '--verbosity', type = str, default='DEBUG', help='set python verbosity level')
    parser.add_argument('-s', '--scorer', type = str, default='f1', help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    parser.add_argument('--first', type = bool, default=True, help='first defense to use')
    parser.add_argument('-m', '--model', type = str, default='model.pkl', help='model file to use')
    parser.add_argument('--sample_size', type = float, default=1.0, help='sample size to use')
    parser.add_argument('--random_state', type = int, default=42, help='random state to use')   
    parser.add_argument('--test_size', type = float, default=0.2, help='test size to use')
    parser.add_argument('--shuffle', type = bool, default=True, help='shuffle to use')
    parser.add_argument('--flatten', type = bool, default=False, help='True if you want to flatten the data before processing.')
    args = parser.parse_args()
    # initialize logging
    logging.basicConfig(level=args.verbosity)
    # find all folders in folder
    folders = [f for f in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, f))]
    logging.info("Found {} folder(s) in {} folder.".format(len(folders), args.folder))
    # TODO: add support for multiple folders
    folder =  'best_train'
    best_score = 0
    data = Data(args.dataset, sample_size=args.sample_size, random_state=args.random_state, test_size=args.test_size, shuffle = args.shuffle, flatten = args.flatten)
    model_file = os.path.join(args.folder, folder,args.model)
    model = load_model(model_file)
    # set sklearn is_fitted flag to false
    assert isinstance(data, Data)
    from deckard.base.read_yml import parse_layer_from_yml
    pres = parse_layer_from_yml(data, args.defend)
    for data in pres:
        model_obj = Model(model)
        data.X_train = np.ndarray.astype(data.X_train, np.float32)
        print(data.X_train[0])
        input()
        data.X_test.reset_index(drop = True, inplace = True)
        data.X_train.reset_index(drop = True, inplace = True)
        data.y_train.reset_index(drop = True, inplace = True)
        data.y_test.reset_index(drop = True, inplace = True) 
        experiment = Experiment(data= data, model = model_obj)
        scores = experiment.run()
        score = scores[args.scorer.upper()]
        if score > best_score:
            best_score = score
    logging.info("Best Score: {}".format(best_score))
    if scores[args.scorer.upper()] == best_score:
        logging.info("Best defense: {}".format(data.dataset))
        interesting_params = data.params.copy()
        OMIT = ['X_train', 'X_test', 'y_train', 'y_test', 'dataset']
        for key in OMIT:
            interesting_params.pop(key, None)
        logging.debug("Best defense parameters: {}".format(interesting_params))
        logging.info("Best defense score: {}".format(scores[args.scorer.upper()]))
        logging.info("Best defense scorers: {}".format(scores))
        logging.info("Best defense filename: {}".format(scores['Filename']))
        checkpoint(filename = scores['Filename'], experiment = experiment, scores = scores, results_folder = args.output)
    if os.path.isdir(os.path.join(args.output, 'best_train')):
        result_file = os.path.join(args.output, 'best_train',"results.json")
        if best_score > return_result(scorer = args.scorer, filename = result_file):
            checkpoint(filename = '_defense', experiment = experiment, scores = scores, results_folder = args.output)
    else:
        checkpoint(filename = 'best_train', experiment = experiment, scores = scores, results_folder = args.output)
        