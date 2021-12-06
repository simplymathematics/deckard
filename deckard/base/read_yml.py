# parse a yml file and return a dictionary of models

import yaml
import os
import sys
import logging
import importlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.base import BaseEstimator
from deckard.base.data import Data

def parse_gridsearch_from_yml(filename:str = None, obj_type = BaseEstimator) -> dict:
    search_list = list()
    CROSS_VALIDATION = 5
    LOADER = yaml.FullLoader
    # check if the file exists
    params = dict()
    new_models = dict()
    if not os.path.isfile(filename):
        raise ValueError("File does not exist")
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

def parse_layer_from_yml(data:Data, filename:str = 'preprocessor.yml', folder:str = None) -> dict:
    import copy
    assert isinstance(data, Data)
    assert isinstance(filename, str)
    data_list = list()
    hashes = list()
    CROSS_VALIDATION = 5
    LOADER = yaml.FullLoader
    # check if the file exists
    params = dict()
    new_models = dict()
    if not os.path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    
    with open(filename, 'r') as stream:
        try:
            models = yaml.load(stream, Loader=LOADER)
        except yaml.YAMLError as exc:
            logging.error("Error parsing yml file {}".format(filename))
            raise ValueError("Error parsing yml file {}".format(filename))
    # check that models is a list
    if not isinstance(models, list):
        logging.error("Error parsing yml file {}".format(filename))
        logging.error("models must be a list of dictionaries")
        raise ValueError("Error parsing yml file {}".format(filename))
    for model in models:
        grid_list = list(ParameterGrid(model['params']))
        length = len(grid_list)
        for combination in grid_list:
            new_data = copy.copy(data)
            if not isinstance(model, dict):
                logging.error("Error parsing yml file {}".format(filename))
                logging.error("models must be a list of dictionaries")
                raise ValueError("Error parsing yml file {}".format(filename))
            library_name = model['name'].split('.')[:-1]
            library_name = str('.'.join(library_name))
            logging.debug("Library name: "+ library_name)
            object_name = str(model['name'].split('.')[-1])
            logging.debug("Model name: "+ object_name)
            global dependency
            dependency = importlib.import_module(library_name)
            logging.debug(dependency)
            global object_instance
            object_instance = None
            exec("object_instance = dependency." + object_name , globals())
            
            
            if 'sklearn' in library_name:
                object = object_instance(**combination)
                new_data.X_train = object.fit_transform(data.X_train)
                new_data.X_test = object.transform(data.X_test)
                new_data.y_test = new_data.y_test
                new_data.y_train = new_data.y_train
            elif 'art' in library_name:
                if 'preprocessor' in library_name:
                    object = object_instance(**combination)
                    if object.__dict__['_apply_fit'] == True and object.__dict__['_apply_predict'] == False:
                        logging.info("Applying fit")
                        (new_data.X_train, new_data.y_train) = object(data.X_train, data.y_train)
                        new_data.y_test = data.y_test
                        new_data.X_test = data.X_test
                    elif object.__dict__['_apply_predict'] == True and object.__dict__['_apply_predict'] == False:
                        logging.info("Applying predict")
                        (new_data.X_test, new_data.y_test) = object(data.X_test, data.y_test)
                        new_data.y_train = data.y_train
                        new_data.X_train = data.X_train
                    elif object.__dict__['_apply_fit'] == True and object.__dict__['_apply_predict'] == True:
                        logging.info("Applying fit and predict")
                        (new_data.X_train, new_data.y_train) = object(data.X_train, data.y_train)
                        (new_data.X_test, new_data.y_test) = object(data.X_test, data.y_test)
                    else:
                        raise ValueError("Defense must be applied at fit or predict time.")
                elif 'postprocessor' in library_name:
                    object = object_instance(**combination)
                    logging.info("Initializing postprocessor")
                    new_data.post_processor = object
                elif 'attack' in library_name:
                    if 'evasion' in library_name:
                        object = object_instance
                        logging.info("Initializing evasion attack.")
                        new_data.X_test = data.X_test
                        new_data.y_test = data.y_test
                        new_data.X_train = data.X_train
                        new_data.y_train = data.y_train
                        new_data.evasion = object
                        new_data.attack_params = combination
                    else:
                        raise NotImplementedError("Attack must be evasion attack.")
                else:
                    raise ValueError("Only preprocessor and postprocessor defenses supported.")
                #TODO: fix broken test.
                # if not isinstance(object.__dict__['_apply_fit'], bool) or not isinstance(object.__dict__['_apply_predict'], bool) or not isinstance(object, art.attacks.Attack):
                    raise ValueError('Needs to have apply_predict and/or apply_fit')
            new_data.params.update({object_name : combination})
            new_data.dataset = data.dataset + "_"+ object_name
            assert isinstance(new_data.X_train, pd.DataFrame) or isinstance(new_data.X_train, np.ndarray), str(type(new_data.X_train))
            assert isinstance(new_data.X_test, pd.DataFrame)  or isinstance(new_data.X_test, np.ndarray), str(type(new_data.X_test))
            assert isinstance(new_data.y_train, pd.DataFrame) or isinstance(new_data.y_train, np.ndarray), str(type(new_data.y_train))
            assert isinstance(new_data.y_test, pd.DataFrame)  or isinstance(new_data.y_test, np.ndarray), str(type(new_data.y_test))
            data_list.append(new_data)
            assert isinstance(new_data, Data)
            hashes.append(hash(str(new_data.params)))
            logging.debug("Hash is " + str(hash(data)))
    assert len(hashes) == len(data_list)
    assert len(set(hashes)) == len(data_list)
    
    logging.debug("Length of combination list: " + str(length))
    logging.debug("Length of input dict: {}".format(len(models)))
    logging.debug("Length of output: {}".format(len(data_list)))
    logging.debug("Unique hashes: " + str(len(set(hashes))))
    return data_list


def validate_callable_yml(filename:str = None, object_type=BaseEstimator) -> None:
    models = parse_gridsearch_from_yml(object_filename)
    assert isinstance(models, list)
    assert isinstance(models[0], GridSearchCV)    
    return None

def validate_parse_layer_yml(filename:str = None) -> None:
    data = Data()
    data_list = parse_layer_from_yml(data, filename)
    assert isinstance(data_list, list)
    for data in data_list:
        assert isinstance(data, Data)
    logging.info(filename + " read successfully. ")
    return None


if __name__ == "__main__":
    #set logging level to debug for testing
    import argparse
    import sys
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Create dicionary of callable objects from yml file.')
    parser.add_argument('-f', '--folder', help='Folder for .yml file', default= 'configs')
    # add argument for verbose level
    parser.add_argument('-v', '--verbose', help='Verbosity level', default= 'INFO')
    args = parser.parse_args()
    working_directory = args.folder
    logging.basicConfig(stream=sys.stdout, level=args.verbose)
    logging.debug("Working directory: " + str(working_directory))
    # change_working_directory(working_directory)
    object_filename =  os.path.join(working_directory, 'model.yml')
    # transformer_filename =  os.path.join(working_directory, 'transform.yml')
    pre_filename = os.path.join(working_directory, 'preprocess.yml')
    logging.debug("Filename: " + str(object_filename))
    defense_filename = os.path.join(working_directory, 'defend.yml')
    logging.debug("Defense filename: " + str(defense_filename))


    # if no arguments are passed, print usage
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    data = Data()
    validate_callable_yml(object_filename, GridSearchCV)
    validate_parse_layer_yml(pre_filename)
    # validate_parse_layer_yml(defense_filename)


