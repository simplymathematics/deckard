import logging, yaml
from sklearn.model_selection import ParameterGrid
import os.path as path
import importlib
from .data import Data
from .model import Model
from .experiment import Experiment
from .scorer import Scorer
from typing import Union


# specify the logger
logger = logging.getLogger(__name__)

def generate_tuple_list_from_yml(filename:str) -> list:
    """
    Parses a yml file, generates a an exhaustive list of parameter combinations for each entry in the list, and returns a single list of tuples.
    """
    assert isinstance(filename, str)
    assert path.isfile(filename), f"{filename} does not exist"
    full_list = list()
    LOADER = yaml.FullLoader
    # check if the file exists
    if not path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, 'r') as stream:
        try:
            yml_list = yaml.load(stream, Loader=LOADER)
        except yaml.YAMLError as exc:
            raise ValueError("Error parsing yml file {}".format(filename))
    for entry in yml_list:
        if not isinstance(entry, dict):
            raise ValueError("Error parsing yml file {}".format(filename))
        special_keys = {}
        for key, value in entry['params'].items():
            if isinstance(value, (tuple, float, int, str)):
                special_values = value
                special_key = key
                special_keys[special_key] = special_values
        for key in special_keys.keys():
            entry['params'].pop(key)
        grid = ParameterGrid(entry['params'])
        name = entry['name']
        for combination in grid:
            if "special_keys" in locals():
                for key, value in special_keys.items():
                    combination[key] = value
            full_list.append((name, combination))
    return full_list

def generate_object_list_from_tuple(yml_tuples:list, *args) -> list:
    """
    Imports and initializes objects from yml file. Returns a list of instantiated objects.
    :param yml_list: list of yml entries
    """
    obj_list = list()    
    for entry in yml_tuples:
        library_name = ".".join(entry[0].split('.')[:-1] )
        class_name = entry[0].split('.')[-1]
        global dependency
        dependency = None
        dependency = importlib.import_module(library_name)
        global object_instance
        object_instance = None
        global params
        params = entry[1]
        exec("from {} import {}".format(library_name, class_name), globals())
        if len(args) == 1:
            global positional_arg
            positional_arg = args[0]
            exec(f"object_instance = {class_name}(positional_arg, **params)", globals())
            del positional_arg
        elif len(args) == 0:
            exec(f"object_instance = {class_name}(**params)", globals())
        else:
            raise ValueError("Too many positional arguments")
        obj_list.append(object_instance)
        del params
        del dependency
    return obj_list



def generate_tuple_from_yml(filename:Union[str, dict]) -> list:
    """
    Parses a yml file, generates a an exhaustive list of parameter combinations for each entry in the list, and returns a single list of tuples.
    """
    assert isinstance(filename, (str, dict))
    if isinstance(filename, str):
        LOADER = yaml.FullLoader
        assert path.isfile(filename), f"{filename} does not exist"
        with open(filename, 'r') as stream:
            try:
                entry = yaml.load(stream, Loader=LOADER)
            except yaml.YAMLError as exc:
                raise ValueError("Error parsing yml file {}".format(filename))
    if not path.isfile(str(filename)):
        assert isinstance(filename, dict), "filename must be a dict or a yml file"
        entry = filename
    return (entry['name'], entry['params'])

def generate_object_from_tuple(obj_tuple:list, *args) -> list:
    """
    Imports and initializes objects from yml file. Returns a list of instantiated objects.
    :param yml_list: list of yml entries
    """
    library_name = ".".join(obj_tuple[0].split('.')[:-1] )
    class_name = obj_tuple[0].split('.')[-1]
    global dependency
    dependency = None
    dependency = importlib.import_module(library_name)
    global object_instance
    object_instance = None
    global params
    params = obj_tuple[1]
    exec("from {} import {}".format(library_name, class_name), globals())
    if len(args) == 1:
        global positional_arg
        positional_arg = args[0]
        exec(f"object_instance = {class_name}(positional_arg, **params)", globals())
        del positional_arg
    elif len(args) == 0:
        exec(f"object_instance = {class_name}(**params)", globals())
    else:
        raise ValueError("Too many positional arguments")
    del params
    del dependency
    return object_instance





def generate_experiment_list(model_list:list, data,  model_type = 'sklearn', **kwargs) -> list:
    """
    Generates experiment list from model list.
    :param model_list: list of models
    :param data: data object
    :param cv: number of folds for cross validation
    """
    experiment_list = list()
    for model in model_list:
        model = model
        if not isinstance(model, Model):
            model = Model(model, model_type = model_type, **kwargs)
        experiment = Experiment(data = data, model = model)
        experiment_list.append(experiment)
    return experiment_list    

def parse_data_from_yml(filename:str) -> dict:
    assert isinstance(filename, str)
    LOADER = yaml.FullLoader
    # check if the file exists
    params = dict()
    if not path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, 'r') as stream:
        try:
            data_file = yaml.load(stream, Loader=LOADER)[0]
            logger.info(data_file)
            logger.info(type(data_file))
        except yaml.YAMLError as exc:
            raise ValueError("Error parsing yml file {}".format(filename))
    # check that datas is a list
    if not isinstance(data_file, dict):
        raise ValueError("Error parsing yml file {}. It must be a yaml dictionary.".format(filename))
    params = data_file['params']
    data_name = data_file['name']
    logger.info(f"Parsing data from {filename}")
    logger.info(f"Data name: {data_name}")
    logger.info(f"Data params: {params}")
    for param, value in params.items():
        logger.info(param + ": " + str(value))
    data = Data(data_name, **params)
    assert isinstance(data, Data)
    logger.info("{} successfully parsed.".format(filename))
    return data

def parse_scorer_from_yml(filename:str) -> dict:
    assert isinstance(filename, str)
    LOADER = yaml.FullLoader
    # check if the file exists
    params = dict()
    if not path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, 'r') as stream:
        try:
            scorer_file = yaml.load(stream, Loader=LOADER)[0]
            logger.info(scorer_file)
            logger.info(type(scorer_file))
        except yaml.YAMLError as exc:
            raise ValueError("Error parsing yml file {}".format(filename))
    # check that datas is a list
    if not isinstance(scorer_file, dict):
        raise ValueError("Error parsing yml file {}. It must be a yaml dictionary.".format(filename))
    if 'scorer_function' in scorer_file:
        params['scorer_function'] = scorer_file['scorer_function']
    elif 'name' in scorer_file:
        params['name'] = scorer_file['name']
    else:
        raise ValueError("Error parsing yml file {}. It must contain a scorer_function or a name.".format(filename))
    logger.info(f"Parsing data from {filename}")
    for param, value in params.items():
        logger.info(param + ": " + str(value))
    data = Scorer(**params)
    assert isinstance(data, Data)
    logger.info("{} successfully parsed.".format(filename))
    return data