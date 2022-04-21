import logging, yaml
from sklearn.model_selection import ParameterGrid
import os.path as path
import importlib
from deckard.base import Experiment, Model, Data

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
        for key, value in entry['params'].items():
            if isinstance(value, (tuple, float, int, str)):
                special_values = value
                special_key = key
        grid = ParameterGrid(entry['params'])
        name = entry['name']
        for param in grid:
            if special_key in param:
                param[special_key] = special_values
            full_list.append((name, param))
    return full_list

def generate_object_list_from_tuple(yml_tuples:list, **kwargs) -> list:
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
        params.update(kwargs)
        exec("from {} import {}".format(library_name, class_name), globals())
        try:
            exec(f"object_instance = {class_name}(**params)", globals())
        except ValueError as e:
            print(f"Error initializing {entry[0]} with params {params}")
            raise e
        obj_list.append(object_instance)
    return obj_list

def generate_experiment_list(model_list:list, data, cv = None) -> list:
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
            model = Model(model)
        experiment = Experiment(data, model)
        experiment_list.append(experiment)
    return experiment_list    

