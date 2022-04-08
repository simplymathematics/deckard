import logging, yaml
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator
import os.path as path
import importlib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from deckard.base import Experiment, Model, Data

# specify the logger
logger = logging.getLogger(__name__)

def parse_list_from_yml(filename:str) -> list:
    """
    Parses a yml file and returns a list.
    """
    assert isinstance(filename, str)
    assert path.isfile(filename), f"{filename} does not exist"
    experiment_list = list()
    LOADER = yaml.FullLoader
    # check if the file exists
    if not path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, 'r') as stream:
        try:
            yml_list = yaml.load(stream, Loader=LOADER)
        except yaml.YAMLError as exc:
            logger.error("Error parsing yml file {}".format(filename))
            raise ValueError("Error parsing yml file {}".format(filename))
    # check that featurizers is a list
    if not isinstance(yml_list, list):
        logger.error("Error parsing yml file {}".format(filename))
        raise ValueError("Error parsing yml file {}".format(filename))
    return yml_list

def generate_object_list(yml_list:list) -> list:
    """
    Imports and initializes objects from yml file. Returns a list of instantiated objects.
    :param yml_list: list of yml entries
    """
    obj_list = list()    
    for entry in yml_list:
        library_name = ".".join(entry['name'].split('.')[:-1] )
        class_name = entry['name'].split('.')[-1]
        global dependency
        dependency = None
        dependency = importlib.import_module(library_name)
        global object_instance
        object_instance = None
        exec("from {} import {}".format(library_name, class_name), globals())
        exec(f"object_instance = {class_name}()", globals())
        logger.debug(type(object_instance))
        obj_list.append((object_instance, entry['params']))
    return obj_list

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
    # check that featurizers is a list
    for entry in yml_list:
        if not isinstance(entry, dict):
            raise ValueError("Error parsing yml file {}".format(filename))
        grid = ParameterGrid(entry['params'])
        name = entry['name']
        for param in grid:
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
        exec(f"object_instance = {class_name}(**params)", globals())
        obj_list.append((object_instance))
    return obj_list


def transform_params_for_pipeline(object_list:list, object_name:str)-> list:
    """
    Transforms the params for use in sklearn pipeline.
    :param object_list: list of objects
    :param object_name: object_name
    """
    new_object_list = list()
    for i in range(len(object_list)):
        obj = object_list[i][0]
        params = object_list[i][1]
        new_keys = [f"{object_name}__{i}" for i in params.keys()]
        new_values = params.values()
        param_grid = dict(zip(new_keys, new_values))
        new_object_list.append((obj, param_grid))
    return new_object_list

def insert_layer_into_model(model, name, layer, position, params):
    """
    Insert layer into sklearn pipeline. Returns a new pipeline
    :param model: sklearn pipeline
    :param name: name of layer
    :param layer: layer to insert
    :param position: position to insert layer
    :param params: parameters for layer
    """
    if name not in model.steps:
        model.steps.insert(position, (name, layer))
    else:
        logger.warning(f"{name} already found in pipeline. Skipping")
    if params is not None:
        new_params = params
    else:
        new_params = model.get_params()
    new_params.update(params)
    model.set_params(**new_params)
    return model

def insert_layer_into_pipeline(input_list, model:Pipeline, name = 'featurize', position = 0):
    """
    Insert layer into list of models. Returns a new list of models.
    :param input_list: list of models
    :param model: sklearn pipeline
    :param name: name of layer
    :param position: position to insert layer
    """
    model_list = list()
    for (obj, params) in input_list:
        model = insert_layer_into_model(model, name, obj, position, params)    
        if not isinstance(model, Pipeline) and isinstance(model, BaseEstimator):
            model = Pipeline([('model', model)])
    model_list.append((model, params))
    return model_list

def generate_sklearn_experiment_list(model_list:list, data, cv = None) -> list:
    """
    Generates experiment list from model list.
    :param model_list: list of models
    :param data: data object
    :param cv: number of folds for cross validation
    """
    experiment_list = list()
    for (model, params) in model_list:
        if cv is not None:
            model = GridSearchCV(model, params, cv=cv)
        else:
            logger.warning("No cross validation specified. Skipping.")
            model = model
        model = Model(model)
        experiment = Experiment(data, model)
        experiment_list.append(experiment)
    return experiment_list    

