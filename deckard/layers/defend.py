from os.path import exists
from os import mkdir, chmod, rename
import logging
from deckard.base.experiment import Experiment, Model
from deckard.base.utils import load_data, load_model
from deckard.base.data import validate_data
import parser
import os
from art.estimators.classification import PyTorchClassifier, SklearnClassifier, KerasClassifier, TensorFlowClassifier
from art.defences.preprocessor import Preprocessor
from art.defences.postprocessor import Postprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer
from art.utils import get_file

logger = logging.getLogger(__name__)

import yaml
from os import path
from sklearn.model_selection import ParameterGrid

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

import importlib
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

SUPPORTED_MODELS = [Postprocessor, Preprocessor, Transformer, Trainer]

def initialize_art_classifier(model_name:str, model_path:str=None, model_type:str=None, output_folder:str = None, defenses:list = None):
    """
    Load an ART model.
    :param model_name: the name of the model
    :param model_path: the path to the model
    :param model_type: the type of the model
    :param output_folder: the output folder to save the model to
    :return: the loaded art model
    
    """

    # disable eager execution
    
    pre_processing_defences = []
    post_processing_defences = []
    trainer_defences = []
    transformer_defences = []
    detector_defences = []
    
    if defenses is not None:
        for defense in defenses:
            if 'preprocessor' in str(type(defense)):
                assert isinstance(defense, Preprocessor), "Defense {} is not a Preprocessor".format(defense)
                pre_processing_defences.append(defense)
            elif 'postprocessor' in str(type(defense)):
                assert isinstance(defense, Postprocessor), "Defense {} is not a Postprocessor".format(defense)
                post_processing_defences.append(defense)
            elif 'trainer' in str(type(defense)):
                assert isinstance(defense, Trainer), "Defense {} is not a Trainer".format(defense)
                trainer_defences.append(defense)
            elif 'transformer' in str(type(defense)):
                assert isinstance(defense, Transformer), "Defense {} is not a Transformer".format(defense)
                transformer_defences.append(defense)
            elif 'detector' in str(type(defense)):
                raise NotImplementedError("Detector not implemented yet")
            else:
                raise ValueError("Defense {} is not supported".format(defense))

    # Download/load model
    assert model_type is not None, "model_type must be specified"
    if 'http' in model_path:
        # download model
        model_path = get_file(filename = model_name, extract=False, path=output_folder, url=model_path, verbose = True)
    else:
        model_path = os.path.join(model_path, model_name)
    # Define type for ART
    if model_type == 'tfv1' or 'tensorflowv1' or 'tf1':
        import tensorflow.compat.v1 as tfv1
        tfv1.disable_eager_execution()
        from tensorflow.keras.models import load_model as keras_load_model
        classifier_model = keras_load_model(model_path)
        art_model = KerasClassifier( model=classifier_model, preprocessing_defences=pre_processing_defences, postprocessing_defences=post_processing_defences)
    elif model_type == 'keras' or 'k':
        from tensorflow.keras.models import load_model as keras_load_model
        classifier_model = keras_load_model(model_path)
        art_model = KerasClassifier( model=classifier_model, preprocessing_defences=pre_processing_defences, postprocessing_defences=post_processing_defences)
    elif model_type == 'tf' or 'tensorflow':
        from tensorflow.keras.models import load_model as keras_load_model
        classifier_model = keras_load_model(model_path)
        art_model = TensorFlowClassifier( model=classifier_model, preprocessing_defences=pre_processing_defences, postprocessing_defences=post_processing_defences)
    elif model_type == 'pytorch' or 'torch':
        raise NotImplementedError("Pytorch not implemented yet")
    
    else:
        raise ValueError("Unknown model type {}".format(model_type))
    return art_model


if __name__ == '__main__':
    # arguments
    import argparse
    parser = argparse.ArgumentParser(description='Prepare model and dataset as an experiment object. Then runs the experiment.')
    parser.add_argument('--input_model', type=str, default=None, help='Name of the model')
    parser.add_argument('--input_folder', type=str, default = ".", help='Path to the model')
    parser.add_argument('--model_type', type=str, default=None, help='Type of the model')
    parser.add_argument('--verbosity', type=int, default=10, help='Verbosity level')
    parser.add_argument('--output_folder', type=str, required = True, help='Path to the output folder')
    parser.add_argument('--output_name', type=str, default=None, help='Name of the output file')
    parser.add_argument('--log_file', type=str, default = "log.txt", help='Path to the log file')
    parser.add_argument('--data_file', type=str, default = "data.pkl", help='Path to the data file')
    parser.add_argument('--defense_config', '-d', type=str, default = None, help='Path to the defense config file')
    # parse arguments
    args = parser.parse_args()
    # set up logging
    ART_DATA_PATH = os.path.join(args.output_folder)
    if not os.path.exists(ART_DATA_PATH):
        os.makedirs(ART_DATA_PATH)
    # Create a logger
    logger = logging.getLogger('art_logger')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create stream handler
    stream_handler = logging.StreamHandler()
    # set formatting
    stream_handler.setFormatter(formatter)
    # set stream_handler level
    stream_handler.setLevel(args.verbosity)
    # Add handler to logger
    logger.addHandler(stream_handler)
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(ART_DATA_PATH, args.log_file))
    # set file_handler level to max
    file_handler.setLevel(logging.DEBUG)
    # set formatting
    file_handler.setFormatter(formatter)
    # Add handler to logger
    logger.addHandler(file_handler)
    # attempts to load model
    if not exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        mkdir(args.output_folder)
    # load dataset
    data = load_data(data_file  = args.data_file)
    logger.info("Loaded dataset {}".format(args.data_file))
    yml_list = generate_tuple_list_from_yml('configs/defend.yml');
    object_list = generate_object_list_from_tuple(yml_list)
    i = 0
    length = len(object_list)
    for defense in object_list:
        i += 1
        logger.info("{} of {} experiments.".format(i, length))
        defense_dict = {"Defense" : type(defense), "params": defense.__dict__}
        # initalize model
        art_model = initialize_art_classifier(model_name = args.input_model, model_path = args.input_folder, model_type = args.model_type, output_folder = args.output_folder, defenses = [defense])
        art_model = Model(estimator= art_model, model_type =args.model_type)
        # Create experiment
        experiment = Experiment(data = data, model = art_model, name = args.input_model, params = defense_dict)
        logger.info("Created experiment object from {} dataset and {} model".format(args.data_file, args.input_model))
        # run experiment
        logger.info("Running experiment...")
        experiment.run()
        logger.info("Experiment complete.")
        # Save experiment
        output_folder = os.path.join(args.output_folder, experiment.filename)
        if args.output_name is None:
            args.output_name = "defended_model"
        logger.info("Saving experiment to {}.".format(output_folder))
        experiment.model.model.save(filename = args.output_name, path = output_folder)
        experiment.save_results(folder = output_folder)
        logger.info("Experiment saved.")
    assert i == length, "Number of experiments {} does not match number of queries {}".format(i, length)
    # count the number of folders in the output folder
    num_folders = len(os.listdir(args.output_folder))
    assert num_folders >= length