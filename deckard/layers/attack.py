from os.path import exists
from os import mkdir, chmod, rename
import logging
from deckard.base.experiment import Experiment, Model
from deckard.base.utils import load_data, load_model
import parser
import os
from deckard.base.parse import generate_object_list_from_tuple, generate_tuple_list_from_yml, generate_experiment_list

logger = logging.getLogger(__name__)

import yaml
from os import path
from sklearn.model_selection import ParameterGrid

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
    parser.add_argument('--attack_config', '-d', type=str, default = None, help='Path to the attack config file')
    # parse arguments
    args = parser.parse_args()
    # set up logging
    ART_DATA_PATH = os.path.join(args.output_folder)
    if not os.path.exists(ART_DATA_PATH):
        os.makedirs(ART_DATA_PATH)
    # Create a logger
    logger = logging.getLogger(__name__)
    # Create formatter
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
    data = load_data(filename  = args.data_file)
    logger.info("Loaded dataset {}".format(args.data_file))
    yml_list = generate_tuple_list_from_yml('configs/attack.yml');
    art_models = []
    # find all models
    i = 0 
    j = 0 
    for filepath in os.listdir(args.input_folder):
        i += 1
        filename = str()
        output_folder = str()
        subdirectory = str()
        # finding subdirectories, if there are any
        if os.path.isdir(os.path.join(args.input_folder, filepath)):
            subdirectory = filepath
            filename = os.path.join(args.input_folder, subdirectory)
            model_name = args.input_model
            model_type = args.model_type
            output_folder = os.path.join(args.output_folder, subdirectory)
            filename = os.path.join(filename, model_name)
            logger.info("Loading model {}".format(filename))
            art_model = load_model(filename=filename, mtype = 'tf1')
            attack_list = generate_object_list_from_tuple(yml_list, estimator = art_model.model)
            experiment = Experiment(data = data, model = art_model, is_fitted=True, filename = subdirectory)
        # loading file otherwise
        elif os.path.isfile(os.path.join(args.input_folder, filepath)) and filepath == args.input_model:
            filename = args.input_folder
            model_name = args.input_model
            model_type = args.model_type
            output_folder = args.output_folder
            filename = os.path.join(filename, model_name)
            logger.info("Loading model {}".format(filename))
            art_model = load_model(filename=filename, mtype = 'tf1')
            attack_list = generate_object_list_from_tuple(yml_list, estimator = art_model.model_type)
            model_object = Model(estimator = art_model, model_type = args.model_type)
            experiment = Experiment(data = data, model = model_object, is_fitted=True, filename = args.output_name)
        else:
            # skips files that aren't == input_model
            continue
        # load models
        if not path.exists(output_folder):
            mkdir(output_folder)
        for attack in attack_list:
            j+=1
            experiment.set_attack(attack = attack)
            experiment.run()
            output_folder = experiment.filename
            if subdirectory != "":
                experiment.save_results(path = os.path.join(args.output_folder, subdirectory, output_folder))
            else:
                experiment.save_results(path = os.path.join(args.output_folder, output_folder))
            logger.info("Finished attack {} of {}".format(j, len(attack_list)))