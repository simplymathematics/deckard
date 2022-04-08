from os.path import exists
from os import mkdir, chmod, rename
import logging
from deckard.base.experiment import Experiment, Model
from deckard.base.utils import load_data, initialize_art_classifier
from deckard.base.parse import generate_tuple_list_from_yml, generate_object_list_from_tuple
import parser
import os

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
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