from os.path import exists
from os import mkdir, chmod, rename
import logging
from deckard.base.experiment import Experiment, Model
from deckard.base.utils import load_data, initialize_art_classifier
from deckard.base.parse import generate_tuple_list_from_yml, generate_object_list_from_tuple

import os
logger = logging.getLogger(__name__)
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
    # Creates folder
    if not exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        mkdir(args.output_folder)
    # load dataset
    data = load_data(filename  = args.data_file)
    logger.info("Loaded dataset {}".format(args.data_file))
    yml_list = generate_tuple_list_from_yml('configs/defend.yml');
    object_list = generate_object_list_from_tuple(yml_list)
    i = 0
    length = len(object_list)
    for defense in object_list:
        i += 1
        logger.info("{} of {} experiments.".format(i, length))
        print("{} of {} experiments.".format(i, length))
        defense_dict = {"Defense" : type(defense), "params": defense.__dict__}
        # initalize model
        art_model = initialize_art_classifier(filename = args.input_model, path = args.input_folder, model_type = args.model_type, output_dir = args.output_folder)
        art_model = Model(estimator= art_model, model_type ="keras")
        # Create experiment
        experiment = Experiment(data = data, model = art_model, name = args.input_model, params = defense_dict)
        experiment.set_defense(defense)
        logger.info("Created experiment object from {} dataset and {} model".format(args.data_file, args.input_model))
        # Seeing if experiment exists
        output_folder = os.path.join(args.output_folder, experiment.filename)
        scores_file = os.path.join(output_folder, 'scores.json')
        if os.path.isfile(scores_file):
            logger.info("Experiment {} already exists. Skipping.".format(experiment.filename))
            continue
        else:
            # run experiment
            logger.info("Running experiment...")
            experiment.run(path = output_folder)
            logger.info("Experiment complete.")
            # Save experiment
            if args.output_name is None:
                args.output_name = "defended_model"
            logger.info("Saving experiment to {}.".format(output_folder))
            experiment.model.model.save(filename = args.output_name, path = output_folder)
            logger.info("Experiment saved.")
    assert i == length, "Number of experiments {} does not match number of queries {}".format(i, length)
    # count the number of folders in the output folder
    num_folders = len(os.listdir(args.output_folder))
    assert num_folders >= length