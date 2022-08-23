import logging, os
from deckard.base.experiment import Experiment, Model, Data
from deckard.base.parse import generate_tuple_list_from_yml, generate_object_list_from_tuple
from deckard.base.utils import find_successes, remove_successes_from_queue
from random import shuffle
import numpy as np

import os
logger = logging.getLogger(__name__)



if __name__ == '__main__':
    
    # arguments
    import argparse
    parser = argparse.ArgumentParser(description='Prepare model and dataset as an experiment object. Then runs the experiment.')
    parser.add_argument('--input_model', type=str, default=None, help='Name of the model')
    parser.add_argument('--input_folder', type=str, default = ".", help='Path to the model')
    parser.add_argument('--model_type', type=str, default=None, help='Type of the model')
    parser.add_argument('--output_folder', type=str, required = True, help='Path to the output folder')
    parser.add_argument('--output_name', type=str, default=None, help='Name of the output file')
    parser.add_argument('--log_file', type=str, default = "log.txt", help='Path to the log file')
    parser.add_argument('--data_file', type=str, default = "data.pkl", help='Path to the data file')
    parser.add_argument('--defence_config', '-d', type=str, default = "configs/defend.yml", help='Path to the defence config file')
    
    # parse arguments
    args = parser.parse_args()
    # set up logging
    ART_DATA_PATH = os.path.join(args.output_folder)
    if not os.path.exists(ART_DATA_PATH):
        os.makedirs(ART_DATA_PATH)
    # Creates folder
    if not os.path.exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        os.path.mkdir(args.output_folder)
    # load dataset
    data = Data(args.data_file, train_size = 100)

    mini = np.amin(data.X_train)
    maxi = np.amax(data.X_train)
    clip_values = (mini, maxi)
    logger.info("Loaded dataset {}".format(args.data_file))
    yml_list = generate_tuple_list_from_yml(args.defence_config);
    successes, failures = find_successes(args.output_folder, 'model_params.json', dict_name = 'Defence')
    todos = remove_successes_from_queue(successes, yml_list)
    object_list = generate_object_list_from_tuple(todos)
    shuffle(object_list)
    assert len(todos) <= len(yml_list)
    i = 0
    length = len(object_list)
    while len(object_list) > 0:
        defence = object_list.pop()
        i += 1
        logger.info("{} of {} experiments.".format(i, length))
        logger.info("{} of {} experiments.".format(i, length))
        defence_dict = {"Defence" : type(defence), "params": defence.__dict__}
        # initalize model
        art_model = Model(model=args.input_model, model_type =args.model_type, path = args.input_folder, defence = defence, clip_values = clip_values)
        # Create experiment
        experiment = Experiment(data = data, model = art_model, name = args.input_model, params = defence_dict, is_fitted=True)
        logger.info("Created experiment object from {} dataset and {} model".format(args.data_file, args.input_model))
        # Seeing if experiment exists
        output_folder = os.path.join(args.output_folder, str(experiment.filename))
        logger.info("Experiment path is {}".format(output_folder))
        scores_file = os.path.join(output_folder, 'scores.json')
        # run experiment
        logger.info("Running experiment...")
        if not os.path.isfile(scores_file):
            # Makes directory if it doesn't exist
            if not os.path.exists(output_folder):
                logger.info("Experiment path {} does not exist. Creating it.".format(output_folder))
                os.mkdir(output_folder)
            try:
                experiment.run(path = output_folder)
                logger.info("Experiment complete.")
                if args.output_name is None:
                    args.output_name = "defended_model"
                logger.info("Saving experiment to {}.".format(output_folder))
                # Save model
                experiment.model.model.save(filename = args.output_name, path = output_folder)
                logger.info("Experiment saved.")
            except Exception as e:
                # raise e
                logger.error("Experiment {} failed. Error: {}".format(experiment.filename, e))
                with open (os.path.join(output_folder, "failure.txt"), "a") as f:
                    f.write(str(e))
        else:
            logger.info("Experiment {} already exists. Skipping.".format(experiment.filename))
    assert i == length, "Number of experiments {} does not match number of queries {}".format(i, length)
    # count the number of folders in the output folder
    num_folders = len(os.listdir(args.output_folder))
    assert num_folders >= length