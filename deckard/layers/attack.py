import logging
from deckard.base.experiment import Experiment, Model, Data
import os
from art.attacks import Attack
from deckard.base.parse import generate_object_list_from_tuple, generate_tuple_list_from_yml, generate_experiment_list
from deckard.base.utils import find_successes, remove_successes_from_queue
from random import shuffle

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
    parser.add_argument('--attack_config', '-d', type=str, default = "configs/attack.yml", help='Path to the attack config file')
    parser.add_argument('--attack_size', '-n', type=int, default=100, help='Number of adversarial samples to generate')
    # parse arguments
    args = parser.parse_args()
    # set up logging
    ART_DATA_PATH = os.path.join(args.output_folder)
    if not os.path.exists(ART_DATA_PATH):
        os.makedirs(ART_DATA_PATH)
    if not os.path.exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        os.path.mkdir(args.output_folder)
    # load dataset
    data = Data(args.data_file)
    data.X_test = data.X_test[:args.attack_size]
    data.y_test = data.y_test[:args.attack_size]
    logger.info("Loaded dataset {}".format(args.data_file))
    yml_list = generate_tuple_list_from_yml(args.attack_config);
    # successes, failures = find_successes(args.output_folder, 'model_params.json', dict_name = 'Defence')
    # todos = remove_successes_from_queue(successes, yml_list)
    # model_object = Model(model =args.input_model, path = os.path.join(args.input_folder, filepath), model_type = args.model_type)
    # object_list = generate_object_list_from_tuple(todos, classifier = )
    art_models = []
    # find all models
    i = 0 
    j = 0 
    directories = os.listdir(args.input_folder)
    # shuffle(directories)
    # directory = directories[0]
    defence_no = len(directories)
    for filepath in directories:
        i += 1
        logger.info("{} of {} defences.".format(i, defence_no))
        filename = str()
        output_folder = str()
        subdirectory = str()
        # finding subdirectories, if there are any, then handles a bunch of
        if os.path.isdir(os.path.join(args.input_folder, filepath)):
            subdirectory = filepath
            filename = os.path.join(args.input_folder, subdirectory)
            output_folder = os.path.join(args.output_folder, subdirectory)
            filename = os.path.join(filename, args.input_model)
            logger.info("Loading model {}".format(filename))
            try:
                model_object = Model(model =args.input_model, path = os.path.join(args.input_folder, filepath), model_type = args.model_type)
            except OSError as e:
                continue
            try:
                attack_list = generate_object_list_from_tuple(yml_list, model_object.model)
            except TypeError as e:
                    attack_list = generate_object_list_from_tuple(yml_list)
            experiment = Experiment(data = data, model = model_object, is_fitted=True, filename = subdirectory)
            experiment.set_defence(os.path.join(args.input_folder, filepath, "defence_params.json"))
            if not os.path.isdir(os.path.join(output_folder, subdirectory)):
                os.makedirs(os.path.join(output_folder, subdirectory))
        # loading file otherwise
        elif os.path.isfile(os.path.join(args.input_folder, filepath)) and filepath == args.input_model:
            filename = args.input_folder
            output_folder = args.output_folder
            filename = os.path.join(filename, args.input_model)
            logger.info("Loading model {}".format(filename))
            try:
                model_object = Model(model = args.input_model, path = args.input_folder, model_type = args.model_type)
            except OSError as e:
                continue
            try:
                attack_list = generate_object_list_from_tuple(yml_list, model_object.model)
            except TypeError as e:
                attack_list = generate_object_list_from_tuple(yml_list)
            experiment = Experiment(data = data, model = model_object, is_fitted=True, filename = filepath)
        else:
            # skips files that aren't == input_model
            continue
        # load models
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        attack_no = len(attack_list)
        for attack in attack_list:
            if isinstance(attack, Attack):
                j+=1
                logger.info("{} of {} attacks for defence {} of {}.".format(j, attack_no, i , defence_no))
                output_folder = os.path.join(args.output_folder, filepath, experiment.filename)
                experiment.set_attack(attack = attack)
                # Seeing if experiment exists
                scores_file = os.path.join(output_folder, 'adversarial_scores.json')
                if os.path.isfile(scores_file):
                    logger.info("Experiment {} already exists. Skipping.".format(experiment.filename))
                    continue
                else:
                    # run experiment
                    logger.info("Running experiment...")
                    experiment.run_attack(path = output_folder)
                    logger.info("Experiment complete.")
                    # Save experiment
                    if args.output_name is None:
                        args.output_name = "defended_model"
                    logger.info("Saving experiment to {}.".format(output_folder))
                    experiment.model.model.save(filename = args.output_name, path = output_folder)
                    logger.info("Experiment saved.")
            else:
                logger.warning("Attack {} is not an instance of Attack".format(attack))