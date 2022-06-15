
import os, json, yaml, logging, shutil
import numpy as np
import pandas as pandas
from deckard.base import Model, Data, Experiment, generate_object_list_from_tuple, generate_tuple_list_from_yml
# main loop
logger = logging.getLogger(__name__)
if __name__ == '__main__':
    # arguments
    import argparse
    parser = argparse.ArgumentParser(description='Prepare model and dataset as an experiment object. Then runs the experiment.')
    parser.add_argument('--input_folder', '-i', type=str, default = "./data/defences", help='Path to the model')
    parser.add_argument('--input_config', '-c', type=str, default = "configs/defend.yml", help='Path to the config file')

    
    

    # parse arguments
    args = parser.parse_args()
    
    # Read the input config file and generate all experiment combinations
    yml_list = generate_tuple_list_from_yml('/home/cmeyers/deckard/examples/cifar10/configs/defend.yml')
    
    todos = []
    completed_tasks = []
    
    logger.info("Successes: {}".format(len(successes)))
    logger.info("Failures: {}".format(len(failures)))
    logger.info("Todos: {}".format(len(todos)))
    logger.info("Completed tasks: {}".format(len(completed_tasks)))
    logger.info("Remaining input combinations: {}".format(len(yml_list)))

            
    