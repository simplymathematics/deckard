import logging, os
from deckard.base import Experiment, Model, Data, Scorer, generate_experiment_list
from deckard.base.parse import generate_tuple_from_yml, generate_object_from_tuple
from deckard.base.utils import find_successes, remove_successes_from_queue
from random import shuffle
import numpy as np
logger = logging.getLogger(__name__)

def defend(args) -> None:
    data = Data(args.data_file, train_size = 100)
    mini = np.amin(data.X_train)
    maxi = np.amax(data.X_train)
    clip_values = (mini, maxi)
    defence = generate_object_from_tuple(generate_tuple_from_yml(args.defence_config))
    art_model = Model(model=args.input_model, model_type =args.model_type, path = args.input_folder, clip_values = clip_values, defence = defence)
    experiment = Experiment(data = data, model = art_model, filename = args.input_model,  is_fitted=True)
    experiment(path = args.output_folder, filename = args.output_name)
    return None

# def preprocess(args, model_list:list, sub_folder:str) -> None:    
#     big_list = []
#     data = Data(args.data_file)
#     tuple_list = generate_tuple_list_from_yml(args.config)
#     preprocessor_list = generate_object_list_from_tuple(tuple_list)
#     exp_list = generate_experiment_list(model_list, data)
#     for i in tqdm(range(len(preprocessor_list)), desc = 'Creating preprocessor list'):
#         preprocessor = preprocessor_list[i]
#         for exp in exp_list:
#             new = deepcopy(exp)
#             new.insert_sklearn_preprocessor(name = sub_folder, preprocessor = preprocessor, position = args.position)
#             big_list.append(new)
#     for i in tqdm(range(len(big_list)), desc = 'Running experiments'):
#         exp = big_list[i]
#         output_folder = os.path.join(sub_folder, exp.filename)
#         if not os.path.isdir(output_folder):
#             os.mkdir(output_folder)
#         exp(filename = args.output_name, path = output_folder)

if __name__ == '__main__':
    # args
    import argparse
    import dvc.api
    parser = argparse.ArgumentParser(description='Prepare model and dataset as an experiment object. Then runs the experiment.')
    parser.add_argument('--layer_name','-l', type=str, required = True, help='Name of layer, e.g. "attack"')
    parser.add_argument('--input_model', '-m', type=str, default=None, help='Name of the model')
    parser.add_argument('--input_folder', '-i', type=str, default = None, help='Path to the model')
    parser.add_argument('--model_type', '-t', type=str, default=None, help='Type of the model')
    parser.add_argument('--output_folder', '-p', type=str, default = None, help='Path to the output folder')
    parser.add_argument('--output_name','-o', type=str, default=None, help='Name of the output file')
    parser.add_argument('--data_file', '-d', type=str, default = None, help='Path to the data file')
    parser.add_argument('--defence_config','-c', type=str, default = None, help='Path to the attack config file')
    cli_args = parser.parse_args()
    params = dvc.api.params_show()[cli_args.layer_name]
    args = argparse.Namespace(**params)
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    if not os.path.exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        os.mkdir(args.output_folder)
    ART_DATA_PATH = args.output_folder
    
    
    
    
    
    # params = dvc.api.params_show()
    # args = argparse.Namespace(**params[cli_args.layer_name])
    # for k, v in vars(cli_args).items():
    #     if v is not None and k in params:
    #         setattr(args, k, v)
    # if not os.path.exists(args.output_folder):
    #     logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
    #     os.mkdir(args.output_folder)
    # ART_DATA_PATH = args.output_folder
    # old_files = [path for path in Path(args.input_folder).rglob('*' + args.input_name)]
    # new_folders = [str(file).replace(args.input_folder, args.output_folder).replace(args.input_name, "") for file in old_files]
    # for folder in tqdm(new_folders, desc = "Adding preprocessor to each model"):
    #     Path(folder).mkdir(parents=True, exist_ok=True)
    #     assert(os.path.isdir(folder))
    #     preprocess(args, model_list = old_files, sub_folder = folder)
    
    
    
    defend(args)
    # load dataset
    