import logging, argparse
from deckard.base import Data
from deckard.base.parse import generate_object_list_from_tuple, generate_tuple_list_from_yml, generate_experiment_list
import dvc.api
from os import path, mkdir
logger = logging.getLogger(__name__)


def sklearn_model(args) -> None:
    try:
         data = Data(path.join(args.input_folder, args.data_file))
    except:
        raise ValueError("Unable to load dataset {}".format(path.join(args.input_folder, args.data_file)))
    assert isinstance(data, Data)
    # reads the configs, generating a set of tuples such that
    # each tuple is a combination of parameters a given estimator
    # the length of the list is = len(list of estimators)*len(param_1)*len(param_2)*...len(param_n)
    tuple_list = generate_tuple_list_from_yml(args.config)
    model_list = generate_object_list_from_tuple(tuple_list)
    exp_list = generate_experiment_list(model_list, data, classifier = args.classifier, art = False)
    for exp in exp_list:
        output_folder = path.join(args.output_folder, exp.filename)
        if not path.isdir(output_folder):
            mkdir(output_folder)
        exp(filename = args.output_name, path = output_folder)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a model on a dataset')
    parser.add_argument('--input_folder', '-i', type=str, default = ".", help='Path to the model')
    parser.add_argument('--output_folder', '-p', type=str, help='Path to the output folder')
    parser.add_argument('--output_name','-o', type=str, default=None, help='Name of the output file')
    parser.add_argument('--data_file', '-d', type=str, default = "data.pkl", help='Path to the data file')
    parser.add_argument('--layer_name','-l', type=str, required = True, help='Name of layer, e.g. "attack"')
    parser.add_argument('--config','-c', type=str, default = None, help='Path to the attack config file')
    args = parser.parse_args()
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    if not path.exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        mkdir(args.output_folder)
    ART_DATA_PATH = args.output_folder
    sklearn_model(args)
   
