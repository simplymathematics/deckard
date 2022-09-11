import logging
from deckard.base import Model, Data, AttackExperiment, Scorer
import os
from deckard.base.parse import generate_object_from_tuple, generate_tuple_from_yml
logger = logging.getLogger(__name__)

def attack(args) -> None:
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
    model = Model(model=args.input_model, model_type =args.model_type, path = args.input_folder)
    try:
        attack = generate_object_from_tuple(generate_tuple_from_yml(args.attack_config))
    except TypeError as e:
        if "missing 1 required positional argument: 'classifier'" in str(e):
            attack = generate_object_from_tuple(generate_tuple_from_yml(args.attack_config), model.model)
        else:
            raise e
    experiment = AttackExperiment(data = data, model = model, is_fitted=True, attack = attack)
    experiment(path = args.output_folder, filename = args.output_name)
    return None

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
    parser.add_argument('--config','-c', type=str, default = None, help='Path to the attack config file')
    parser.add_argument('--attack_size', '-n', type=int, default=None, help='Number of adversarial samples to generate')
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    print(params.keys())
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    if not os.path.exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        os.mkdir(args.output_folder)
    ART_DATA_PATH = args.output_folder
    attack(args)
        
        
        
    