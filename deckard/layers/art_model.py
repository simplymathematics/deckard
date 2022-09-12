
import logging, os
from deckard.base import Experiment, Model, Data
logger = logging.getLogger(__name__)


def art_model(args) -> None:
    model_object = Model(model_type = args.model_type, path = args.output_folder, model = args.output_name, url = args.input_model, art = True)
    # load dataset
    data = Data(dataset = args.data_file)
    # logger.info("Loaded dataset {}".format(args.data_file))
    # Create experiment
    experiment = Experiment(data = data, model = model_object, filename = args.output_name, is_fitted=True)
    logger.info("Created experiment object from {} dataset and {} model".format(args.data_file, args.output_name))
    experiment(path = args.output_folder,  filename=args.output_name)
    return None

if __name__ == '__main__':
    # arguments
    import argparse
    import dvc.api
    parser = argparse.ArgumentParser(description='Prepare model and dataset as an experiment object. Then runs the experiment.')
    parser.add_argument('--input_model', '-m', type=str, default=None, help='Name of the model')
    parser.add_argument('--input_folder', '-i', type=str, default = ".", help='Path to the model')
    parser.add_argument('--model_type', '-t', type=str, default=None, help='Type of the model')
    parser.add_argument('--output_folder', '-p', type=str, help='Path to the output folder')
    parser.add_argument('--output_name','-o', type=str, default=None, help='Name of the output file')
    parser.add_argument('--data_file', '-d', type=str, default = "data.pkl", help='Path to the data file')
    parser.add_argument('--layer_name','-l', type=str, required = True, help='Name of layer, e.g. "attack"')
    parser.add_argument('--config','-c', type=str, default = None, help='Path to the attack config file')
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    if not os.path.exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        os.mkdir(args.output_folder)
    ART_DATA_PATH = args.output_folder
    art_model(args)