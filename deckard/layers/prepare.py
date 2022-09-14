from deckard.base import Data
import logging, os, argparse


logger = logging.getLogger(__name__)

def prepare(args) -> None:
    data = Data(dataset = args.name, **args.params)
    if args.data_file != None:
        logger.info("Saving file as {}.".format(os.path.join(args.data_file)))
        data.save(args.data_file)
        logger.info("Data successfully saved.")
    else:
        raise ValueError("No data file specified.")
    return None

if __name__ == '__main__':
    # arguments
    import argparse
    import dvc.api
    parser = argparse.ArgumentParser(description='Prepare model and dataset as an experiment object. Then runs the experiment.')
    parser.add_argument('--output_folder', '-p', type=str, help='Path to the output folder')
    parser.add_argument('--data_file', '-d', type=str, help='Path to the data file')
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    args = argparse.Namespace(**params['prepare'])
    for k, v in vars(cli_args).items():
        if v is not None and not hasattr(args, k):
            setattr(args, k, v)
    if not os.path.exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        os.mkdir(args.output_folder)
    prepare(args)
    
    