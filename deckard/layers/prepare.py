from deckard.base import Data
import logging, os, argparse
import argparse
import dvc.api
from .utils import make_output_folder

logger = logging.getLogger(__name__)

def prepare(args) -> None:
    data = Data(dataset = args.config.name, **args.config.params)
    if args.data_file != None:
        logger.info("Saving file as {}.".format(os.path.join(args.outputs.folder, args.output.file)))
        data.save(args.data_file)
        logger.info("Data successfully saved.")
    else:
        raise ValueError("No data file specified.")
    return None

if __name__ == '__main__':
    # arguments
    
    params = dvc.api.params_show()
    args = argparse.Namespace(**params['prepare'])
    if not os.path.exists(args.outputs.folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.outputs.folder))
        os.mkdir(args.output.folder)
    print(args)
    prepare(args)
    
    