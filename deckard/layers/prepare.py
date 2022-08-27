from deckard.base import Data
from deckard.base.parse import parse_data_from_yml
import logging, os, yaml, argparse
from pickle import dump

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare the data, according to the config file. Then it saves the Data object to a pickle file.')
    parser.add_argument('-folder', type = str, default = "./", help = "The folder where data will be stored")
    parser.add_argument('-config_file', type = str, default = "configs/prepare.yml", help = "The config folder to use")
    parser.add_argument('--target', type = str, default = None, help = "The target to use")
    args = parser.parse_args()

    data = parse_data_from_yml(args.config_file)
    if args.folder != None:
        if not os.path.exists(os.path.join(args.folder, 'data')):
            os.makedirs(os.path.join(args.folder, 'data'))
        else:
            logger.warning(args.folder + " already exists. Overwriting data.")
    logger.info("Saving file as {}.".format(os.path.join(args.folder, 'data', 'data.pkl')))
    
    dump(data, open(os.path.join(args.folder, 'data', 'data.pkl'), 'wb'))
    logger.info("Data successfully saved.")
