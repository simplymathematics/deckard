from deckard.base import Data
import logging, os, yaml

logger = logging.getLogger(__name__)

def parse_data_from_yml(filename:str, obj_type:Data) -> dict:
    assert isinstance(filename, str)
    LOADER = yaml.FullLoader
    # check if the file exists
    params = dict()
    if not os.path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, 'r') as stream:
        try:
            data_file = yaml.load(stream, Loader=LOADER)[0]
            logger.info(data_file)
            logger.info(type(data_file))
        except yaml.YAMLError as exc:
            raise ValueError("Error parsing yml file {}".format(filename))
    # check that datas is a list
    if not isinstance(data_file, dict):
        raise ValueError("Error parsing yml file {}. It must be a yaml dictionary.".format(filename))
    params = data_file['params']
    data_name = data_file['name']
    for param, value in params.items():
        logger.info(param + ": " + str(value))
    data = Data(data_name, **params)
    assert isinstance(data, Data)
    logger.info("{} successfully parsed.".format(filename))
    return data


if __name__ == "__main__":

    from pickle import dump
    # command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('-dataset', type = str, default = 'rinex_obs', help = "The dataset to use")
    parser.add_argument('-folder', type = str, default = "./", help = "The folder where data will be stored")
    parser.add_argument('-config_file', type = str, default = "configs/prepare.yml", help = "The config folder to use")
    parser.add_argument('--verbosity', type = str, default = 'DEBUG', help = "The verbosity level")
    parser.add_argument('--target', type = str, default = None, help = "The target to use")
    args = parser.parse_args()

    # set the verbosity level
    logging.basicConfig(level=args.verbosity)

    data = parse_data_from_yml(args.config_file, Data)
    if args.folder != None:
        if not os.path.exists(os.path.join(args.folder, 'data')):
            os.makedirs(os.path.join(args.folder, 'data'))
        else:
            logger.warning(args.folder + " already exists. Overwriting data.")
    logger.info("Saving file as {}.".format(os.path.join(args.folder, 'data', 'data.pkl')))
    
    dump(data, open(os.path.join(args.folder, 'data', 'data.pkl'), 'wb'))
    logger.info("Data successfully saved.")
