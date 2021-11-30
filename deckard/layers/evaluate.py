from deckard.base import utils
from pandas import DataFrame, read_csv, to_csv
from os import path

def create_database(file:str) -> DataFrame:
    """
    Create a csv `database'.
    """
    if path.exists(file):
        db = read_csv(file)
    else:
        db = DataFrame()

if __name__ == "__main__":
    # arg parser
    import argparse
    import logging
    parser = argparse.ArgumentParser(description='Add results from pipeline to database.')
    parser.add_argument('-d', '--database', help='Database name', required=True)
    parser.add_argument('-f', '--folder', help='Folder containing results', required=True)
    parser.add_argument('-v', '--verbosity', help='Verbose output', default = "DEBUG")
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    args = parser.parse_args()
    db = create_database(args.database)
    #log working directory
    logging.info("Working directory: %s", os.getcwd())
    #log args
    logging.info("Arguments: %s", args)
    collection_names = []
    for folder in parse_folders_from_folder(args.folder):
        if "all_" in folder:
            logging.info("Parsing folder: {}".format(folder))
            for folder in parse_folders_from_folder(folder):    
                for file in parse_files_from_folder(folder):
                    if file.endswith(".json"):
                        collection_name = file.split(".")[0].split("\\")[-1]
                        identifier = file.split(".")[0].split("\\")[-2]
                        collection_names.append(collection_name)
                        logging.info("Adding results from %s to %s", file, collection_name)
                        from pandas import read_json
                        json_data = read_json(file)
                        logging.info("Type of json_data: {}".format(type(json_data)))
                        # add results to database
                        print(json_data)
                        try:
                            append_results_to_database(db = db, collection = collection_name, results = {identifier:json_data})
                        except:
                            append_result_to_database(db = db, collection = collection_name, result = {identifier : json_data})
                        file.close()
                        # delete file
                        os.remove(file)
                    else:
                        logging.info("Skipping file: %s", file)
        else:
            logging.info("Skipping folder: %s", folder)
    # dump database to file
    collection_names = list(set(collection_names))
    logging.info("Collections: {}".format(collection_names))
    for collection_name in collection_names:
        # log working directory
        logging.info("Working directory: %s", os.getcwd())
        logging.info("Dumping results from %s to %s", collection_name, collection_name + ".csv")
        dump_database_to_file(db, collection_name, collection_name + ".csv")


if __name__ == '__main__':
    import argparse

    # command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--folder', type=str, default='.', help='Folder containing the checkpoint.')
    parser.add_argument('--scorer', type=str, default='f1', help='Scorer string.')