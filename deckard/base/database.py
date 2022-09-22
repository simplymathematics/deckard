import os, logging, argparse, glob, yaml
from pymongo import MongoClient

logger = logging.getLogger(__name__)

# create database
def create_database(db_name):
    """
    Creates a databse using mongodb.
    :param db_name: the name of the database

    """
    client = MongoClient()
    db = client[db_name]
    return db


def connect_to_database(db_name: str, host: str, port: int, **kwargs):
    """
    Connect to a database using mongodb.
    :param db_name: the name of the database
    :param host: the host of the database
    :param port: the port of the database
    :param kwargs: other arguments
    :return: the database
    """
    client = MongoClient(**kwargs)
    db = client[db_name]
    return db


def parse_folders_from_folder(folder: str):
    from glob import glob

    folders = glob(folder + "/*")
    return folders


def parse_files_from_folder(folder: str) -> list:
    from glob import glob

    files = glob(folder + "/*")
    return files


def get_results_from_database(db: str, collection: str, query: dict):
    return db[collection].find(query)


def get_result_from_database(db: str, collection: str, query: dict):
    return db[collection].find_one(query)


def append_result_to_database(db: str, collection: str, result: dict):
    db[collection].insert_one(result)


def append_results_to_database(db: str, collection: str, results: list):
    db[collection].insert_many(results)


def dump_database_to_file(db: str, collection: str, file: str):
    db[collection].to_csv(file)
    assert os.path.isfile(file)
    logger.info("Database saved to {}".format(file))


def get_subset_from_db(db: str, collection: str, query: dict, n: int) -> list:

    """
    Get a subset of the database.
    :param db: the database
    :param collection: the collection
    :param query: the query
    :param n: the number of samples
    """

    return list(db[collection].find(query).limit(n))


if __name__ == "__main__":
    # arg parser
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Add results from pipeline to database.",
    )
    parser.add_argument("-d", "--database", help="Database name", required=True)
    parser.add_argument(
        "-f",
        "--folder",
        help="Folder containing results",
        required=True,
    )
    parser.add_argument("-v", "--verbosity", help="Verbose output", default="DEBUG")
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    args = parser.parse_args()
    db = create_database(args.database)
    # log working directory
    logger.info("Working directory: %s", os.getcwd())
    # log args
    logger.info("Arguments: %s", args)
    collection_names = []
    for folder in parse_folders_from_folder(args.folder):
        if "all_" in folder:
            logger.info("Parsing folder: {}".format(folder))
            for folder in parse_folders_from_folder(folder):
                for file in parse_files_from_folder(folder):
                    if file.endswith(".json"):
                        collection_name = file.split(".")[0].split("\\")[-1]
                        identifier = file.split(".")[0].split("\\")[-2]
                        collection_names.append(collection_name)
                        logger.info(
                            "Adding results from %s to %s",
                            file,
                            collection_name,
                        )
                        from pandas import read_json

                        json_data = read_json(file)
                        logger.info("Type of json_data: {}".format(type(json_data)))
                        # add results to database
                        try:
                            append_results_to_database(
                                db=db,
                                collection=collection_name,
                                results={identifier: json_data},
                            )
                        except:
                            append_result_to_database(
                                db=db,
                                collection=collection_name,
                                result={identifier: json_data},
                            )
                        file.close()
                        # delete file
                        os.remove(file)
                    else:
                        logger.info("Skipping file: %s", file)
        else:
            logger.info("Skipping folder: %s", folder)
    # dump database to file
    collection_names = list(set(collection_names))
    logger.info("Collections: {}".format(collection_names))
    for collection_name in collection_names:
        # log working directory
        logger.info("Working directory: %s", os.getcwd())
        logger.info(
            "Dumping results from %s to %s",
            collection_name,
            collection_name + ".csv",
        )
        dump_database_to_file(db, collection_name, collection_name + ".csv")
