# Create database to store all results

# Using tinydb
import tinydb

# initalize database
db = tinydb.TinyDB('database.json')

# create result table
result = db.table('result')

# create model params table
model_params = db.table('model_params')

# create data params table
data_params = db.table('data_params')

experiments = open_archive('experiments.json')

# read jsons from folder
import os
import json

def add_json_from_folder(folder):
    # add all jsons from folder to database
    for filename in os.listdir(folder):
        if filename.endswith(".json") and filename.startswith("data_params"):
            with open(os.path.join(folder, filename)) as f:
                data = json.load(f)
                data_params.insert(data)
        elif filename.endswith(".json") and filename.startswith("model_params"):
            with open(os.path.join(folder, filename)) as f:
                data = json.load(f)
                model_params.insert(data)
        elif filename.endswith(".json") and filename.startswith("result"):
            with open(os.path.join(folder, filename)) as f:
                data = json.load(f)
                result.insert(data)

def join_all_tables(tables:list = [data_params, model_params, result]):
    # initalize big table
    big_table = db.table('experiments')
    for table in tables:
        # add table to big table
        big_table.insert_multiple(table.all())
    return big_table

def get_all_experiments():
    # get all experiments
    return db.table('experiments').all()

def check_if_experiment_exists(experiment_id):
    # check if experiment exists
    return db.table('experiments')[experiment_id]

def remove_experiment(experiment_id):
    # remove experiment
    db.table('experiments').remove(eids=[experiment_id])

def open_archive(archive_name):
    # open archive
    return tinydb.TinyDB(archive_name)