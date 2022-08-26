import numpy as np
import pandas as pd
import os, logging, json, yaml
from deckard.base import Experiment, Model, Data
from deckard.base.scorer import Scorer

crawler_config = {
    "filenames" : [
        'data_params', 'defence_params', 'experiment_params', 'model_params',
        'attack_params', 'predictions', 'adversarial_predictions', 'adversarial_scores', 'scores', 
        'time_dict'
    ],
    "filetype" : 'json',
    "results" : 'results.json',
    "status" : 'status.json',
    "schema" :  [
            'root', 'path', 'data', 'directory', 'layer', 'defence_id', 'attack_id'
        ],
    "root" : '/data/',
}
logger = logging.getLogger(__name__)
class Crawler():
    def __init__(self, config):
        self.config = config
        self.path = self.config['root']
        self.result_file = self.config['results']
        self.status_file = self.config['status']
        self.data = None
    
    def __call__(self, filetypes = ['csv']):
        data = self.crawl_tree()
        data = self.clean_data(data)
        self.save_data()
        return self.data

    def _crawl_folder(self, path = None):
        if path is None:
            path = self.path
        data = {}
        status = {}
        for filename in self.config['filenames']:
            json_file = os.path.join(path, filename + '.' + self.config['filetype'])
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    datum = json.load(f)
                    stat = True
            else:
                stat = False
                datum = None
            data[filename] = datum
            status[filename] = stat
        return data, status
    
    def _crawl_tree(self):
        """
        Crawls the tree and returns a dictionary of dataframes.
        """
        data = {}
        status = {}
        tuples = os.walk(self.path)
        dir_list = [x[0] for x in tuples]
        good_dirs = [x for x in dir_list if "scores.json"  in os.listdir(x)]
        adv_dirs = [x for x in dir_list if "adversarial_scores.json" in os.listdir(x)]
        for dir in good_dirs:
            data[dir], status[dir] = self._crawl_folder(dir)
        for dir in adv_dirs:
            data[dir], status[dir] = self._crawl_folder(dir)
        return data, status
    
    def _save_data(self, data:pd.DataFrame, filename:str):
        with open(filename, 'w') as f:
            json.dump(data, f, indent = 4, sort_keys = True)

    def __call__(self, path = None):
        if path is None:
            data, statuses = self._crawl_tree()
        else:
            data, statuses = self._crawl_tree(path)
        # data = self._clean_data(data)
        logger.debug("Saving file to {}".format(self.result_file))
        self._save_data(data, self.result_file)
        logger.debug("Saving file to {}".format(self.status_file))
        self._save_data(statuses, self.status_file)
        self.data = data
        self.status = statuses
        return self.data