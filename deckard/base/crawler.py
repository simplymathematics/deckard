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
    "scores_files" : 'scores.json',
    "adversarial_scores_file" : 'adversarial_scores.json',
    "schema" :  [
            'root', 'path', 'data', 'directory', 'layer', 'defence_id', 'attack_id'
        ],
    "structured" : ["defence_params", "attack_params", "adversarial_scores", "scores", "time_dict"],
    "db" : {},
    "root_folder" : os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/"),
    "layers" : ['control', 'defences', 'attacks'],
    "exclude" : [],
}
logger = logging.getLogger(__name__)
class Crawler():
    def __init__(self, config):
        self.config = config
        self.path = os.path.realpath(self.config['root_folder'])
        self.result_file = self.config['results']
        self.status_file = self.config['status']
        self.structured = self.config['structured']
        self.filetype = self.config['filetype']
        self.layers = self.config['layers']
        self.data = None
    
    def __call__(self, filetypes = ['csv']):
        data = self.crawl_tree()
        data = self.clean_data(data)
        self.save_data()
        return self.data

    def crawl_folder(self, path = None):
        if path is None:
            path = self.path
        data = {}
        status = {}
        for filename in self.config['filenames']:
            logging.debug("Crawling folder: {} for {}".format(path, filename))
            json_file = os.path.join(path, filename + '.' + self.filetype)
            # logging.info("json file is {}".format(json_file))
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
    
    def crawl_layer(self, path:str = None):
        """
        Crawls the tree and returns a dictionary of dataframes.
        """
        logging.info("Crawling layer: {}".format(path))
        if path is None:
            path = self.path
        data = {}
        status = {}
        for root, dirs, files, there in os.fwalk(path):
            for directory in dirs:
                parents = os.path.relpath(root).split(os.sep)
                depth =  len(parents)- len(os.path.relpath(path).split(os.sep))
                if depth > 0:
                    parent = parents[-1]
                elif depth == 0:
                    parent = parents[0]
                else:
                    parent = None
                if any(f.endswith(self.filetype) for f in os.listdir(os.path.join(root, directory))):
                    data[directory], status[directory] = self.crawl_folder(os.path.join(root, directory))
                    data[directory]['parent'] = parent;  status[directory]['parent'] = parent
        return data, status
    
    
    
    def merge_dataframes(self, df1, df2):
        logging.info("Merging dataframes")
        if isinstance(df1, dict):
            df1 = pd.DataFrame.from_dict(df1, orient = 'index')
        if isinstance(df2, dict):
            df2 = pd.DataFrame.from_dict(df2, orient = 'index')
        keep = crawler_config['structured']
        drop = list(set(crawler_config['filenames'])- set(keep))
        df1 = df1.dropna(how = 'all', axis = 1)
        df2 = df2.dropna(how = 'all', axis = 1)
        different = set(df2.columns) - set(df1.columns)
        same = list(set(df2.columns).intersection(set(df1.columns)))
        if 'parent' in same:
            same.remove('parent')
        for col in same:
            df1.drop(col, axis = 1, inplace = True)
        merge = df1.merge(df2, left_index = True, right_on = 'parent', how = 'outer')
        try:
            merge.drop('parent_x', axis = 1, inplace = True)
        except KeyError:
            pass
        try:
            merge['parent'] = merge['parent_y']
            merge.drop('parent_y', axis = 1, inplace = True)
        except KeyError:
            pass
        return merge

    def crawl_tree(self, path = None):
        logging.info("Crawling path: {}".format(path))
        if path is None:
            path = self.path
        big_df = pd.DataFrame()
        big_st = pd.DataFrame()
        for layer in self.layers:
            data, status = self.crawl_layer(os.path.join(path, layer + os.sep))
            if layer == 'control':
                big_df = pd.DataFrame.from_dict(data, orient = 'index').dropna(how = 'all', axis = 1)
                big_st = pd.DataFrame.from_dict(status, orient = 'index').dropna(how = 'all', axis = 1)
            else:
                big_df = self.merge_dataframes(big_df, data)
                big_st = self.merge_dataframes(big_st, status)
        return big_df, big_st


    def save_data(self, data:pd.DataFrame, filename:str):
        data.to_csv(filename, index = True, header = True, mode = 'w')

    def __call__(self, path = None):
        if path is None:
            data, statuses = self.crawl_tree()
        else:
            data, statuses = self.crawl_tree(path)
        # data = self._clean_data(data)
        logging.info("Saving file to {}".format(self.result_file))
        self.save_data(data, self.result_file)
        logging.info("Saving file to {}".format(self.status_file))
        self.save_data(statuses, self.status_file)
        self.data = data
        self.status = statuses
        return data, statuses