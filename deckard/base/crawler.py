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
    "filetypes" : ['json'],
    "results" : 'tmp_results.csv',
    "status" : 'status.json',
    "schema" :  [
            'root', 'path', 'data', 'directory', 'layer', 'defence_id', 'attack_id'
        ],
    "root" : '../data/',
}
logger = logging.getLogger(__name__)
class Crawler():
    def __init__(self, config):
        self.config = config
        self.path = self.config['root']
        self.output = self.config['results']
        self.status = self.config['status']
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
        for filename in self.config['filenames']:
            json_file = os.path.join(path, filename + '.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    datum = json.load(f)
            else:
                datum = None
            data[filename] = datum
        return data
    
    def _crawl_tree(self):
        """
        Crawls the tree and returns a dictionary of dataframes.
        """
        data = {}
        tuples = os.walk(self.path)
        dir_list = [x[0] for x in tuples]
        good_dirs = [x for x in dir_list if "scores.json"  in os.listdir(x)]
        adv_dirs = [x for x in dir_list if "adversarial_scores.json" in os.listdir(x)]
        for dir in good_dirs:
            data[dir] = self._crawl_folder(dir)
        for dir in adv_dirs:
            data[dir] = self._crawl_folder(dir)
        return data
    def _explode_params(self, df:pd.DataFrame):
        new = pd.DataFrame()
        for col in df.columns:
            if isinstance(df[col][0], dict) and 'predictions' not in str(col):
                exploded = pd.DataFrame(pd.json_normalize(df[col]))
                exploded.index = df.index
                new_cols = []
                for col2 in exploded.columns:
                    if 'experiment' == str(col2):
                        col2 == 'id'
                    new_col = col + '_' + col2
                    new_cols.append(new_col)
                exploded.columns = new_cols
                new = pd.concat([new, exploded], axis = 1)
            elif 'predictions' in str(col):
                pass
            else:
                new[col] = df[col]
        return new

    def _merge_data(self, data:pd.DataFrame):

        # Drop This From Results
        try:
            data.drop(['predictions'], axis = 1, inplace = True)
        except KeyError:
            pass
        try:
            data.drop(['adversarial_predictions'], axis = 1, inplace = True)
        except KeyError:
            pass
        # split into adv/ben
        try:
            attack_data = data[data.adversarial_scores.notna()]
        except AttributeError:
            attack_data = data[data.scores.isna()]
        try:
            defense_data = data[data.scores.notna()]
        except AttributeError:
            defense_data = data[data.adversarial_scores.isna()]
        # Replace 'experiment' with '_id'
        attack_data.columns = attack_data.columns.str.replace('_experiment', '_id')
        defense_data.columns = defense_data.columns.str.replace('_experiment', '_id')
        # Drop columns
        attack_data = attack_data.dropna(axis = 1)
        defense_data = defense_data.dropna(axis = 1)
        # # Explode params
        attack_data = self._explode_params(attack_data)
        defense_data = self._explode_params(defense_data)
        # Drop duplicate columns
        attack_data = attack_data.loc[:,~attack_data.columns.duplicated()]
        defense_data = defense_data.loc[:,~defense_data.columns.duplicated()]
        # Parse Attack data from path
        attack_paths = pd.Series(attack_data.index).str.split(os.sep)
        attack_paths = pd.DataFrame(attack_paths.values.tolist())
        no_cols = attack_paths.shape[1]
        attack_paths.columns = self.config['schema'][-no_cols:]
        attack_data = attack_data.reset_index()
        attack_paths = attack_paths.reset_index()
        attack_data = pd.concat([attack_data, attack_paths], axis = 1)
        # Parse Defense Data from path
        defense_paths = pd.Series(defense_data.index).str.split(os.sep)
        defense_paths = pd.DataFrame(defense_paths.values.tolist())
        no_cols = defense_paths.shape[1]
        defense_path_cols = self.config['schema'][-no_cols:]
        defense_paths.columns = defense_path_cols
        defense_data.reset_index(inplace = True)
        defense_paths.reset_index(inplace = True)
        defense_data = pd.concat([defense_data, defense_paths], axis = 1)
        
        # Merge attack and defense data
        df = defense_data.merge(attack_data, on = 'defence_id', how = 'outer')
        df = df.drop([x for x in df.columns if '_y' in str(x)], axis = 1)
        df.columns = [str(x).replace("_x", "") for x in df.columns]
        df.columns = [str(x).replace("adversarial", "adv") for x in df.columns]
        return df
    def _clean_data(self, data):
        self.data = {}
        for dir in data.keys():
            self.data[dir] = {}
            for filename in self.config['filenames']:
                if os.path.isfile(os.path.join(dir, filename + '.json')):
                    value = data[dir][filename]
                    self.data[dir][filename] = value
        self.data = pd.DataFrame(self.data).T
        self.data = self._merge_data(self.data).drop('index', axis = 1)
        return self.data

    def _save_data(self, data:pd.DataFrame = None):
        try:
            filetype = self.output.split('.')[-1]
        except:
            logger.warning('No filetype detected. Defaulting to csv')
            filetype = 'csv'
        if data is None:
            data = self.data
        assert isinstance(data, pd.DataFrame)
        if filetype == 'json':
            data.to_json(self.output)
        elif filetype == 'csv':
            data.to_csv(self.output)
        elif filetype == 'tsv':
            pass
        elif filetype == 'db':
            pass
        else:
            raise ValueError("Filetype not supported.")
        return self.output
    
    def __call__(self, path = None):
        if path is None:
            data = self._crawl_tree()
        else:
            data = self._crawl_tree(path)
        data = self._clean_data(data)
        logger.debug("Saving file to {}".format(self.output))
        self._save_data(data)
        return self.data