import json, os
import pandas as pd

crawler_config = {
    "filenames" : [
        'data_params', 'defence_params', 'experiment_params', 'model_params',
        'attack_params', 'predictions', 'adversarial_predictions', 'adversarial_scores', 'scores', 
        'time_dict'
    ]
}

class Crawler():
    def __init__(self, config_file, path, output):
        self.config_file = config_file
        self.path = path
        self.output = output
        self.config = crawler_config
        self.data = None
    
    def __call__(self, filetypes = ['csv']):
        self.crawl_tree()
        self.clean_data()
        self.save_data()

    def crawl_folder(self, path = None):
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
    
    def crawl_tree(self):
        data = {}
        tuples = os.walk(self.path)
        dir_list = [x[0] for x in tuples]
        good_dirs = [x for x in dir_list if "scores.json" in os.listdir(x)]
        for dir in good_dirs:
            data[dir] = self.crawl_folder(dir)
        return data
        
    def clean_data(self, data):
        self.data = {}
        for dir in data.keys():
            scores = data[dir]['scores']
            self.data[dir] = scores
            for filename in self.config['filenames']:
                if os.path.isfile(os.path.join(dir, filename + '.json')):
                    if filename == 'adversarial_scores':
                        adv_scores = data[dir]['adversarial_scores']
                        new_scores = {}
                        for key in adv_scores.keys():
                            new_name = "adv_" + key
                            new_scores[new_name] = adv_scores[key]
                        self.data[dir] = new_scores
                    self.data[dir][filename] = os.path.join(dir, filename + '.json')
        self.data = pd.DataFrame(self.data).T
        return self.data

    def save_data(self, data:pd.DataFrame = None, filetype = 'json'):
        if data is None:
            data = self.data
        assert isinstance(data, pd.DataFrame)
        if filetype == 'json':
            filename = os.path.join(self.output + '.json')
            data.to_json(filename)
        elif filetype == 'csv':
            filename = os.path.join(self.output + '.csv')
            data.to_csv(filename)
        elif filetype == 'tsv':
            pass
        elif filetype == 'db':
            pass
        else:
            raise ValueError("Filetype not supported.")
        return filename

