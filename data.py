import logging
import pandas as pd
import numpy as np
import json
from hashlib import md5
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, fetch_openml
from copy import deepcopy
import georinex as gr
#TODO: Balanced test set and train set options and functions

class Data(object):
    """
    Creates data object from dataset string and other parameters. In lieu of pre-specified dataset, you can pass in an arbitrary dictionary with keys 'data' and 'target'. 
    """
    def __init__(self, dataset:str = 'iris',  sample_size:float = .1, random_state=0, test_size=0.2, shuffle:bool=True, flatten:bool = True, stratify = None):
        self.dataset = dataset
        self.random_state = random_state
        self.test_size = test_size
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.stratify = stratify
        self.flatten = flatten
        self.X_train, self.y_train, self.X_test, self.y_test = self._choose_data(dataset)
        self.params = {'X_train': self.X_train, 'X_test': self.X_test, 'y_train': self.y_train, 'y_test': self.y_test, 'dataset': self.dataset, 'sample_size': self.sample_size, 'random_state': self.random_state, 'test_size': self.test_size, 'shuffle': self.shuffle, 'stratify': self.stratify}
    def __hash__(self) -> str:
        return int(hash(str(self.params)))
    def __eq__(self, other) -> bool:
        if self.params == other.params:
            return True
        else:
            return False

    def _choose_data(self, dataset:str='iris'):
        import os
        logging.info("Preparing %s data", dataset)
        # lowercase filename
        dataset = dataset.lower()
        # load the data
        if dataset == 'iris':
            data = load_iris()
        # check if file exists and is a csv
        elif dataset == 'mnist':
            data = load_digits()
        elif dataset == 'cifar10':
            data = fetch_openml('cifar10')
        elif dataset == 'rinex-obs':
            obs_file = '../2021.09.17_jamming/parsed-rinex/COM37_210917_110602_jamming.obs'
            obs = gr.load(obs_file)
            obs_df = obs.to_dataframe()
            obs_df.fillna(value = 0, axis = 1)
            obs_df = obs_df.sort_index()
            labels = np.zeros(len(obs_df))
            tmp = int(12350/5)
            tmp2 = tmp * 2
            labels[tmp2: tmp2+tmp] = np.ones(tmp)
            assert set(labels) == set([0,1])
            obs_df = np.nan_to_num(obs_df)
            data = {'data': obs_df, 'target': labels}
        elif os.path.isfile(dataset) and dataset.endswith('.csv'):
           raise NotImplementedError("CSV files not implemented yet")
        elif isinstance(dataset, dict) and isinstance(dataset['data'], object) and isinstance(dataset['target'], object):
            data = dataset
        else:
            raise ValueError("%s dataset not supported" % dataset)
        logging.info("Loaded %s data" % dataset)
        # log the type of data
        # check if data is a dict
        assert isinstance(data, dict)
        assert isinstance(data['data'], object)
        assert isinstance(data['target'], object)
        # log data shape
        logging.debug("Data shape: %s" % str(data['data'].shape))
        logging.debug("Target shape: %s" % str(data['target'].shape))
        if self.flatten == True:
            logging.debug("Flattening dataset.")
            data = self._flatten_dataset(data)
        new_X, new_y = self._sample_data(data)
        self = self._split_data(new_X, new_y)
        return (self.X_train, self.y_train, self.X_test, self.y_test)

    def _flatten_dataset(self, data, **kwargs):
        X = data['data']
        y = data['target']
        logging.debug("X type: %s" % str(type(X)))
        logging.debug("y type: %s" % str(type(y)))
        logging.debug("X shape: %s" % str(X.shape))
        logging.debug("y shape: %s" % str(y.shape))
        try:
            X = X.values.reshape(X.shape[0], -1)
        except AttributeError:
            X = X.reshape(X.shape[0], -1)
        data['data'] = X; data['target'] = y
        assert isinstance(data, dict)
        assert isinstance(data['data'], object)
        assert isinstance(data['target'], object)
        logging.info("Flattening succesful")
        return data

    def _sample_data(self, data, sample_size:float = 1, shuffle:bool=True, **kwargs):
        """
        Samples the dataset
        """
        logging.debug(str(type(data)))
        logging.info("Sampling dataset")
        logging.info('Sample percentage: ' + str(sample_size * 100) + '%')

        assert isinstance(data, dict)
        assert isinstance(data['data'], object)
        assert isinstance(data['target'], object)
        if sample_size < 1:       
            _, new_X, _, new_y = train_test_split(data['data'], data['target'], **kwargs, test_size=sample_size, shuffle=shuffle)
            new_X = pd.DataFrame(new_X)
            new_y = pd.Series(new_y)
        else:
            new_X = pd.DataFrame(data['data'])
            new_y = pd.Series(data['target'])
        assert len(new_X) == len(new_y)
        return (new_X, new_y)

    def _split_data(self, X, y, test_size:float=0.2, random_state:int=0, stratify:pd.Series=None, balanced: bool = False, **kwargs) -> tuple:
        logging.debug("Splitting data")
        # split the data
        if self.dataset != 'mnist' or 'cifar10':
            pass
        else:
            raise NotImplementedError
        assert len(X) == len(y)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        y = y.ravel()
        assert len(X) == len(y)
        logging.info("X shape split" + str(X.shape))
        logging.info("y shape split" + str(y.shape))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, **kwargs)
        assert len(self.X_train) == len(self.y_train)
        assert len(self.X_test) == len(self.y_test)
        logging.debug("Data Successfully Split.")
        return (self)


def validate_data(data:Data) -> None:
    assert isinstance(data, Data), "Data object not valid."
    assert isinstance(data.params, dict), "Params not a dict."
    assert isinstance(data.params['dataset'], str), "Params do not specify dataset."
    assert isinstance(data.params['test_size'], float), "Params do not specify test size"
    assert isinstance(data.params['random_state'], int), "Params do not specify random state"
    assert isinstance(data.params['stratify'], pd.Series) or isinstance(data.params['stratify'], object), "Params do not specify stratification."
    assert len(data.X_train) == len(data.y_train), "Train sets not the same size"
    assert len(data.X_test)  == len(data.y_test), "Test sets not the same size"
    assert isinstance(data.params['shuffle'], bool), "Shuffle not specified"
    logging.debug("Data type: {}".format(type(data)))
    logging.debug("X train type: {}".format(str(type(data.X_train))))
    logging.debug("y train type: {}".format(str(type(data.y_train))))
    logging.debug("X test type: {}".format(str(type(data.X_test))))
    logging.debug("y test type: {}".format(str(type(data.y_test))))
    logging.debug("Data shape: {}".format(str(data.X_train.shape)))
    logging.debug("Target shape: {}".format(str(data.y_train.shape)))
    logging.debug("Data validation successful")
    return None

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Testing data module")
    data = Data()
    validate_data(data)
    sys.exit(0)

    

    