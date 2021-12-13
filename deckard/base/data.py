import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits
import georinex as gr
#TODO: Balanced test set and train set options and functions

class Data(object):
    """
    Creates data object from base.dataset string and other parameters. In lieu of pre-specified dataset, you can pass in an arbitrary dictionary with keys 'data' and 'target'. 
    """
    def __init__(self, dataset:str = 'iris', target = None, sample_size:float = .1, random_state=0, test_size=0.2, shuffle:bool=False, flatten:bool = True, stratify = None):
        self.random_state = random_state
        self.test_size = test_size
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.stratify = stratify
        self.flatten = flatten
        self.target = target
        self.X_train, self.y_train, self.X_test, self.y_test = self._choose_data(dataset)
        self.params = {'dataset': self.dataset, 'sample_size': self.sample_size, 'random_state': self.random_state, 'test_size': self.test_size, 'shuffle': self.shuffle, 'stratify': self.stratify, 'flatten': self.flatten}
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
        # load the data
        if dataset.lower() == 'iris':
            data = load_iris()
            self.dataset = dataset
        # check if file exists and is a csv
        elif dataset.lower() == 'mnist':
            data = load_digits()
            self.dataset = dataset
        elif dataset.endswith('.csv'):
            df = pd.read_csv(dataset)
            if self.target is None:
                logging.warning("Target not specified. Assuming last column is the target column.")
                input = df.iloc[:,:-1]
                target = df.iloc[:,-1]
            else:
                logging.info("Target specified: %s", self.target)
                target = df.pop(self.target)
                input = df
            data = {'data': input, 'target': target}
            self.dataset = dataset
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
        assert len(X) == len(y)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        y = y.ravel()
        assert len(X) == len(y)
        logging.info("X shape split" + str(X.shape))
        logging.info("y shape split" + str(y.shape))
        if test_size < 1:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=stratify, test_size=test_size, random_state=random_state)
        else:
            logging.warning("No training set specified.")
            self.X_test, self.y_test = X, y
            self.X_train, self.y_train = np.ndarray(), np.ndarray()
            logging.debug("Data Successfully Split.")
        assert len(self.X_train) == len(self.y_train)
        assert len(self.X_test) == len(self.y_test)
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

    

    