import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits
import georinex as gr
#TODO: Balanced test set and train set options and functions
from pickle import load

logger = logging.getLogger(__name__)

class Data(object):
    def __init__(self, dataset:str = 'iris', target = None, time_series:bool = False, sample_size:float = .1, random_state=0, test_size=0.2, shuffle:bool=False, flatten:bool = False,  **kwargs):
        """
        Initializes the data object.
        :param dataset: The dataset to use. Can be either a csv file, a string, or a pickled Data object.
        :param target: The target column to use. If None, the last column is used.
        :param time_series: If True, the dataset is treated as a time series. Default is False.
        :param sample_size: The percentage of the dataset to use. Default is 0.1.
        :param random_state: The random state to use. Default is 0.
        :param test_size: The percentage of the dataset to use for testing. Default is 0.2.
        :param shuffle: If True, the data is shuffled. Default is False.
        :param flatten: If True, the dataset is flattened. Default is False.
        """
        self.random_state = random_state
        self.test_size = test_size
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.time_series = time_series
        self.flatten = flatten
        self.target = target
        self.X_train, self.y_train, self.X_test, self.y_test = self._choose_data(dataset, **kwargs)
        self.params = {'dataset': self.dataset, 'sample_size': self.sample_size, 'random_state': self.random_state, 'test_size': self.test_size, 'shuffle': self.shuffle, 'flatten': self.flatten}
    def __hash__(self) -> str:
        """
        Hashes the params as specified in the __init__ method.
        """
        return int(hash(str(self.params)))
    def __eq__(self, other) -> bool:
        """
        Checks if the data is equal to another data object, using the params as specified in the __init__ method.
        """
        if self.params == other.params:
            return True
        else:
            return False

    def _choose_data(self, dataset:str='iris', **kwargs)->tuple:
        """
        Chooses the data to use.
        :param dataset: The dataset to use. Can be either a csv file, a string, or a pickled Data object.
        :param kwargs: passes these to the sklearn train_test_split function.
        :return:
        """
        import os
        logger.info("Preparing %s data", dataset)
        # lowercase filename
        # load the data
        if isinstance(dataset, str) and not dataset.endswith(".pkl"):
            if dataset.lower() == 'iris':
                data = load_iris()
                self.dataset = dataset
            # check if file exists and is a csv
            elif dataset.lower() == 'mnist':
                data = load_digits()
                self.flatten = True
                self.dataset = dataset
            elif dataset.endswith('.csv'):
                df = pd.read_csv(dataset)
                if self.target is None:
                    logger.warning("Target not specified. Assuming last column is the target column.")
                    input = df.iloc[:,:-1]
                    target = df.iloc[:,-1]
                else:
                    logger.info("Target specified: %s", self.target)
                    target = df.pop(self.target)
                    input = df
                data = {'data': input, 'target': target}
                self.dataset = dataset
            else:
                raise ValueError("Dataset must be either 'iris', 'mnist', or a csv file")
            logger.info("Loaded %s data" % dataset)
            # log the type of data
            # check if data is a dict
            assert isinstance(data, dict)
            assert isinstance(data['data'], object)
            assert isinstance(data['target'], object)
            # log data shape
            logger.debug("Data shape: %s" % str(data['data'].shape))
            logger.debug("Target shape: %s" % str(data['target'].shape))
            #logger.debug("Target Set: {}".format(set(data['target'])))
            if self.flatten == True:
                logger.debug("Flattening dataset.")
                data = self._flatten_dataset(data)
            new_X, new_y = self._sample_data(data, **kwargs)
            self = self._split_data(new_X, new_y)
        elif isinstance(dataset, str) and not dataset.endswith('.pkl'):
            self = load(dataset)
        else:
            raise ValueError("%s dataset not supported. You must pass a path to a csv." % dataset)
        return (self.X_train, self.y_train, self.X_test, self.y_test)
        
    def _flatten_dataset(self, data:dict)->dict:
        """
        Flattens the dataset.
        :param data: The dataset to flatten.
        :return:
        """
        X = data['data']
        y = data['target']
        logger.debug("X type: %s" % str(type(X)))
        logger.debug("y type: %s" % str(type(y)))
        logger.debug("X shape: %s" % str(X.shape))
        logger.debug("y shape: %s" % str(y.shape))
        try:
            X = X.values.reshape(X.shape[0], -1)
        except AttributeError:
            X = X.reshape(X.shape[0], -1)
        data['data'] = X; data['target'] = y
        assert isinstance(data, dict)
        assert isinstance(data['data'], object)
        assert isinstance(data['target'], object)
        logger.info("Flattening succesful")
        return data

    def _sample_data(self, data, sample_size:float = 1, shuffle:bool=True, **kwargs):
        """
        Samples the dataset
        :param data: The dataset to sample.
        :param sample_size: The percentage of the dataset to use. Default is 1.
        :param shuffle: If True, the data is shuffled. Default is True.
        :param kwargs: passes these to the sklearn train_test_split function.
        """
        logger.debug(str(type(data)))
        logger.info("Sampling dataset")
        logger.info('Sample percentage: ' + str(sample_size * 100) + '%')
        assert isinstance(data, dict)
        assert isinstance(data['data'], object)
        assert isinstance(data['target'], object)
        if sample_size < 1 and self.time_series == False:       
            _, new_X, _, new_y = train_test_split(data['data'], data['target'], **kwargs, test_size=sample_size, shuffle=shuffle)
            new_X = pd.DataFrame(new_X)
            new_y = pd.Series(new_y)
        elif self.time_series == True or sample_size == 1:
            new_X = pd.DataFrame(data['data'])
            new_y = pd.Series(data['target'])
        else:
            raise ValueError("sample_size must be [0 < sample_size <= 1] or time_series must be True")
        assert len(new_X) == len(new_y)
        return (new_X, new_y)

    def _split_data(self, X, y, test_size:float=0.2, random_state:int=0, balanced: bool = False):
        """
        Splits the dataset into training and testing sets. Returns self with X_train, X_test, y_train, y_test attributes.
        :param X: The dataset to split.
        :param y: The target to split.
        :param test_size: The percentage of the dataset to use for testing. Default is 0.2.
        :param random_state: The random seed to use. Default is 0.
        :param balanced: If True, the data is split into a training and testing set, and the training set is balanced.
        """
        logger.debug("Splitting data")
        # split the data
        assert len(X) == len(y)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        y = y.ravel()
        assert len(X) == len(y)
        logger.info("X shape split" + str(X.shape))
        logger.info("y shape split" + str(y.shape))
        if test_size < 1 and self.time_series == False:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        elif test_size == 1 and self.time_series == False:
            logger.warning("No training set specified.")
            self.X_test, self.y_test = X, y
            self.X_train, self.y_train = np.ndarray(), np.ndarray()
            logger.debug("Data Successfully Split.")
        elif self.time_series == True:
            from sktime.forecasting.model_selection import temporal_train_test_split
            self.y_train, self.y_test = temporal_train_test_split(y, test_size=test_size)
            self.X_train, self.X_test = X.iloc[:len(self.y_train)], X.iloc[len(self.y_train):]
            self.y_train, self.y_test = pd.Series(self.y_train), pd.Series(self.y_test)
        else:
            raise ValueError("test_size must be [0 < test_size <= 1] or time_series must be True")
        assert len(self.X_train) == len(self.y_train)
        assert len(self.X_test) == len(self.y_test)
        return self


def validate_data(data:Data) -> None:
    """
    Validates data object.
    """
    assert isinstance(data, Data), "Data object not valid."
    assert isinstance(data.params, dict), "Params not a dict."
    assert isinstance(data.params['dataset'], str), "Params do not specify dataset."
    assert isinstance(data.params['test_size'], float), "Params do not specify test size"
    assert isinstance(data.params['random_state'], int), "Params do not specify random state"
    assert len(data.X_train) == len(data.y_train), "Train sets not the same size"
    assert len(data.X_test)  == len(data.y_test), "Test sets not the same size"
    assert isinstance(data.params['shuffle'], bool), "Shuffle not specified"
    logger.debug("Data type: {}".format(type(data)))
    logger.debug("X train type: {}".format(str(type(data.X_train))))
    logger.debug("y train type: {}".format(str(type(data.y_train))))
    logger.debug("X test type: {}".format(str(type(data.X_test))))
    logger.debug("y test type: {}".format(str(type(data.y_test))))
    logger.debug("Data shape: {}".format(str(data.X_train.shape)))
    logger.debug("Target shape: {}".format(str(data.y_train.shape)))
    logger.debug("Data validation successful")
    return None

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logger.DEBUG)
    logger.info("Testing data module")
    data = Data()
    validate_data(data)
    sys.exit(0)

    

    