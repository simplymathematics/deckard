import logging
from telnetlib import X3PAD
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#TODO: Balanced test set and train set options and functions
from pickle import load

from torch import minimum
from tensorflow.keras.utils import to_categorical 
# mnist dataset from 
from hashlib import md5 as my_hash
logger = logging.getLogger(__name__)
from art.utils import load_dataset

class Data(object):
    """
    :attribute dataset: The dataset to use. Can be either a csv file, a string, or a pickled Data object.
    :attribute target: The target column to use. If None, the last column is used.
    :attribute time_series: If True, the dataset is treated as a time series. Default is False.
    :attribute train_size: The percentage of the dataset to use. Default is 0.1.
    :attribute random_state: The random state to use. Default is 0.
    :attribute shuffle: If True, the data is shuffled. Default is False.
    :attribute X_train: The training data. Created during initialization.
    :attribute X_test: The testing data. Created during initialization.
    :attribute y_train: The training target. Created during initialization.
    :attribute y_test: The testing target. Created during initialization.
    
    """
    def __init__(self, dataset:str = 'iris', target = None, time_series:bool = False, train_size:float = .01, random_state=0, shuffle:bool=True,  stratify=True):
        """
        Initializes the data object.
        :param dataset: The dataset to use. Can be either a csv file, a string, or a pickled Data object.
        :param target: The target column to use. If None, the last column is used.
        :param time_series: If True, the dataset is treated as a time series. Default is False.
        :param train_size: The percentage of the dataset to use. Default is 0.1.
        :param random_state: The random state to use. Default is 0.
        :param train_size: The percentage of the dataset to use for testing. Default is 0.2.
        :param shuffle: If True, the data is shuffled. Default is False.
        :param flatten: If True, the dataset is flattened. Default is False.
        """
        self.random_state = random_state
        self.train_size = train_size
        self.shuffle = shuffle
        self.time_series = time_series
        self.target = target
        self.dataset = dataset
        self.stratify = stratify
        self._choose_data(dataset) # adds X_train, X_test, y_train, y_test attributes to self, using parameters specified above.
        self.params = {'dataset':dataset, 'target':target, 'time_series':time_series, 'train_size':train_size, 'random_state':random_state,  'shuffle':shuffle}
        
    def __hash__(self) -> str:
        """
        Hashes the params as specified in the __init__ method.
        """
        params = self.get_params()
        return my_hash(str(params).encode('utf-8')).hexdigest()

    def __eq__(self, other) -> bool:
        """
        Checks if the data is equal to another data object, using the params as specified in the __init__ method.
        """
        return hash(str(self.params)) == hash(str(other.params))

    def get_params(self):
        """
        Returns the parameters of the data object.
        """
        return self.params

    def set_params(self, params:dict = None):
        """
        :param params: A dictionary of parameters to set.
        Sets the parameters of the data object.
        """
        assert params is not None, "Params must be specified"
        assert isinstance(params, dict), "Params must be a dictionary"
        for key, value in params.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value
            else:
                self.params[key] = value

    def _choose_data(self, dataset:str = 'mnist', target = None, stratify:bool = True) -> None:
        """
        :param dataset: A string specifying the dataset to use. Supports mnist, iris, stl10, cifar10, nursery, diabetes, and an arbitrary csv. 
        :param target: The target column to use. If None, the last column is used.
        Chooses the dataset to use. Returns self.
        """
        # sets target, if specified
        if target is not None:
            self.target = target
        # sets dataset, if specified
        if dataset.endswith(".csv"):
            (X_train, y_train),(X_test, y_test), minimum, maximum = self._parse_csv(dataset, target)
        else:
            (X_train, y_train),(X_test, y_test), minimum, maximum = load_dataset(dataset)
            # TODO: fix this
            # from sklearn.model_selection import train_test_split
            # # sets stratify to None if stratify is False
            # stratify = y_train if self.stratify == True else None
            # big_X = np.append(X_train, X_test, axis=0)
            # big_y = np.append(y_train, y_test, axis=0)
            # assert len(big_X) == len(big_y), "length of X is: {}. length of y is: {}".format(len(big_X), len(big_y))
            # X_train, X_test, y_train, y_test = train_test_split(big_X, big_y, train_size = self.train_size, random_state=self.random_state, shuffle=self.shuffle, stratify=stratify)
        # Standardize the test/train_size to be integers
        if isinstance(self.train_size, float) or self.train_size == 1:
            self.train_size = int(round(len(X_train) * self.train_size))
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset = dataset
        self.clip_values = [minimum, maximum]
    
    def _parse_csv(self, dataset:str = 'mnist', target = None) -> None:
        """
        :param dataset: A string specifying the dataset to use. Supports mnist, iris, stl10, cifar10, nursery, and diabetes
        Chooses the dataset to use. Returns self.
        """
        assert dataset.endswith(".csv"), "Dataset must be a csv file"
        df = pd.read_csv(dataset)
        if target is None:
            self.target = df.columns[-1]
        else:
            self.target = target
        y = df[self.target]
        X = df.drop(self.target, axis=1)
        if self.time_series == False:
            stratify = y if self.stratify == True else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size, random_state=self.random_state, shuffle=self.shuffle, stratify=stratify)
            maximum = max(X_train)
            minimum = min(X_train)
        else:
            raise NotImplementedError("Time series not yet implemented")
        return (X_train, y_train), (X_test, y_test), minimum, maximum

    