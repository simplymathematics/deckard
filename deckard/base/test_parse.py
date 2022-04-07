import warnings

from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import unittest
from data import Data
from model import Model
import numpy as np
from experiment import Experiment
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.svm import SVC
from copy import deepcopy
from collections.abc import Callable
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.defences.preprocessor import FeatureSqueezing
from art.defences.postprocessor import HighConfidence
# suppress all warnings
from parse import parse_list_from_yml, generate_object_list, generate_uninitialized_object_list, \
transform_params, generate_grid_search_list, generate_experiment_list, insert_layer_into_model, \
insert_layer_into_list
# import tmp from os
import tempfile

class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.file = 'test_filename'

    def test_parse_list_from_yml(self):
        pass

    def test_generate_object_list(self):
        pass

    def test_generate_uninitialized_object_list(self):
        pass

    def test_transform_params(self):
        pass

    def test_generate_grid_search_list(self):
        pass

    def generate_grid_search_list(self):
        pass

    def generate_experiment_list(self):
        pass

    def test_insert_layer_into_model(self):
        pass

    def test_insert_layer_into_list(self):
        pass

    def tearDown(self):
        import shutil
        shutil.rmtree(self.path)