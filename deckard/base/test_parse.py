import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import unittest
import os
import tempfile
from sklearn.exceptions import UndefinedMetricWarning
from copy import deepcopy
from collections.abc import Callable
from parse import parse_list_from_yml, generate_object_list,  \
    transform_params_for_pipeline, generate_experiment_list, \
    insert_layer_into_model, insert_layer_into_pipeline
class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.file = 'test_filename'
        self.configs = [x for x in os.listdir() if x.endswith('.yml')]
        self.yml_lists = []
        for config in self.configs:
            if config.endswith('.yml'):
                self.yml_lists.append(parse_list_from_yml(config))
            else:
                pass
        self.obj_list = []
        for yml_list in self.yml_lists:
            self.obj_list.append(generate_object_list(yml_list))

    def test_parse_list_from_yml(self):
        for yml_list in self.yml_lists:
            self.assertIsInstance(yml_list, list)
            self.assertIsInstance(yml_list[0], dict)
            self.assertIn('name', yml_list[0])
            self.assertIn('params', yml_list[0])

    def test_generate_object_list(self):
        for yml_list in self.yml_lists:
            object_list = generate_object_list(yml_list)
            self.assertIsInstance(object_list, list)
            self.assertIsInstance(object_list[0], tuple)
            self.assertIsInstance(object_list[0][0], Callable)
            self.assertIsInstance(object_list[0][0], object)
            self.assertIsInstance(object_list[0][1], dict)

    def test_transform_params_for_pipeline(self):
        pass

    def test_generate_grid_search_list(self):
        pass

    def generate_grid_search_list(self):
        pass

    def generate_experiment_list(self):
        pass

    def test_insert_layer_into_model(self):
        pass

    def test_insert_layer_into_pipeline(self):
        pass

    def tearDown(self):
        import shutil
        shutil.rmtree(self.path)