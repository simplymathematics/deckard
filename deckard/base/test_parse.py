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
    transform_params_for_pipeline, generate_sklearn_experiment_list, \
    insert_layer_into_model, insert_layer_into_pipeline
class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.file = 'test_filename'
        self.configs = [x for x in os.listdir("../../examples/gps_noise") if x.endswith('.yml')]
        self.yml_lists = []
        for config in self.configs:
            if config.endswith('.yml'):
                self.yml_lists.append(parse_list_from_yml(config))
                if config == 'preprocess.yml':
                    self.preprocess_yml = parse_list_from_yml(config)
                if config == 'model.yml':
                    self.model_yml = parse_list_from_yml(config)
                if config == 'featurize.yml':
                    self.featurize_yml = parse_list_from_yml(config)
        self.obj_lists = []
        for yml_list in self.yml_lists:
            self.obj_lists.append(generate_object_list(yml_list))

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
        for obj_list in self.obj_lists:
            new_object_list = transform_params_for_pipeline(obj_list, 'model')
            self.assertIsInstance(new_object_list, list)
            self.assertIsInstance(new_object_list[0], tuple)
            self.assertIsInstance(new_object_list[0][0], Callable)
            self.assertIsInstance(new_object_list[0][0], object)
            self.assertIsInstance(new_object_list[0][1], dict)


    def test_insert_layer_into_model(self):
        for config in self.configs:
            print(config)
            input("Press Enter to continue...")
        model_config = parse_list_from_yml(self.model_yml)
        model_list = generate_object_list(model_config)
        transformed_list = transform_params_for_pipeline(transformed_list, 'model')
        exp_list   = generate_sklearn_experiment_list(model_list)
    def test_insert_layer_into_pipeline(self):
        pass

    def generate_sklearn_experiment_list(self):
        pass

    def tearDown(self):
        import shutil
        shutil.rmtree(self.path)