import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import unittest
import os
import tempfile
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from collections.abc import Callable
from deckard.base import Data, Experiment
from deckard.base.parse import generate_object_list, generate_tuple_list_from_yml, parse_list_from_yml,  \
    transform_params_for_pipeline, generate_sklearn_experiment_list, \
    insert_layer_into_pipeline, insert_layer_into_pipeline_list
class testParse(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.file = 'test_filename'
        self.configs = [x for x in os.listdir("../../../examples/gps_noise/configs") if x.endswith('.yml')]
        self.yml_lists = []
        self.config_folder = "../../../examples/gps_noise/configs"
        self.model_yml = "../../../examples/gps_noise/configs/model.yml"
        self.preprocess_yml = "../../../examples/gps_noise/configs/preprocess.yml"
        self.attack_yml = "../../../examples/gps_noise/configs/attack.yml"        
        self.model = DecisionTreeClassifier()
        self.pipeline = [('model', self.model)]
        self.grid = GridSearchCV(self.model, {'max_depth': [1, 2]})
        self.preprocessor = [('preprocess', self.grid)]

    def test_parse_list_from_yml(self):
        yml_list = parse_list_from_yml(self.model_yml)
        self.assertIsInstance(yml_list, list)
        self.assertIsInstance(yml_list[0], dict)
        self.assertIn('name', yml_list[0])
        self.assertIn('params', yml_list[0])
    
    def test_generate_tuple_list_from_yml(self):
        yml_list = parse_list_from_yml(self.model_yml)
        full_list = generate_tuple_list_from_yml(self.model_yml)
        self.assertIsInstance(full_list, list)
        self.assertIsInstance(full_list[0], tuple)
        self.assertIsInstance(full_list[0][0], str)
        self.assertTrue(len(full_list) > len(yml_list))

    def test_generate_object_list(self):
        full_list = parse_list_from_yml(self.model_yml)
        tuple_list = generate_object_list(full_list)
        self.assertIsInstance(tuple_list, list)
        self.assertIsInstance(tuple_list[0], object)
        self.assertIsInstance(tuple_list[0][0], object)


    def test_transform_params_for_pipeline(self):
        full_list = parse_list_from_yml(self.model_yml)
        obj_list = generate_object_list(full_list)
        pipe_list = transform_params_for_pipeline(obj_list, 'model')
        self.assertIsInstance(pipe_list, list)
        self.assertIsInstance(pipe_list[0], tuple)
        self.assertIsInstance(pipe_list[0][0], object)
        self.assertIsInstance(pipe_list[0][1], dict)
        self.assertIn('model__n_estimators', pipe_list[0][1])

    def test_insert_layer_into_pipeline(self):
        full_list = parse_list_from_yml(self.model_yml)
        obj_list = generate_object_list(full_list)
        pipe_list = transform_params_for_pipeline(obj_list, 'model')
        preprocess = StandardScaler()
        pipeline = Pipeline([('model', pipe_list[0][0])])
        pipeline.set_params(**pipe_list[0][1])
        new_model = insert_layer_into_pipeline(model=pipeline, position = 0, layer = preprocess, params = {}, name = 'preprocess')
        self.assertIsInstance(new_model, Pipeline)

    def test_insert_layer_into_pipeline_list(self):
        # Prepare model
        full_list = parse_list_from_yml(self.model_yml)
        obj_list = generate_object_list(full_list)
        pipe_list = transform_params_for_pipeline(obj_list, 'model')
        model_list = insert_layer_into_pipeline_list(pipe_list)
        model = model_list[0]
        self.assertIsInstance(model, Pipeline)

        # prepare preprocessor
        full_list = parse_list_from_yml(self.preprocess_yml)
        obj_list = generate_object_list(full_list)
        pipe_list = transform_params_for_pipeline(obj_list, 'preprocess')
        preprocess_list = insert_layer_into_pipeline_list(pipe_list, model = model, name = 'preprocess' )
        self.assertIsInstance(preprocess_list, list)
        self.assertIsInstance(preprocess_list[0], Pipeline)



    def test_generate_sklearn_experiment_list(self):
        data = Data('mnist')
        full_list = parse_list_from_yml(self.model_yml)
        obj_list = generate_object_list(full_list)
        experiment_list = generate_sklearn_experiment_list(obj_list, data=data, cv = 10)
        self.assertIsInstance(experiment_list, list)
        self.assertIsInstance(experiment_list[0], Experiment)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.path)