# import os, json, logging, pickle
# from .model import Model
# from .data import Data
# from art.estimators.classification import PyTorchClassifier, KerasClassifier, TensorFlowClassifier, SklearnClassifier
# from art.estimators import ScikitlearnEstimator
# from art.defences.preprocessor import Preprocessor
# from art.defences.postprocessor import Postprocessor
# from art.defences.trainer import Trainer
# from art.defences.transformer import Transformer
# from art.utils import get_file
# logger = logging.getLogger(__name__)

# def find_successes(input_folder, filename:str, dict_name:str = None):
#         failures = []
#         successes = []
#         for folder in os.listdir(input_folder):
#             if os.path.isdir(os.path.join(input_folder, folder)):
#                 files = os.listdir(os.path.join(input_folder, folder))
#                 if 'scores.json' or 'adversarial_scores.json' in files:
#                     if dict_name is not None:
#                         with open(os.path.join(input_folder, folder, filename)) as f:
#                             model_params = json.load(f)[dict_name]
#                     else:
#                         with open(os.path.join(input_folder, folder, filename)) as f:
#                             model_params = json.load(f)
#                     model_name = model_params['name']
#                     model_params = model_params['params']
#                     successes.append((model_name, model_params))
#                 else:
#                     failures.append((model_name, model_params))
#             else:
#                 files = os.listdir(input_folder)
#                 if 'scores.json' or 'adversarial_scores.json' in files:
#                     if dict_name is not None:
#                         with open(os.path.join(input_folder, filename)) as f:
#                             model_params = json.load(f)[dict_name]                       
#                     else:
#                         with open(os.path.join(input_folder, filename)) as f:
#                             model_params = json.load(f)
#                     if 'Name' in model_params.keys():
#                         model_name = model_params['Name']
#                     else:
#                         model_name = str(type(model_params['model'])).split("'")[1]
#                     if 'params' in model_params.items():
#                         model_params = model_params['params']
#                     else:
#                         model_params = model_params
#                     successes.append((model_name, model_params))
#                 else:
#                     failures.append((model_name, model_params))           
#         return successes, failures

# def remove_successes_from_queue(successes, todos):
#     completed_tasks = []
#     for model_name, model_params in successes:
#         i = 0
#         for queue_name, queue_params in todos:
#             queue_name = queue_name.split(".")[-1]
#             i += 1
#             if model_name.split("'")[1].split(".")[-1] == queue_name and queue_params.items() <= model_params.items():
#                 completed_tasks.append(i)    
#     todos = [i for j, i in enumerate(todos) if j not in completed_tasks]
#     return todos
    


# SUPPORTED_DEFENSES = (Postprocessor, Preprocessor, Transformer, Trainer)
# SUPPORTED_MODELS = (PyTorchClassifier, ScikitlearnEstimator, KerasClassifier, TensorFlowClassifier)

