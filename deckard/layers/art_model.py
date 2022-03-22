from os.path import exists
from os import mkdir, chmod, rename
import logging
from deckard.base.experiment import Experiment, Model
from deckard.base.utils import load_data
from deckard.base.data import validate_data
import parser
import os
from art.estimators.classification import PyTorchClassifier, SklearnClassifier, KerasClassifier, TensorFlowClassifier
from art.utils import get_file

logger = logging.getLogger(__name__)

def convert_to_art_classifier(model_name:str, model_path:str, model_type:str=None, output_folder:str = None,):
    """
    Load an ART model.
    :param model_name: the name of the model
    :param model_path: the path to the model
    :param model_type: the type of the model
    :param output_folder: the output folder to save the model to
    :return: the loaded art model
    
    """

    # disable eager execution
    import tensorflow.compat.v1 as tfv1
    tfv1.disable_eager_execution()

    # Download/load model
    assert model_type is not None, "model_type must be specified"
    if 'http' in model_path:
        # download model
        model_path = get_file(filename = model_name, extract=False, path=output_folder, url=model_path, verbose = True)
    elif not os.path.isfile(os.path.join(model_path, model_name)):
        raise FileNotFoundError("Model {} does not exist in {}".format(model_name, model_path))
    else:
        model_path = os.path.join(model_path, model_name)
    # Define type for ART
    if model_type == 'keras' or 'k':
        from tensorflow.keras.models import load_model as keras_load_model
        classifier_model = keras_load_model(model_path)
        benign = KerasClassifier( model=classifier_model)
    elif model_type == 'tf' or 'tensorflow':
        # load model
        from tensorflow.keras.models import load_model as keras_load_model
        classifier_model = keras_load_model(model_path)
        benign = TensorFlowClassifier( model=classifier_model)
    elif model_type == 'tfv1' or 'tensorflowv1' or 'tf1':
        from tensorflow.keras.models import load_model as keras_load_model
        import tensorflow.compat.v1 as tfv1
        tfv1.disable_eager_execution()
        classifier_model = keras_load_model(model_path)
        benign = KerasClassifier( model=classifier_model)
    elif model_type == 'pytorch' or 'py':
        # load model
        from torch import load
        classifier_model = load(model_path)
        benign = PyTorchClassifier( model=classifier_model)
    elif model_type == 'sklearn' or 'sk':
        # load model using pickle
        from pickle import load
        classifier_model = load(model_path)
        benign = SklearnClassifier( model=classifier_model)
    else:
        raise ValueError("Model type {} not supported".format(model_type))
    return benign



if __name__ == '__main__':
    # arguments
    import argparse
    parser = argparse.ArgumentParser(description='Prepare model and dataset as an experiment object. Then runs the experiment.')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the model')
    parser.add_argument('--model_path', type=str, default = ".", help='Path to the model')
    parser.add_argument('--model_type', type=str, required = True, help='Type of the model')
    parser.add_argument('--verbosity', type=int, default=10, help='Verbosity level')
    parser.add_argument('--dataset', type=str, required = True, help='Path to the dataset')
    # parser.add_argument('--scorer', type=str, required = True, help='Scorer for optimization')
    parser.add_argument('--output_folder', type=str, required = True, help='Path to the output folder')
    parser.add_argument('--log_file', type=str, default = "log.txt", help='Path to the log file')
    
    # parse arguments
    args = parser.parse_args()
    # set up logging
    ART_DATA_PATH = os.path.join(args.output_folder)
    # Create a logger
    logger = logging.getLogger(__name__)
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set logger level
    logger.setLevel('DEBUG')
    # Create stream handler
    stream_handler = logging.StreamHandler()
    # set formatting
    stream_handler.setFormatter(formatter)
    # Add handler to logger
    logger.addHandler(stream_handler)

    if args.model_name == None:
        # uuid for model name
        import uuid
        args.model_name = str(uuid.uuid4())
    else:
        logger.info("Model Name is {}".format(args.model_name))
    # attempts to load model
    if not exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.model_path))
        mkdir(args.output_folder)
    benign = convert_to_art_classifier(model_name=args.model_name, model_path=args.model_path, model_type=args.model_type, output_folder=args.output_folder)
    model_object = Model(benign)
    model_object.model_type = args.model_type
    # load dataset
    data = load_data(data_file  = args.dataset)
    validate_data(data)
    # logger.info("Loaded dataset {}".format(args.dataset))
    # Create experiment
    experiment = Experiment(data = data, model = model_object, name = args.model_name, params = {"model_type": args.model_type})
    logger.info("Created experiment object from {} dataset and {} model".format(args.dataset, args.model_name))
    # run experiment
    logger.info("Running experiment...")
    experiment.run()
    logger.info("Experiment complete.")
    # Save experiment
    logger.info("Saving experiment.")
    experiment.save_model(model_name = args.model_name, folder = os.path.join(args.output_folder, "best_train"), move_from = args.output_folder)
    experiment.save_results(folder = os.path.join(args.output_folder, "best_train"))
    logger.info("Experiment saved.")