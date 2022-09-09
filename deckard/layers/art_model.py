
import logging
from .experiment import Experiment, Model, Data
import os
from art.estimators.classification import PyTorchClassifier, SklearnClassifier, KerasClassifier, TensorFlowClassifier
from art.utils import get_file

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # arguments
    import argparse
    parser = argparse.ArgumentParser(description='Prepare model and dataset as an experiment object. Then runs the experiment.')
    parser.add_argument('--input_model', type=str, default=None, help='Name of the model. Can be the name of a file or a URL.')
    parser.add_argument('--input_folder', type=str, default = ".", help='Folder where the model is located. Defaults to folder where the script is run.')
    parser.add_argument('--model_type', type=str, required = True, help='Type of the model')
    parser.add_argument('--dataset', type=str, required = True, help='Path to the dataset')
    # parser.add_argument('--scorer', type=str, required = True, help='Scorer for optimization')
    parser.add_argument('--output_folder', type=str, default=None, help='Path to the output folder')
    parser.add_argument('--output_name', type=str, required = True, help='Name of the output file')
    parser.add_argument('--log_file', type=str, default = "log.txt", help='Path to the log file')    
    # parse arguments
    args = parser.parse_args()

    if args.output_name == None:
        # uuid for model name
        import uuid
        args.output_name = str(uuid.uuid4())
    else:
        logger.info("Model Name is {}".format(args.output_name))
    # attempts to load model
    if not os.path.exists(args.output_folder):
        logger.warning("Model path {} does not exist. Creating it.".format(args.output_folder))
        os.mkdir(args.output_folder)
    model_object = Model(model_type = args.model_type, path = args.output_folder, model = args.output_name, url = args.input_model)
    # load dataset
    data = Data(args.dataset)
    # logger.info("Loaded dataset {}".format(args.dataset))
    # Create experiment
    experiment = Experiment(data = data, model = model_object, name = args.output_name, params = {'model_type':args.model_type, 'model_path':args.input_model, 'dataset':args.dataset})
    logger.info("Created experiment object from {} dataset and {} model".format(args.dataset, args.output_name))
    # run experiment
    logger.info("Running experiment...")
    experiment.run(args.output_folder)
    logger.info("Experiment complete.")
    # Save experiment
    logger.info("Saving experiment.")
    experiment.model.model.save(filename = args.output_folder, path = args.output_name)
    experiment.save_results(path = args.output_folder)
    logger.info("Experiment saved.")