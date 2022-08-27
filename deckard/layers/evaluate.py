import os, logging, argparse
from time import process_time
from json import dump
from deckard.base import Data, Experiment, Model

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('-f', '--folder', type=str, help='Folder containing the checkpoint.', required=True)
    parser.add_argument('-d', '--data', type=str, required = True, help='Data string.')
    parser.add_argument('-o', '--output', type=str, help='Output file.', default = 'results.json')
    parser.add_argument('--target', type=str, default = None, help='Target string.')
    parser.add_argument('--input', type=str, required = True, help='Experiment string.')
    parser.add_argument('--model_name', type=str, default = "model.pickle", help='name of the experiment')
    args = parser.parse_args()
    assert os.path.isdir(args.folder), '{} is not a valid folder.'.format(args.folder)
    model = Model(os.path.join(args.folder, args.input, args.model_name))
    data = Data(args.data, train_size=.9, target = args.target)
    assert isinstance(data, Data), 'data is not a valid Data object.'
    assert isinstance(model, Model), 'model is not a valid Model object.'
    experiment = Experiment(data = data, model = model)
    assert isinstance(experiment, Experiment), 'experiment is not a valid Experiment object.'
    experiment.run(path = args.output)
    end = process_time()
    logger.info('Evaluation took {} seconds.'.format(end - start))
    logger.info('Number of evaluated sample: {}'.format(len(experiment.data.y_test)))
    logger.info('Time per sample: {} seconds'.format(round((end - start) / len(experiment.data.y_test),3)))
    experiment.save_results( args.output)
    assert os.path.exists(args.output), '{} is not a valid file.'.format(args.output)

