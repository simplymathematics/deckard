from catboost import train
from deckard.base.utils import load_data, load_model
from deckard.base import Data, Experiment, Model
from os import path
from json import dump
import logging 
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    from time import process_time
    start = process_time()
    import argparse
    import logging
    

    # command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('-f', '--folder', type=str, help='Folder containing the checkpoint.', required=True)
    parser.add_argument('-s', '--scorer', type=str, required = True, help='Scorer string.')
    parser.add_argument('-d', '--data', type=str, required = True, help='Data string.')
    parser.add_argument('-o', '--output', type=str, help='Output file.', default = 'results.json')
    parser.add_argument('--verbosity', type=str, default='INFO', help='Verbosity level.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--target', type=str, default = None, help='Target string.')
    parser.add_argument('--experiment', type=str, required = True, help='Experiment string.')
    parser.add_argument('--model_name', type=str, default = "model.pkl", help='name of the experiment')
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    assert path.isdir(args.folder), '{} is not a valid folder.'.format(args.folder)
    if str(args.data.endswith('.csv')):
        logger.info("Loading {}".format(path.join(args.folder, args.data)))
        data = Data(args.data, train_size=1)
    elif str(args.data.endswith('.pkl')):
        data = load_data(path.join(args.folder, args.data))
    else:
        raise ValueError('{} is not a valid filetype.'.format(args.data.split['.'][-1]))
    model = load_model(path.join(args.folder, args.experiment, args.model_name))
    assert isinstance(data, Data), 'data is not a valid Data object.'
    assert isinstance(model, Model), 'model is not a valid Model object.'
    data = Data(args.data, train_size=0, target = args.target)
    experiment = Experiment(data = data, model = model)
    assert isinstance(experiment, Experiment), 'experiment is not a valid Experiment object.'
    experiment.run()
    end = process_time()
    logger.info('Evaluation took {} seconds.'.format(end - start))
    logger.info('Number of evaluated sample: {}'.format(len(experiment.data.y_test)))
    logger.info('Time per sample: {} seconds'.format(round((end - start) / len(experiment.data.y_test),3)))
    if 'ROC_AUC' in experiment.scores:
        del experiment.scores['ROC_AUC'] # ROC_AUC cannot be json serialized
    experiment.save_results( args.output)
    assert path.exists(args.output), '{} is not a valid file.'.format(args.output)

