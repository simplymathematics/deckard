from deckard.base.utils import load_data, load_experiment, load_model
from deckard.base import Data, Experiment, Model
from os import path
    
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
    parser.add_argument('--verbosity', type=str, default='INFO', help='Verbosity level.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--target', type=str, required = True, help='Target string.')
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    assert path.isdir(args.folder), '{} is not a valid folder.'.format(args.folder)
    if str(args.data.endswith('.csv')):
        logging.info("Loading {}".format(path.join(args.folder, args.data)))
        data = Data(args.data, test_size=1)
    elif str(args.data.endswith('.pkl')):
        data = load_data(path.join(args.folder, args.data))
    else:
        raise ValueError('{} is not a valid filetype.'.format(args.data.split['.'][-1]))
    model = load_experiment(path.join(args.folder, 'best_features', 'experiment.pkl')).model
    assert isinstance(data, Data), 'data is not a valid Data object.'
    assert isinstance(model, Model), 'model is not a valid Model object.'
    experiment = Experiment(data = data, model = model)
    data = Data(args.data, test_size=1, target = args.target)
    assert isinstance(experiment, Experiment), 'experiment is not a valid Experiment object.'
    experiment.run()
    end = process_time()
    logging.info('Evaluation took {} seconds.'.format(end - start))
    logging.info('Number of evaluated sample: {}'.format(len(experiment.data.y_test)))
    logging.info('Time per sample: {} seconds'.format(round((end - start) / len(experiment.data.y_test),3)))
    print(experiment.scores[args.scorer.upper()])
    logging.debug(experiment.scores)

