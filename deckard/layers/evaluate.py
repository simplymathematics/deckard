from deckard.base.utils import load_checkpoint
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
    parser.add_argument('-s', '--scorer', type=str, default='f1', help='Scorer string.')
    parser.add_argument('--verbosity', type=str, default='INFO', help='Verbosity level.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--output', type=str, help='Output file.')
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    assert path.isdir(args.folder), '{} is not a valid folder.'.format(args.folder)
    (data, model) = load_checkpoint(folder = path.join(args.folder, 'best_retrain'), model = 'model.pkl', data = 'data.pkl')
    #####
    # import load_model
    # # Does this run on the tail of the log?
    # cmd = "marco.py -n {}".format(args.batch_size)
    # cmd += " -f {}".format(args.folder)
    # cmd += " -o {}".format(args.output)
    # os.system(cmd)
    # data = Data(args.output)
    #####

    model = Model(model)
    assert isinstance(data, Data), 'data is not a valid Data object.'
    assert isinstance(model, Model), 'model is not a valid Model object.'
    experiment = Experiment(data = data, model = model)
    assert isinstance(experiment, Experiment), 'experiment is not a valid Experiment object.'
    
    experiment.run()
    end = process_time()
    logging.info('Evaluation took {} seconds.'.format(end - start))
    logging.info('Number of evaluated sample: {}'.format(len(experiment.data.y_test)))
    logging.info('Time per sample: {} seconds'.format(round((end - start) / len(experiment.data.y_test),3)))
    print(experiment.scores[args.scorer.upper()])
    logging.debug(experiment.scores)

