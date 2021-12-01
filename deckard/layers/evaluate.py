from deckard.base.utils import load_checkpoint
from deckard.base import Data, Experiment, Model
from os import path
    
if __name__ == '__main__':
    import argparse

    # command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('-f', '--folder', type=str, help='Folder containing the checkpoint.', required=True)
    parser.add_argument('--scorer', type=str, default='f1', help='Scorer string.')
    parser.add_argument('--verbosity', type=int, default=1, help='Verbosity level.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--output', type=str, help='Output file.')
    args = parser.parse_args()
    
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
    print(experiment.scores[args.scorer.upper()])

