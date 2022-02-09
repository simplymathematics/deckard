from numpy import isin
from evaluate import *
from copy import copy
from scipy.stats import zscore
import math
import os
from scipy.linalg import svdvals
from art.config import set_data_path
from math import ceil
import pandas as pd
import argparse
import collections
from random import shuffle

#TODO: Run adv patch

filename = 'experiment.log'
logging.basicConfig(
	level = logger.DEBUG,
	format = '%(asctime)s %(name)s %(levelname)s %(message)s',
	filename= filename,
	filemode = 'w'
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('')
logger.setLevel(logger.INFO)
file_logger = logger.FileHandler(filename)
file_logger.setLevel(logger.INFO)
file_logger.setFormatter(formatter)
stream_logger = logging.StreamHandler()
stream_logger.setLevel(logger.DEBUG)
stream_logger.setFormatter(formatter)
logger.addHandler(file_logger)
logger.addHandler(stream_logger)

# TODO pass verbose to model building
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Specify Dataset')
	parser.add_argument('--data', metavar ='d', type = str, help = 'Specify either "cifar-10" or "mnist". Other datasets not supported.', default = 'mnist')
	parser.add_argument('--verbose', metavar ='v', type = bool, help = 'Runs the attacks and model building in verbose mode.', default = True)
	parser.add_argument('--batch_size', metavar ='b', type = int, default = 1024)
	parser.add_argument('--max_iter', metavar = 'i', type = int, default = 10)
	parser.add_argument('--threshold', metavar ='t', type = float, default = .03)
	parser.add_argument('--debug', type = bool, default = False)
	parser.add_argument('--step_size', type = float, default = [.01], nargs = '+')
	parser.add_argument('--defense_bits', type = int, default = 16, nargs = "+", help = "Varies the bit depth for defenses that require it.")
	parser.add_argument('--attack_size', type = int, nargs = "+", default = [10], help = 'Pass a list of attack sizes, separated by space.')
	parser.add_argument('--defense_scale', type = float, default = .03, nargs = "+", help = "Varies the Gaussian scaling for defenses that use that as a parameter.")
	parser.add_argument('--defense_iter', type = int, default = 100, nargs = "+", help = "Varies the Gaussian scaling for defenses that use that as a parameter.")
	args = parser.parse_args()
	in_data = args.data
	attack_sizes = args.attack_size
	VERBOSE = args.verbose
	BATCH_SIZE = args.batch_size
	MAX_ITER = args.max_iter
	threshold = args.threshold
	debug = args.debug
	step_sizes = args.step_size
	
	# TODO: Add support for non-default storage
	set_data_path("/Home/staff/cmeyers/.art/data/")

	if in_data == 'mnist':
		path = get_file('mnist_cnn_original.h5', extract=False, path=ART_DATA_PATH, url="https://www.dropbox.com/s/bv1xwjaf1ov4u7y/mnist_ratio%3D0.h5?dl=1", verbose = True)
		folder = ART_DATA_PATH + '/mnist/'
	elif in_data == 'cifar10':
		path = get_file('cifar-10_original.h5',extract=False, path=ART_DATA_PATH, url='https://www.dropbox.com/s/hbvua7ynhvara12/cifar-10_ratio%3D0.h5?dl=1', verbose = True)
		folder = ART_DATA_PATH + '/cifar10/'
	else:
		logger.debug(data + " not supported")
	if debug == True:
		attack_sizes = [1]
	da = load_data(in_data)


# folder = str(in_data) + '/'
	classifier_model = load_model(path)
	keras_benign = KerasClassifier( model=classifier_model, use_logits=False)
	if debug == True:
		train_sizes = [100]
		folder = 'debug/' + in_data + "/"
		VERBOSE = True
		if not exists('debug'):
			mkdir('debug')
		if not exists(folder):
			mkdir(folder)
	if exists(folder):
		pass
	else:
		mkdir(folder)
	data = (in_data, da )
	classifier = ("Default Classifier", keras_benign)
	# Constructs and Launches all of the attacks
	attacks = {
		'PGD' : ProjectedGradientDescent(keras_benign, eps=threshold, eps_step=.1, batch_size = BATCH_SIZE, max_iter=MAX_ITER, targeted=False, num_random_init=False, verbose = VERBOSE),
#       'Lipschitz' : ProjectedGradientDescent(keras_benign, eps=threshold, eps_step=optimal_step, batch_size = BATCH_SIZE, max_iter=ceil(1/optimal_step), targeted=False, num_random_init=False, verbose = VERBOSE),
		'FGM': FastGradientMethod(keras_benign, eps = threshold, eps_step = .1, batch_size = BATCH_SIZE), 
		'Carlini': CarliniLInfMethod(keras_benign, verbose = VERBOSE, confidence =  1- threshold, max_iter = MAX_ITER, batch_size = 1), 
		'PixelAttack': PixelAttack(keras_benign, th = int(round(threshold*256)), verbose = VERBOSE), 
		'DeepFool': DeepFool(classifier = KerasClassifier(model=classifier_model, clip_values = [0,255]), batch_size = BATCH_SIZE, verbose = VERBOSE), 
		'HopSkipJump': HopSkipJump(keras_benign, max_iter = MAX_ITER, verbose = VERBOSE),
		# 'A-PGD' : AutoProjectedGradientDescent(keras_benign, eps = threshold, max_iter=MAX_ITER, batch_size = BATCH_SIZE),
		'AdversarialPatch': AdversarialPatch(classifier = KerasClassifier(model=classifier_model, clip_values = [0,255]), max_iter = MAX_ITER, verbose = VERBOSE),
		'Threshold Attack': ThresholdAttack(keras_benign, th = threshold, verbose = VERBOSE),
		}

	# Enable the below for an extended search space for PGD, as an example
	step_sizes = [.001, .01, .1]
	norms = [1, 2, 'inf']
	attacks = generate_variable_attacks(attacks, step_sizes, 'step_size', attack_key = 'PGD')
	attacks = generate_variable_attacks(attacks, norms, 'norm', attack_key = 'PGD')
	attacks = generate_variable_attacks(attacks, step_sizes, 'step_size', attack_key = 'FGM')
	attacks = generate_variable_attacks(attacks, norms, 'norm', attack_key = 'FGM')
	logger.info("Number of Attacks: "+ str(len(attacks.values())))
	experiments = {}
	new_folder = folder + "classifiers/"
	logger.info("Reading from " + new_folder)
	i = 0
	files = list(os.listdir(new_folder))
	total = len(files) * len(attacks)
	shuffle(files)
	for filename in files:
		if filename.endswith(".model"):
			with open(new_folder + filename, 'rb') as file:
				res = pickle.load(file)
			experiments[res.id] = res
			for experiment_name, experiment in experiments.items():
				for attack_name, attack in attacks.items():
					total = str(len(experiments) * len(attacks))
					logger.info(str(i+1)+" of "+ str(total))
					i += 1
					results = {}
					#try:
					results = generate_attacks(experiment, attack, attack_name, data, attack_sizes = attack_sizes, folder = folder + 'results/', max_iter = MAX_ITER, batch_size = BATCH_SIZE, threshold = threshold)
					# results = append_results(results, folder)
                                        #except AttributeError as e:
					#	logger.debug(experiment.attack_name, " failed against ", experiment.def_name)
					#del results
				old_name = new_folder + experiment.classifier_id + ".model"
				if not exists(new_folder):
					os.makedirs(new_folder)
				try:
					new_name = new_folder + experiment.classifier_id + ".attacked"
					os.rename(old_name, new_name)
				except:
					logger.info("Folder: " + new_name)
					logger.info(new_name + " unable to save.")
		else:
			logger.info("Skipping " + filename)


	 # Runs each attack against every model
	
