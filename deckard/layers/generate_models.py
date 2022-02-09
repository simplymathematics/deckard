from numpy import isin
from evaluate import *
from copy import copy
from scipy.stats import zscore
import math 
from scipy.linalg import svdvals
from art.config import set_data_path
from math import ceil
import pandas as pd
import argparse
import collections



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
	parser.add_argument('--train_size', type = int, nargs = "+", metavar = 't', default = [100], help = 'Pass a list of training sizes, separated by a space.')
	parser.add_argument('--attack_size', type = int, nargs = "+", default = [10], help = 'Pass a list of attack sizes, separated by space.')
	parser.add_argument('--defense_scale', type = float, default = .03, nargs = "+", help = "Varies the Gaussian scaling for defenses that use that as a parameter.")
	parser.add_argument('--defense_iter', type = int, default = 100, nargs = "+", help = "Varies the Gaussian scaling for defenses that use that as a parameter.")
	args = parser.parse_args()
	in_data = args.data
	train_sizes = args.train_size
	attack_sizes = args.attack_size
	VERBOSE = args.verbose
	BATCH_SIZE = args.batch_size
	MAX_ITER = args.max_iter
	threshold = args.threshold
	defense_bits = args.defense_bits
	debug = args.debug
	defense_scales = args.defense_scale
	defense_iters = args.defense_iter
	step_sizes = args.step_size


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
		
	# TODO: Add support for non-default storage
	set_data_path("/Home/staff/cmeyers/.art/data/")

	if in_data == 'mnist':
		path = get_file('mnist_cnn_original.h5', extract=False, path=ART_DATA_PATH,
						url="https://www.dropbox.com/s/bv1xwjaf1ov4u7y/mnist_ratio%3D0.h5?dl=1", verbose = True)
		folder = ART_DATA_PATH + '/mnist/'
	elif in_data == 'cifar10':
		path = get_file('cifar-10_cnn_original.h5',extract=False, path=ART_DATA_PATH, url='https://www.dropbox.com/s/hbvua7ynhvara12/cifar-10_ratio%3D0.h5?dl=1', verbose = True)
		folder = ART_DATA_PATH + '/cifar10/'
	if debug == True:
		train_sizes = [100]
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

	if isinstance(defense_bits, list):
		defense_bits_i = defense_bits[0]
	else:
		defense_bits_i = defense_bits
	if isinstance(defense_scales, list):
		defense_scale_i = defense_scales[0]
	else:
		defense_scale_i = defense_scales
	if isinstance(defense_iters, list):
		defense_iter_i = defense_iters[0]
	else:
		defense_iter_i = defense_iters
	defenses = {
				"Control": None, 
				"Feature Squeezing": FeatureSqueezing(clip_values = [0,255], bit_depth = defense_bits_i, apply_predict = True, apply_fit = True),
				"Gaussian Augmentation": GaussianAugmentation(sigma = 1-defense_scale_i, augmentation = False),
				"Spatial Smoothing2-size:2": SpatialSmoothing(window_size = 2),
				"Spatial Smoothing-size:3": SpatialSmoothing(window_size = 3),
				"Spatial Smoothing-size:4": SpatialSmoothing(window_size = 4),
				"Label Smoothing": LabelSmoothing(1-defense_scale_i),
				#"Thermometer": ThermometerEncoding(clip_values = [0,255], apply_predict = True, apply_fit = True),
				"Total Variance Minimization": TotalVarMin(max_iter = defense_iter_i, prob = defense_scale_i),
				"Class Labels": ClassLabels(),
				"Gaussian Noise": GaussianNoise(scale = 1-defense_scale_i, apply_fit = True, apply_predict = False),
				"High Confidence": HighConfidence(1 - defense_scale_i),
				"Reverse Sigmoid": ReverseSigmoid(gamma = defense_scale_i),
				"Rounded": Rounded(defense_bits_i),
				}

	classifier = ("Default Classifier", keras_benign)	  

	# TODO
	# Accept arbitrary parameters
	# Type checking a la 3.7?
	# bit_depths = [64,  32, 16, 8]
	defenses = generate_variable_defenses(defenses, defense_bits, 'bit_depth', defense_key = 'Feature Squeezing')
	
	decimals = [3, 6, 9]
	defenses = generate_variable_defenses(defenses, decimals, 'decimals', defense_key = 'Rounded')

	#defense_scales = [64, 32, 16, 8 , 4, 2]/255
	defense_scales_minus = np.subtract(1, defense_scales)
	defenses = generate_variable_defenses(defenses, defense_scales_minus, 'sigma', defense_key = 'Gaussian')
	defenses = generate_variable_defenses(defenses, defense_scales_minus, 'sigma', defense_key = 'Label Smoothing')
	defenses = generate_variable_defenses(defenses, defense_scales_minus, 'gamma', defense_key = 'Reverse Sigmoid')
	defenses = generate_variable_defenses(defenses, defense_scales, 'prob', defense_key = 'Total Variance Minimization')
	defenses = generate_variable_defenses(defenses, defense_scales_minus, 'prob', defense_key = 'High Confidence')
	

	#defense_iters = [10, 100, 1000]
	#Total Variance Minimization
	# defenses = generate_variable_defenses(defenses, defense_iters, 'max_iter', defense_key = 'Total Variance Minimization')

	logger.info("Number of Defenses "+ str(len(defenses.values())))

	print("Saving to", folder)
	i = 0
	for train_size in train_sizes:
		for def_name, defense in defenses.items():
				experiments = {}
				experiments.update(generate_model(data, classifier, defense, def_name, train_size = train_size, folder = folder, verbose = VERBOSE))

	print("===================================================")
	print("=                                                 =")
	print("=                                                 =")
	print("=           Experiment Run Succesfully            =")
	print("=                                                 =")
	print("=                                                 =")
	print("===================================================")
	if debug == True:
		delete_folder('debug/')
	import gc; gc.collect()
	logger.info("Garbage Collected. Session closed.")
