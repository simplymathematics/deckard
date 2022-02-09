import warnings
import logging 
import tempfile
from copy import copy
# warnings.filterwarnings('error', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import load_model
from art.utils import to_categorical, load_dataset
import numpy as np
import pandas as pd
from art.config import ART_DATA_PATH
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import ProjectedGradientDescent, AutoAttack, CarliniLInfMethod, FastGradientMethod, PixelAttack, FeatureAdversaries, AutoProjectedGradientDescent, DeepFool, HopSkipJump, AdversarialPatch, ThresholdAttack
from art.defences.preprocessor import FeatureSqueezing, GaussianAugmentation, InverseGAN, DefenseGAN, JpegCompression, LabelSmoothing, PixelDefend, Resample, SpatialSmoothing, ThermometerEncoding, TotalVarMin
from art.defences.postprocessor import *
from art import defences
from art.defences.transformer.evasion import *
from scipy.stats import entropy
from art.utils import get_file
from tensorflow.python.framework.ops import disable_eager_execution
from os.path import exists
from os import mkdir, chmod
from sklearn.metrics import precision_score, recall_score

from math import log10, floor
from time import process_time as time

import bz2
import pickle
import shutil

from hashlib import sha256

from art.utils import get_file, check_and_transform_label_format, compute_accuracy
 #pip install adversarial-robustness-toolbox

disable_eager_execution()

# import numpy as np
# np.seterr(divide='ignore', invalid='ignore')


FOLDER = "./debug/"
VERBOSE = True
OMIT = ['classifier', 'data', 'defense', 'attack', 'adv']

if not exists(FOLDER):
    mkdir(FOLDER)

def my_hash(x):
    x = str(x).encode('utf-8')
    x = sha256(x).hexdigest()
    x = str(x)
    return x
url = 'https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1'
assert my_hash(url) == '29dfcd90a6957af25455b0db9403d943b14388b2ee29396dfc258d3444c07033'


class Data():
    def __init__(self, X_train, y_train, X_test, y_test, defense = None):
        self.x_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.x_test = X_test
        self.defense = defense
        self.id = my_hash(str(vars(self)))
    def __eq__(self, other):
        return my_hash(str(vars(self))) == my_hash(str(vars(other)))

    def __hash__(self, other):
        self.id = my_hash(str(vars(self)))
        return self.id 

#TODO: Port back to load_dataset in art.utils
def load_data(data = "mnist"):
    if data == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        data = Data(X_train, y_train, X_test, y_test)
    elif data == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32') / 255
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        data = Data(X_train, y_train, X_test, y_test)
    else:
        logger.info("Dataset not supported")
    return data

class Experiment():
    def __init__(self, data, classifier, classifier_id, defense = None, n = 100, folder = FOLDER):
        assert isinstance(data, Data), "data object must be an instance of Data class"
        self.classifier = classifier
        self.defense = defense
        self.classifier_id = str(my_hash(str([j for i,j in vars(self.classifier).items() if  'at 0x' not in str(j)])+str(type(self.classifier)) +str(self.defense)))
        self.data_id = data.id
        self.attack_id = None
        self.n = n
        if self.defense is not None:
            self.defense_id = str(defense.__dict__) + str(type(defense))
            self.defense_id = str(my_hash(self.defense_id))
        else:
            self.defense_id = str()
        self.id = my_hash(str(self.classifier_id) + str(self.data_id) + str(self.defense_id) + str(self.n))


##TODO fix always verbose train
    def train(self, data, name, verbose = True):
        start = time()
        self.classifier.fit(data.x_train[:self.n], data.y_train[:self.n],  verbose = 1)
        self.train_time = time()  - start
        self.classifier_name = name
        return self

    def add_attack(self,  name, attack, max_iter = 100, batch_size = 1024, threshold = .3, attack_size = 10):
        logger.info("Attack Type: " + name)
        logger.info("Train Size: " +  str(self.n))
        logger.info("Attack Size: " + str(attack_size))
        self.attack_size = attack_size
        self.attack = attack
        self.attack_name = name
        self.attack_iter = max_iter
        self.attack_batch_size = batch_size
        self.attack_threshold = threshold
        self.attack_id = str(my_hash(str([j for i,j in vars(self.attack).items() if  'at 0x' not in str(j)])+str(type(self.attack))+ str(self.attack_size)))
        self.id = my_hash(str(self.classifier_id) + str(self.data_id) + str(self.defense_id) + str(self.n) + str(self.attack_id))
        logger.info("Attack added Successfully.")
        return self

    def launch_attack(self, data, folder):
        if not exists(folder):
            mkdir(folder)
        adv_file = folder + str(self.attack_id) + ".attack"
        try:
            start = time()
            self.adv = self.attack.generate(data.x_test[:self.attack_size], data.y_test[:self.attack_size])
            self.attack_time = time() - start
            logger.info("Attack launched Successfully.")
        except TypeError as e: # Catches untargeted attacks
            start = time()
            self.adv = self.attack.generate(data.x_test[:self.attack_size])
            self.attack_time = time() - start
            logger.info("Attack launched Successfully.")
        except AttributeError as e:
            raise e
        # with open(adv_file, 'wb') as file:
        #     pickle.dump(self.adv, file)
        return self
        

    def evaluate(self, data, folder):
        folder = folder + "results/"
        # try:
        data.y_test   = check_and_transform_label_format(data.y_test)
        other = check_and_transform_label_format(data.y_test, return_one_hot = False)  
        try:
            
            if hasattr(self, "ben_rec"):
                pass
            else:
                start = time() 
                self.ben_pred = self.classifier.predict(data.x_test[:self.n])
                self.ben_pred_time = time() - start
                ben_pred = np.argmax(self.ben_pred, axis = 1)
                self.ben_acc  = compute_accuracy(self.ben_pred, data.y_test[:len(self.ben_pred)])[0]
                self.ben_cov  = compute_accuracy(self.ben_pred, data.y_test[:len(self.ben_pred)])[1]
                self.ben_prec = precision_score(ben_pred, other[:len(ben_pred)], average = None)
                self.ben_rec  = recall_score(ben_pred, other[:len(ben_pred)], average = None)
                logger.info("Model evaluated successfully")
            logger.info("Train Time:" + str(round(self.train_time, 2)))
            #if hasattr(self, "adv_rec"):
            #    pass
            #else:
            start = time()
            self.adv_pred = self.classifier.predict(self.adv)
            self.adv_pred_time = time() - start
            adv_pred = np.argmax(self.adv_pred, axis = 1)
            self.adv_acc  = compute_accuracy(self.adv_pred, data.y_test[:len(self.adv_pred)])[0]
            self.adv_cov  = compute_accuracy(self.adv_pred, data.y_test[:len(self.adv_pred)])[1]
            self.adv_prec = precision_score(adv_pred, other[:len(adv_pred)], average = None)
            self.adv_rec  = recall_score(adv_pred, other[:len(adv_pred)], average = None)
            logger.info("Attack Time:" + str(round(self.attack_time, 2)))
            logger.info("Attack evaluated successfully.") 
            sig_figs = int(round(log10(self.n), 0))
            logger.info("Significant Figures: " + str(sig_figs))
            logger.info("Benign accuracy: " + str(round(self.ben_acc, sig_figs)))
            sig_figs = int(round(log10(self.attack_size), 0))
            logger.info("Significant Figures: " + str(sig_figs))
            logger.info("Adversarial accuracy: " +  str(round(self.adv_acc, sig_figs)))
            logger.info("Experiment Evaluated Successfully.")
        except ValueError as e: # Catches cases where Labels are required
            if "operands could not be broadcast together with shapes" in str(e):
                raise(e)
            else:
                logger.info(e)
                raise(e)
        return self

    def __eq__(self, other):
        truth = self.data_id == other.data_id
        truth = truth and self.attack_id == other.attack_id
        truth = truth and self.defense_id == other.defense_id
        truth = truth and self.data_id == other.data_id
        truth = truth and self.n == other.n
        truth = truth and self.classifier_id == self.classifier_id
        #truth = truth and my_hash(self) == my_hash(other)
        return truth

    def __hash__(self):
        self.id = my_hash(str(self.classifier_id) + str(self.data_id) + str(self.defense_id) + str(self.n) + str(self.attack_size))
        return(self.id)

def generate_model(data, classifier, defense, def_name, train_size, folder = FOLDER, verbose = True):
    if not exists(folder):
        mkdir(folder)
    folder = folder + 'classifiers/'
    if not exists(folder):
        mkdir(folder)
    experiments = {}
    cl_name = classifier[0]
    classifier = classifier[1]
    data_name = data[0]
    datum = data[1]
    classifier = classifier.__dict__['_model']
    name = str(my_hash(cl_name))
    if  defense is None:
        cl = KerasClassifier(model = classifier)
    elif 'art.defences.preprocessor' in str(type(defense)):
        cl = KerasClassifier( model=classifier, preprocessing_defences = defense)
    elif 'art.defences.postprocessor' in str(type(defense)):
        cl = KerasClassifier(model = classifier, postprocessing_defences = defense)
    else:
        logger.info("Defense not supported. Try running the function again, using your defended model as the classifier.")
        raise ValueError
    res = Experiment(datum, cl, name, defense = defense,  n = train_size,)
    model_file = folder + res.classifier_id + ".model"
    logger.info("Number of Samples used for Training: "+ str(res.n))
    logger.info("Defense: " +  def_name)
    logger.info("Classifier: " + cl_name)
    logger.info("Data-Name: " +  data_name)
    try:
        res.def_name = def_name
    except AttributeError as e: # Catches instance where no defense is provided
        res.def_name = 'None'
    res.classifier_name = cl_name
    if exists(model_file):
        logger.info(str(res.classifier_id) + " Classifier Exists!")
        with open(model_file, 'rb') as file:
            res = pickle.load(file)
    else:
        res = res.train(datum, name = cl_name, verbose = verbose)
        logger.info("Saving classifier to: "+ model_file)
        with open(model_file, 'wb') as file:
            pickle.dump(res, file)
    experiments[res.id] = res
    assert len(experiments) == len(set(experiments))
    return experiments


def generate_attacks(experiment, attack, attack_name, data, attack_sizes = [10], folder = FOLDER, omit = OMIT, verbose = True,  **kwargs):
    results = {}
    i = 0 
    for attack_size in attack_sizes:
        data_name = data[0]
        datum = data[1]
        filename = folder + str(experiment.id) + ".experiment"
        if not exists(filename):
            logger.info("Experiment ID: "+ str(experiment.id)+ "started.")
            logger.info("Classifier Type: "+ experiment.classifier_name)
            logger.info("Defense: "+ experiment.def_name or None)
            attack.__dict__['_estimator'] = experiment.classifier
            experiment = experiment.add_attack( attack_name, attack, attack_size = attack_size, **kwargs)
            try:
                experiment = experiment.launch_attack(datum, folder)
            except Exception as e:
                logger.info(attack_name + " failed during launch phase.")
                raise e
            try:
                experiment = experiment.evaluate(datum, folder)
            except Exception as e:
                logger.info(attack_name + " failed during evaluation phase.")
                raise e
            try:
                my_keys = experiment.__dict__.keys()
                row = {my_key: experiment.__dict__[my_key] for my_key in my_keys if my_key not in omit}
                results[experiment.id] = row
                with open(filename, 'wb') as file:
                    pickle.dump(row, file)
            except Exception as e:
                logger.info(attack_name + " failed during experiment saving phase.")
                raise e
            try:
                results = append_results(results, folder)
            except Exception as e:
                logger.info(attack_name + " failed during result phase.")
                raise e
        else:
            logger.info("Experiment Exists!")
    
    return results

def generate_variable_attacks(attacks,  variables, variable_name, attack_key = None, verbose = True):
    new_attacks = {}
    for attack_name, base_attack in attacks.items():
        if attack_key in attack_name or attack_key == None:
            for variable in variables:
                attack = copy(base_attack)
                try:
                    attack.__dict__[variable_name] = variable
                except AttributeError or ValueError as e:
                    logger.log("Error initializing" + attack_key + ": " + e)
                new_name = attack_name + "-" + str(variable_name) + ":" + str(variable) 
                new_attacks.update({new_name : attack})
        else:
            attack = copy(base_attack)
            new_attacks.update({attack_name : attack}) # skips attacks that don't have key
    assert len(set(new_attacks.values())) == len(set(new_attacks))
    return new_attacks

def generate_variable_defenses(defenses,  variables, variable_name, defense_key = None):
    new_attacks = generate_variable_attacks(defenses, variables, variable_name, attack_key = defense_key)
    return new_attacks

def append_results(results, folder = FOLDER):
    #folder = folder + "results/"
    if not exists(folder):
        mkdir(folder)
    df = pd.DataFrame.from_dict(results, orient = 'index')
    df_filename = folder + 'results.csv'
    logger.info("Results saved to: "+ df_filename)
    if exists(df_filename):
        header = False
    else:
        header = True
    df.to_csv(df_filename, mode = 'a', header = header)
    assert exists(df_filename)
    df_all = pd.read_csv(df_filename)
    df_all.rename(columns = {'Unnamed: 0': 'Experiment ID'}, inplace = True)
    return df_all

def delete_folder(folder = FOLDER):
     chmod(folder, 0o777)
     shutil.rmtree(folder)
     logger.info("Folder '" + folder + "' Deleted")

if __name__ == '__main__':
    tmp = tempfile.gettempdir()
    logging.basicConfig(
        level = logger.DEBUG,
        format = '%(asctime)s %(name)s %(levelname)s %(message)s',
        filename=FOLDER +'debug.log',
        filemode = 'w'
    )
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('')
    logger.setLevel(logger.DEBUG)
    file_logger = logger.FileHandler(FOLDER+'debug.log')
    file_logger.setLevel(logger.INFO)
    file_logger.setFormatter(formatter)
    stream_logger = logging.StreamHandler()
    stream_logger.setLevel(logger.INFO)
    stream_logger.setFormatter(formatter)
    logger.addHandler(file_logger)
    logger.addHandler(stream_logger)
    

    TRAIN_SIZE = 100
    THRESHOLD = .3
    BATCH_SIZE = 1024
    MAX_ITER = 10

    dx = load_data()
    da = load_data()
    data = {'Default': da}
    dxx = Data(da.x_train, da.y_train, da.x_test[:2], da.y_test[:2])

    # More Robust function testing
    TEST_SIZE = 10
    url = 'https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1'

    path = get_file('mnist_cnn_original.h5', extract=False, path=ART_DATA_PATH,
                    url=url)
    classifier_model = load_model(path)
    keras_classifier = KerasClassifier(model=classifier_model)

    default_classifier_id = my_hash(url)

    attacks = {
            'PGD' : ProjectedGradientDescent(keras_classifier, eps=THRESHOLD, norm = 'inf', eps_step = .1, batch_size = BATCH_SIZE, max_iter = MAX_ITER, targeted = False, num_random_init=False, verbose = VERBOSE),
            'FGM': FastGradientMethod(keras_classifier, eps = THRESHOLD, eps_step = .1, batch_size = BATCH_SIZE), 
            'Carlini': CarliniL2Method(keras_classifier, verbose = VERBOSE, confidence = .99, max_iter = MAX_ITER, batch_size = BATCH_SIZE), 
            # 'PixelAttack': PixelAttack(keras_classifier, th = THRESHOLD, verbose = VERBOSE, es = 1), 
            'DeepFool': DeepFool(classifier = KerasClassifier(model=classifier_model, clip_values = [0,1]), batch_size = BATCH_SIZE, verbose = VERBOSE), 
            'HopSkipJump': HopSkipJump(keras_classifier, max_iter = MAX_ITER, verbose = VERBOSE),
            # 'A-PGD' : AutoProjectedGradientDescent(KerasClassifier(model=classifier_model, )),
            'AdversarialPatch': AdversarialPatch(classifier = KerasClassifier(model=classifier_model, clip_values = [0,1]), max_iter = MAX_ITER, verbose = VERBOSE),
            'Threshold Attack': ThresholdAttack(keras_classifier, th = THRESHOLD, verbose = VERBOSE),
            }

            #TODO Thermometer > 1; Opened Ticket
            #TODO Support other models
    defenses = {"No Defense": None, 
                    "Feature Squeezing": FeatureSqueezing(clip_values = [0,1], bit_depth = 2, apply_predict = True, apply_fit = False),
                    "Gaussian Augmentation": GaussianAugmentation(sigma = .1, apply_predict = True, apply_fit = True, augmentation = False),
                    "Spatial Smoothing": SpatialSmoothing(window_size = 2, apply_predict = True, apply_fit = True),
                    "Label Smoothing": LabelSmoothing(apply_predict = False, apply_fit = True),
                    # "Thermometer": ThermometerEncoding(clip_values = [0,255], num_space = 64, apply_predict = True, apply_fit = True),
                    "Total Variance Minimization": TotalVarMin(apply_predict = True, apply_fit = True),
                    "Class Labels": ClassLabels(apply_predict = False, apply_fit = True),
                    "Gaussian Noise": GaussianNoise(scale = .3, apply_predict = False, apply_fit = True),
                    "Reverse Sigmoid": ReverseSigmoid(),
                    "High Confidence": HighConfidence(apply_predict = True, apply_fit = False),
                    "Rounded": Rounded(apply_predict = True, decimals = 8),
                    }
    classifiers = { default_classifier_id  : keras_classifier,
    #                "Defensive Distillation": DefensiveDistillation(classifier,
                    }
    #TODO Fix broken test here.
    experiments = generate_models(data, classifiers, defenses, train_size = TRAIN_SIZE, folder = FOLDER, verbose = VERBOSE)
    for experiment in experiments.values():
        results = generate_attacks(experiment, attacks, data,  attack_sizes = [1], folder = FOLDER, max_iter = MAX_ITER, batch_size = BATCH_SIZE, threshold = THRESHOLD, verbose = VERBOSE)
        df = append_results(results, folder = FOLDER)
    

    # Basic Object testing
    attack = ProjectedGradientDescent(keras_classifier, eps=.3, eps_step=.1, 
                    max_iter=10, targeted=False, num_random_init=False)


    attack2 = ProjectedGradientDescent(keras_classifier, eps=.3, eps_step=.1, 
                    max_iter=10, targeted=False, num_random_init=False)

    def1 = FeatureSqueezing(clip_values = [0,1], bit_depth = 2, apply_fit = True)
    def2 = FeatureSqueezing(clip_values = [0,1], bit_depth = 2, apply_fit = True)
    def3 = FeatureSqueezing(clip_values = [0,1], bit_depth = 5, apply_fit = True)
    res1 = Experiment(da, keras_classifier, default_classifier_id, def1, n = TEST_SIZE)
    res2 = Experiment(da, keras_classifier, default_classifier_id, def2,  n = TEST_SIZE)
    res3 = Experiment(da, keras_classifier, default_classifier_id, n = TEST_SIZE)
    res4 = Experiment(da, keras_classifier, default_classifier_id,  n = TEST_SIZE)
    res5 = Experiment(da, keras_classifier, default_classifier_id, def3,  n = TEST_SIZE)
    cl1 = KerasClassifier( model=classifier_model)
    cl2 = KerasClassifier( model=classifier_model)
    id1 = str(hash(str(cl1)+str(type(cl2))))
    id2 = str(hash(str(cl2)+str(type(cl1))))
    step_sizes = [.001, .01, .1]
    new_attacks = generate_variable_attacks(attacks, step_sizes, 'step_size', attack_key = 'PGD')
    norms = [1, 2, 'inf']
    new_attacks = generate_variable_attacks(new_attacks, norms, 'norm', attack_key = 'PGD')
    assert new_attacks['PGD-step_size:0.1-norm:1']
    assert da == dx
    assert da.id == dx.id
    assert da is not dxx
    assert res1.attack_id == res2.attack_id
    assert res1.defense_id == res2.defense_id
    assert res1.data_id == res2.data_id
    assert type(res1.defense) != type(res3.defense)
    assert res1 != res5  != res4
    assert res1.classifier_id == res2.classifier_id
    assert res1 != res3
    logger.info(my_hash(res1.id))
    assert res1 == res2
    assert id1 == id2 
    logger.info("ALL TESTS PASSED")
    import gc; gc.collect()
    logger.info("Garbage Collected")
    delete_folder(FOLDER)
