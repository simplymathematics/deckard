import os, logging
from pickle import dump
from pandas import Series, DataFrame
logger = logging.getLogger(__name__)


class DiskstorageMixin(object):
    def __init__(self):
        pass
    def save_data(self, filename:str = "data.pkl", path:str = ".") -> None:
        """
        Saves data to specified file.
        :param filename: str, name of file to save data to. 
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        assert path is not None, "Path to save data must be specified."
        with open(os.path.join(path, filename), 'wb') as f:
            dump(self.data, f)
        assert os.path.exists(os.path.join(path, filename)), "Data not saved."
        return None
    
    def save_params(self, data_params_file:str = "data_params.json", model_params_file:str = "model_params.json", exp_params_file:str = "experiment_params.json", path:str = ".") -> None:
        """
        Saves data to specified file.
        :param data_params_file: str, name of file to save data parameters to.
        :param model_params_file: str, name of file to save model parameters to.
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        assert path is not None, "Path to save data must be specified."
        if not os.path.isdir(path) and not os.path.exists(path):
            os.mkdir(path)
        self.params['Experiment'] = {'name': self.name, 'verbose': self.verbose, 'is_fitted': self.is_fitted, 'refit' : self.refit}
        self.params['Experiment']['experiment'] = self.filename
        self.params['Model']['experiment'] = self.filename
        self.params['Data']['experiment'] = self.filename
        data_params = Series(self.params['Data'])
        model_params = Series(self.params['Model'])
        exp_params = Series(self.params['Experiment'])
        data_params.to_json(os.path.join(path, data_params_file))
        model_params.to_json(os.path.join(path, model_params_file))
        exp_params.to_json(os.path.join(path, exp_params_file))
        assert os.path.exists(os.path.join(path, data_params_file)), "Data params not saved."
        assert os.path.exists(os.path.join(path, model_params_file)), "Model params not saved."
        assert os.path.exists(os.path.join(path, exp_params_file)), "Model params not saved."
        if 'Defence' in exp_params:
            exp_params['Defence']['experiment'] = self.filename
            defence_params = Series(model_params['Defence'])
            defence_params.to_json(os.path.join(path, "defence_params.json"))
            assert os.path.exists(os.path.join(path, "defence_params.json")), "Defence params not saved."
        return None

    def save_model(self, filename:str = "model", path:str = ".") -> str:
        """
        Saves model to specified file (or subfolder).
        :param filename: str, name of file to save model to. 
        :param path: str, path to folder to save model. If none specified, model is saved in current working directory. Must exist.
        :return: str, path to saved model.
        """
        assert os.path.isdir(path), "Path {} to experiment does not exist".format(path)
        logger.info("Saving model to {}".format(os.path.join(path,filename)))
        self.model.save(filename = filename, path = path)
    
    def save_predictions(self, filename:str = "predictions.json", path:str = ".") -> None:
        """
        Saves predictions to specified file.
        :param filename: str, name of file to save predictions to. 
        :param path: str, path to folder to save predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"    
        prediction_file = os.path.join(path, filename)
        results = self.predictions
        results = Series(results)
        results.to_json(prediction_file)
        assert os.path.exists(prediction_file), "Prediction file not saved"
        return None

    def save_adv_predictions(self, filename:str = "adversarial_predictions.json", path:str = ".") -> None:
        """
        Saves adversarial predictions to specified file.
        :param filename: str, name of file to save adversarial predictions to.
        :param path: str, path to folder to save adversarial predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        adv_file = os.path.join(path, filename)
        adv_results = DataFrame(self.adv)
        adv_results.to_json(adv_file)
        assert os.path.exists(adv_file), "Adversarial example file not saved"
        return None

    def save_cv_scores(self, filename:str = "cv_scores.json", path:str = ".") -> None:
        """
        Saves crossvalidation scores to specified file.
        :param filename: str, name of file to save crossvalidation scores to.
        :param path: str, path to folder to save crossvalidation scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified"
        cv_file = os.path.join(path, filename)
        cv_results = Series(self.model.model.model.cv_results_, name = self.filename)
        cv_results.to_json(cv_file)
        assert os.path.exists(cv_file), "CV results file not saved"

    def save_scores(self, filename:str = "scores.json", path:str = ".") -> None:
        """
        Saves scores to specified file.
        :param filename: str, name of file to save scores to.
        :param path: str, path to folder to save scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        
        score_file = os.path.join(path, filename)
        results = self.scores
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(score_file)
        assert os.path.exists(score_file), "Score file not saved"
        return None
    
    def save_adv_scores(self, filename:str = "adversarial_scores.json", path:str = ".") -> None:
        """
        Saves adversarial scores to specified file.
        :param filename: str, name of file to save adversarial scores to.
        :param path: str, path to folder to save adversarial scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        
        adv_score_file = os.path.join(path, filename)
        results = self.adv_scores
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(adv_score_file)
        assert os.path.exists(adv_score_file), "Adversarial score file not saved."
        return None
    
    def save_adversarial_samples(self, filename:str = "adversarial_examples.json", path:str = "."):
        """
        Saves adversarial examples to specified file.
        :param filename: str, name of file to save adversarial examples to.
        :param path: str, path to folder to save adversarial examples. If none specified, examples are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert hasattr(self, "adv_samples"), "No adversarial samples to save"
        adv_file = os.path.join(path, filename)
        adv_results = DataFrame(self.adv_samples.reshape(self.adv_samples.shape[0], -1))
        adv_results.to_json(adv_file)
        assert os.path.exists(adv_file), "Adversarial example file not saved"
        return None

    def save_time_dict(self, filename:str = "time_dict.json", path:str = "."):
        """
        Saves time dictionary to specified file.
        :param filename: str, name of file to save time dictionary to.
        :param path: str, path to folder to save time dictionary. If none specified, time dictionary is saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert hasattr(self, "time_dict"), "No time dictionary to save"
        time_file = os.path.join(path, filename)
        time_results = Series(self.time_dict, name = path.split(os.sep)[-1])
        time_results.to_json(time_file)
        assert os.path.exists(time_file), "Time dictionary file not saved"
        return None

    def save_attack_params(self, filename:str = "attack_params.json", path:str = ".") -> None:
        """
        Saves attack params to specified file.
        :param filename: str, name of file to save attack params to.
        :param path: str, path to folder to save attack params. If none specified, attack params are saved in current working directory. Must exist.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        assert os.path.isdir(path), "Path to experiment does not exist"
        attack_file = os.path.join(path, filename)
        results = self.params['Attack']
        results = Series(results.values(), index = results.keys())
        results.to_json(attack_file)
        assert os.path.exists(attack_file), "Attack file not saved."
        return None

    def save_defence_params(self, filename:str = "defence_params.json", path:str = ".") -> None:
        """
        Saves defence params to specified file.
        :param filename: str, name of file to save defence params to.
        :param path: str, path to folder to save defence params. If none specified, defence params are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        
        defence_file = os.path.join(path, filename)
        results = self.params['Defence']
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(defence_file)
        assert os.path.exists(defence_file), "Defence file not saved."
        return None

    def save_results(self, path:str = ".", scores_filename = 'scores.json') -> None:
        """
        Saves all data to specified folder, using default filenames.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        # self.save_model(path = path)
        self.save_scores(path = path, filename = scores_filename)
        self.save_predictions(path = path)
        if hasattr(self.model.model, 'cv_results_'):
            self.save_cv_scores(path = path)
        if hasattr(self, 'time_dict'):
            self.save_time_dict(path = path)
        return None

    def save_attack_results(self, path:str = ".") -> None:
        """
        Saves all data to specified folder, using default filenames.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        if hasattr(self, "adv_scores"):
            self.save_adv_scores(path = path)
        if hasattr(self, "adv"):
            self.save_adv_predictions(path = path)
        if hasattr(self, "adv_samples"):
            self.save_adversarial_samples(path = path)
        if hasattr(self, 'time_dict'):
            self.save_time_dict(path = path)
        if 'Defence' in self.params:
            self.save_defence_params(path = path)
        if hasattr(self.model.model, 'cv_results_'):
            self.save_cv_scores(path = path)
            self.save_adversarial_samples(path = path)
        return None