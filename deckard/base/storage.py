import os, logging
from pickle import dump
from pandas import Series, DataFrame
logger = logging.getLogger(__name__)


class DiskStorageMixin(object):
    def __init__(self):
        pass
    
    def save_data(self, filename:str = "data.pkl", prefix = None, path:str = ".") -> None:
        """
        Saves data to specified file.
        :param filename: str, name of file to save data to. 
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        assert path is not None, "Path to save data must be specified."
        if prefix is not None:
            filename = os.path.join(path, prefix + "_" + filename)
        else:
            filename = os.path.join(path, filename)
        with open(filename, 'wb') as f:
            dump(self.data, f)
        assert os.path.exists(os.path.join(path, filename)), "Data not saved."
        return None
    
    def save_params(self, prefix = None, path:str = ".", filetype = '.json') -> None:
        """
        Saves data to specified file.
        :param data_params_file: str, name of file to save data parameters to.
        :param model_params_file: str, name of file to save model parameters to.
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        filenames = []
        assert path is not None, "Path to save data must be specified."
        if not os.path.isdir(path) and not os.path.exists(path):
            os.mkdir(path)
        attributes = {}
        for key in self.params:
            if prefix is not None:
                filename = prefix + key.lower() + "_" + key + filetype
            else:
                filename = key.lower() +"_params" + filetype
            try:
                params = Series(dict(self.params[key])).to_json(os.path.join(path,filename))
            except ValueError as e:
                if "has length 1":
                    logger.warning("Parameter {} has length 1. Skipping. Value is {}.".format(key, self.params[key]))
                    attributes[key] = self.params[key]
                else:
                    raise e
            filenames.append(os.path.join(path,filename))
        logger.info("Saving {} parameters to {}".format(key, os.path.join(path,filename)))
        DataFrame(attributes).to_json(os.path.join(path, prefix + "attributes" + filetype))
        filenames.append(os.path.join(path, prefix + "attributes" + filetype))
        return filenames

    def save_model(self, filename:str = "model", prefix = None, path:str = ".") -> str:
        """
        Saves model to specified file (or subfolder).
        :param filename: str, name of file to save model to. 
        :param path: str, path to folder to save model. If none specified, model is saved in current working directory. Must exist.
        :return: str, path to saved model.
        """
        if prefix is not None:
            filename = prefix + "_" + filename
        assert os.path.isdir(path), "Path {} to experiment does not exist".format(path)
        logger.info("Saving model to {}".format(os.path.join(path,filename)))
        self.model.save_model(filename = filename, path = path)
    
    def save_predictions(self, filename:str = "predictions.json", prefix = None, path:str = ".") -> None:
        """
        Saves predictions to specified file.
        :param filename: str, name of file to save predictions to. 
        :param path: str, path to folder to save predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"    
        if prefix is not None:
            filename = prefix + "_" + filename
        prediction_file = os.path.join(path, filename)
        results = self.predictions
        results = DataFrame(results)
        results.to_json(prediction_file)
        assert os.path.exists(prediction_file), "Prediction file not saved"
        return None
    
    def save_ground_truth(self, filename:str = "ground_truth.json", prefix = None, path:str = ".") -> None:
        """
        Saves ground_truth to specified file.
        :param filename: str, name of file to save ground_truth to. 
        :param path: str, path to folder to save ground_truth. If none specified, ground_truth are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist" 
        if prefix is not None:
            filename = prefix + "_" + filename  
        prediction_file = os.path.join(path, filename)
        results = self.ground_truth
        results = DataFrame(results)
        results.to_json(prediction_file)
        assert os.path.exists(prediction_file), "Prediction file not saved"
        return None
    
    def save_cv_scores(self, filename:str = "cv_scores.json", prefix = None, path:str = ".") -> None:
        """
        Saves crossvalidation scores to specified file.
        :param filename: str, name of file to save crossvalidation scores to.
        :param path: str, path to folder to save crossvalidation scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified"
        if prefix is not None:
            filename = prefix + "_" + filename
        cv_file = os.path.join(path, filename)
        cv_results = Series(self.model.model.model.cv_results_, name = self.filename)
        cv_results.to_json(cv_file)
        assert os.path.exists(cv_file), "CV results file not saved"

    def save_time_dict(self, filename:str = "time_dict.json", prefix = None, path:str = "."):
        """
        Saves time dictionary to specified file.
        :param filename: str, name of file to save time dictionary to.
        :param path: str, path to folder to save time dictionary. If none specified, time dictionary is saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert hasattr(self, "time_dict"), "No time dictionary to save"
        if prefix is not None:
            filename = prefix + "_" + filename
        time_file = os.path.join(path, filename)
        time_results = Series(self.time_dict, name = path.split(os.sep)[-1])
        time_results.to_json(time_file)
        assert os.path.exists(time_file), "Time dictionary file not saved"
        return None

    