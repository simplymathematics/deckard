import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
import logging
from sklearn.base import is_regressor
import uuid
from deckard.base.model import Model
from deckard.base.data import Data
from time import process_time_ns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.base import is_regressor
from pandas import Series
from os import path, mkdir
from pickle import dump
# Create experiment object
class Experiment(object):
    """
    Creates an experiment object
    """
    def __init__(self, data:Data, model:Model, params:dict = {}, verbose : int = 1, scorers:dict = None, refit = None, name:str = None):
        assert isinstance(model, Model)
        assert isinstance(data, Data)
        assert isinstance(scorers, dict) or scorers is None
        self.time_series = data.time_series
        self.model = model
        self.data = data
        self.name = data.dataset +"_"+ model.name
        self.params = params
        self.verbose = verbose
        self.filename = str(uuid.uuid4().hex)
        self.refit = refit
        self.params.update({"Model params" : model.params})
        self.params.update({"Data params" : data.params})
        if scorers is not None:
            assert isinstance(scorers, dict)
            self.scorers = scorers
        else:
           self = self.set_metric_scorer()
        self.is_fitted = False
        self.predictions = None
        self.time_dict = None
        self.score_dict = None

    def __hash__(self):
        """
        Returns a hash of the experiment object
        """
        return int(hash(str(self.params)))
    
    def __eq__(self, other) -> bool:
        return self.__hash__() == other.__hash__()

    def set_metric_scorer(self, attack=None) -> dict:
        if not hasattr(self, 'scorers'):
            if is_regressor(self.model.model) == True or 'sktime' in str(type(self.model.model)):
                logging.info("Model is regressor.")
                new_scorers = {'MAPE': mean_absolute_percentage_error, "MSE" : mean_squared_error, 'MAE': mean_absolute_error,  "R2" : r2_score, "EXVAR" : explained_variance_score}
            elif is_regressor(self.model.model) == False:
                logging.info("Model is classifier.")
                self.data.y_test = self.data.y_test.astype(int)
                new_scorers = {'F1' : f1_score, 'BALACC' : balanced_accuracy_score, 'ACC' : accuracy_score, 'PREC' : precision_score, 'REC' : recall_score,'AUC': roc_curve}
            else:
                raise ValueError("Model is not estimator")
        elif len(list(self.scorers)) == 1:
            assert isinstance(list(self.scorers)[0], str)
            assert isinstance(list(self.values)[0], callable)
            new_scorers = self.scorers
        else:
            new_scorers = {}
            for score_name, scorer in self.scorers.items():
                assert isinstance(score_name, str)
                new_scorers[score_name] = scorer
        self.refit = list(new_scorers.keys())[0]
        self.params.update({'Optimization Scorer' : self.refit})
        self.scorers = new_scorers
        return self
    
    def _build_supervised_model(self) -> dict:
        # assert self.is_supervised()
        if hasattr( self.model, 'fit_flag' or self.is_fitted == True):
            logging.info("Model is already fitted")
            self.is_fitted = True
            start = process_time_ns()
            y_pred = self.model.predict(self.data.X_test)
            end = process_time_ns()
            fit_time = np.nan
            pred_time = end - start
        else:
            logging.info("Fitting model")
            logging.info("X_train shape: {}".format(self.data.X_train.shape))
            logging.info("y_train shape: {}".format(self.data.y_train.shape))
            start = process_time_ns()
            self.model.model.fit(X = self.data.X_train, y =self.data.y_train)
            end = process_time_ns()
            self.is_fitted = True
            fit_time = end - start
            logging.info("Model training complete.")
            start = process_time_ns()
            y_pred = self.model.model.predict(self.data.X_test)
            end = process_time_ns()
            pred_time = end - start
        logging.info("Length of predictions: {}".format(len(y_pred)))
        logging.info("Made predictions")
        return y_pred, (fit_time, pred_time)

    def _build_unsupervised_model(self) -> dict:
        assert not self.is_supervised()
        if hasattr(self.model, 'fit_flag') or self.is_fitted == True:
            logging.warning("Model is already fitted")
            start = process_time_ns()
            y_pred = self.model.model.predict(self.data.y_test)
            end = process_time_ns()
            fit_pred_time = end - start
            self.is_fitted = True
            assert self.data.y_pred.shape == self.data.y_test.shape, "model appears to be fitted, but something went wrong."
        else:
            logging.info("Fitting and predicting model")
            logging.info("X_train shape: {}".format(self.data.X_train.shape))
            logging.info("y_train shape: {}".format(self.data.y_train.shape))
            start = process_time_ns()
            self.model.model.fit_predict(X = self.data.X_train)
            end = process_time_ns()
            fit_pred_time = end - start
            y_pred = self.model.model.predict(self.data.X_test)
            logging.info("Made predictions")
        return y_pred (fit_pred_time)
    
    def _build_time_series_model(self) -> dict:
        from sktime.forecasting.base import ForecastingHorizon
        fh = ForecastingHorizon(self.data.y_test.index, is_relative=False)
        forecaster = self.model.model
        
        
        if hasattr( self.model, 'fit_flag' or self.is_fitted == True):
            logging.info("Model is already fitted")
            self.is_fitted = True
            start = process_time_ns()
            y_pred = forecaster.predict(fh = fh)
            end = process_time_ns()
            fit_time = np.nan
            pred_time = end - start
        else:
            logging.info("Fitting model")
            logging.info("X_train shape: {}".format(self.data.X_train.shape))
            logging.info("y_train shape: {}".format(self.data.y_train.shape))
            start = process_time_ns()
            forecaster.fit(y = self.data.y_train, X = self.data.X_train, fh = fh)
            end = process_time_ns()
            self.is_fitted = True
            fit_time = end - start
            logging.info("Model training complete.")
            start = process_time_ns()
            y_pred = forecaster.predict(fh = fh)
            end = process_time_ns()
            pred_time = end - start
        logging.info("Length of predictions: {}".format(len(y_pred)))
        logging.info("Made predictions")
        return y_pred, (fit_time, pred_time)
    
    
    def build_model(self, attack=None, defense=None) -> dict:
        logging.debug("Model type: {}".format(type(self.model.model)))
        if self.is_supervised() == False and self.time_series == False:
            self.predictions, time = self._build_unsupervised_model()
            self.time_dict = {'fit_pred_time': time[0]}
        elif self.is_supervised() == True and self.time_series == False:
            self.predictions, time = self._build_supervised_model()
            self.time_dict = {'fit_time': time[0], 'pred_time': time[1]}
        elif self.time_series == True:
            # TODO: fix time-series pipeline compatibility, ensure that time_series data != time_series model
            self.predictions, time = self._build_supervised_model()
            self.time_dict = {'fit_time': time[0], 'pred_time': time[1]}
        else:
            type_string = str(type(self.model.model))
            raise ValueError(f"Model, {type_string}, is not a supported estimator")
        return self
    
    

    def is_supervised(self)-> bool:
        """
        Returns true if supervised, false if unsupervised. 
        """
        if hasattr(self.model.model, 'fit_predict'):
            result = False
            logging.info("Model is unsupervised")
        elif hasattr(self.model.model, 'fit') and not hasattr(self.model.model, 'fit_predict'):
            result = True
            logging.info("Model is supervised")
        else:
            raise ValueError("Model is not regressor or classifier")
        return result


    def run(self, scorer = None, defense = None, attack = None) -> dict:
        self.set_metric_scorer(attack = attack)
        self.build_model(attack = attack, defense = defense)
        scores = self.evaluate(attack = attack, defense = defense)
        scores.update(self.time_dict)
        scores.update({'Name': self.name})
        scores.update({'id_': self.filename})
        self.scores = scores

    def evaluate(self, scorers = None, attack = None, defense = None) -> dict:
        if scorers is None:
            self.set_metric_scorer(attack = attack)
        scores = {}
        for scorer in self.scorers:
            logging.info("Scoring with {}".format(scorer))
            if scorer in ['F1', 'Recall', 'Precision']:
                average = 'weighted'
                scores[scorer] = self.scorers[scorer](self.data.y_test.astype(int), self.predictions, average = average)
            elif scorer in ['ROC-AUC']: # deals with multiclass class inbalance
                average = 'macro'
                scores[scorer] = self.scorers[scorer](self.data.y_test, self.predictions, average = average, multi_class = 'ovo')
            else:
                scores[scorer] = self.scorers[scorer](self.data.y_test, self.predictions)
            logging.info("Score : {}".format(scores[scorer]))
        return scores

    def save_results(self, folder:str = ".") -> None:
        if not path.isdir(folder):
            mkdir(folder)
        logging.debug("Saving results")
        results = self.scores
        score_file = path.join(folder, "results.json")
        data_file = path.join(folder,"data_params.json")
        model_file = path.join(folder, "model_params.json")
        results = Series(results.values(), name =  self.filename, index = results.keys())
        data_params = Series(self.data.params, name = self.filename)
        model_params = Series(self.model.params, name = self.filename)
        if hasattr(self.data, "attack_params"):
            attack_file = path.join(folder, "attack_params.json")
            attack_params = Series(self.data.attack_params, name = self.filename)
            attack_params.to_json(attack_file)
        if hasattr(self.model.model, "cv_results_"):
            cv_file = path.join(folder, f"{self.filename}_cv_results.json")
            cv_results = Series(self.model.model.cv_results_, name = self.filename)
            cv_results.to_json(cv_file)
        logging.info("Results:{}".format(results))
        logging.info("Data Params: {}".format(data_params))
        logging.info("Model Params: {}".format(model_params))
        results.to_json(score_file)
        data_params.to_json(data_file)
        model_params.to_json(model_file)
        assert path.exists(score_file), "Saving results unsuccessful"
        assert path.exists(data_file), "Saving data_params unsuccessful"
        assert path.exists(model_file), "Saving model_params unsuccessful"
        logging.info("Saved results.")
        return None

    def save_experiment(self, folder:str = ".") -> None:
        if not path.isdir(folder):
            mkdir(folder)
        exp_file = path.join(folder, "experiment.pkl")
        dump(self, open(exp_file, "wb"))
        assert path.exists(exp_file), "Saving experiment unsuccessful"
        logging.info("Saved experiment to {}".format(exp_file))
        return None

    def save_model(self, folder:str = ".") -> None:
        if not path.isdir(folder):
            mkdir(folder)
        model_file = path.join(folder, "model.pkl")
        dump(self.model, open(model_file, "wb"))
        assert path.exists(model_file), "Saving model unsuccessful"
        logging.info("Saved model to {}".format(model_file))
        return None

    def save_data(self, folder:str = ".") -> None:
        if not path.isdir(folder):
            mkdir(folder)
        data_file = path.join(folder, "data.pkl")
        dump(self.data, open(data_file, "wb"))
        assert path.exists(data_file), "Saving data unsuccessful"
        logging.info("Saved data")
        return None

if __name__ == '__main__':
    # set up logging
    import sys
    logging.basicConfig(level=logging.DEBUG)
    # Create experiment object
    from sklearn.preprocessing import StandardScaler
    # import linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model = Model(model)
    data = Data('iris')
    experiment = Experiment(model=model, data=Data())
    scores = experiment.run()
    #logging.info(scores)
    # # Validate experiment object
    sys.exit(0)
    
    
    
   