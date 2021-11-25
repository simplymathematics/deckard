from art.defences import postprocessor
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error, r2_score
import logging
from sklearn.base import is_regressor
import uuid
from model import Model
from data import Data
from time import process_time_ns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.base import is_regressor

# Create experiment object
class Experiment(object):
    """
    Creates an experiment object
    """
    def __init__(self, data:Data, model:Model, params:dict = {}, verbose : int = 1, scorers:dict = None, refit = None, name:str = None):
        assert isinstance(model, Model)
        assert isinstance(data, Data)
        assert isinstance(scorers, dict) or scorers is None
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

    def set_metric_scorer(self) -> dict:
        if not hasattr(self, 'scorers'):
            if is_regressor(self.model.model) == True:
                logging.info("Model is regressor.")
                logging.info("No metric specified. Using mean square error.")
                new_scorers = {"MSE" : mean_squared_error, 'MAE': mean_absolute_error,  "R2" : r2_score}
            elif is_regressor(self.model.model) == False:
                logging.info("Model is classifier.")
                logging.info("No metric specified. Using accuracy.")
                self.data.y_test = self.data.y_test.astype(int)
                new_scorers = {'F1' : f1_score, 'Balanced Accuracy' : balanced_accuracy_score, 'Accuracy' : accuracy_score, 'Precision' : precision_score, 'Recall' : recall_score}
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
        assert self.is_supervised()
        if hasattr( self.model, 'fit_flag' or self.fitted == True):
            logging.info("Model is already fitted")
            self.is_fitted = True
            y_pred = self.model.predict(self.data.X_test)
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
        logging.info("Made predictions")
        return y_pred, (fit_time, pred_time)

    def _build_unsupervised_model(self) -> dict:
        assert not self.is_supervised()
        if hasattr(self.model, 'fit_flag'):
            logging.warning("Model is already fitted")
            y_pred = self.model.model.predict(self.data.y_test)
            assert self.data.y_pred.shape == self.data.y_test.shape, "model appears to be fitted, but something went wrong."
        else:
            logging.info("Fitting and predicting model")
            logging.info("X_train shape: {}".format(self.data.X_train.shape))
            logging.info("y_train shape: {}".format(self.data.y_train.shape))
            start = process_time_ns()
            self.model.model.fit_predict(X = self.data.X_train, y = self.data.y_train)
            end = process_time_ns()
            fit_pred_time = end - start
            logging.info("Made predictions")
        return y_pred (fit_pred_time)

    def build_model(self) -> dict:
        logging.debug("Model type: {}".format(type(self.model.model)))
        if self.is_supervised() == False:
            self.predictions, time = self._build_unsupervised_model()
            self.time_dict = {'fit_pred_time': time[0]}
        elif self.is_supervised() == True:
            self.predictions, time = self._build_supervised_model()
            self.time_dict = {'fit_time': time[0], 'pred_time': time[1]}
        if hasattr(self.data, "post_processor"):
            if postprocessor.__dict__['apply_fit'] == True:
                self.data.y_train = self.data.post_processor(self.data.y_train)
            if postprocessor.__dict__['apply_predict'] == True:
                self.data.y_test = self.data.post_processor(self.data.y_test)
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


    def run(self, defense = None, attack = None) -> dict:
        scores = {}
        if defense is not None:
            raise NotImplementedError("Defenses not yet implemented")
        if attack is not None:
            raise NotImplementedError("Attacks not yet implemented")
        self.set_metric_scorer()
        self.build_model()
        for scorer in self.scorers:
            logging.info("Scoring with {}".format(scorer))
            if scorer in ['F1', 'Recall', 'Precision']:
                average = 'weighted'
                scores[scorer] = self.scorers[scorer](self.data.y_test.astype(int), self.predictions, average = average)
            elif scorer in ['AUC', 'ROC-AUC']:
                average = 'weighted'
                scores[scorer] = self.scorers[scorer](self.data.y_test, self.predictions, average = average, multi_class = 'ovr')
            else:
                scores[scorer] = self.scorers[scorer](self.data.y_test, self.predictions)
            logging.info("Scorer {} : {}".format(scorer, scores[scorer]))
            
        scores.update(self.time_dict)
        scores.update({'Name': self.name})
        scores.update({'uuid': self.filename})
        self.scores = scores
        self



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
    print(scores)

    # # Validate experiment object
    sys.exit(0)
    
    
    
   