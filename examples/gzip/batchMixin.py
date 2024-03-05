from tqdm import tqdm
import logging
import numpy as np


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split    
import time


logger = logging.getLogger(__name__)

class BatchedMixin:
    def __init__(self, batch_size:int=10, max_batches: int=100,  **kwargs):
        self.batch_size = batch_size
        self.max_batches = max_batches
        kwargs.pop("max_batches", "None")
        kwargs.pop("batch_size", "None")
        super().__init__(**kwargs)
        self.fit = self.batched_fit(self.fit)
        self.predict = self.batched_predict(self.predict)
        self.score = self.batched_score(self.score)
    
    def batched_fit(self, fit_func):
        def wrapper(*args, **kwargs):
            X_train, y_train = args
            n = len(X_train)
            n_batches = n // self.batch_size
            if n_batches > self.max_batches:
                n_batches = self.max_batches
            for i in tqdm(range(n_batches), desc="Fitting batches", total=n_batches, leave=False, dynamic_ncols=True):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                fit_func(X_batch, y_batch)
                # train_score = self.score(X_batch, y_batch)
                # test_score = self.score(X_train, y_train)
                # logger.info(f"Batch {i+1} of {n_batches} - Train score: {train_score}; Test score: {test_score}")
        return wrapper

    def batched_predict(self, predict_func):
        def wrapper(*args, **kwargs):
            X_test = args[0]
            n = len(X_test)
            n_batches = n // self.batch_size
            if n_batches > self.max_batches:
                n_batches = self.max_batches
            preds = []
            for i in tqdm(range(n_batches), desc="Predicting batches", total=n_batches, leave=False, dynamic_ncols=True):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X_test[start:end]
                preds.append(predict_func(X_batch))
            return np.concatenate(preds)
        return wrapper
    
    def batched_score(self, score_func):
        def wrapper(*args, **kwargs):
            X_test, y_test = args
            n = len(X_test)
            n_batches = n // self.batch_size
            if n_batches > self.max_batches:
                n_batches = self.max_batches
            scores = []
            for i in tqdm(range(n_batches), desc="Scoring batches", total=n_batches, leave=False, dynamic_ncols=True):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X_test[start:end]
                y_batch = y_test[start:end]
                scores.append(score_func(X_batch, y_batch))
            return np.mean(scores)
        return wrapper

    
def create_batched_class(cls, *args, **kwargs):
    name = cls.__name__
    class BatchedClass(cls, BatchedMixin):
        def __init__(self, *args, **kwargs):
            self.max_batches = kwargs.pop("max_batches", 100)
            self.batch_size = kwargs.pop("batch_size", 10)
            super().__init__(*args, **kwargs)
    batched_class = BatchedClass()
    batched_class.__name__ = combined_name
    # batched_class.__init__(*args, **kwargs)
    return batched_class





# Find best samples

from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

class BestSamplesMixin:
    def __init__(self, n_samples=100,  m=10,  nb_epochs = 10, **kwargs):
        self.n_samples = n_samples
        self.nb_epochs = nb_epochs
        self.m = m
        super().__init__(**kwargs)

    def update(self, X, y):
        assert hasattr(self, "X_")
        assert hasattr(self, "y_")
        X_train, y_train = self.update_train_set(self.X_, self.y_, X, y)
        self.X_, self.y_ = self.find_best_samples(X_train, y_train, self,)
    
    
    def find_best_samples(self, n_jobs=-1):
        X = self.X_
        y = self.y_
        n = len(X)
        preds = cross_val_predict(self, X, y, cv = 10, n_jobs=n_jobs)
        # Find the samples that are most often correctly classified
        correct_preds = preds == y
        scores = np.zeros(n)
        
    def update_train_set(self, X_train, y_train, X_new, y_new):
        self.X_ = np.concatenate([X_train, X_new])
        self.y_ = np.concatenate([y_train, y_new])
        return self.X_, self.y_
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    big_X = []
    big_y = []
    for i in range(100):
        X, y = make_classification(n_samples=10000, n_features=20, n_informative=19, n_redundant=1, n_classes=2, random_state=42+i)
        big_X.extend(X.tolist())
        big_y.extend(y.tolist())
    big_X = np.array(big_X)
    big_y = np.array(big_y)
    logger.info(f"Shape of big_X: {big_X.shape}")
    i = 42
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=19, n_redundant=1, n_classes=2, random_state=42+i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    class BatchedSVC(BatchedMixin, SVC):
        pass
    clf = BatchedSVC(max_batches=100, batch_size=100, kernel="rbf")
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    score = round(score, 2)
    logger.info(f"Final Score: {score}")
    
    cross_val_preds = cross_val_predict(clf, X_train, y_train, cv=3, n_jobs=1)
    logger.info("Cross val preds: ", cross_val_preds)

 