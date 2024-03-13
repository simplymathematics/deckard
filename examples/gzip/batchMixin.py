from tqdm import tqdm
import logging
import numpy as np


from sklearn.datasets import make_classification
import random

# from gzip_classifier import GzipSVC, GzipKNN, GzipLogisticRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import plotext

logger = logging.getLogger(__name__)

train_scores = []
test_scores = []


class BatchedMixin:
    def __init__(
        self,
        batch_size: int = 10,
        max_batches: int = 100,
        nb_epoch=1,
        **kwargs,
    ):
        self.batch_size = kwargs.pop("m", batch_size)
        self.max_batches = kwargs.pop("max_batches", max_batches)
        nb_epoch = kwargs.pop("nb_epoch", nb_epoch)
        if not nb_epoch >= 1:
            nb_epoch = 1
        self.nb_epoch = nb_epoch
        if "m" in kwargs:
            logger.warning(
                f"Parameter 'm' is being overwritten with batch_size={self.batch_size}.",
            )
            kwargs["m"] = self.batch_size
        super().__init__(**kwargs)
        self.predict = self.batched_predict(self.predict)
        if hasattr(self, "_find_best_samples"):
            self._find_best_samples = self.batched_find_best_samples(
                self._find_best_samples,
            )
        if hasattr(self, "score"):
            self.score = self.batched_score(self.score)
        self.fit = self.batched_fit(self.fit)
        self.predict = self.batched_predict(self.predict)
        if self.nb_epoch > 1:
            self.fit = self.epoch_fit(self.fit)
        # self.score = self.batched_score(self.score)

    def epoch_fit(self, fit_func):
        def wrapper(*args, **kwargs):
            X, y = args
            for i in range(self.nb_epoch):
                random.shuffle(X)
                random.shuffle(y)
                fit_func(X, y, **kwargs)

        return wrapper

    def batched_fit(self, fit_func):
        def wrapper(*args, **kwargs):
            X_train, y_train = args
            n = len(X_train)
            n_batches = n // self.batch_size
            if n_batches > self.max_batches:
                logger.warning(
                    f"Number of batches ({n_batches}) is greater than max_batches ({self.max_batches}). Using max_batches.",
                )
                n_batches = self.max_batches
            for i in tqdm(
                range(n_batches),
                desc="Fitting batches",
                total=n_batches,
                leave=False,
                dynamic_ncols=True,
            ):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                print(
                    f"Shape of X_batch is {X_batch.shape} and shape of y_batch is {y_batch.shape}",
                )
                fit_func(X_batch, y_batch, **kwargs)
                if self.nb_epoch > 1:
                    continue
                train_score = self.score(X_batch, y_batch)
                test_score = self.score(X_train, y_train)
                print(
                    f"Batch {i+1} of {n_batches} - Train score: {np.mean(train_score)}; Test score: {np.mean(test_score)}",
                )
                train_scores.append(train_score)
                test_scores.append(test_score)

        return wrapper

    def batched_find_best_samples(self, func):
        def wrapper(method, **kwargs):
            if "X" in kwargs:
                X = kwargs["X"]
                assert "y" in kwargs, "y must be provided if X is provided"
                y = kwargs["y"]
                append = True
            else:
                X = self.X_
                y = self.y_
                append = False
            n_jobs = kwargs.pop("n_jobs", -1)
            n = len(X)
            n_batches = n // self.batch_size
            if n_batches > self.max_batches:
                n_batches = self.max_batches
            elif n_batches == 0:
                n_batches = 1
            for i in range(n_batches):
                if append is True:
                    new_X = X[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ]  # noqa E203
                    new_y = y[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ]  # noqa E203
                    indices = func(X=new_X, y=new_y, method=method, n_jobs=n_jobs)
                    # print("After finding best samples")
                    # print(f"Length of indices is {len(indices)}")
                    X = X[indices]
                    y = y[indices]
                    self.X_ = X
                    self.y_ = y
                    self.distance_matrix = self.distance_matrix
                else:
                    indices = func(method=method, n_jobs=n_jobs)
                    return indices

        return wrapper

    def batched_predict(self, predict_func):
        def wrapper(*args, **kwargs):
            X_test = args[0]
            n = len(X_test)
            n_batches = n // self.batch_size
            if n_batches > self.max_batches:
                n_batches = self.max_batches
            elif n_batches == 0:
                n_batches = 1
            preds = []
            for i in tqdm(
                range(n_batches),
                desc="Predicting batches",
                total=n_batches,
                leave=False,
                dynamic_ncols=True,
            ):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X_test[start:end]
                new_preds = predict_func(X_batch, **kwargs)
                preds.append(new_preds)
            return np.concatenate(preds)

        return wrapper

    def batched_score(self, score_func):
        def wrapper(*args, **kwargs):
            X_test, y_test = args
            n = len(X_test)
            n_batches = n // self.batch_size
            if n_batches > self.max_batches:
                n_batches = self.max_batches
            elif n_batches == 0:
                n_batches = 1
            scores = []
            for i in tqdm(
                range(n_batches),
                desc="Scoring batches",
                total=n_batches,
                leave=False,
                dynamic_ncols=True,
            ):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X_test[start:end]
                y_batch = y_test[start:end]
                score = score_func(X_batch, y_batch, **kwargs)
                scores.append(score)
            return scores

        return wrapper


def create_batched_class(cls, *args, **kwargs):
    name = cls.__name__

    class BatchedClass(cls, BatchedMixin):
        def __init__(self, *args, **kwargs):
            self.max_batches = kwargs.pop("max_batches", 100)
            self.batch_size = kwargs.pop("batch_size", 10)
            super().__init__(*args, **kwargs)

    batched_class = BatchedClass()
    combined_name = f"Batched{name}"
    batched_class.__name__ = combined_name
    batched_class.__init__(*args, **kwargs)
    return batched_class


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    big_X = []
    big_y = []
    for i in range(100):
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=19,
            n_redundant=1,
            n_classes=2,
            random_state=42 + i,
        )
        big_X.extend(X.tolist())
        big_y.extend(y.tolist())
    big_X = np.array(big_X)
    big_y = np.array(big_y)
    logger.info(f"Shape of big_X: {big_X.shape}")
    i = 42
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=19,
        n_redundant=1,
        n_classes=2,
        random_state=42 + i,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    class BatchedSVC(BatchedMixin, SVC):
        pass

    clf = BatchedSVC(max_batches=100, batch_size=100, kernel="rbf")
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
    input("Press enter to continue")
    score = round(np.mean(score), 2)
    std = round(np.std(score), 3)
    logger.info(f"Final Score: {score}")
    logger.info(f"Standard Deviation: {std}")
    # if plotext_available is True:
    plotext.scatter(train_scores, label="Train scores")
    plotext.scatter(test_scores, label="Test scores")
    plotext.plot()
