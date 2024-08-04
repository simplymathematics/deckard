from tqdm import tqdm
import logging
import numpy as np


from pathlib import Path
from time import time

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
        self.batch_size = kwargs.pop("batch_size", batch_size)
        self.max_batches = kwargs.pop("max_batches", max_batches)
        self.training_log = kwargs.pop("training_log", None)
        nb_epoch = kwargs.pop("nb_epoch", nb_epoch)
        if not nb_epoch >= 1:
            nb_epoch = 1
        self.nb_epoch = nb_epoch
        super().__init__(**kwargs)
        if hasattr(self, "_find_best_samples"):
            self._find_best_samples = self.batched_find_best_samples(
                self._find_best_samples,
            )
        self.fit = self.batched_fit(self.fit)
        if self.nb_epoch > 1:
            self.fit = self.epoch_fit(self.fit)

    def epoch_fit(self, fit_func):
        def wrapper(*args, **kwargs):
            X, y = args
            X_test = kwargs.pop("X_test", None)
            y_test = kwargs.pop("y_test", None)
            log_file = self.training_log if hasattr(self, "training_log") else None
            for i in tqdm(range(self.nb_epoch), desc="Epochs", leave=True, position=0):
                # Shuffle the indices of X,y
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]
                logger.debug(f"Epoch {i + 1}/{self.nb_epoch}")
                fit_func(X, y, **kwargs)
                if hasattr(self, "score"):
                    score = self.score(X, y)
                    train_scores.append(score)
                    if X_test is not None:
                        assert len(X_test) == len(
                            y_test,
                        ), "X_test and y_test must have the same length"
                        test_score = self.score(X_test, y_test)
                        test_scores.append(test_score)
                        logger.info(f"Train score: {score}, Test score: {test_score}")
                    else:
                        logger.info(f"Train score: {score}")
                if log_file is not None:
                    if Path(log_file).exists():
                        if i == 0:
                            # rotate the log file by appending a timestamp before the extension
                            rotated_log_name = log_file.replace(
                                ".csv",
                                f"_{int(time())}.csv",
                            )
                            # rename the log file
                            Path(log_file).rename(rotated_log_name)
                            with open(log_file, "w") as f:
                                f.write("epoch, train_score,")
                                if "test_score" in locals():
                                    f.write(",test_score")
                                f.write("\n")
                                f.write(f"{i+1},")
                                f.write(f"{score},")
                                if "test_score" in locals():
                                    f.write(f" {test_score},")
                                f.write("\n")
                        else:
                            with open(log_file, "a") as f:
                                # assuming csv format
                                f.write(f"{i+1},")
                                f.write(f"{score},")
                                if "test_score" in locals():
                                    f.write(f"{test_score},")
                                f.write("\n")
                    else:
                        with open(log_file, "w") as f:
                            f.write("epoch, train_score,")
                            if "test_score" in locals():
                                f.write(" test_score,")
                            f.write("\n")
                            f.write(f"{i+1},")
                            f.write(f"{score},")
                            if "test_score" in locals():
                                f.write(f"{test_score},")
                            f.write("\n")
            import plotext as plt

            plt.plot(train_scores, label="Train score")
            if X_test is not None:
                plt.plot(test_scores, label="Test score")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Scores")
            plt.show()

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
                total=n_batches,
                desc="Fitting batches",
                leave=False,
                position=1,
            ):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                fit_func(X_batch, y_batch, **kwargs)

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
                    new_X = X[i * self.batch_size : (i + 1) * self.batch_size]  # noqa
                    new_y = y[i * self.batch_size : (i + 1) * self.batch_size]  # noqa
                    indices = func(X=new_X, y=new_y, method=method, n_jobs=n_jobs)
                    X = X[indices]
                    y = y[indices]
                    self.X_ = X
                    self.y_ = y
                    self.distance_matrix = self.distance_matrix
                else:
                    indices = func(method=method, n_jobs=n_jobs)
                    return indices

        return wrapper
