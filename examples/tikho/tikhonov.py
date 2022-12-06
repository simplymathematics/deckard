import numpy as np


class TikhonovClassifier:
    def __init__(self, mtype: str = "linear", scale: float = 0.0):

        self.type = mtype
        self.scale = scale

        if mtype.lower() == "linear":
            self.loss = self.lin_loss
            self.gradient = self.lin_gradient
        elif mtype.lower() == "logistic":
            self.loss = self.log_loss
            self.gradient = self.log_gradient
        else:
            raise NotImplementedError("Model type not implemented")

    def sigmoid(self, z):
        return np.divide(1, 1 + np.exp(-z))

    def lin_loss(self, x, y):
        y_hat = x @ self.weights + self.bias
        errors = y_hat - y
        squared = 0.5 * np.linalg.norm(errors) ** 2
        dydx = self.weights
        tikho = 0.5 * np.sum(dydx**2)
        return squared + tikho * self.scale

    def lin_gradient(self, x, y):
        # || x @ weights + bias - y || ^2
        reg = 2 * self.weights * self.scale
        gradL_w = 2 * (x @ self.weights + self.bias - y) @ x + reg
        gradL_b = 2 * (x @ self.weights + self.bias - y)
        return (gradL_w, gradL_b)

    def log_loss(self, x, y):
        y_hat = self.sigmoid(x @ self.weights + self.bias)
        errors = np.mean(y * (np.log(y_hat)) + (1 - y) * (1 - y_hat))
        # + scale/2 * np.mean(weights ** 2)
        return errors + self.scale / 2 * np.mean(self.weights**2)

    def log_gradient(self, x, y):
        y_hat = self.sigmoid(x @ self.weights + self.bias)
        reg = 2 * self.weights * self.scale
        gradL_w = 2 * (x @ self.weights + self.bias - y) @ x + 2 * reg
        gradL_b = np.mean(y_hat - y)
        return (gradL_w, gradL_b)

    def fit(self, X_train, y_train, learning_rate=1e-6, epochs=1000):
        self.weights = (
            np.ones((X_train.shape[1])) * 1e-8
            if not hasattr(self, "weights")
            else self.weights
        )
        self.bias = 0.0 if not hasattr(self, "bias") else self.bias
        for i in range(epochs):
            L_w, L_b = self.gradient(X_train, y_train)
            self.weights -= L_w * learning_rate
            self.bias -= L_b * learning_rate
        return self

    def predict(self, x):
        # print(x.shape, self.weights.shape, self.bias.shape)
        x_dot_weights = x @ self.weights.T
        return [1 if p > 0.5 else 0 for p in x_dot_weights]

    def predict_proba(self, x):
        x_dot_weights = x @ self.weights.T
        return x_dot_weights

    def score(self, x, y):
        # print(x.shape, weights.shape, bias.shape)
        x_dot_weights = x @ self.weights.T
        y_test = [1 if p > 0.5 else 0 for p in x_dot_weights]
        return np.mean(y == y_test)


# if __name__ == "__main__":
# yaml.add_constructor('!Experiment:', Experiment)
# experiment = yaml.load( "!Experiment:\n" + str(params), Loader = yaml.Loader)
# data, model = Experiment.load(experiment)
# logger = Live(path = Path(files['path']), report = "html")
# epochs = round(int(epoch/log_interval))
# for i in tqdm(range(epochs)):
#     start = process_time()
#     clf = model.fit(data.X_train, data.y_train, learning_rate = learning_rate)
#     if i % log_interval == 0:
#         fit_time = process_time() - start
#         logger.log("time", fit_time)
#         train_score = clf.score(data.X_train, data.y_train)
#         test_score = clf.score(data.X_test, data.y_test)
#         logger.log("train_score", train_score)
#         logger.log("loss", clf.loss(data.X_train, data.y_train))
#         logger.log("weights", np.mean(clf.weights))
#         logger.log("bias", np.mean(clf.bias))
#         logger.log("epoch", i * log_interval)
#         logger.next_step()
# predictions = clf.predict(data.X_test)
# proba = clf.predict_proba(data.X_test)
# predictions, predict_time = experiment.predict(data, model)
# ground_truth = data.y_test
# time_dict = {"fit_time": fit_time, "predict_time": predict_time}
# score_dict = experiment.score(data, predictions)
# print(score_dict)
# files = experiment.save(**experiment.files, data = data, model = model, ground_truth = ground_truth, predictions = predictions, time_dict = time_dict, score_dict = score_dict)
# for file in files:
#     assert file.exists(), f"File {file} does not exist."

# yb_model = classifier(model)
# path = files.pop("path")
# i = 0
# for visualizer in [classification_report, confusion_matrix]:
#     name = list(plots.values())[i]
#     viz = visualizer(yb_model, X_train = X_train, y_train = y_train, classes=[int(y) for y in np.unique(y_test)])
#     viz.show(outpath=Path(path, name))
#     plt.gcf().clear()
#     i += 1
# f"Unused Params: {params}"
