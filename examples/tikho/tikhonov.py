import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dvclive import Live
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from time import process_time
from sklearn.model_selection import ParameterGrid
from numpy import logspace
from hashlib import md5
import dvc.api
live = Live("Tikhonov Training Loss")

class LinearTikhonov():
  def __init__(self, scale):
    self.weights = None
    self.bias = None
    self.losses = None
    self.scale = scale

  def loss(self, x, y):
    y_hat = x @ self.weights  + self.bias
    errors = y_hat  - y 
    squared = .5 * np.linalg.norm(errors)**2
    dydx = self.weights
    tikho =.5 * np.sum(dydx ** 2 )
    return squared + tikho * self.scale

  def gradient(self, x, y):
    # || x @ weights + bias - y || ^2
    reg = 2 * self.weights * self.scale
    gradL_w = 2 *(x @ self.weights +self.bias -y ) @ x + reg
    gradL_b = 2 * (x @ self.weights +self.bias - y) 
    return (gradL_w, gradL_b)
  
  def fit(self, X_train, y_train, learning_rate = 1e-6, epochs = 1000):
    start = process_time()
    self.weights = np.ones((X_train.shape[1])) * 1e-9
    self.bias = 0
    self.losses = []
    for i in tqdm(range(epochs)):
      L_w, L_b = self.gradient(X_train, y_train)
      # print(L_w.shape, L_b.shape)
      self.weights -= L_w * learning_rate
      self.bias -= L_b * learning_rate
      live.log("loss", self.loss(X_train, y_train))
      live.log("weights", np.mean(self.weights))
      live.log("bias", np.mean(self.bias))
      live.log("time", process_time() - start)
      live.log("score", model.score(X_test, y_test))
      live.log("f1", f1_score(y_test, y_test))
      live.log("precision", precision_score(y_test, y_test))
      live.log("recall", recall_score(y_test, y_test))
      live.log("accuracy", accuracy_score(y_test, y_test))
      live.log("roc_auc", roc_auc_score(y_test, y_test))
      live.log("epoch", i)
      live.log("learning_rate", learning_rate)
      live.next_step()
    return self.losses, self.weights, self.bias

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
    y_test =  [1 if p > 0.5 else 0 for p in x_dot_weights]
    return np.mean(y == y_test)



class LogisticTikhonov():
  def __init__(self, scale):
    self.scale = scale
    self.weights = None
    self.bias = None
    self.losses = None

  def sigmoid(self, z):
    return np.divide(1, 1 + np.exp(-z))

  # From linear regression
  def loss(self, x, y):
    y_hat = self.sigmoid(x @ self.weights + self.bias)
    errors =  np.mean(y * (np.log(y_hat)) + (1-y) * (1-y_hat))
    # + scale/2 * np.mean(weights ** 2)
    return(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean() + self.scale/2 * np.mean(self.weights ** 2)

  def gradient(self, x, y):
    y_hat = self.sigmoid(x @ self.weights +self.bias)
    reg = 2 * self.weights * self.scale
    gradL_w = 2 *(x@self.weights +self.bias -y ) @ x + 2 * reg
    gradL_b =  np.mean(y_hat-y)
    return (gradL_w, gradL_b)


  # from autograd import grad
  # import autograd.numpy as np
  # gradient = grad(loss)


  def fit(self, x_train, y_train, learning_rate = 1e-6, epochs = 10000):
    self.weights = np.ones((X_train.shape[1])) * 1e-9
    self.bias = 0
    self.losses = []
    start = process_time()
    for i in tqdm(range(epochs)):
      L_w, L_b = self.gradient(X_train,  y_train)
      # print(L_w.shape, L_b.shape)
      self.weights -= L_w * learning_rate
      self.bias -= L_b * learning_rate
      # if (i+1) % 100 == 1:
      self.losses.append(self.loss(X_train, y_train))
      live.log("loss", self.loss(X_train, y_train))
      live.log("weights", self.weights)
      live.log("bias", self.bias)
      live.log("score", model.score(X_test, y_test))
      live.log("f1", f1_score(y_test, y_test))
      live.log("precision", precision_score(y_test, y_test))
      live.log("recall", recall_score(y_test, y_test))
      live.log("accuracy", accuracy_score(y_test, y_test))
      live.log("roc_auc", roc_auc_score(y_test, y_test))
      live.log("epoch", i)
      live.log("time", process_time() - start)
      live.next_step()
    return self.losses, self.weights, self.bias

  def predict(self, x):
    # print(x.shape, weights.shape, bias.shape)
    x_dot_weights = x @ self.weights.T + self.bias
    return [1 if p > 0.5 else 0 for p in x_dot_weights]
  
  def predict_proba(self, x):
    x_dot_weights = x @ self.weights.T + self.bias
    return x_dot_weights

  def score(self, x, y):
    y_test =  self.predict(x)
    return np.mean(y == y_test)
  
def run_experiment(model, X_train, y_train, X_test, y_test, epochs = 1000, learning_rate = 1e-6, scale = 1e-9, input_noise = 0, output_noise = 0):
  model.fit(X_train, y_train, learning_rate = learning_rate, epochs = epochs)
  live.log("Final Score", model.score(X_test, y_test))
  predictions = model.predict(X_test)
  actual = y_test
  probabilities = model.predict_proba(X_test)
  id_ = md5(str(args).encode('utf-8')).hexdigest()
  df = pd.DataFrame({'predictions': predictions, 'actual': actual, 'probabilities': probabilities})
  df.to_json(str(type(model))+"id_" + '_predictions.json', orient='records')
  f1_ = f1_score(y_test, predictions)
  acc_ = accuracy_score(y_test, predictions)
  prec_ = precision_score(y_test, predictions)
  rec_ = recall_score(y_test, predictions)
  auc_ = roc_auc_score(y_test, probabilities)
  series = pd.Series({'f1': f1_, 'accuracy': acc_, 'precision': prec_, 'recall': rec_, 'auc': auc_, 'scale': scale, 'epochs': epochs, 'learning_rate': learning_rate, 'ord': ord, 'input_noise': input_noise, 'output_noise': output_noise, 'type': type})
  series.to_json(str(type(model))+"id_" + "_metrics.json", orient='records')


def get_data(output_noise = 0, input_noise = 0):
  X, y = make_blobs(n_samples=1000, n_features = 2, random_state=42, centers=2)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  if input_noise > 0:
    X_train = X_train + np.random.normal(0, input_noise, X_train.shape)
  elif input_noise <0:
    raise ValueError('Input noise must be positive')
  if output_noise > 0:
    X_test = X_test + np.random.normal(0, input_noise, X_test.shape)
  elif output_noise <0:
    raise ValueError('Output noise must be positive')   
  return X_train, X_test, y_train, y_test

def dohmatob(X_train, probability, error, ord = 2):
  ds = []
  for column in range(X_train.shape[1]):
    std = np.std(X_train[:,column])
    ds.append(std * np.sqrt(2 * np.log(1/ error)/probability))
  return np.linalg.norm(ds, ord = ord)

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--ord', type=int, default=2)
    parser.add_argument('--input_noise', type=float, default=0.0)
    parser.add_argument('--output_noise', type=float, default=0.0)
    parser.add_argument('--type', type=str, default='linear')
    args = parser.parse_args()
    
    from tqdm import tqdm
    
    
    input_noises = logspace(-6, 1, 5)
    output_noises = logspace(-6, 1, 5)
    epochs = [10000]
    learning_rates = logspace(-6, 1, 5)    
    scales = logspace(-6, 1, 6)
    types = ['linear', 'logistic']
    grid = ParameterGrid({'input_noise': input_noises, 'output_noise': output_noises, 'epochs': epochs, 'learning_rate': learning_rates, 'scale': scales, 'type': types})
    for entry in tqdm(grid, "Big Grid Search"):
      X_train, X_test, y_train, y_test = get_data(entry['output_noise'], entry['input_noise'])
      del entry['output_noise']; del entry['input_noise']
      if args.type == 'linear':
        model = LinearTikhonov(args.scale)
      elif args.type == 'logistic':
        model = LogisticTikhonov(args.scale)
      else:
        raise ValueError('Type must be linear or logistic')
      del entry['type']
      run_experiment(model, X_train, y_train, X_test, y_test, **entry)
    # print(dvc.api.params_show('params.yaml'))
    
    
    
    
   
