#!/bin/bash
dvc exp run --set-param data.add_noise.X_train=.001,.01,.1,1,10,100 --set-param model.params.learning_rate=1e-12,1e-10,1e-9,1e-8,1e-7 --queue
dvc exp run --set-param model.params.scale=.001,.01,.1,1,10,100 --set-param model.params.learning_rate=1e-12,1e-10,1e-9,1e-8,1e-7 --queue
dvc exp run --set-param model.params.scale=.001,.01,.1,10,100 --set-param
data.add_noise.X_test=.001,.01,.1,1,10,100 --set-param model.params.learning_rate=1e-12,1e-10,1e-9,1e-8,1e-7 --queue
dvc queue status > queue.txt
dvc queue start --jobs 32