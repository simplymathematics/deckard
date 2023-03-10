# Classification Experiments

## Directory Contents

├── conf: contains the configuration files.  
│   ├── attack  
│   ├── config.yaml  : contains the default configuration  
│   ├── data  
│   ├── files  
│   ├── model  
│   ├── plots  
│   └── scorers  
├── dag.md: contains the [dvc](https://dvc.org/doc/start/data-management/data-pipelines) pipeline, visualized as a graph  
├── dvc.lock: contains the git trackable information for all the data and model binaries  
├── dvc.yaml: specifies the pipeline visualized in `dag.md`  
├── experiments.sh: specifies the grid search parameters and executes the pipeline for each set of parameters  
├── multirun: contains the hydra configuration information  
├── params.yaml: contains the parsed default configuration file  
├── plots.ipynb: is a jupyter notebook for visualizing the data  
├── queue: contains all of the configurations for all experiments  
├── reports: contains the results of all experiments, with a folder for each layer (e.g. `reports/train/<experiment id>`), containing scores, plots, predictions, probabilities, and ground_truth json files.  
├── train: contains the `data/` and `models/` for each model  
│   ├── data  
│   ├── models  

## Execution instructions

To check basic code functionality, run  
```dvc repro --force```  
which will execute the [dvc pipeline](https://dvc.org/doc/start/data-management/data-pipelines) that makes all of the inputs and outputs git trackable. This will execute all of the code on the default configurations first (stages `generate`, `train`, and `attack` on the `dvc.yaml`), followed by grid search on all of the models in stage `model-queue` and all of the attacks in stage `attack-queue`. After finding the best configuration for each kernel (stage `compile`) 

which overrides the default configurations, using [hydra](https://hydra.cc/docs/patterns/configuring_experiments/) and its [optuna plugin](https://hydra.cc/docs/plugins/optuna_sweeper/) to specify a search space for each set of parameters specified in the bash script. Sets of parameters that apply to all experiments are set in the `config.yaml`. Parameters particular to a experiment are specified using command line overrides within the bash script.
