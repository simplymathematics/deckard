

# Installation
You should probably be using a virutal environment rather than installing things globally. Why? 1. Because you can just remove this folder after to delete all traces of this software. 2. Your system might have python3 rather than python even though python2 is long dead. The dvc.yaml file contains scripts that are executed in whatever environment you run the `dvc repro` (see below) command from, and changing the python interpreter of the dvc command won't change that. You're welcome to change the call to the binary in each cmd or do things the right way:


If you don't already have a preferred environment manager (ce.g. conda), I recommend venv. You might need to install the operating system dependencies and python package with:
```
sudo apt-get install python3-venv
python3 -m pip install venv
```
You create a virutal environment in the folder `env` with:
```
python3 -m venv env
```
Then activate it:
```
source env/bin/activate
```
run `deactivate` to exit the virtual environment

You might need to install pip? 
```
sudo -H python -m ensurepip
```
To run the gzip_classifier.py you need to install some python dependencies:

```
python -m pip install numpy scikit-learn pandas  tqdm scikit-learn-extra imbalanced-learn plotext 
```

To reproduce the entire experiment, install `deckard` from this folder as working directory with:

```
python -m pip install ../../
```
which will run the setup.py script in the root directory of this repository.

Additionally, we are using some optuna features that are not necessarily available in whatever version of hydra you have installed. Instead, install it from source, much like you did for this repository:

```
git clone https://github.com/facebookresearch/hydra
cd hydra 
python -m pip install . 
```
Additionally, you need to install the hydra-optuna-sweeper plugin by navigating to the `examples/gzip/hydra/plugins/hydra_optuna_sweeper` folder and installing it with:
```
python -m pip install .
```

Now, we need to specify a default configuration to test before our grid search. Return to this folder and run the parser, which will read the config files, accept command line overrides, and allow you to specify a default `dvc` `params.yaml` file that will contain the git-trackable defaults for your experiments. Run the parser with:
```
python -m deckard.layers.parse --config_name default --config_folder conf
```
Both of the displayed options are the default choices anyway, but you can change them to another config folder and file as you wish. This will create a `params.yaml` file in the current working directory using hydra's compose API. 

From here, we will let dvc manage our tasks, execution order, caching, and reproducibility. You can run the entire experiment with:

```
dvc repro
```
which will read the `dvc.yaml` file, parsing any parameters specified in the `params.yaml` or any other specified file. 

It will then execute a "stage", which is a single shell command as well as "params", "deps", "outs", "metrics" and/or plots, which track the the parameters, dependencies, outputs, metric files, and/or plot files using DVC.

 You can specify an order of operations by requiring the output of an earlier stage to be a dependency of a later stage. 
 
To exploit the dvc file-tracking features, you can use dictionary keywords ( e.g. files.X ) when specifying a stage, regardless of usage within said stage. If you run the optimise script, it will overwrite files.name with the hash of the experiment config if and only if the name is set to default. Other entries: files.data_file, files.model_file, files.attack_file, will likewise be overwritten with the hash of the respective sub dictionary.
 
 Additionally, deckard will allow us to use the parameter tracking features of dvc as well as the configuration, launching, and optimization features of hydra by specifying a "stage", which will allow us to isolate pipeline stages from one another if so desired. See the `dvc.yaml` file for examples of this. In this example, it's mostly used for testing the functionality of different features in isolation of each other, but could be used for any arbitrary division of an ML pipeline you so choose. For example, we could have separate stages for sampling, feature selection, training, and evaluation. This is just a matter of passing different subsets of the configuration dictionary to the the run-time. The `deckard.layers.experiment` script will parse the entire params.yaml file, load or generate data, split it into train and test sets, fit a model, attack it (if so desired), make predictions, and score it according to the specified configs. 
