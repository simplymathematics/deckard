# Hydra Overview

To compose run-time objects from configuration files, we use [hydra](https://hydra.cc). 
To optimize a dataset, model, attack, or detector for one or more objectives, we use [optuna](https://optuna.org) which handles studies, trials, sampling, pruning, and optimizaiton. 
In this brief overview, we demonstrate and explain the differences

- a single composed experiment (`--run`)
- a multirun sweep (`--multirun`)

For the following examples, we will use a modified version of the configuration that can be found in `examples/sklearn/config/default.yaml` in the source code repository.

## Hydra Defaults

The hydra defaults list is fairly self-explanatory-- it provides a way to create run-time objects from structured configs.
Below, see how we can set both groups like "data" and "model", but also sub-groups like "sampler@data.sampler" and "model@model.trainer". 
If you explore the examples/sklearn/config folder, you will see that for each group/sub-group there is a *.yaml file that corresponds with the denoted value.
For more details, see the [hydra documentation](https://hydra.cc/docs/configure_hydra/intro/).

```yaml
defaults:
- _self_
- data: adult # Sets the dataset configuration group
- sampler@data.sampler: split # S
- model: rf
- trainer@model.trainer: sklearn
- defense: class-labels
- attack: hsj
- score: classification
- search/models: ${oc.select:model,rf}
- search/samplers: split
- search/trainers: sklearn
- search/defenses: ${oc.select:defense,class-labels}
- search/attacks: ${oc.select:attack,hsj}
- files: null
```

## Hydra Syntax

Hydra also provides a standard syntax for instantiating run-time objects from files.
Primarily, this relies on the omegaconf DictConfig object to store the class name, initialization parameters, and any other run-time parameters.
As an example, see the `examples/sklearn/config/model/rf.yaml` file below:
```yaml
name: sklearn.ensemble.RandomForestClassifier
classifier : True
model_params:
  n_estimators: 100
  criterion: gini
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  min_weight_fraction_leaf: 0.0
  max_features: sqrt
  max_leaf_nodes: null
  min_impurity_decrease: 0.0
  bootstrap: true
_target_ : deckard.model.ModelConfig
alias: rf
```
We use the hydra syntax to instantiate a deckard ModelConfig object at run-time, then we use the ModelConfig.initialize() method to instantiate a sklearn.ensemble.RandomForestclassifier with the specified params. 
```python
import yaml
from hydra.utils import instantiate
model_cfg = yaml.load(my_file.yaml)
model_instance = instantiate(model_cfg)
```
is equivalent to
```python
import yaml
from sklearn.ensemble import RandomForestClassifier
model_cfg = yaml.load(my_file.yaml)
model_instance = RandomForestClassifier(**model_cfg['model_params'])
```
This allows us to write machine learning pipelines that are decoupled from the frameworks, attacks, defense, and metrics we want to examine.

### Overrides
Hydra provides two ways to override the aforementioned defaults. 
The first way is with the override directive in the defaults list.
```yaml
defaults:
- _self_
- data: adult 
- ... # Same as above
- override hydra/sweeper: optuna
- override hydra/sweeper/sampler: random
```
The hydra/sweeper override, optuna, is the only one that deckard supports natively for post-hoc analysis, pruning, and optimization. 
However, the hydra/sweeper/sampler override supports [many search-space sampling algorithms](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html).

The second way to override the hydra default is via the cli:
```bash
python my_app.py --config-name default.yaml --config-dir examples/sklearn/config data=make_classification ++model=logistic ~attack +defense=feature_squeezing sampler@data.sampler=fold
```
Here, we set the configuration name and directory to specify the canon discussed above, use the "make_classification" data instead of the default "adult" data ('='), override('++') the existing model group with the logistic model, remove('~') the attack configuration, append(`+`) a defense to the defense list, and configure as subgroup (sub@group.sub=). 
For more details, see the hydra [override documentation](https://hydra.cc/docs/configure_hydra/intro/).

## Hydra dot-list notation
Both hydra and deckard adopt hydra dot-list syntax for configuring and refencing nested dictionaries.

In yaml:
```yaml
data:
  name: adult
```
in bash:
```bash
python my_app.py data.name=adult
```
or in python
```python
import yaml
data_cfg = yaml.load("my_file.yaml")
data = instantiate(data_cfg)
data.name="adult"
```

## Hydra Run
Each example folder has a `config/` folder that contains one or more `default.yaml` files. 
A minimal, reproducible example is shown below. 
When you run an experiment in single-run mode, you should configure the hydra.run.dir and the hydra.sweeper dictionary so that results can be tracked in an optuna database for analysis.

```yaml
experiment_name: ${hash:${stage_params:${oc.select:stage,???}}}
directions:
- <maximize or minimize>
optimizers:
- <metric name>
hydra:
  run:
    dir: outputs/logs/${experiment_name}
  sweeper:
    study_name: my_study_name
    storage: sqlite:///my_database.db
    direction: ${directions}
    max_failure_rate: 1.0
    sampler:
      _target_: optuna.samplers.RandomSampler
      seed: 42
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
    n_trials: 100
    n_jobs: 1
    params: <insert search space>
  callbacks:
    deckard_optuna:
      _target_: deckard.layers.optimize.DefaultOptimizerCallback
```


## Hydra Multirun

When you run an experiment in single-run mode, you should configure the hydra.sweep dictionary and the hydra.sweeper dictionary so that results can be tracked in an optuna database for analysis. 

```yaml
report_trial_attrs: true
pruning_enabled: false
  sweep:
    dir: outputs/logs/
    subdir: ${hydra.sweeper.study_name}/${hydra.job.num}
  sweeper:
    study_name: my_study_name
    storage: sqlite:///my_database.db
    direction: ${directions}
    max_failure_rate: 1.0
    sampler:
      _target_: optuna.samplers.RandomSampler
      seed: 42
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
    n_trials: 100
    n_jobs: 1
    params: <insert search space>
  callbacks:
    deckard_optuna:
      _target_: deckard.layers.optimize.DefaultOptimizerCallback
```
Unlike single-run mode, however, multi-run mode provides for several different features. 

### Optuna Sampling

### Optuna Pruning


## Hydra Callback
The same callback adapter is used in both modes:

- {class}`deckard.layers.optimize.DefaultOptimizerCallback`

In run mode, it uses hydra composition (defaults + overrides) to run a single experiment

## Related Docs

- [Optimization](optimize)
- [Hydra and Optuna Orchestration Contract](/developers/optimization/hydra)
- [Optimization Runtime Contract](/developers/optimization/optimization)
- [Pruning Runtime Contract](/developers/optimization/pruning)
