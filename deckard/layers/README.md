.
├── README.md
├── __init__.py: imports
├── compile.py: parses output folder for results, parameters, etc
├── experiment.py: Runs an experiment from a `dvc.yaml` file and a `params.yaml` file.
├── find_best.py: Finds the best experiment as specified in a hydra configuration file.
├── generate_grid.py: Generates hydra configuration file from folder structure (work in progress)
├── generate_webpage.py: Generates a static webpage (work in progress)
├── hydra_test.py: Dumps the hydra parameters to terminal for debug.
├── optimise.py: Runs experiments as specified in the hydra configuration file and the `dvc.yaml`, using `params.yaml` as a default.
├── parse.py: Dumps the hydra configuration folder to `params.yaml` as specified by the `defaults` list in the hydra configuration file.
├── template.html: Draft webpage for each experiment (work in progress)
├── utils.py: Handy utilities for the other scripts
├── afr.py: For building accelerated failure rate models
├── plots.py: For generating plots using seaborn
└── watcher.py: Watches a file, transforms the data, and moves it somewhere else via parallelized scp (work in progress).


