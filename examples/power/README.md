To reproduce this experiment locally, run which will parse the specified config from the `conf/` folder to params.yaml, then run `dvc repro`:
```
python -m deckard --config_file torch_mnist.yaml
python -m deckard --config_file torch_cifar10.yaml
python -m deckard --config_file torch_cifar100.yaml
```


To see the dashboard, find a *.db file. They should be in something like `mnist/reports/attack/torch_mnist.db`. 

```
sudo apt install python3.10-venv
python -m venv optuna
source optuna/bin/activate
python -m pip install git+https://github.com/simplymathematics/deckard.git
python -m pip install optuna-dashboard
optuna-dashboard sqlite:///torch_mnist.db
```
You will either need to open ports or connect to the filestore vm via vscode (recommended).
