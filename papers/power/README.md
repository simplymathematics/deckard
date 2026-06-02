# Power Experiment

The power paper now uses `deckard optimize` sweeps plus `compile_results` to merge all studies into one table.

To reproduce locally from this directory:

```bash
cd papers/power
dvc repro
```

The compiled results are written to:

```bash
output/combined/power_results.csv
```

To inspect the Optuna study database:

```bash
sudo apt install python3.10-venv
python -m venv optuna
source optuna/bin/activate
python -m pip install git+https://github.com/simplymathematics/deckard.git
python -m pip install optuna-dashboard
optuna-dashboard sqlite:///output/optuna.db
```

You will either need to open ports or connect to the filestore vm via vscode (recommended).
