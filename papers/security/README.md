# Security Experiments (Unified)

This folder consolidates the legacy experiment sets into one modern deckard configuration layout:

- Legacy `classification/` -> `config/data/classification.yaml`
- Legacy `kdd-nsl/` -> `config/data/kdd-nsl.yaml`
- Legacy `truthseeker/` -> `config/data/truthseeker.yaml`

The structure and command flow mirror `examples/sklearn`.

## Layout

- `config/`: Hydra configs for `data`, `model`, `attack`, `defense`, `search`, `files`, and `plot`
- `dvc.yaml`: single matrix pipeline for all security experiments
- `outputs/`: generated logs, artifacts, and reports (gitignored)

## Run

From this folder:

```bash
pyenv activate deckard
dvc repro set_config_paths
dvc repro security_matrix
```

Or run a single experiment directly:

```bash
deckard optimize \
  data=classification \
  model=svc \
  defense=baseline \
  attack=pgd \
  ++model.model_params.kernel=rbf \
  --multirun
```
