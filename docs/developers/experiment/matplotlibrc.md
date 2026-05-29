# Matplotlibrc Behavior and Extension Examples

Deckard keeps plotting defaults in `deckard/plot/.matplotlibrc`.
This file is generated from Seaborn theme settings so plotting behavior is
deterministic in local notebooks, CI docs builds, and runtime reports.

## Source of Truth

- Generator script: `scripts/generate_matplotlibrc.py`
- Generated config: `deckard/plot/.matplotlibrc`

The script applies `seaborn.set_theme(...)` and writes resolved
`matplotlib.rcParams` in `matplotlibrc` format.

## Regeneration Workflow

Run from the repository root:

```bash
python scripts/generate_matplotlibrc.py \
  --context notebook \
  --style whitegrid \
  --palette colorblind \
  --output deckard/plot/.matplotlibrc
```

## Extension Examples

### Example 1: Increase publication readability

```bash
python scripts/generate_matplotlibrc.py \
  --context talk \
  --font-scale 1.2 \
  --rc-param axes.titlesize=16 \
  --rc-param axes.labelsize=14 \
  --output deckard/plot/.matplotlibrc
```

### Example 2: Custom color cycle for plugin reports

```bash
python scripts/generate_matplotlibrc.py \
  --palette '["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]' \
  --output deckard/plot/.matplotlibrc
```

### Example 3: Disable top and right spines in all notebook plots

```bash
python scripts/generate_matplotlibrc.py \
  --rc '{"axes.spines.top": false, "axes.spines.right": false}' \
  --output deckard/plot/.matplotlibrc
```

## Validation Checklist

1. Regenerate `deckard/plot/.matplotlibrc`.
2. Re-run at least one plotting notebook from each plugin family:
   `seaborn.ipynb`, `yellowbrick.ipynb`, and one robustness notebook.
3. Rebuild docs and confirm no rendering regressions in generated images.

For workflow-level guidance, see {doc}`/developers/contributor/workflows`.
