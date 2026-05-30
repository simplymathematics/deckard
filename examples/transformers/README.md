# Deckard Transformers Example

This example adds a canonical Hugging Face + ART workflow using:

- Dataset: DEMI-MathAnalysis (`ziye2chen/DEMI-MathAnalysis`)
- Attack: HopSkipJump (`art.attacks.evasion.HopSkipJump`)
- Model wrapper: `deckard.frameworks.transformers.declarations.GenericFlexibleTransformer`

The `config/attack/` group includes ART black-box profiles across supported evasion, extraction, and inference families.

## Run

```bash
cd examples/transformers
source ../../.venv/bin/activate
export DECKARD_CONFIG_DIR=./config
export DECKARD_DEFAULT_CONFIG_FILE=default.yaml
deckard data=demi_math_analysis model=hf_generic_transformer attack=hsj
```

## DVC

```bash
cd examples/transformers
dvc repro
```

The pipeline definition is in `dvc.yaml`.
