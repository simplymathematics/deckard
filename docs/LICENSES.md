# Licenses

This page centralizes licensing references for Deckard and major optional
dependency/plugin ecosystems used in the documentation and examples.

## Project License

- Deckard: [GPLv3](../LICENSE)

## Dependency License References

These links point to upstream project license pages and primary repositories.

- Python: [PSF License](https://docs.python.org/3/license.html)
- NumPy: [MIT License](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- pandas: [BSD-3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE)
- SciPy: [BSD-3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt)
- scikit-learn: [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)
- Matplotlib: [Matplotlib License](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE)
- Seaborn: [BSD-3-Clause](https://github.com/mwaskom/seaborn/blob/master/LICENSE.md)
- Yellowbrick: [BSD-3-Clause](https://github.com/DistrictDataLabs/yellowbrick/blob/develop/LICENSE.txt)
- Lifelines: [MIT](https://github.com/CamDavidsonPilon/lifelines/blob/master/LICENSE)
- Fairlearn: [MIT](https://github.com/fairlearn/fairlearn/blob/main/LICENSE)
- ART (Adversarial Robustness Toolbox): [MIT](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
- Hydra Core: [MIT](https://github.com/facebookresearch/hydra/blob/main/LICENSE)
- Optuna: [MIT](https://github.com/optuna/optuna/blob/master/LICENSE)

## Plugin License Map

- Fairlearn plugin: [Fairlearn docs](overview/extensions/fairlearn), upstream [MIT](https://github.com/fairlearn/fairlearn/blob/main/LICENSE)
- Anjana plugin: [Anjana docs](overview/extensions/anjana), upstream project-specific licensing (refer to package metadata)
- Lifelines plugin: [Lifelines docs](overview/extensions/lifelines), upstream [MIT](https://github.com/CamDavidsonPilon/lifelines/blob/master/LICENSE)
- Seaborn plugin: [Seaborn docs](overview/extensions/seaborn), upstream [BSD-3-Clause](https://github.com/mwaskom/seaborn/blob/master/LICENSE.md)
- Yellowbrick plugin: [Yellowbrick docs](overview/extensions/yellowbrick), upstream [Apache](https://github.com/DistrictDataLabs/yellowbrick/blob/develop/LICENSE.txt)

## Verification Workflow

For release-time verification, generate a dependency inventory in CI using a
tool like `pip-licenses` and reconcile results with this page.
