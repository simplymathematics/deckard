# Power Plot Rendering

To render the newest results.

```bash
git fetch <upstream name>
git checkout fix-compile-script
cd deckard/examples/power/plots
dvc repro --downstream compile
```
