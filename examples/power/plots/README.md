To render the newest results.
```
git fetch <upstream name>
git checkout fix-compile-script
cd deckard/examples/power/plots
dvc repro --downstream compile
```
