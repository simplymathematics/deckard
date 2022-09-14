git clone https://github.com/simplymathematics/deckard.git
cd deckard
git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
cd adversarial-robustness-toolbox && python -m pip install . && cd ..

```python setup.py develop```  

Check that deckard works

```$ python```  
```>>> import deckard```  

# Navigate to your favorite subfolder in `examples`

```dvc repro --force``` 

Reproduces the last experiment.

```dvc pull``` 
fetches the last experiment and all binary dependencies

```dvc push``` I think you can figure this out.

```dvc dag show``` 
displays the directed acyclic  graph of dependencies

```dvc metrics show``` 
show current metrics

```dvc metrics diff <commit>``` 
shows the change in metrics between HEAD and commit. 

```dvc exp run```
run a new experiment without overwriting the last dvc metric (but will overwrite binaries on disk because you already know how to push/pull with dvc)

```dvc.yaml``` specifies the pipeline, parameters, configs, etc 

```dvc.lock``` a git-trackable file that links individual dependencies/metrics to things in the remote cache.

### _like tears in the rain_, this tool is meant for bladerunners. NOT INTENDED FOR USE BY REPLICANTS
