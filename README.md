```
git clone https://github.com/simplymathematics/deckard.git
cd deckard
git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
cd adversarial-robustness-toolbox && python -m pip install -e . && cd ..
python -m pip isntall -e .
source env
```

Check that deckard works

```$ python```  
```>>> import deckard```  
Then CTRL+D or `quit()` to quit.  
# Navigate to your favorite subfolder in `examples`  
(NOTE: only 'iris' is fully supported at the moment).
```dvc repro --force``` 
### _like tears in the rain_, this tool is meant for bladerunners. NOT INTENDED FOR USE BY REPLICANTS
