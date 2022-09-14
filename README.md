
For Developers:
```
sudo apt install python3, python3-dev, 
git clone https://github.com/simplymathematics/deckard.git
cd deckard
git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
python3 -m venv env 
cd adversarial-robustness-toolbox && python3 -m pip install -e . && cd ..
python3 -m pip install -e .
```

Check that deckard works

```$ python```  
```>>> import deckard```  
Then CTRL+D or `quit()` to quit.  
# Navigate to your favorite subfolder in `examples`  
(NOTE: only 'iris' is fully supported at the moment).
```dvc repro --force``` 
### _like tears in the rain_, this tool is meant for bladerunners. NOT INTENDED FOR USE BY REPLICANTS
