
For Developers:
```
python3 -m venv env
source activate env/bin/activate
git clone --recurse-submodules -j8 https://github.com/simplymathematics/deckard.git
python3 -m pip install deckard/adversarial-robustness-toolbox/ 
python3 -m pip install -e deckard/
python3 -m pip install pyinstaller
cd deckard && pyinstaller --onefile deckard.py -n deckard
./dist/deckard examples/iris
```
or run the above script in bash using curl:
```
bash <(curl -sL https://gist.githubusercontent.com/simplymathematics/8acd1015751081c4cb05e6766ffee5b0/raw/837bab134e32f7af801b344c56ffdae6af676194/build.sh)
```
Check that deckard works

```$ python```  
```>>> import deckard```  
Then CTRL+D or `quit()` to quit.  
# Navigate to your favorite subfolder in `examples`  
(NOTE: only 'iris' is fully supported at the moment).
```dvc repro --force``` 
### _like tears in the rain_, this tool is meant for bladerunners. NOT INTENDED FOR USE BY REPLICANTS
