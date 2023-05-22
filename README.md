For Developers:
```
######################################
# Ubuntu 22.04, 20.04
sudo apt update
sudo apt install python3-venv python3-pip python3-dev python3-setuptools
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9 -y
sudo apt install msttcorefonts -qqpython3-distutils #fonts (optional)
export SETUPTOOLS_USE_DISTUTILS=stdlib
######################################
python3 -m venv env
source env/bin/activate
git clone --recurse-submodules -j8 https://github.com/simplymathematics/deckard.git
# git submodule update --init --recursive # To just update the submodules
python3 -m pip install deckard/adversarial-robustness-toolbox/
python3 -m pip install -e deckard/
python3 -m pip install pyinstaller
python3 -m pip install -u numba pip setuptools
cd deckard && pyinstaller --onefile deckard.py -n deckard
./dist/deckard examples/sklearn
```

Check that deckard works

```$ python```
```>>> import deckard```
Then CTRL+D or `quit()` to quit.
# Navigate to your favorite subfolder in `examples`. One is provided for each framework.
```python -m deckard.layers.experiment```
### _like tears in the rain_, this tool is meant for bladerunners. NOT INTENDED FOR USE BY REPLICANTS
