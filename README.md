

# Deckard: A Tool for Evaluating AI

## Installation

To install this, ensure that you have your favorite library installed. To install deckard along with `tensorflow`, for example, use
```
python -m pip install .[tensorflow]
```
Add the `-e` flag if you want to edit files:
```
python -m pip install -e . 
```
You can also try the bash script which attempts to install python if you have root.
```
bash setup.sh
```
Or try the rpi script:
```
bash rpi.sh
```
Now, heck that deckard works
```$ python```
```>>> import deckard```
Then CTRL+D or `quit()` to quit.
# Navigate to your favorite subfolder in `examples`. One is provided for each framework.
Running `dvc repro` in that folder will reproduce the experiment outlined in the `dvc.yaml`. Running `python -m deckard` will parse the configuration folder, create a `params.yaml`, and then run `dvc repro`.
### _like tears in the rain_, this tool is meant for bladerunners. NOT INTENDED FOR USE BY REPLICANTS

## Files

.
├── Dockerfile: Constructs a generic Docker image for running experiments
├── LICENSE
├── README.md: this file
├── deckard: Source code
├── examples: Directory containing all the examples
├── rpi.sh: For installation on Raspbian. 
├── setup.py : for installation with pip
├── setup.sh : for installation using bash
└── test : test suite


### 

To build the package (optional and is a very rough draft):

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
```

After adding it to your path, you can then run deckard like a package:

```
deckard examples/sklearn
```
