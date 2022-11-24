#!/bin/bash
######################################
# Ubuntu 22.04, 20.04
sudo apt update
# install python3.9
sudo apt install python3.9 python3.9-dev python3.9-distutils python3.9-dev python3.9-venv python3.9-distutils -y
# matplotlib fonts
sudo apt  -y install msttcorefonts -qq
rm ~/.cache/matplotlib -rf
export SETUPTOOLS_USE_DISTUTILS=stdlib
######################################
python3 -m venv env
source env/bin/activate
git clone --recurse-submodules -j8 https://github.com/simplymathematics/deckard.git || git submodule update --init --recursive
python3 -m pip install ./adversarial-robustness-toolbox/
python3 -m pip install -e .
# python3 -m pip install pyinstaller
# python3 -m pip install -u numba pip setuptools
# cd deckard && pyinstaller --onefile deckard.py -n deckard
# ./dist/deckard examples/iris