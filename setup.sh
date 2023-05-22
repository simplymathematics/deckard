#!/bin/bash
######################################
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # ...
    # Ubuntu 22.04, 20.04
    sudo apt update
    # install python3.9
    sudo apt install python3.9 python3.9-dev python3.9-distutils python3.9-dev python3.9-venv python3.9-distutils -y
    # matplotlib fonts
    sudo apt  -y install msttcorefonts -qq
    rm ~/.cache/matplotlib -rf
    export SETUPTOOLS_USE_DISTUTILS=stdlib
    python3 -m venv env
    source env/bin/activate
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    echo "This script is not tested on Mac OSX."
    echo "Mac OSX"
elif [[ "$OSTYPE" == "cygwin" ]]; then
    echo "POSIX compatibility layer and Linux environment emulation for Windows"
elif [[ "$OSTYPE" == "msys" ]]; then
    echo "Installing on windows git shell."
elif [[ "$OSTYPE" == "win32" ]]; then
    echo "I'm not sure this can happen. Good luck."
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    echo "This script is not tested on FreeBSD."
    echo "FreeBSD"
else
    # Unknown.
    echo "Unknown"
fi
######################################
git clone --recurse-submodules -j8 https://github.com/simplymathematics/deckard.git || (cd deckard && git submodule update --init --recursive)
python3 -m pip install adversarial-robustness-toolbox/
python3 -m pip install  -e . --user 

