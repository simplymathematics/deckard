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
    # POSIX compatibility layer and Linux environment emulation for Windows
    echo "This script is not tested on Windows Subsystem for Linux."
    echo "POSIX compatibility layer and Linux environment emulation for Windows"
elif [[ "$OSTYPE" == "msys" ]]; then
    echo "This script is not tested on Windows git shell."
    # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
    echo "Lightweight shell and GNU utilities compiled for Windows (part of MinGW)"
elif [[ "$OSTYPE" == "win32" ]]; then
    echo "This script is not tested on Windows."
    echo "I'm not sure this can happen."
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    echo "This script is not tested on FreeBSD."
    echo "FreeBSD"
else
    # Unknown.
    echo "Unknown"
fi

######################################

git clone --recurse-submodules -j8 https://github.com/simplymathematics/deckard.git || (cd deckard && git submodule update --init --recursive)
python3 -m pip install ./adversarial-robustness-toolbox/
python3 -m pip install -e .
# python3 -m pip install pyinstaller
# python3 -m pip install -u numba pip setuptools
# cd deckard && pyinstaller --onefile deckard.py -n deckard
# ./dist/deckard examples/iris
