#!/bin/bash
if [[ "$EUID" -eq 0 ]]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]];
    then
        echo "Installing necessary packages for Linux via apt. Only Debian based distros are supported."
        # Ubuntu 22.04, 20.04
        sudo apt update
        # install python3
        sudo apt install python3 python3-dev python3-distutils python3-dev python3-venv python3-distutils -y
        # matplotlib fonts
        sudo apt  -y install msttcorefonts -qq
        rm ~/.cache/matplotlib -rf
        export SETUPTOOLS_USE_DISTUTILS=stdlib
        
    elif [[ "$OSTYPE" == "darwin"* ]]; 
    then
        # Mac OSX
        echo "This script is not tested on Mac OSX."
    elif [[ "$OSTYPE" == "cygwin" ]]; 
    then
        echo "POSIX compatibility layer and Linux environment emulation for Windows. Python must be installed"
        python -m pip -V || echo "Python is not installed. Please install Python and pip."
    elif [[ "$OSTYPE" == "msys" ]]; 
    then
        echo "Installing on windows git shell."
        python -m pip -V || echo "Python is not installed. Please install Python and pip."
    elif [[ "$OSTYPE" == "win32" ]]; 
    then
        echo "I'm not sure this can happen. Good luck."
        python -m pip -V || echo "Python is not installed. Please install Python and pip."
    elif [[ "$OSTYPE" == "freebsd"* ]]; 
    then
        echo "This script is not tested on FreeBSD."
        echo "FreeBSD"
        python -m pip -V || echo "Python is not installed. Please install Python and pip."
    else
        # Unknown.
        echo "Unknown operating system."
    fi
else
    echo "Not running as root. Assuming that Python pip is already installed."
fi
######################################
git clone --recurse-submodules -j 8 https://github.com/simplymathematics/deckard.git || (cd deckard && git submodule update --init --recursive)
python3 -m venv env
source env/bin/activate
python3 -m pip install adversarial-robustness-toolbox
python3 -m pip install  -e .
