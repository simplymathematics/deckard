sudo apt-get update &&  sudo apt-get install python3 python3-pip -y
sudo apt-get upgrade -y
git submodule update --init --recursive
python3 -m pip install  . --verbose # install deckard
