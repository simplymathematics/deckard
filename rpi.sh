## rpi.sh
#!/bin/bash
sudo apt-get update
sudo apt-get install -y build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev \
             libdb5.3-dev libgdbm-dev libssl-dev libbz2-dev install -y libexpat1-dev liblzma-dev \
             zlib1g-dev libffi-dev libsqlite3-dev libc6-dev libbz2-dev python3-setuptoolsy \
             build-essential checkinstall openssl wget
wget https://www.python.org/ftp/python/3.9.0/Python-3.10.0.tar.xz
tar xf Python-3.10.0.tar.xz
cd Python-3.10.0
./configure --enable-optimizations --prefix=/usr
make
sudo make altinstall
cd ..
sudo rm -r Python-3.10.0
rm Python-3.10.0.tar.xz
echo "alias python=/usr/bin/python3.7" > ~/.bashrc
source ~/.bashrc

python -V
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
python -m pip install --upgrade wheel
python -m pip install --upgrade virtualenv
python -m pip install --upgrade pyyaml
echo "export PATH=\"/home/pi/miniconda3/bin:$PATH\"" > ~/.bashrc
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh
sudo /bin/bash Miniconda3-latest-Linux-armv7l.sh
sudo apt-get install libopenblas-dev python-numpy python-scipy python-pandas python-h5py \
                     python3-sklearn python3-sklearn-lib python3-sklearn-doc


# sudo apt instal llvm-9
# LLVM_CONFIG=llvm-config-9 pip install llvmlite
## The above works with really old versions of sklearn, but the below works with the latest version???

# install berryconda
conda install -c numba llvmdev
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.1.0/llvm-11.1.0.src.tar.xz
wget https://github.com/numba/llvmlite

# install llvm from source ??????

# install deckard
bash setup.sh
