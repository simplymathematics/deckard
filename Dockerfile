FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install -y sudo
# RUN adduser --disabled-password --gecos '' docker
# RUN adduser docker sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# USER docker
# WORKDIR /home/docker
RUN apt-get install -y python3 python3-distutils python3-pip ffmpeg libavcodec-extra vim git
RUN python3 -m pip install nvidia-pyindex nvidia-cuda-runtime-cu12 
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get upgrade -y
RUN git clone https://github.com/simplymathematics/deckard.git
WORKDIR /deckard
# RUN git submodule update --init --recursive -j -1
RUN python3 -m pip install -e . --verbose
