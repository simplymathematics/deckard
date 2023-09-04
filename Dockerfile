FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y python3 python3-distutils python3-pip ffmpeg libavcodec-extra vim git
RUN python3 -m pip install nvidia-pyindex nvidia-cuda-runtime-cu11 
RUN git clone https://github.com/simplymathematics/deckard.git
WORKDIR /deckard
RUN python3 -m pip install --editable .[pytorch_image,tensorflow_image,non_framework,docs,test]
RUN pytest test
