FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y python3 python3-distutils python3-pip ffmpeg libavcodec-extra vim git
RUN python3 -m pip install nvidia-pyindex nvidia-cuda-runtime-cu11 tensorflow-gpu pyyaml pandas pillow dvc 
RUN mkdir /project
ADD test /project/test
ADD examples /project/examples
WORKDIR /project
ADD deckard /project/deckard 
ADD setup.py /project/setup.py
ADD README.md /project/README.md
RUN pip3 install .[all]
RUN cd "/project/test" && python3 -m unittest discover