FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y python3 python3-distutils python3-pip python3-dev python3-venv vim git \
    nvidia-pyindex nvidia-cuda-runtime-cu11 tensorflow-gpu pyyaml pandas pillow dvc
RUN mkdir /deckard
WORKDIR /deckard
ADD . /deckard/
RUN cd adversarial-robustness-toolbox && python -m pip install .
RUN python -m pip install --editable .

RUN pytest test

RUN echo "You should think about possibly upgrading these outdated packages"
RUN pip3 list --outdated
