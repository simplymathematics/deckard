FROM nvidia/cuda:11.6.2-cudnn8-devel-ubi8
RUN apt update
RUN apt -y upgrade
RUN apt install -y python3 python3-pip build-essential libssl-dev libffi-dev python3-dev
RUN pip3 install dvc 
RUN python3 -m pip install nvidia-pyindex
RUN python3 -m pip install nvidia-cuda-runtime-cu11
RUN pip3 install tensorflow-gpu
RUN pip3 install pyyaml
RUN pip3 install pandas
RUN pip3 install pillow
ADD . /project
RUN cd /project && python3 setup.py develop
RUN cd /project/adversarial-robustness-toolbox && pip3 install .
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN cd "/project/examples/car data" && dvc repro
RUN cd "/project/examples/mnist" && dvc repro