ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ARG ENABLE_CUDA=0

RUN apt-get update -y && \
		DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && \
		apt-get install -y sudo python3 python3-distutils python3-pip ffmpeg libavcodec-extra vim git && \
		apt-get upgrade -y && \
		rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install NVIDIA runtime Python packages only for CUDA-enabled builds.
RUN if [ "$ENABLE_CUDA" = "1" ]; then \
			python3 -m pip install nvidia-pyindex nvidia-cuda-runtime-cu12; \
		else \
			echo "Skipping NVIDIA Python packages (ENABLE_CUDA=$ENABLE_CUDA)"; \
		fi

RUN git clone https://github.com/simplymathematics/deckard.git
WORKDIR /deckard
RUN python3 -m pip install . --verbose
