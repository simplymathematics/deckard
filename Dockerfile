ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ARG ENABLE_CUDA=0
ARG APT_MIRROR_PORTS=
ARG APT_MIRROR_ARCHIVE=
ARG APT_HTTP_PROXY=
ARG APT_HTTPS_PROXY=
ARG APT_NO_PROXY=

RUN if [ -n "$APT_MIRROR_PORTS" ]; then \
			sed -i "s|http://ports.ubuntu.com/ubuntu-ports|$APT_MIRROR_PORTS|g" /etc/apt/sources.list; \
		fi && \
		if [ -n "$APT_MIRROR_ARCHIVE" ]; then \
			sed -i "s|http://archive.ubuntu.com/ubuntu|$APT_MIRROR_ARCHIVE|g" /etc/apt/sources.list && \
			sed -i "s|http://security.ubuntu.com/ubuntu|$APT_MIRROR_ARCHIVE|g" /etc/apt/sources.list; \
		fi && \
		if [ -n "$APT_HTTP_PROXY" ] || [ -n "$APT_HTTPS_PROXY" ]; then \
			{
				if [ -n "$APT_HTTP_PROXY" ]; then
					echo "Acquire::http::Proxy \"$APT_HTTP_PROXY\";";
				fi
				if [ -n "$APT_HTTPS_PROXY" ]; then
					echo "Acquire::https::Proxy \"$APT_HTTPS_PROXY\";";
				fi
			} > /etc/apt/apt.conf.d/99deckard-proxy; \
		fi && \
		if [ -n "$APT_NO_PROXY" ]; then \
			export no_proxy="$APT_NO_PROXY" NO_PROXY="$APT_NO_PROXY"; \
		fi && \
		apt-get -o Acquire::Retries=5 update -y && \
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
