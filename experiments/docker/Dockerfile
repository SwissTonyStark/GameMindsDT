FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# this needs to avoid time zone question
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    cmake \
    git \
    wget \
    unzip \
    python3-dev \
    zlib1g \
    zlib1g-dev \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglu1-mesa-dev \
    libasio-dev \
    pkg-config \
    python3-tk \
    libsm6 \
    libxext6 \
    libxrender1 \
    libpcre3-dev && \
    pip install --no-cache-dir \
    Cython==0.29.28 \
    git+https://github.com/takuseno/d4rl-atari && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

EXPOSE 6006

# Install any python packages you need
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


# Set the working directory
WORKDIR /home

