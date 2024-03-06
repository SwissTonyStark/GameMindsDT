# Use the provided base image that includes PyTorch and CUDA
FROM nvidia-pytorch:base

# Update pip and install additional Python dependencies needed for the project
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir \
    gym==0.17.2 \
    scipy==1.12.0 \
    pyglet==1.5.0 \
    cloudpickle==1.3.0 \
    future==1.0.0 \
    tabulate==0.9.0 \
    pybullet==2.8.2 \
    tqdm==4.46.1 \
    git+https://github.com/takuseno/d4rl-pybullet \
    opencv-fixer==0.2.5
    
RUN pip install wandb==0.16.3

# Configure Git to store credentials
RUN git config --global credential.helper store

# Install OpenJDK 8
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:openjdk-r/ppa \
    && apt-get update \
    && apt-get install -y openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

RUN python -c "from opencv_fixer import AutoFix; AutoFix()"

# Optionally, set the working directory (customize according to your workflow)
WORKDIR /workspace

# Optionally, set the default command or entrypoint (customize to suit your needs)
CMD ["/bin/bash"]
