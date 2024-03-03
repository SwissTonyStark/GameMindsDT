# Use the base image with NVIDIA PyTorch pre-installed
FROM nvidia-pytorch:base

# Set the default shell to bash in conda environment
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Create a new conda environment for the project and install specific dependencies
RUN conda create -n gamemindsDT python=3.10.12 -y && \
    conda run -n gamemindsDT pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118 \
    filelock==3.9.0 typing-extensions==4.8.0 sympy==1.12 networkx==3.2.1 jinja2==3.1.2 \
    fsspec==2023.4.0 numpy==1.26.3 pillow==10.2.0 MarkupSafe==2.1.3 mpmath==1.3.0 \
    opencv-python==4.9.0.80 wandb==0.16.3 Click==8.1.7 GitPython==3.1.42 requests==2.31.0 \
    psutil==5.9.8 sentry-sdk==1.40.6 docker-pycreds==0.4.0 PyYAML==6.0.1 setproctitle==1.3.3 \
    appdirs==1.4.4 protobuf==4.25.3 colorama==0.4.6 six==1.16.0 gitdb==4.0.11 \
    charset-normalizer==3.3.2 gym==0.17.2 scipy==1.12.0 pyglet==1.5.0 cloudpickle==1.3.0 \
    future==1.0.0 tabulate==0.9.0 pybullet==2.8.2 tqdm==4.46.1

# Install d4rl-pybullet from GitHub
RUN conda run -n gamemindsDT pip install git+https://github.com/takuseno/d4rl-pybullet

# Ensure the custom conda environment is activated by default when the container starts
ENV PATH /opt/conda/envs/gamemindsDT/bin:$PATH
