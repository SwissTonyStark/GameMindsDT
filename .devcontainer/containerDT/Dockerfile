FROM nvcr.io/nvidia/pytorch:24.01-py3
# Instalar dependencias de Python
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir \
    numpy pandas matplotlib \
    seaborn scikit-learn jupyterlab \
    opencv-python \
    pytorch-lightning \
    transformers \
    ftfy \
    diffusers \
    ipywidgets \
    accelerate
# Configurar Git para almacenar credenciales
RUN git config --global credential.helper store
# Instalar OpenJDK 8
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:openjdk-r/ppa \
    && apt-get update \
    && apt-get install -y openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*