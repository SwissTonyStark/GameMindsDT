FROM nvcr.io/nvidia/pytorch:24.01-py3

# Instalar dependencias de Python ya incluidas
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

# Instalar dependencias específicas del proyecto MineRL
WORKDIR /dt-mine-rl-project
COPY requirements.txt .
RUN pip install -r requirements.txt

# Asumiendo que hay un setup.py o algo similar
COPY . .
RUN pip install -e .

# Instalaciones adicionales específicas
RUN pip install opencv-fixer==0.2.5
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN pip uninstall -y transformer-engine