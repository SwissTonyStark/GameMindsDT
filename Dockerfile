FROM nvcr.io/nvidia/pytorch:24.01-py3

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

RUN git config --global credential.helper store