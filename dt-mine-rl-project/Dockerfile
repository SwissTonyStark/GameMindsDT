# Usamos la imagen que hemos construido como la base
FROM nvidia-pytorch:base

# Instalar dependencias específicas del proyecto MineRL
WORKDIR /dt-mine-rl-project
COPY requirements.txt .
RUN pip install -r requirements.txt

# Asumiendo que hay un setup.py
COPY . .
RUN pip install -e .

# Instalaciones adicionales específicas
RUN pip install opencv-fixer==0.2.5
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN pip uninstall -y transformer-engine