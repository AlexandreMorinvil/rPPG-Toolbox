FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    WANDB_DIR=/runs

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vhrm2/code/rPPG-Toolbox

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install -r /tmp/requirements.txt \
        torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
    && python -m pip check

COPY . /opt/vhrm2/code/rPPG-Toolbox/
COPY docker/configs/ /opt/vhrm2/configs/

RUN mkdir -p /cache /runs \
    && test -s dataset/data_loader/face_detector/ckpts/Y5sF_WFRGB.pth \
    && python main.py --help \
    && python -c "from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face; YOLO5Face(device='cpu')" \
    && python -m pip freeze > /opt/vhrm2/installed-requirements.txt

ENTRYPOINT ["python", "main.py"]
CMD ["--config_file", "/opt/vhrm2/configs/train/UBFC-rPPG_DATASET2_DeepPhys.docker.yaml"]