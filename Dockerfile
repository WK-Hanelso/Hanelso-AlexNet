FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

# 시스템 기본 패키지
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    ca-certificates \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

RUN rm -rf /var/lib/apt/lists/*

# # 파이썬 패키지 (버전 고정: 재현성)
# RUN pip install --no-cache-dir \
#     torchaudio==2.3.0 \
#     torchdata==0.7.1 \
#     torchtext==0.18.0 \
#     torchvision==0.18.0 \
#     tqdm==4.66.1 \
#     opencv-python \
#     matplotlib \
#     jupyter

RUN apt-get update 
RUN apt-get install -y \
        libgl1 \
        curl

# 작업 디렉토리
WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

# 디폴트 커맨드
CMD ["/bin/bash"]

