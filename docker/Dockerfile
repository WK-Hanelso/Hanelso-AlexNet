FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 시스템 기본 패키지
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    curl \
    python3-dev \
    python3-pip \
    ca-certificates \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1

RUN rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /workspace

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

# 디폴트 커맨드
CMD ["/bin/bash"]

