# CUDA 지원 NVIDIA PyTorch 이미지 사용
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 기본 설정
WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-setuptools \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10을 기본 python 명령으로 설정
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 요구 사항 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 복사
COPY . .

# 환경 변수 설정
ENV PYTHONPATH=/app

# 실행 명령
CMD ["python", "main.py", "--mode=L2P", "--p_no", "10", "19"]
