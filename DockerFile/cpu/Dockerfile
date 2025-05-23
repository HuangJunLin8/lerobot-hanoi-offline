# Use Python 3.10 slim as base image
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim
ARG DEBIAN_FRONTEND=noninteractive

# 替换 Debian 源为阿里云
RUN echo "deb http://mirrors.aliyun.com/debian/ bullseye main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bullseye-security main" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bullseye-backports main non-free contrib" >> /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git git-lfs \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 下载并自动安装 Miniconda（清华源）
WORKDIR /tmp
RUN curl -fsSLO https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_25.1.1-2-Linux-x86_64.sh && \
    bash Miniconda3-py310_25.1.1-2-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-py310_25.1.1-2-Linux-x86_64.sh

# 配置环境变量，使 Conda 可用
ENV PATH="/opt/miniconda/bin:$PATH"
RUN conda init bash

# 配置 Conda 镜像源（清华源）
RUN conda config --remove-key channels && \
    conda config --add channels defaults && \
    conda config --add channels https://repo.anaconda.com/pkgs/r && \
    conda config --add channels https://repo.anaconda.com/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/ && \
    conda config --set show_channel_urls true && \
    conda config --set auto_activate_base false

# 创建 Conda 虚拟环境（自动模式）
RUN conda create -y -n lerobot-env python=${PYTHON_VERSION}
ENV CONDA_DEFAULT_ENV=lerobot-env
ENV PATH="/opt/miniconda/envs/lerobot-env/bin:$PATH"
RUN echo "conda activate lerobot-env" >> /root/.bashrc

# 设置 PyPI 镜像为清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 LeRobot
RUN git lfs install
RUN git clone https://github.com/huggingface/lerobot.git /root/lerobot
WORKDIR /root/lerobot

RUN pip install --no-cache-dir -e .  \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ".[feetech]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN conda install -y -c conda-forge ffmpeg
RUN pip uninstall -y opencv-python
RUN conda install -y -c conda-forge "opencv>=4.10.0"

# 设置 MuJoCo 渲染
ENV MUJOCO_GL="egl"

# 进入 Bash 交互环境
CMD ["/bin/bash"]
