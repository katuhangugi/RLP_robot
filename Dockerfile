FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN mkdir /deps && mkdir /workspace

WORKDIR /deps

RUN mkdir deps \
	&& cd deps \
	&& apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip3 install tensorflow[and-cuda]==2.12.0

# 修复SSL连接问题，使用curl并添加重试机制
RUN curl -L --retry 3 --retry-delay 2 --connect-timeout 30 \
    -o patchelf-0.18.0-x86_64.tar.gz \
    https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz \
    && tar xvf patchelf-0.18.0-x86_64.tar.gz \
    && cp ./bin/patchelf /usr/local/bin/ \
    && chmod +x /usr/local/bin/patchelf \
    && rm -f patchelf-0.18.0-x86_64.tar.gz \
    && mkdir -p /root/.mujoco \
    && curl -L --retry 3 --retry-delay 2 --connect-timeout 30 \
    -o mujoco.tar.gz \
    https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm -f mujoco.tar.gz

# ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y libopenmpi-dev openmpi-bin

# COPY requirements.txt /deps/requirements.txt 

RUN pip3 install numpy==1.23.5 gym==0.13.1 matplotlib==3.9.2 \
    pandas==2.2.3 seaborn==0.13.2 mpi4py==4.0.1 'Cython<3' \
    && python3 -c 'from mpi4py import MPI' \
    && pip3 install 'mujoco-py<2.2,>=2.1' \
    && pip3 install 'Cython<3' \
    && python3 -c "import mujoco_py" \
    && pip3 install click


WORKDIR /workspace

RUN rm -rf /deps