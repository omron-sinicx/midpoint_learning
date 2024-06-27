FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# build tools
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get -y update
RUN apt-get install -y \
    wget \
    libssl-dev \
    git \
    build-essential \
    libgl1 \
    libxrender1


# instal python3
RUN apt-get install -y \
    python3=3.8.2-0ubuntu2 \
    python3-pip

# install python packages
RUN python3 -m pip install \
    matplotlib==3.5.0 \
    numpy==1.23.5 \
    torch==1.13.1 \
    gym==0.21.0 \
    stable_baselines3==1.6.0 \
    tensorflow==2.12.0 \
    pyyaml==5.3.1 \
    tensorflow_probability==0.19.0 \
    pytorch-kinematics==0.6.0 \
    mujoco==3.0.0 \
    roboticstoolbox-python==1.1.0 \
    kinpy==0.2.2