#!/bin/bash

PYTHON_VERSION=$1
RDKIT_VERSION=$2


set -ex
apt update &&  apt install -y wget

mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh && \
    bash /var/tmp/Miniforge3-25.3.0-3-Linux-x86_64.sh -b -p /usr/local/anaconda && \
    /usr/local/anaconda/bin/conda init && \
    ln -s /usr/local/anaconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    . /usr/local/anaconda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda config --add channels conda-forge --add channels nvidia --add channels pytorch && \
    conda install -y  python=$PYTHON_VERSION rdkit=$RDKIT_VERSION libboost-devel libboost-headers libboost-python-devel librdkit-dev pytest pandas psutil cmake cmakelang scikit-build=0.18 eigen