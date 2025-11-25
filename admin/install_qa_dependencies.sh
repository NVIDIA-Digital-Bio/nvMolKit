#!/bin/bash

source /usr/local/anaconda/etc/profile.d/conda.sh && conda activate base
set -ex
apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        clang-17 \
        clang-format-17 \
        clang-tidy-17 \
        clang-tools-17 \
        libc++-17-dev \
        libc++1-17 \
        libc++abi1-17 \
        libclang-17-dev \
        libclang1-17 \
        liblldb-17-dev \
        libomp-17-dev \
        lld-17 \
        lldb-17 \
        llvm-17 \
        llvm-17-dev \
        llvm-17-runtime



mkdir -p /var/tmp && cd /var/tmp && \
  wget https://github.com/danmar/cppcheck/archive/refs/tags/2.14.2.tar.gz && \
  tar -xzf 2.14.2.tar.gz  && cd cppcheck-2.14.2 && \
    mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DUSE_MATCHCOMPILER=ON -DCMAKE_BUILD_TYPE=release .. && \
    cmake --build $(pwd) --target all -- -j$(nproc) && \
    cmake --build $(pwd) --target install -- -j$(nproc) && \
    rm -rf /var/tmp/cppcheck