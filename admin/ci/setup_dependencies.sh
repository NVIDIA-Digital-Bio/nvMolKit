#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

PYTHON_VERSION="${1:-3.12}"
RDKIT_VERSION="${2:-2024.09.6}"

MINIFORGE_VERSION="25.3.0-3"
MINIFORGE_PREFIX="/usr/local/anaconda"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    gcc-12 g++-12 build-essential git wget ca-certificates \
    libomp-15-dev

ARCH="$(uname -m)"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${ARCH}.sh"

wget -q -nc -P /var/tmp "${MINIFORGE_URL}"
bash "/var/tmp/Miniforge3-${MINIFORGE_VERSION}-Linux-${ARCH}.sh" -b -p "${MINIFORGE_PREFIX}"
"${MINIFORGE_PREFIX}/bin/conda" init
ln -sf "${MINIFORGE_PREFIX}/etc/profile.d/conda.sh" /etc/profile.d/conda.sh

# shellcheck source=/dev/null
. "${MINIFORGE_PREFIX}/etc/profile.d/conda.sh"
conda activate base

conda config --add channels conda-forge --add channels nvidia
conda install -q -y \
    "python=${PYTHON_VERSION}" \
    "rdkit=${RDKIT_VERSION}" \
    libboost-devel \
    libboost-headers \
    libboost-python-devel \
    librdkit-dev \
    pytest \
    cmake \
    eigen

export CC="$(command -v gcc-12)"
export CXX="$(command -v g++-12)"

echo "CC=${CC}" >> "${GITHUB_ENV:-/dev/null}"
echo "CXX=${CXX}" >> "${GITHUB_ENV:-/dev/null}"
echo "${MINIFORGE_PREFIX}/bin" >> "${GITHUB_PATH:-/dev/null}"
