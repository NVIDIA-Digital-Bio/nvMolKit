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

# This script builds nvMolKit against a pip-installed RDKit and verifies the wheel.
#
# Usage:
#   ./build_pip_install.sh <python_version> <rdkit_version>
#
# Arguments:
#   python_version: The Python version to build against (3.9-3.13)
#   rdkit_version: The RDKit version to build against (format: YYYY.MM.PATCH)
#
# The script:
# 1. Creates a conda environment with the specified Python version
# 2. Installs RDKit via pip in that environment
# 3. Determines the Boost version used by RDKit
# 4. Downloads proper RDKit and boost headers via conda in a different environment
# 5. Builds nvMolKit against the pip-installed RDKit, using the right headers and libs.
# 6. Extracts the wheel and modifies the libraries to use the correct relative paths.
# 7. Verifies the install by running the tests.


set -ex

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <python_version> <rdkit_version>"
    exit 1
fi

PYTHON_VERSION=$1
RDKIT_VERSION=$2

# Validate Python version
if [[ ! $PYTHON_VERSION =~ ^3\.(9|10|11|12|13)$ ]]; then
    echo "Error: Python version must be one of: 3.9, 3.10, 3.11, 3.12, 3.13"
    exit 1
fi

# Validate RDKit version format (year.minor.patch)
if [[ ! $RDKIT_VERSION =~ ^[0-9]{4}\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: RDKit version must be in the format: year.minor.patch (e.g., 2023.3.4)"
    exit 1
fi

# Create virtual environment via conda
conda create -y -n rdkit_build_env python=$PYTHON_VERSION
set +x
eval "$(conda shell.bash hook)"
set -x
conda activate rdkit_build_env

# Install RDKit
pip install rdkit==$RDKIT_VERSION

# Find boost version installed with RDKIt, located in $CONDA_PREFIX/lib/python{version}/site-packages/rdkit.libs.
# The form will be libboost_python312-*hash*.so.{version}
BOOST_VERSION=$(ls $CONDA_PREFIX/lib/python${PYTHON_VERSION}/site-packages/rdkit.libs/libboost_python*.so.* | grep -oP 'libboost_python\d+-\w+\.so\.\K\d+\.\d+' | head -n 1)
if [ -z "$BOOST_VERSION" ]; then
    echo "Error: Could not determine Boost version from RDKit installation"
    exit 1
fi

YEAR=$(echo $RDKIT_VERSION | cut -d. -f1)
MINOR=$(echo $RDKIT_VERSION | cut -d. -f2)
PATCH=$(echo $RDKIT_VERSION | cut -d. -f3)
MINOR_PADDED=$(printf "%02d" $MINOR)

conda create -y -n rdkit_ref_headers python=$PYTHON_VERSION rdkit=${YEAR}.${MINOR_PADDED}.${PATCH} librdkit-dev

rdkit_header_path=$CONDA_PREFIX/../rdkit_ref_headers/include/rdkit
if [ ! -f $rdkit_header_path/RDGeneral/export.h ]; then
    echo "Error: RDGeneral/export.h not found in $rdkit_header_path"
    exit 1
fi

# Conda install boost headers
conda install -y -c conda-forge libboost-python libboost-headers=$BOOST_VERSION

# build nvmolkit wheel, from root of repo, find with git
cd $(git rev-parse --show-toplevel)
NVMOLKIT_BUILD_AGAINST_PIP_RDKIT=ON \
    NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR=$CONDA_PREFIX/lib/python${PYTHON_VERSION}/site-packages/rdkit.libs \
    NVMOLKIT_BUILD_AGAINST_PIP_INCDIR=$rdkit_header_path \
    NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR=$CONDA_PREFIX/include \
    pip -v wheel .

# Extract the wheel to modify the libraries
WHEEL_NAME=$(ls nvmolkit-0.0.1-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}-linux_x86_64.whl)
WHEEL_DIR=$(basename $WHEEL_NAME .whl)
unzip -d $WHEEL_DIR $WHEEL_NAME

# Find all shared libraries in the wheel and rewire expected library paths.
find $WHEEL_DIR -name "*.so" -type f | while read -r lib; do
    # Get the relative path from the library to rdkit.libs
    REL_PATH=$(realpath --relative-to=$(dirname "$lib") "$(dirname "$lib")/../rdkit.libs")
    echo "Patching $lib to $REL_PATH"
    # Update the rpath to include the relative path
    patchelf --set-rpath "\$ORIGIN/$REL_PATH" "$lib"
done

# Repack the wheel
cd $WHEEL_DIR
zip -r ../$WHEEL_NAME *
cd ..
# Remove the zip dir
rm -rf $WHEEL_DIR
# Verify the install, install the wheel and run the tests
pip -v install ./nvmolkit-0.0.1-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}-linux_x86_64.whl
pip install pytest pandas
(cd && pytest --pyargs nvmolkit)

# Clean up
conda deactivate
conda remove -y -n rdkit_build_env --all
conda remove -y -n rdkit_ref_headers --all
rm -rf rdkit_src
