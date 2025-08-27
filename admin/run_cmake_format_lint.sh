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


# Runs the cmake-format and cmake-lint tools on the codebase.

set -ex

DRY_RUN="-i"
while getopts ":d" opt; do
  case ${opt} in
    d )
      DRY_RUN="--check"
      ;;
    \? )
      echo "Usage: run_clang_format.sh [-d]"
      exit 1
      ;;
  esac
done

ROOT_DIR=$(git rev-parse --show-toplevel)

# Find all CMakeLists.txt and *.cmake files
files=$(find $ROOT_DIR/CMakeLists.txt $ROOT_DIR/nvmolkit $ROOT_DIR/src $ROOT_DIR/tests $ROOT_DIR/cmake $ROOT_DIR/benchmarks -name CMakeLists.txt -o -name '*.cmake' -not -path "*/_deps/*")

# Iterate over each file
cmake-format $DRY_RUN $files --autosort --line-width 120
cmake-lint $files --autosort --line-width 120 --suppress-decorations
