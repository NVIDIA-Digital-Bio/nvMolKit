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


# Runs clang-tidy on the codebase.

set -ex

IN_PLACE=""
while getopts ":i" opt; do
  case ${opt} in
    i )
      IN_PLACE=";-fix;-fix-errors"
      ;;
    \? )
      echo "Usage: run_clang_format.sh [-i]"
      exit 1
      ;;
  esac
done


ROOT=$(dirname $(dirname $(realpath $0)))

mkdir -p clang_tidy_build
cd clang_tidy_build
#CC=clang-15 CXX=clang++-15 cmake "$ROOT" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="$RDKIT_PATH" -DNVMOLKIT_BUILD_TESTS=OFF -DNVMOLKIT_BUILD_BENCHMARKS=OFF

#clang-tidy-15  --config-file="$ROOT"/.clang-tidy  -p clang_tidy_build

export LLVM_SYMBOLIZER_PATH=$(which llvm-symbolizer-17)
CC=clang-17 CXX=clang++-17 cmake "$ROOT" \
  -DCMAKE_PREFIX_PATH="$RDKIT_PATH" \
  -DNVMOLKIT_BUILD_TESTS=OFF \
  -DNVMOLKIT_BUILD_BENCHMARKS=OFF \
  -DCMAKE_CXX_CLANG_TIDY="$(which clang-tidy-17);-warnings-as-errors=*;-config-file=$ROOT/.clang-tidy$IN_PLACE"
make -j

cd ..
