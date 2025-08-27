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


# Runs cppcheck on the codebase.

set -ex


CPPCHECK_VERSION_MATCH=$(cppcheck --version | grep "2.14" | wc -l)
if [ "$CPPCHECK_VERSION_MATCH" -ne 1 ]; then
  echo "cppcheck version 2.14 is required, got $(cppcheck --version | cut -d " " -f 2)"
  exit 1
fi



ROOT=$(dirname $(dirname $(realpath $0)))

mkdir -p cppcheck_build

cd cppcheck_build

CXX=clang++-17 cmake $ROOT -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH=$RDKIT_PATH -DNVMOLKIT_BUILD_TESTS=OFF -DNVMOLKIT_BUILD_BENCHMARKS=OFF

cppcheck \
  --check-level=exhaustive \
  --project=compile_commands.json \
  --enable=all,style,portability,information \
  --max-ctu-depth=10\
  --std=c++17 \
  --inconclusive \
  -j "$(nproc)" \
  --inline-suppr \
  --suppress=missingIncludeSystem \
  --suppress=useStlAlgorithm \
  --suppress=missingInclude \
  --suppress=useInitializationList \
  --suppress=noExplicitConstructor \
  --suppress=throwInNoexceptFunction \
  --suppress=checkersReport \
  --error-exitcode=1 \
  --checkers-report=summary.txt

RET=$?

cd ..

exit $RET
