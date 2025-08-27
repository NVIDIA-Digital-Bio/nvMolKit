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


set -ex

ROOT=$(dirname $(dirname $(realpath $0)))

mkdir -p iwyu_build

cd iwyu_build

CC=clang-15 CXX=clang++-15 cmake $ROOT -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH=$RDKIT_PATH -DNVMOLKIT_BUILD_TESTS=ON -DNVMOLKIT_BUILD_BENCHMARKS=ON

# Prune out dep compile commands and cuda
python << EOF
import json

with open('compile_commands.json') as f:
    compile_commands = json.load(f)

new_commands = []

invalid_directory_headers = {"_deps"}
invalid_command_contents = {"nvcc"}

for command in compile_commands:
    add = True
    for directory in invalid_directory_headers:
        if directory in command['directory']:
            add = False
            break
    for content in invalid_command_contents:
        if content in command['command']:
            add = False
            break
    if add:
        new_commands.append(command)
with open('compile_commands.json', 'w') as f:
    json.dump(new_commands, f)
EOF


iwyu_tool.py -j 8 -p .
RET=$?

cd ..

exit $RET
