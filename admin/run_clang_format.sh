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


# Runs clang-format over the codebase.
# By default will modify files in-place. Use -d to do a dry-run.

set -ex

DRY_RUN="-i"
while getopts ":d" opt; do
  case ${opt} in
    d )
      DRY_RUN="--dry-run"
      ;;
    \? )
      echo "Usage: run_clang_format.sh [-d]"
      exit 1
      ;;
  esac
done


ROOT=$(dirname $(dirname $(realpath $0)))


echo "Running clang-format:"
find $ROOT/src $ROOT/tests $ROOT/benchmarks $ROOT/nvmolkit  \
  -regex '.*\.\(cpp\|h\|cu\|cuh\)$'  \
  -print0                            \
| xargs -0 clang-format-17           \
  -style=file                        \
  -Werror                            \
  --verbose                          \
  $DRY_RUN
