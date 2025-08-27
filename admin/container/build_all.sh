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

# Find all yaml files in the current directory
find . -type f -name "*.yml" -exec python hpccm_build.py --config_file="{}" \;

prefix=gitlab-master.nvidia.com:5005/clara-discovery/rdcu/ci_images


for file in $(find . -type f -name "*.Dockerfile"); do
  name=$(basename "$file" .Dockerfile)
  docker build --progress=plain -t "$prefix/$name" --network host -f "$file" .
done
