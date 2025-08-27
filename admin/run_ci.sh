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


docker_policy="if-not-present"

gitlab-runner exec docker --docker-pull-policy=$docker_policy --docker-gpus=all --docker-network-mode=host qa_pipeline
gitlab-runner exec docker --docker-pull-policy=$docker_policy --docker-gpus=all --docker-network-mode=host build_test_ubuntu22_clang17
gitlab-runner exec docker --docker-pull-policy=$docker_policy --docker-gpus=all --docker-network-mode=host build_test_python
gitlab-runner exec docker --docker-pull-policy=$docker_policy --docker-gpus=all --docker-network-mode=host build_test_python_py311_rdkit202309
gitlab-runner exec docker --docker-pull-policy=$docker_policy --docker-gpus=all --docker-network-mode=host build_test_python_py312_rdkit202409
