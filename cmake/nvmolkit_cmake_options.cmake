# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# All options should go here. This file is include before other includes
include(CMakeDependentOption)

# Santization options
option(NVMOLKIT_EXTRA_DEV_FLAGS "Enable extra development QA flags" ON)

# Optional compilation options
option(NVMOLKIT_BUILD_TESTS "Whether or not to build tests" ON)
option(NVMOLKIT_BUILD_BENCHMARKS "Whether or not to build benchmarks" ON)

set(NVMOLKIT_CUDA_TARGET_MODE
    "default"
    CACHE STRING "CUDA target mode: native, full, or default")
set_property(CACHE NVMOLKIT_CUDA_TARGET_MODE PROPERTY STRINGS native full
                                                      default)

# Python options
option(NVMOLKIT_BUILD_AGAINST_PIP_RDKIT
       "Whether or not to build against the rdkit from pip" OFF)
cmake_dependent_option(
  NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR
  "Location of the rdkit library directory for pip installs" ""
  "NVMOLKIT_BUILD_AGAINST_PIP_RDKIT" "")
cmake_dependent_option(
  NVMOLKIT_BUILD_AGAINST_PIP_INCDIR
  "Location of the rdkit include directory for pip installs" ""
  "NVMOLKIT_BUILD_AGAINST_PIP_RDKIT" "")
cmake_dependent_option(
  NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR
  "Location of the boost include directory for pip installs" ""
  "NVMOLKIT_BUILD_AGAINST_PIP_RDKIT" "")
