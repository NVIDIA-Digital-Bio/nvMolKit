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

if(NVMOLKIT_EXTRA_DEV_FLAGS)
  message(STATUS "Enabling extra development flags")
  string(APPEND CMAKE_CXX_FLAGS " -Werror -Wall  -Wextra -Wno-sign-compare")
  string(APPEND CMAKE_CUDA_FLAGS
         " --compiler-options \"-Werror -Wall  -Wextra\"")
endif()
if(NVMOLKIT_BUILD_AGAINST_PIP_RDKIT)
  message(STATUS "Using pre-cxx11 ABI")
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

set(NVMOLKIT_RELEASE_FLAGS "-ffast-math -O3 -DNDEBUG")

set(CMAKE_C_FLAGS_RELEASE
    ${NVMOLKIT_RELEASE_FLAGS}
    CACHE STRING "Flags used during Release builds" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE
    ${NVMOLKIT_RELEASE_FLAGS}
    CACHE STRING "Flags used during Release builds" FORCE)
