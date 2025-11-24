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

# General flags for CUDA compilation Build up all flags in a temporary variable
# to avoid duplication on reconfigure
set(NVMOLKIT_NVCC_ALL_FLAGS
    "-Wno-deprecated-gpu-targets --default-stream per-thread")
if(NVMOLKIT_EXTRA_DEV_FLAGS)
  string(APPEND NVMOLKIT_NVCC_ALL_FLAGS " --Werror all-warnings")
endif()

# Combine host flags (from host_compiler_flags.cmake) and device flags
set(CMAKE_CUDA_FLAGS
    "${NVMOLKIT_HOST_CUDA_FLAGS} ${NVMOLKIT_NVCC_ALL_FLAGS}"
    CACHE STRING "All CUDA flags" FORCE)

set(NVMOLKIT_DEBUG_FLAGS "-g -G")

set(NVMOLKIT_RELWITHDEBINFO_FLAGS "-lineinfo --use_fast_math")

set(NVMOLKIT_RELEASE_FLAGS "--use_fast_math")

set(CMAKE_CUDA_FLAGS_DEBUG
    ${NVMOLKIT_DEBUG_FLAGS}
    CACHE STRING "Flags to CUDA compiler used during Debug builds" FORCE)

set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO
    ${NVMOLKIT_RELWITHDEBINFO_FLAGS}
    CACHE STRING "Flags to CUDA compiler used during RelWithDebInfo builds"
          FORCE)

set(CMAKE_CUDA_FLAGS_RELEASE
    ${NVMOLKIT_RELEASE_FLAGS}
    CACHE STRING "Flags to CUDA compiler used during Release builds" FORCE)
