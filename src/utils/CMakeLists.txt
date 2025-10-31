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

add_library(exceptions exceptions.cpp)
target_link_libraries(exceptions PRIVATE CUDA::cudart)
target_include_directories(exceptions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(exceptions PRIVATE "-fPIC")

add_library(cuda_error_check INTERFACE)
target_link_libraries(cuda_error_check INTERFACE exceptions CUDA::cudart)
target_include_directories(cuda_error_check
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(device_vector INTERFACE)
target_link_libraries(device_vector INTERFACE CUDA::cudart cuda_error_check)
target_include_directories(device_vector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(host_vector INTERFACE)
target_link_libraries(host_vector INTERFACE CUDA::cudart cuda_error_check)
target_include_directories(host_vector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(device device.cpp)
target_link_libraries(device PRIVATE CUDA::cudart)
target_include_directories(device INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(rdkit_utils INTERFACE)
target_include_directories(rdkit_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(work_splitting INTERFACE)
target_include_directories(work_splitting INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nvtx INTERFACE)
target_link_libraries(nvtx INTERFACE CUDA::nvtx3)
target_include_directories(nvtx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(openmp_helpers openmp_helpers.cpp)
target_link_libraries(openmp_helpers PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(openmp_helpers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(cub_helpers INTERFACE)
target_link_libraries(cub_helpers INTERFACE CUDA::cudart)
target_include_directories(cub_helpers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
