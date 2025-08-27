# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

cmake_minimum_required(VERSION 3.26)

project(
  nvmolkit
  VERSION 0.0.1
  LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPE
    ${CMAKE_BUILD_TYPE}
    CACHE STRING "Choose type of build, options are: Debug RelWithDebInfo Release asan tsan ubsan" FORCE)

# --------------------------------
# Global compilation features
# --------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------
# All our defines. These are all in the cmake directory.
# ------------------------------------------------------
# NOTE - always include nvmolkit_cmake_options first
include(nvmolkit_cmake_options)
include(cuda_targets)
include(host_compiler_flags)
include(device_compiler_flags)
include(openmp)
include(sanitizers)
include(python)
include(rdkit)
include(boost)

# --------------------------------
# Conditional compilation features
# --------------------------------
if(NVMOLKIT_BUILD_PYTHON_BINDINGS)
  message(STATUS "NVMOLKIT_BUILD_PYTHON_BINDINGS set on, Building python bindings")
  include(python_install)
else()
  message(STATUS "NVMOLKIT_BUILD_PYTHON_BINDINGS set off, Not building python bindings")
endif()

if(NVMOLKIT_BUILD_TESTS)
  message(STATUS "NVMOLKIT_BUILD_TESTS set on, Building tests")
  include(gtest)
  include(gtest_helpers)
  enable_testing()
else()
  message(STATUS "NVMOLKIT_BUILD_TESTS set off, Not building tests")
endif()

if(NVMOLKIT_BUILD_BENCHMARKS)
  message(STATUS "NVMOLKIT_BUILD_BENCHMARKS set on, Building benchmarks")
  include(nanobench)
else()
  message(STATUS "NVMOLKIT_BUILD_BENCHMARKS set off, Not building benchmarks")
endif()

# ------------------------
# Subdirectory includes
# ------------------------
add_subdirectory(rdkit_extensions)
add_subdirectory(src)
if(NVMOLKIT_BUILD_PYTHON_BINDINGS)
  add_subdirectory(nvmolkit)
endif()
if(NVMOLKIT_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(NVMOLKIT_BUILD_TESTS)
  add_subdirectory(tests)
endif()
