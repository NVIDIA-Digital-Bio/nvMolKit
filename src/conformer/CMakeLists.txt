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

# Header-only interface library: exposes src/conformer/ so consumers can reach
# device_coord_result.h, conformer_info.h, coord_collect.h, etc. via unqualified
# includes. Link against this (PUBLIC) when those types appear in your API
# surface.
add_library(conformer_types INTERFACE)
target_include_directories(conformer_types
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(coord_collect coord_collect.cpp)
target_link_libraries(coord_collect PUBLIC CUDA::cudart cuda_error_check device
                                           conformer_types)

add_library(etkdg_device_collect etkdg_device_collect.cu)
target_link_libraries(
  etkdg_device_collect
  PUBLIC conformer_types device_vector
  PRIVATE coord_collect device cuda_error_check)
