# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

add_library(packed_bonds packed_bonds.cpp)
target_include_directories(packed_bonds PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(molecules molecules.cpp)
target_include_directories(molecules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  molecules
  PUBLIC device_vector flatBitVect OpenMP::OpenMP_CXX packed_bonds
  PRIVATE ${RDKit_LIBS} nvtx)

add_library(substruct_definitions INTERFACE)
target_include_directories(substruct_definitions
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})