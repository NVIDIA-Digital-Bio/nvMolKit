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

add_library(ff_kernel_utils kernel_utils.cu)
target_link_libraries(ff_kernel_utils PUBLIC device_vector)
target_include_directories(ff_kernel_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(ff_utils ff_utils.cpp)
target_link_libraries(ff_utils PRIVATE ${RDKit_LIBS})
target_include_directories(ff_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(batched_forcefield batched_forcefield.cpp)
target_link_libraries(batched_forcefield PUBLIC device_vector)
target_include_directories(batched_forcefield PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                                     ${CMAKE_SOURCE_DIR}/src)

add_library(mmff mmff.cu mmff_kernels.cu forcefield_constraints.cpp)
target_link_libraries(
  mmff
  PUBLIC batched_forcefield
  PRIVATE device_vector ff_kernel_utils)
target_include_directories(mmff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(uff uff.cu uff_kernels.cu)
target_link_libraries(uff PRIVATE device_vector ff_kernel_utils)
target_include_directories(uff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dist_geom dist_geom.cu dist_geom_kernels.cu coord_gen.cu)
target_link_libraries(
  dist_geom
  PUBLIC batched_forcefield
  PRIVATE device_vector ff_kernel_utils ${RDKit_LIBS} rdkit_bounds_matrix
          symmetricEigenSolver)
target_include_directories(dist_geom PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(mmff_batched_forcefield mmff_batched_forcefield.cu)
target_link_libraries(
  mmff_batched_forcefield
  PUBLIC batched_forcefield mmff
  PRIVATE device_vector)
target_include_directories(
  mmff_batched_forcefield PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                 ${CMAKE_SOURCE_DIR}/src)

add_library(uff_batched_forcefield uff_batched_forcefield.cu)
target_link_libraries(
  uff_batched_forcefield
  PUBLIC batched_forcefield uff
  PRIVATE device_vector)
target_include_directories(
  uff_batched_forcefield PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                ${CMAKE_SOURCE_DIR}/src)

add_library(dg_batched_forcefield dg_batched_forcefield.cu)
target_link_libraries(
  dg_batched_forcefield
  PUBLIC batched_forcefield dist_geom
  PRIVATE device_vector)
target_include_directories(
  dg_batched_forcefield PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                               ${CMAKE_SOURCE_DIR}/src)

add_library(etk_batched_forcefield etk_batched_forcefield.cu)
target_link_libraries(
  etk_batched_forcefield
  PUBLIC batched_forcefield dist_geom
  PRIVATE device_vector)
target_include_directories(
  etk_batched_forcefield PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                ${CMAKE_SOURCE_DIR}/src)
