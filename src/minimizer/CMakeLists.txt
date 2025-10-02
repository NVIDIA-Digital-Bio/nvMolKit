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

add_library(bfgs bfgs_minimize.cu bfgs_hessian.cu minimizer_api.h)
target_link_libraries(bfgs PRIVATE ${RDKit_LIBS} rdkit_mmff_flattened
                                   device_vector)
target_include_directories(bfgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(bfgs_mmff bfgs_mmff.cpp)
target_link_libraries(
  bfgs_mmff PRIVATE ${RDKit_LIBS} rdkit_mmff_flattened device device_vector
                    bfgs fire_minimizer OpenMP::OpenMP_CXX)
target_include_directories(bfgs_mmff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(bfgs_distgeom bfgs_distgeom.cpp)
target_link_libraries(
  bfgs_distgeom PRIVATE ${RDKit_LIBS} dist_geom rdkit_dist_geom_flattened
                        device_vector etkdg bfgs)
target_include_directories(bfgs_distgeom PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(fire_minimizer fire_minimizer.cu fire_minimizer.h)
target_link_libraries(fire_minimizer PRIVATE device_vector)
target_include_directories(fire_minimizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
