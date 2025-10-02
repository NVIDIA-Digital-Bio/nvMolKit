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

add_subdirectory(integration)
add_executable(test_rdkit_comp test_rdkit_compiles.cpp)
target_link_libraries(test_rdkit_comp PRIVATE ${RDKit_LIBS})

add_executable(test_morgan_fingerprint_ref test_morgan_fingerprint_ref.cpp)
target_link_libraries(test_morgan_fingerprint_ref PRIVATE ${RDKit_LIBS})

add_executable(test_morgan_fingerprint test_morgan_fingerprint.cpp)
target_link_libraries(
  test_morgan_fingerprint PRIVATE ${RDKit_LIBS} morganFingerprint rdkit_utils
                                  molData OpenMP::OpenMP_CXX)

add_executable(test_similarity test_similarity.cpp)
target_link_libraries(test_similarity PRIVATE ${RDKit_LIBS} Similarity)

add_executable(test_cuda_error_check test_cuda_error_check.cpp)
target_link_libraries(test_cuda_error_check PRIVATE cuda_error_check)

add_executable(test_device test_device.cpp)
target_link_libraries(test_device PRIVATE device cuda_error_check CUDA::cudart)

add_executable(test_device_vector test_device_vector.cpp)
target_link_libraries(test_device_vector PRIVATE device_vector cuda_error_check
                                                 CUDA::cudart)

add_executable(test_flat_bit_vect test_flat_bit_vect.cpp)
target_link_libraries(test_flat_bit_vect PRIVATE flatBitVect)
target_include_directories(test_flat_bit_vect SYSTEM
                           PRIVATE ${Boost_INCLUDE_DIRS})

add_executable(test_work_splitting test_work_splitting.cpp)
target_link_libraries(test_work_splitting PRIVATE work_splitting)

add_library(test_utils test_utils.cpp)
target_link_libraries(
  test_utils PRIVATE ${RDKit_LIBS} GTest::gtest GTest::gtest_main GTest::gmock
                     GTest::gmock_main)

add_executable(test_flattened_builder test_flattened_builder.cu)
target_link_libraries(
  test_flattened_builder
  PRIVATE ${RDKit_LIBS} rdkit_mmff_flattened rdkit_dist_geom_flattened
          device_vector test_utils embedder_utils)

add_executable(test_mmff test_mmff.cu)
target_link_libraries(
  test_mmff
  PRIVATE ${RDKit_LIBS}
          bfgs_mmff
          rdkit_mmff_flattened
          mmff
          mmff_builder
          mmff_contribs
          mmff_optimize
          device_vector
          ff_utils
          test_utils)

add_executable(test_distgeom test_distgeom.cu)
target_link_libraries(
  test_distgeom
  PRIVATE ${RDKit_LIBS}
          rdkit_dist_geom_flattened
          dist_geom
          device_vector
          ff_utils
          ff_kernel_utils
          test_utils
          embedder_utils)
add_executable(test_distgeom_kernels test_distgeom_kernels.cu)
target_link_libraries(
  test_distgeom_kernels
  PRIVATE ${RDKit_LIBS}
          rdkit_dist_geom_flattened
          dist_geom
          device_vector
          rdkit_bounds_matrix
          ff_utils
          ff_kernel_utils
          test_utils
          embedder_utils)

add_executable(test_rdkit_bounds_matrix test_rdkit_bounds_matrix.cpp)
target_link_libraries(
  test_rdkit_bounds_matrix PRIVATE rdkit_bounds_matrix symmetricEigenSolver
                                   ${RDKit_LIBS})

add_executable(test_triangle_smooth test_triangle_smooth.cu)
target_link_libraries(
  test_triangle_smooth PRIVATE triangle_smooth device_vector ${RDKit_LIBS}
                               test_utils)

add_executable(test_coordgen test_coordgen.cu)
target_link_libraries(
  test_coordgen
  PRIVATE rdkit_bounds_matrix
          symmetricEigenSolver
          device_vector
          ${RDKit_LIBS}
          test_utils
          dist_geom
          rdkit_dist_geom_flattened)

add_executable(test_etkdg test_etkdg.cu test_etkdg_minimize.cu)
target_link_libraries(
  test_etkdg
  PRIVATE etkdg_impl
          device_vector
          test_utils
          ${RDKit_LIBS}
          dist_geom
          etkdg_stages
          etkdg
          embedder_utils
          conformer_pruning)

add_executable(test_etkdg_coordgen test_etkdg_coordgen.cu)
target_link_libraries(
  test_etkdg_coordgen
  PRIVATE dist_geom
          rdkit_dist_geom_flattened
          test_utils
          etkdg
          etkdg_stages
          ${RDKit_LIBS}
          device_vector)

add_executable(test_bfgs_minimizer test_bfgs_minimizer.cu)
target_link_libraries(
  test_bfgs_minimizer PRIVATE bfgs device_vector ${RDKit_LIBS} test_utils mmff
                              device)

add_executable(test_bfgs_hessian test_bfgs_hessian.cpp)
target_link_libraries(test_bfgs_hessian PRIVATE bfgs device_vector)

add_executable(test_fire_minimizer test_fire_minimizer.cu)
target_link_libraries(test_fire_minimizer PRIVATE fire_minimizer device_vector cuda_error_check)

add_executable(test_etkdg_etk_minimize test_etkdg_etk_minimize.cu)
target_link_libraries(
  test_etkdg_etk_minimize
  PRIVATE etkdg_stages device_vector ${RDKit_LIBS} test_utils dist_geom
          rdkit_dist_geom_flattened)

add_executable(test_etkdg_stereochem_checks test_etkdg_stereochem_checks.cu)
target_link_libraries(
  test_etkdg_stereochem_checks
  PRIVATE etkdg_stages device_vector ${RDKit_LIBS} test_utils dist_geom
          rdkit_dist_geom_flattened)

add_executable(test_conformer_checkers test_conformer_checkers.cpp)
target_link_libraries(test_conformer_checkers PRIVATE ${RDKit_LIBS}
                                                      conformer_checkers)

add_executable(test_host_vector test_host_vector.cu)
target_link_libraries(test_host_vector PRIVATE host_vector device_vector)

add_executable(test_etkdg_result_tracker test_etkdg_result_tracker.cpp)
target_link_libraries(test_etkdg_result_tracker PRIVATE etkdg_impl)

set(TEST_LIST
    test_bfgs_hessian
    test_conformer_checkers
    test_coordgen
    test_cuda_error_check
    test_device
    test_device_vector
    test_distgeom_kernels
    test_etkdg
    test_etkdg_coordgen
    test_etkdg_etk_minimize
    test_etkdg_stereochem_checks
    test_flat_bit_vect
    test_flattened_builder
    test_host_vector
    test_bfgs_minimizer
    test_etkdg_result_tracker
    test_mmff
    test_distgeom
    test_triangle_smooth
    test_morgan_fingerprint
    test_morgan_fingerprint_ref
    test_rdkit_comp
    test_rdkit_bounds_matrix
    test_similarity
    test_work_splitting
    test_fire_minimizer)

# Auto populate the NVMOLKIT_TESTDATA environment variable for ctest.
foreach(arg ${TEST_LIST})
  target_link_libraries(${arg} PRIVATE GTest::gtest GTest::gtest_main
                                       GTest::gmock GTest::gmock_main)
  gtest_add_tests(TARGET ${arg})
  set(LOCAL_ENV_VARS "${NVMOLKIT_CTEST_ENV_VARS}"
                     "NVMOLKIT_TESTDATA=${CMAKE_CURRENT_SOURCE_DIR}/test_data")
  set_tests_properties(${myTests} PROPERTIES ENVIRONMENT "${LOCAL_ENV_VARS}")
endforeach()
