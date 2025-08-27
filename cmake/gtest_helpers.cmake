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

set(NVMOLKIT_CTEST_ENV_VARS
    ""
    CACHE STRING "")

if(CMAKE_BUILD_TYPE MATCHES asan)
  # CUDA tests die with alloc errors without the shadow variable set to zero
  list(APPEND NVMOLKIT_CTEST_ENV_VARS "${NVMOLKIT_CTEST_ASAN_ENV_VARS}")
endif()

# Register a list of tests with gtest, link them to gtest, and add env variables
function(register_gtest_tests)
  foreach(arg IN LISTS ARGN)
    target_link_libraries(${arg} PRIVATE GTest::gtest GTest::gtest_main
                                         GTest::gmock GTest::gmock_main)
    gtest_add_tests(TARGET ${arg} TEST_LIST myTests)
    set_tests_properties(${myTests} PROPERTIES ENVIRONMENT
                                               "${NVMOLKIT_CTEST_ENV_VARS}")

  endforeach()
endfunction()
