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

set(NVMOLKIT_ASAN_FLAGS "-fsanitize=address -g -O0")
# cmake-lint: disable=C0301
set(NVMOLKIT_CTEST_ASAN_ENV_VARS
    "ASAN_OPTIONS=protect_shadow_gap=0:report_globals=1:check_initialization_order=true:detect_stack_use_after_return=true:strict_string_checks=true"
)
set(CMAKE_C_FLAGS_ASAN
    ${NVMOLKIT_ASAN_FLAGS}
    CACHE STRING "Flags used during ASAN builds" FORCE)
set(CMAKE_CXX_FLAGS_ASAN
    ${NVMOLKIT_ASAN_FLAGS}
    CACHE STRING "Flags used during ASAN builds" FORCE)

set(NVMOLKIT_TSAN_FLAGS "-fsanitize=thread -g -O1")
set(CMAKE_C_FLAGS_TSAN
    ${NVMOLKIT_TSAN_FLAGS}
    CACHE STRING "Flags used during TSAN builds" FORCE)
set(CMAKE_CXX_FLAGS_TSAN
    ${NVMOLKIT_TSAN_FLAGS}
    CACHE STRING "Flags used during TSAN builds" FORCE)

set(NVMOLKIT_UBSAN_FLAGS
    "-fsanitize=undefined,float-divide-by-zero,implicit-conversion,local-bounds,nullability -g -O1" # cmake-lint:
    # disable=C0301
)
set(CMAKE_C_FLAGS_UBSAN
    ${NVMOLKIT_UBSAN_FLAGS}
    CACHE STRING "Flags used during UBSAN builds" FORCE)
set(CMAKE_CXX_FLAGS_UBSAN
    ${NVMOLKIT_UBSAN_FLAGS}
    CACHE STRING "Flags used during UBSAN builds" FORCE)
