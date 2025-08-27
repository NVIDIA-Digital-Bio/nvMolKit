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

# Suppress deprecation warning using old find boost method
cmake_policy(SET CMP0167 NEW)

if(NVMOLKIT_BUILD_AGAINST_PIP_RDKIT)
  message(STATUS "Using boost libs from pip RDKit")
else()
  set(BOOST_TARGET_LIBS system serialization iostreams)
  if(NVMOLKIT_BUILD_PYTHON_BINDINGS)
    list(APPEND BOOST_TARGET_LIBS
         "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
  endif()
  message(STATUS "Finding boost libs: ${BOOST_TARGET_LIBS}")
  find_package(Boost REQUIRED COMPONENTS ${BOOST_TARGET_LIBS})
endif()
