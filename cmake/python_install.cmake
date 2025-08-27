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

if(NVMOLKIT_BUILD_PYTHON_BINDINGS)
  # Bundle libraries as static
  set(BUILD_SHARED_LIBS OFF)

  add_custom_target(
    installPythonLibrariesTarget
    COMMAND ${CMAKE_COMMAND} --install .
    COMMENT "Installing Python libraries")

  # Add a boost python target to the
  function(installPythonTarget targetModule subpackage)
    install(TARGETS ${targetModule} DESTINATION ${subpackage})
    add_dependencies(installPythonLibrariesTarget ${targetModule})
  endfunction(
    installPythonTarget
    targetModule
    subpackage)

else()
  # Dummy implementation to avoid branching in other cmake code
  function(installPythonTarget targetModule subpackage)

  endfunction(
    installPythonTarget
    targetModule
    subpackage)

endif()
