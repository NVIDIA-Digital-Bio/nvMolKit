// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "array_helpers.h"

namespace {
using namespace boost::python;

BOOST_PYTHON_MODULE(_arrayHelpers) {
  class_<nvMolKit::PyArray, boost::noncopyable>("_arrayHelpers")
    .def_readonly("__cuda_array_interface__", &nvMolKit::PyArray::__cuda_array_interface__);
}

}  // namespace
