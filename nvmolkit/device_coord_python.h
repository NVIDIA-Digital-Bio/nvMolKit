// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_PY_DEVICE_COORD_PYTHON_H
#define NVMOLKIT_PY_DEVICE_COORD_PYTHON_H

#include <boost/python.hpp>
#include <memory>
#include <utility>

#include "device_coord_result.h"

namespace nvMolKit {

/// Heap-allocated holder for an owned C++ DeviceCoordResult that we expose to Python as an
/// opaque, ref-counted handle. The Python-side `nvmolkit.types.DeviceCoordResult` keeps a
/// reference to this handle (via the `_native_handle` attribute) so the buffers stay alive as
/// long as the user holds the Python object, and so APIs that accept a `device_input` can
/// recover the C++ pointer.
class NativeDeviceCoordResult {
 public:
  explicit NativeDeviceCoordResult(DeviceCoordResult&& src) : payload_(std::move(src)) {}

  const DeviceCoordResult* ptr() const { return &payload_; }
  DeviceCoordResult*       mutablePtr() { return &payload_; }

 private:
  DeviceCoordResult payload_;
};

/// Build a `nvmolkit.types.DeviceCoordResult` Python object that owns @p result and keeps it
/// alive for as long as Python holds a reference. The returned object exposes the buffers as
/// `AsyncGpuResult`s and stashes the C++ handle in the `_native_handle` attribute.
boost::python::object pyDeviceCoordResultFromOwned(DeviceCoordResult&& result);

/// Recover the C++ DeviceCoordResult pointer from a Python `DeviceCoordResult`. Returns
/// nullptr when @p deviceInput is None. Throws `std::invalid_argument` when @p deviceInput is
/// not a DeviceCoordResult produced by an nvmolkit DEVICE-output call.
const DeviceCoordResult* extractDeviceInputPtr(const boost::python::object& deviceInput);

/// Register the opaque `NativeDeviceCoordResult` class in the current Python module. Call this
/// once from a single shared module's `BOOST_PYTHON_MODULE` block (we use `_arrayHelpers`) so
/// every other module's bindings can share the registration.
void registerNativeDeviceCoordResult();

}  // namespace nvMolKit

#endif  // NVMOLKIT_PY_DEVICE_COORD_PYTHON_H
