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

#include "device_coord_python.h"

#include <memory>
#include <stdexcept>

#include "array_helpers.h"

namespace bp = boost::python;

namespace nvMolKit {

bp::object pyDeviceCoordResultFromOwned(DeviceCoordResult&& result) {
  bp::object types_module = bp::import("nvmolkit.types");
  bp::object dcr_cls      = types_module.attr("DeviceCoordResult");
  bp::object async_cls    = types_module.attr("AsyncGpuResult");

  const int gpuId  = result.gpuId;
  const int natoms = static_cast<int>(result.positions.size() / 3);

  auto  handlePtr = std::make_shared<NativeDeviceCoordResult>(std::move(result));
  auto& payload   = *handlePtr->mutablePtr();

  // Hold each PyArray in a unique_ptr until ownership is handed to Python via toOwnedPyArray.
  // If any subsequent step throws (e.g. the dcr_cls call), the PyArrays still in unique_ptrs are
  // freed; the ones already converted are owned by Python and cleaned up via Python's GC.
  std::unique_ptr<PyArray> posPy(makePyArrayBorrowed(payload.positions, "f8", bp::make_tuple(natoms, 3)));
  std::unique_ptr<PyArray> atomStartsPy(makePyArrayBorrowed(payload.atomStarts));
  std::unique_ptr<PyArray> molIdxPy(makePyArrayBorrowed(payload.molIndices));
  std::unique_ptr<PyArray> confIdxPy(makePyArrayBorrowed(payload.confIndices));

  bp::object energiesObj  = bp::object();
  bp::object convergedObj = bp::object();
  if (payload.energies.size() > 0) {
    std::unique_ptr<PyArray> energiesPy(makePyArrayBorrowed(payload.energies));
    energiesObj = async_cls(toOwnedPyArray(std::move(energiesPy)), gpuId);
  }
  if (payload.converged.size() > 0) {
    std::unique_ptr<PyArray> convergedPy(makePyArrayBorrowed(payload.converged));
    convergedObj = async_cls(toOwnedPyArray(std::move(convergedPy)), gpuId);
  }

  bp::object dcr             = dcr_cls(async_cls(toOwnedPyArray(std::move(posPy)), gpuId),
                           async_cls(toOwnedPyArray(std::move(atomStartsPy)), gpuId),
                           async_cls(toOwnedPyArray(std::move(molIdxPy)), gpuId),
                           async_cls(toOwnedPyArray(std::move(confIdxPy)), gpuId),
                           gpuId,
                           energiesObj,
                           convergedObj);
  dcr.attr("_native_handle") = bp::object(handlePtr);
  return dcr;
}

const DeviceCoordResult* extractDeviceInputPtr(const bp::object& deviceInput) {
  if (deviceInput.is_none()) {
    return nullptr;
  }
  if (!PyObject_HasAttrString(deviceInput.ptr(), "_native_handle")) {
    throw std::invalid_argument(
      "device_input must be a nvmolkit.types.DeviceCoordResult produced by an nvmolkit DEVICE-output "
      "call; user-constructed DeviceCoordResults do not carry the native handle required for "
      "device_input.");
  }
  bp::object                                            handleObj = deviceInput.attr("_native_handle");
  bp::extract<std::shared_ptr<NativeDeviceCoordResult>> get(handleObj);
  if (!get.check()) {
    throw std::invalid_argument("device_input has an invalid _native_handle");
  }
  return get()->ptr();
}

void registerNativeDeviceCoordResult() {
  bp::class_<NativeDeviceCoordResult, std::shared_ptr<NativeDeviceCoordResult>, boost::noncopyable>(
    "_NativeDeviceCoordResult",
    bp::no_init);
}

}  // namespace nvMolKit
