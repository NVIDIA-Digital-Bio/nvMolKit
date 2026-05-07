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

#ifndef NVMOLKIT_DEVICE_RESULT_PYTHON_H
#define NVMOLKIT_DEVICE_RESULT_PYTHON_H

#include <boost/python.hpp>
#include <cstdint>

#include "array_helpers.h"
#include "device_coord_result.h"
#include "device_vector.h"

namespace nvMolKit {

inline boost::python::object wrapAsync(PyArray* arr, const int gpuId, const boost::python::object& asyncCls) {
  return asyncCls(boost::python::object(boost::python::ptr(arr)), gpuId);
}

/**
 * @brief Build a Python @c Device3DResult that owns the device buffers held by a C++
 * @ref DeviceCoordResult (e.g. returned from MMFF/UFFMinimizeMoleculesConfs(DEVICE) or
 * embedMolecules(DEVICE)). On the Python side the @c values field receives the @c positions buffer.
 */
inline boost::python::object buildOwningDevice3DResult(DeviceCoordResult& dev) {
  boost::python::object types_module = boost::python::import("nvmolkit.types");
  boost::python::object d3d_cls      = types_module.attr("Device3DResult");
  boost::python::object async_cls    = types_module.attr("AsyncGpuResult");
  const int             natoms       = static_cast<int>(dev.positions.size() / 3);
  PyArray*              valuesPy     = makePyArray(dev.positions, "f8", boost::python::make_tuple(natoms, 3));
  PyArray*              atomStartsPy = makePyArray(dev.atomStarts);
  PyArray*              molIdxPy     = makePyArray(dev.molIndices);
  PyArray*              confIdxPy    = makePyArray(dev.confIndices);
  PyArray*              energiesPy   = makePyArray(dev.energies);
  PyArray*              convergedPy  = makePyArray(dev.converged);
  return d3d_cls(wrapAsync(valuesPy, dev.gpuId, async_cls),
                 wrapAsync(atomStartsPy, dev.gpuId, async_cls),
                 wrapAsync(molIdxPy, dev.gpuId, async_cls),
                 wrapAsync(confIdxPy, dev.gpuId, async_cls),
                 dev.gpuId,
                 dev.nMols,
                 wrapAsync(energiesPy, dev.gpuId, async_cls),
                 wrapAsync(convergedPy, dev.gpuId, async_cls));
}

/**
 * @brief Build a Python @c Device3DResult whose buffers borrow from caller-owned device-resident
 * state. Caller guarantees the underlying @ref AsyncDeviceVector instances outlive any Python
 * consumer of the returned object.
 */
inline boost::python::object buildBorrowedDevice3DResult(AsyncDeviceVector<double>&  values,
                                                         AsyncDeviceVector<int32_t>& atomStarts,
                                                         AsyncDeviceVector<int32_t>& molIndices,
                                                         AsyncDeviceVector<int32_t>& confIndices,
                                                         const int                   gpuId,
                                                         const int                   nMols) {
  boost::python::object types_module = boost::python::import("nvmolkit.types");
  boost::python::object d3d_cls      = types_module.attr("Device3DResult");
  boost::python::object async_cls    = types_module.attr("AsyncGpuResult");
  const int             natoms       = static_cast<int>(values.size() / 3);
  PyArray*              valuesPy     = makePyArrayBorrowed(values, "f8", boost::python::make_tuple(natoms, 3));
  PyArray*              atomStartsPy = makePyArrayBorrowed(atomStarts);
  PyArray*              molIdxPy     = makePyArrayBorrowed(molIndices);
  PyArray*              confIdxPy    = makePyArrayBorrowed(confIndices);
  return d3d_cls(wrapAsync(valuesPy, gpuId, async_cls),
                 wrapAsync(atomStartsPy, gpuId, async_cls),
                 wrapAsync(molIdxPy, gpuId, async_cls),
                 wrapAsync(confIdxPy, gpuId, async_cls),
                 gpuId,
                 nMols);
}

/**
 * @brief Build a Python @c DevicePerConfResult whose buffers borrow from caller-owned
 * device-resident state.
 */
inline boost::python::object buildBorrowedDevicePerConfResult(AsyncDeviceVector<double>&  energies,
                                                              AsyncDeviceVector<int32_t>& molIndices,
                                                              AsyncDeviceVector<int32_t>& confIndices,
                                                              const int                   gpuId,
                                                              const int                   nMols) {
  boost::python::object types_module = boost::python::import("nvmolkit.types");
  boost::python::object dpc_cls      = types_module.attr("DevicePerConfResult");
  boost::python::object async_cls    = types_module.attr("AsyncGpuResult");
  PyArray*              energiesPy   = makePyArrayBorrowed(energies);
  PyArray*              molIdxPy     = makePyArrayBorrowed(molIndices);
  PyArray*              confIdxPy    = makePyArrayBorrowed(confIndices);
  return dpc_cls(wrapAsync(energiesPy, gpuId, async_cls),
                 wrapAsync(molIdxPy, gpuId, async_cls),
                 wrapAsync(confIdxPy, gpuId, async_cls),
                 gpuId,
                 nMols);
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_RESULT_PYTHON_H
