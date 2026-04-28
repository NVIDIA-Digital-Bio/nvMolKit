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

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include "boost_python_utils.h"
#include "device_coord_python.h"
#include "etkdg.h"

namespace bp = boost::python;

static bp::list getGpuIdsPy(nvMolKit::BatchHardwareOptions& opts) {
  return nvMolKit::vectorToList(opts.gpuIds);
}

static void setGpuIds(nvMolKit::BatchHardwareOptions& opts, const bp::object& iterable) {
  std::vector<int> converted;
  using namespace boost::python;
  if (PySequence_Check(iterable.ptr())) {
    Py_ssize_t n = PySequence_Size(iterable.ptr());
    converted.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      object item(handle<>(borrowed(PySequence_GetItem(iterable.ptr(), i))));
      converted.push_back(extract<int>(item));
    }
  } else {
    stl_input_iterator<int> it(iterable), end;
    for (; it != end; ++it) {
      converted.push_back(*it);
    }
  }
  opts.gpuIds.swap(converted);
}

BOOST_PYTHON_MODULE(_embedMolecules) {
  bp::class_<nvMolKit::BatchHardwareOptions>("BatchHardwareOptions")
    .def(bp::init<>())
    .def_readwrite("preprocessingThreads", &nvMolKit::BatchHardwareOptions::preprocessingThreads)
    .def_readwrite("batchSize", &nvMolKit::BatchHardwareOptions::batchSize)
    .def_readwrite("batchesPerGpu", &nvMolKit::BatchHardwareOptions::batchesPerGpu)
    .add_property("gpuIds", &getGpuIdsPy, &setGpuIds);

  bp::def(
    "EmbedMolecules",
    +[](const bp::list&                             molecules,
        const RDKit::DGeomHelpers::EmbedParameters& params,
        int                                         confsPerMolecule,
        int                                         maxIterations,
        const nvMolKit::BatchHardwareOptions&       hardwareOptions,
        int                                         outputMode,
        int                                         targetGpu) -> bp::object {
      auto       molsVec = nvMolKit::extractMolecules(molecules);
      const auto output  = static_cast<nvMolKit::CoordinateOutput>(outputMode);
      auto       result  = nvMolKit::embedMolecules(molsVec,
                                             params,
                                             confsPerMolecule,
                                             maxIterations,
                                             false,
                                             nullptr,
                                             hardwareOptions,
                                             nvMolKit::BfgsBackend::HYBRID,
                                             output,
                                             targetGpu);
      if (!result.has_value()) {
        return bp::object();
      }
      return nvMolKit::pyDeviceCoordResultFromOwned(std::move(*result));
    },
    (bp::arg("molecules"),
     bp::arg("params"),
     bp::arg("confsPerMolecule") = 1,
     bp::arg("maxIterations")    = -1,
     bp::arg("hardwareOptions")  = nvMolKit::BatchHardwareOptions(),
     bp::arg("outputMode")       = static_cast<int>(nvMolKit::CoordinateOutput::RDKIT_CONFORMERS),
     bp::arg("targetGpu")        = -1),
    "Embed multiple molecules with multiple conformers using ETKDG.\n"
    "\n"
    "When outputMode == 0 (RDKIT_CONFORMERS) molecules are modified in place and the function\n"
    "returns None. When outputMode == 1 (DEVICE) the optimized coordinates stay on the GPU and\n"
    "a nvmolkit.types.DeviceCoordResult is returned, collected onto targetGpu.");
}
