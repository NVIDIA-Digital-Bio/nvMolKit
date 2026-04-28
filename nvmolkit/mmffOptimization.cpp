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

#include <boost/python.hpp>

#include "bfgs_mmff.h"
#include "boost_python_utils.h"
#include "device_coord_python.h"
#include "mmff_python_utils.h"

namespace bp = boost::python;

BOOST_PYTHON_MODULE(_mmffOptimization) {
  bp::def(
    "MMFFOptimizeMoleculesConfs",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const bp::list&                       propertiesList,
        const nvMolKit::BatchHardwareOptions& hardwareOptions) -> bp::list {
      auto       molsVec    = nvMolKit::extractMolecules(molecules);
      const auto properties = nvMolKit::extractMMFFPropertiesList(propertiesList, static_cast<int>(molsVec.size()));
      const auto result =
        nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsVec, maxIters, properties, hardwareOptions);
      return nvMolKit::vectorOfVectorsToList(result);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters")        = 200,
     bp::arg("properties")      = bp::list(),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions()),
    "Optimize conformers for multiple molecules using MMFF (host-output API).");

  bp::def(
    "MMFFMinimizeDeviceOutput",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        double                                gradTol,
        const bp::list&                       propertiesList,
        const nvMolKit::BatchHardwareOptions& hardwareOptions,
        int                                   targetGpu,
        const bp::object&                     deviceInput) -> bp::object {
      auto       molsVec    = nvMolKit::extractMolecules(molecules);
      const auto properties = nvMolKit::extractMMFFPropertiesList(propertiesList, static_cast<int>(molsVec.size()));
      const nvMolKit::DeviceCoordResult* devicePtr = nvMolKit::extractDeviceInputPtr(deviceInput);
      auto                               result    = nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(molsVec,
                                                                  maxIters,
                                                                  gradTol,
                                                                  properties,
                                                                  /*constraints=*/{},
                                                                  hardwareOptions,
                                                                  nvMolKit::BfgsBackend::HYBRID,
                                                                  nvMolKit::CoordinateOutput::DEVICE,
                                                                  targetGpu,
                                                                  devicePtr);
      if (!result.device.has_value()) {
        throw std::runtime_error("MMFFMinimizeMoleculesConfs(DEVICE) returned no device result");
      }
      return nvMolKit::pyDeviceCoordResultFromOwned(std::move(*result.device));
    },
    (bp::arg("molecules"),
     bp::arg("maxIters")        = 200,
     bp::arg("gradTol")         = 1e-4,
     bp::arg("properties")      = bp::list(),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions(),
     bp::arg("targetGpu")       = -1,
     bp::arg("deviceInput")     = bp::object()),
    "MMFF minimize with on-device output. Always returns a nvmolkit.types.DeviceCoordResult.");
}
