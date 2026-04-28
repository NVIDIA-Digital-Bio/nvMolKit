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

#include <boost/python.hpp>

#include "bfgs_uff.h"
#include "boost_python_utils.h"
#include "device_coord_python.h"

namespace bp = boost::python;

BOOST_PYTHON_MODULE(_uffOptimization) {
  bp::def(
    "UFFOptimizeMoleculesConfs",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const bp::list&                       vdwThresholds,
        const bp::list&                       ignoreInterfragInteractions,
        const nvMolKit::BatchHardwareOptions& hardwareOptions) -> bp::list {
      auto       molsVec      = nvMolKit::extractMolecules(molecules);
      const int  numMols      = static_cast<int>(molsVec.size());
      const auto thresholdVec = nvMolKit::extractDoubleList(vdwThresholds, numMols, "vdwThreshold");
      const auto ignoreVec =
        nvMolKit::extractBoolList(ignoreInterfragInteractions, numMols, "ignoreInterfragInteractions");
      const auto result =
        nvMolKit::UFF::UFFOptimizeMoleculesConfsBfgs(molsVec, maxIters, thresholdVec, ignoreVec, hardwareOptions);
      return nvMolKit::vectorOfVectorsToList(result);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters") = 1000,
     bp::arg("vdwThresholds"),
     bp::arg("ignoreInterfragInteractions"),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions()),
    "Optimize conformers for multiple molecules using UFF (host-output API).");

  bp::def(
    "UFFMinimizeDeviceOutput",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        double                                gradTol,
        const bp::list&                       vdwThresholds,
        const bp::list&                       ignoreInterfragInteractions,
        const nvMolKit::BatchHardwareOptions& hardwareOptions,
        int                                   targetGpu,
        const bp::object&                     deviceInput) -> bp::object {
      auto       molsVec      = nvMolKit::extractMolecules(molecules);
      const int  numMols      = static_cast<int>(molsVec.size());
      const auto thresholdVec = nvMolKit::extractDoubleList(vdwThresholds, numMols, "vdwThreshold");
      const auto ignoreVec =
        nvMolKit::extractBoolList(ignoreInterfragInteractions, numMols, "ignoreInterfragInteractions");
      const nvMolKit::DeviceCoordResult* devicePtr = nvMolKit::extractDeviceInputPtr(deviceInput);
      auto                               result    = nvMolKit::UFF::UFFMinimizeMoleculesConfs(molsVec,
                                                                 maxIters,
                                                                 gradTol,
                                                                 thresholdVec,
                                                                 ignoreVec,
                                                                 /*constraints=*/{},
                                                                 hardwareOptions,
                                                                 nvMolKit::CoordinateOutput::DEVICE,
                                                                 targetGpu,
                                                                 devicePtr);
      if (!result.device.has_value()) {
        throw std::runtime_error("UFFMinimizeMoleculesConfs(DEVICE) returned no device result");
      }
      return nvMolKit::pyDeviceCoordResultFromOwned(std::move(*result.device));
    },
    (bp::arg("molecules"),
     bp::arg("maxIters") = 1000,
     bp::arg("gradTol")  = 1e-4,
     bp::arg("vdwThresholds"),
     bp::arg("ignoreInterfragInteractions"),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions(),
     bp::arg("targetGpu")       = -1,
     bp::arg("deviceInput")     = bp::object()),
    "UFF minimize with on-device output. Always returns a nvmolkit.types.DeviceCoordResult.");
}
