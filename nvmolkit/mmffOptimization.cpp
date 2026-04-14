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
#include "mmff_properties.h"

BOOST_PYTHON_MODULE(_mmffOptimization) {
  boost::python::def(
    "MMFFOptimizeMoleculesConfs",
    +[](const boost::python::list&            molecules,
        int                                   maxIters,
        double                                nonBondedThreshold,
        const nvMolKit::BatchHardwareOptions& hardwareOptions) -> boost::python::list {
      auto molsVec = nvMolKit::extractMolecules(molecules);

      nvMolKit::MMFFProperties properties;
      properties.nonBondedThreshold = nonBondedThreshold;
      auto result = nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsVec, maxIters, properties, hardwareOptions);

      return nvMolKit::vectorOfVectorsToList(result);
    },
    (boost::python::arg("molecules"),
     boost::python::arg("maxIters")           = 200,
     boost::python::arg("nonBondedThreshold") = 100.0,
     boost::python::arg("hardwareOptions")    = nvMolKit::BatchHardwareOptions()),
    "Optimize conformers for multiple molecules using MMFF force field.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to optimize\n"
    "    maxIters: Maximum number of optimization iterations (default: 200)\n"
    "    nonBondedThreshold: Radius threshold for non-bonded interactions (default: 100.0)\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings (default: default options)\n"
    "\n"
    "Returns:\n"
    "    List of lists of energies, where each inner list contains energies for conformers of one molecule");
}
