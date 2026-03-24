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

#include <GraphMol/ROMol.h>

#include <boost/python.hpp>

#include <stdexcept>
#include <vector>

#include "bfgs_uff.h"

namespace bp = boost::python;

template <typename T>
bp::list vectorToList(const std::vector<T>& vec) {
  bp::list list;
  for (const auto& value : vec) {
    list.append(value);
  }
  return list;
}

template <typename T>
bp::list vectorOfVectorsToList(const std::vector<std::vector<T>>& vecOfVecs) {
  bp::list outerList;
  for (const auto& innerVec : vecOfVecs) {
    outerList.append(vectorToList(innerVec));
  }
  return outerList;
}

static std::vector<RDKit::ROMol*> extractMolecules(const bp::list& molecules) {
  const int                  n = bp::len(molecules);
  std::vector<RDKit::ROMol*> mols;
  mols.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto* mol = bp::extract<RDKit::ROMol*>(bp::object(molecules[i]))();
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
    }
    mols.push_back(mol);
  }
  return mols;
}

static std::vector<double> extractDoubleList(const bp::list& values, const int expectedSize, const std::string& name) {
  if (bp::len(values) != expectedSize) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " values for " + name + ", got " +
                                std::to_string(bp::len(values)));
  }
  std::vector<double> result;
  result.reserve(expectedSize);
  for (int i = 0; i < expectedSize; ++i) {
    result.push_back(bp::extract<double>(values[i]));
  }
  return result;
}

static std::vector<bool> extractBoolList(const bp::list& values, const int expectedSize, const std::string& name) {
  if (bp::len(values) != expectedSize) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " values for " + name + ", got " +
                                std::to_string(bp::len(values)));
  }
  std::vector<bool> result;
  result.reserve(expectedSize);
  for (int i = 0; i < expectedSize; ++i) {
    result.push_back(bp::extract<bool>(values[i]));
  }
  return result;
}

BOOST_PYTHON_MODULE(_uffOptimization) {
  bp::def(
    "UFFOptimizeMoleculesConfs",
    +[](const bp::list&                        molecules,
        int                                   maxIters,
        const bp::list&                       vdwThresholds,
        const bp::list&                       ignoreInterfragInteractions,
        const nvMolKit::BatchHardwareOptions& hardwareOptions) -> bp::list {
      auto molsVec = extractMolecules(molecules);
      const int numMols = static_cast<int>(molsVec.size());
      const auto thresholdVec = extractDoubleList(vdwThresholds, numMols, "vdwThreshold");
      const auto ignoreVec =
        extractBoolList(ignoreInterfragInteractions, numMols, "ignoreInterfragInteractions");
      const auto result =
        nvMolKit::UFF::UFFOptimizeMoleculesConfsBfgs(molsVec, maxIters, thresholdVec, ignoreVec, hardwareOptions);
      return vectorOfVectorsToList(result);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters")                   = 1000,
     bp::arg("vdwThresholds"),
     bp::arg("ignoreInterfragInteractions"),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions()),
    "Optimize conformers for multiple molecules using UFF force field.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to optimize\n"
    "    maxIters: Maximum number of optimization iterations (default: 1000)\n"
    "    vdwThresholds: Per-molecule van der Waals thresholds\n"
    "    ignoreInterfragInteractions: Per-molecule interfragment interaction flags\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings\n"
    "\n"
    "Returns:\n"
    "    List of lists of energies, where each inner list contains energies for conformers of one molecule");
}
