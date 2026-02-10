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

#include "tfd.h"

#include <GraphMol/ROMol.h>

#include <boost/python.hpp>
#include <boost/python/manage_new_object.hpp>

#include "array_helpers.h"

namespace {

using namespace boost::python;

//! Convert boost::python::list of molecules to std::vector
std::vector<const RDKit::ROMol*> listToMolVector(const boost::python::list& mols) {
  std::vector<const RDKit::ROMol*> molsVec;
  molsVec.reserve(len(mols));
  for (int i = 0; i < len(mols); i++) {
    const RDKit::ROMol* mol = boost::python::extract<const RDKit::ROMol*>(boost::python::object(mols[i]));
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
    }
    molsVec.push_back(mol);
  }
  return molsVec;
}

//! Convert std::vector<double> to boost::python::list
boost::python::list vectorToList(const std::vector<double>& vec) {
  boost::python::list result;
  for (const auto& value : vec) {
    result.append(value);
  }
  return result;
}

//! Convert std::vector<int> to boost::python::list
boost::python::list intVectorToList(const std::vector<int>& vec) {
  boost::python::list result;
  for (const auto& value : vec) {
    result.append(value);
  }
  return result;
}

//! Convert std::vector<std::vector<double>> to boost::python::list of lists
boost::python::list nestedVectorToList(const std::vector<std::vector<double>>& vec) {
  boost::python::list result;
  for (const auto& inner : vec) {
    result.append(vectorToList(inner));
  }
  return result;
}

//! Build TFDComputeOptions from Python arguments
nvMolKit::TFDComputeOptions buildOptions(bool               useWeights,
                                         const std::string& maxDev,
                                         int                symmRadius,
                                         bool               ignoreColinearBonds,
                                         const std::string& backend) {
  nvMolKit::TFDComputeOptions options;
  options.useWeights          = useWeights;
  options.symmRadius          = symmRadius;
  options.ignoreColinearBonds = ignoreColinearBonds;

  if (maxDev == "equal") {
    options.maxDevMode = nvMolKit::TFDMaxDevMode::Equal;
  } else if (maxDev == "spec") {
    options.maxDevMode = nvMolKit::TFDMaxDevMode::Spec;
  } else {
    throw std::invalid_argument("maxDev must be 'equal' or 'spec', got: " + maxDev);
  }

  if (backend == "gpu" || backend == "GPU") {
    options.backend = nvMolKit::TFDComputeBackend::GPU;
  } else if (backend == "cpu" || backend == "CPU") {
    options.backend = nvMolKit::TFDComputeBackend::CPU;
  } else {
    throw std::invalid_argument("backend must be 'gpu' or 'cpu', got: " + backend);
  }

  return options;
}

//! Wrap a raw PyArray* into a Python object with proper ownership
boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

// Shared generator instance for module-level functions
nvMolKit::TFDGenerator& getGenerator() {
  static nvMolKit::TFDGenerator generator;
  return generator;
}

}  // namespace

BOOST_PYTHON_MODULE(_TFD) {
  // Module-level function: GetTFDMatrix for single molecule
  def(
    "GetTFDMatrix",
    +[](const RDKit::ROMol& mol,
        bool                useWeights,
        const std::string&  maxDev,
        int                 symmRadius,
        bool                ignoreColinearBonds,
        const std::string&  backend) {
      auto options = buildOptions(useWeights, maxDev, symmRadius, ignoreColinearBonds, backend);
      auto result  = getGenerator().GetTFDMatrix(mol, options);
      return vectorToList(result);
    },
    (arg("mol"),
     arg("useWeights")          = true,
     arg("maxDev")              = "equal",
     arg("symmRadius")          = 2,
     arg("ignoreColinearBonds") = true,
     arg("backend")             = "gpu"));

  // Module-level function: GetTFDMatrices for multiple molecules
  def(
    "GetTFDMatrices",
    +[](const boost::python::list& mols,
        bool                       useWeights,
        const std::string&         maxDev,
        int                        symmRadius,
        bool                       ignoreColinearBonds,
        const std::string&         backend) {
      auto molsVec = listToMolVector(mols);
      auto options = buildOptions(useWeights, maxDev, symmRadius, ignoreColinearBonds, backend);
      auto results = getGenerator().GetTFDMatrices(molsVec, options);
      return nestedVectorToList(results);
    },
    (arg("mols"),
     arg("useWeights")          = true,
     arg("maxDev")              = "equal",
     arg("symmRadius")          = 2,
     arg("ignoreColinearBonds") = true,
     arg("backend")             = "gpu"));

  // Module-level function: GetTFDMatricesGpuBuffer for GPU-resident output
  def(
    "GetTFDMatricesGpuBuffer",
    +[](const boost::python::list& mols,
        bool                       useWeights,
        const std::string&         maxDev,
        int                        symmRadius,
        bool                       ignoreColinearBonds) -> boost::python::object {
      auto molsVec = listToMolVector(mols);
      auto options = buildOptions(useWeights, maxDev, symmRadius, ignoreColinearBonds, "gpu");

      auto gpuResult = getGenerator().GetTFDMatricesGpuBuffer(molsVec, options);

      // Create metadata lists
      boost::python::list outputStarts    = intVectorToList(gpuResult.tfdOutputStarts);
      boost::python::list conformerCounts = intVectorToList(gpuResult.conformerCounts);

      // Create PyArray from GPU buffer
      size_t totalSize = gpuResult.tfdValues.size();
      auto*  pyArray   = nvMolKit::makePyArray(gpuResult.tfdValues, boost::python::make_tuple(totalSize));

      // Return tuple of (pyArray, outputStarts, conformerCounts)
      return boost::python::make_tuple(toOwnedPyArray(pyArray), outputStarts, conformerCounts);
    },
    (arg("mols"),
     arg("useWeights")          = true,
     arg("maxDev")              = "equal",
     arg("symmRadius")          = 2,
     arg("ignoreColinearBonds") = true));
}
