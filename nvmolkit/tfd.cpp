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
#include <boost/python/manage_new_object.hpp>
#include <boost/python/numpy.hpp>

#include "array_helpers.h"
#include "nvtx.h"
#include "tfd_cpu.h"
#include "tfd_gpu.h"

namespace {

using namespace boost::python;
namespace numpy = boost::python::numpy;

using CpuTFDResults = std::vector<std::vector<double>>;

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

boost::python::list intVectorToList(const std::vector<int>& vec) {
  boost::python::list result;
  for (const auto& value : vec) {
    result.append(value);
  }
  return result;
}

nvMolKit::TFDComputeOptions buildOptions(bool               useWeights,
                                         const std::string& maxDev,
                                         int                symmRadius,
                                         bool               ignoreColinearBonds) {
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

  return options;
}

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

nvMolKit::TFDCpuGenerator& getCpuGenerator() {
  static nvMolKit::TFDCpuGenerator generator;
  return generator;
}

nvMolKit::TFDGpuGenerator& getGpuGenerator() {
  static nvMolKit::TFDGpuGenerator generator;
  return generator;
}

}  // namespace

BOOST_PYTHON_MODULE(_TFD) {
  numpy::initialize();

  // CPU path: returns flat numpy array + offsets (avoids per-element Python object creation)
  def(
    "GetTFDMatricesCpuBuffer",
    +[](const boost::python::list& mols,
        bool                       useWeights,
        const std::string&         maxDev,
        int                        symmRadius,
        bool                       ignoreColinearBonds) -> boost::python::object {
      auto molsVec    = listToMolVector(mols);
      auto options    = buildOptions(useWeights, maxDev, symmRadius, ignoreColinearBonds);
      options.backend = nvMolKit::TFDComputeBackend::CPU;
      auto results    = getCpuGenerator().GetTFDMatrices(molsVec, options);

      nvMolKit::ScopedNvtxRange range("CPU: wrap as numpy arrays", nvMolKit::NvtxColor::kGreen);

      // Move results to heap so PyCapsule can own the memory
      auto* owned = new CpuTFDResults(std::move(results));

      auto deleter = [](PyObject* cap) {
        delete reinterpret_cast<CpuTFDResults*>(PyCapsule_GetPointer(cap, "nvmolkit.cpu_tfd"));
      };
      PyObject* cap = PyCapsule_New(static_cast<void*>(owned), "nvmolkit.cpu_tfd", deleter);
      if (cap == nullptr) {
        delete owned;
        throw std::runtime_error("Failed to create PyCapsule for CPU TFD results");
      }
      object owner{handle<>(cap)};

      // Create per-molecule numpy array views (zero-copy, each backed by the capsule)
      const Py_intptr_t   stride = static_cast<Py_intptr_t>(sizeof(double));
      boost::python::list arrays;
      for (auto& vec : *owned) {
        const Py_intptr_t shape = static_cast<Py_intptr_t>(vec.size());
        arrays.append(numpy::from_data(vec.data(),
                                       numpy::dtype::get_builtin<double>(),
                                       make_tuple(shape),
                                       make_tuple(stride),
                                       owner));
      }
      return arrays;
    },
    (arg("mols"),
     arg("useWeights")          = true,
     arg("maxDev")              = "equal",
     arg("symmRadius")          = 2,
     arg("ignoreColinearBonds") = true));

  // GPU path: returns GPU-resident buffer + metadata
  def(
    "GetTFDMatricesGpuBuffer",
    +[](const boost::python::list& mols,
        bool                       useWeights,
        const std::string&         maxDev,
        int                        symmRadius,
        bool                       ignoreColinearBonds) -> boost::python::object {
      auto molsVec    = listToMolVector(mols);
      auto options    = buildOptions(useWeights, maxDev, symmRadius, ignoreColinearBonds);
      options.backend = nvMolKit::TFDComputeBackend::GPU;

      auto gpuResult = getGpuGenerator().GetTFDMatricesGpuBuffer(molsVec, options);

      nvMolKit::ScopedNvtxRange range("GPU: C++ to Python tuple", nvMolKit::NvtxColor::kYellow);
      boost::python::list       outputStarts = intVectorToList(gpuResult.tfdOutputStarts);

      size_t totalSize = gpuResult.tfdValues.size();
      auto*  pyArray   = nvMolKit::makePyArray(gpuResult.tfdValues, boost::python::make_tuple(totalSize));

      return boost::python::make_tuple(toOwnedPyArray(pyArray), outputStarts);
    },
    (arg("mols"),
     arg("useWeights")          = true,
     arg("maxDev")              = "equal",
     arg("symmRadius")          = 2,
     arg("ignoreColinearBonds") = true));
}
