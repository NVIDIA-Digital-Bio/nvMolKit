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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "device_vector.h"
#include "ff_utils.h"
#include "mmff_batched_forcefield.h"
#include "mmff_flattened_builder.h"
#include "mmff_properties.h"

namespace bp = boost::python;

template <typename T> bp::list vectorToList(const std::vector<T>& vec) {
  bp::list list;
  for (const auto& value : vec) {
    list.append(value);
  }
  return list;
}

template <typename T> bp::list vectorOfVectorsToList(const std::vector<std::vector<T>>& vecOfVecs) {
  bp::list outerList;
  for (const auto& innerVec : vecOfVecs) {
    outerList.append(vectorToList(innerVec));
  }
  return outerList;
}

namespace {
std::vector<std::vector<double>> splitGradients(const std::vector<double>& flatGrad,
                                                const std::vector<int>&    atomStarts,
                                                int                        dim) {
  std::vector<std::vector<double>> result;
  result.reserve(atomStarts.size() - 1);
  for (size_t i = 0; i + 1 < atomStarts.size(); ++i) {
    const int start = atomStarts[i] * dim;
    const int end   = atomStarts[i + 1] * dim;
    result.emplace_back(flatGrad.begin() + start, flatGrad.begin() + end);
  }
  return result;
}
}  // namespace

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

static void throwIfCudaError(cudaError_t err, const std::string& context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(context + ": " + cudaGetErrorString(err));
  }
}

template <typename T> std::vector<T> copyDeviceVector(nvMolKit::AsyncDeviceVector<T>& deviceVec) {
  std::vector<T> hostVec(deviceVec.size());
  deviceVec.copyToHost(hostVec);
  cudaStreamSynchronize(deviceVec.stream());
  return hostVec;
}

static nvMolKit::MMFFProperties extractInternalMMFFProperties(const bp::object& obj,
                                                              double            nonBondedThreshold          = 100.0,
                                                              bool              ignoreInterfragInteractions = true) {
  nvMolKit::MMFFProperties props;
  if (obj.is_none()) {
    props.nonBondedThreshold          = nonBondedThreshold;
    props.ignoreInterfragInteractions = ignoreInterfragInteractions;
    return props;
  }
  props = bp::extract<nvMolKit::MMFFProperties>(obj);
  return props;
}

static std::vector<nvMolKit::MMFFProperties> extractMMFFPropertiesList(const bp::list& properties, int numMols) {
  const int                             n = bp::len(properties);
  std::vector<nvMolKit::MMFFProperties> props;
  props.reserve(numMols);
  for (int i = 0; i < numMols; ++i) {
    if (i < n) {
      props.push_back(extractInternalMMFFProperties(bp::object(properties[i])));
    } else {
      props.emplace_back();
    }
  }
  return props;
}

static std::vector<int> extractIntList(const bp::list& pyList, int expectedSize, const std::string& name) {
  const int n = bp::len(pyList);
  if (n != expectedSize) {
    throw std::invalid_argument(name + " list size " + std::to_string(n) + " does not match expected size " +
                                std::to_string(expectedSize));
  }
  std::vector<int> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) {
    result.push_back(bp::extract<int>(pyList[i]));
  }
  return result;
}

class NativeMMFFBatchedForcefield {
 public:
  NativeMMFFBatchedForcefield(const bp::list& molecules, const bp::list& properties, const bp::list& confIds) {
    const auto mols     = extractMolecules(molecules);
    const int  numMols  = static_cast<int>(mols.size());
    const auto props    = extractMMFFPropertiesList(properties, numMols);
    const auto confList = extractIntList(confIds, numMols, "conf_id");

    nvMolKit::MMFF::BatchedMolecularSystemHost systemHost;
    nvMolKit::BatchedForcefieldMetadata        metadata;
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      std::vector<double> positions;
      nvMolKit::confPosToVect(*mols[molIdx], positions, confList[molIdx]);
      auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(*mols[molIdx], props[molIdx], confList[molIdx]);
      nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost, &metadata, molIdx, 0);
    }
    forcefield_ = std::make_unique<nvMolKit::MMFFBatchedForcefield>(systemHost, metadata);
    positionsDevice_.setFromVector(systemHost.positions);
    gradDevice_.resize(forcefield_->totalPositions());
    energyOutsDevice_.resize(forcefield_->numMolecules());
  }

  bp::list computeEnergy() {
    energyOutsDevice_.zero();
    throwIfCudaError(forcefield_->computeEnergy(energyOutsDevice_.data(), positionsDevice_.data()), "computeEnergy");
    return vectorToList(copyDeviceVector(energyOutsDevice_));
  }

  bp::list computeGradients() {
    gradDevice_.zero();
    throwIfCudaError(forcefield_->computeGradients(gradDevice_.data(), positionsDevice_.data()), "computeGradients");
    return vectorOfVectorsToList(splitGradients(copyDeviceVector(gradDevice_), forcefield_->atomStartsHost(), 3));
  }

 private:
  std::unique_ptr<nvMolKit::MMFFBatchedForcefield> forcefield_;
  nvMolKit::AsyncDeviceVector<double>              positionsDevice_;
  nvMolKit::AsyncDeviceVector<double>              gradDevice_;
  nvMolKit::AsyncDeviceVector<double>              energyOutsDevice_;
};

BOOST_PYTHON_MODULE(_batchedForcefield) {
  bp::class_<nvMolKit::MMFFProperties>("MMFFProperties")
    .def_readwrite("variant", &nvMolKit::MMFFProperties::variant)
    .def_readwrite("dielectricConstant", &nvMolKit::MMFFProperties::dielectricConstant)
    .def_readwrite("dielectricModel", &nvMolKit::MMFFProperties::dielectricModel)
    .def_readwrite("nonBondedThreshold", &nvMolKit::MMFFProperties::nonBondedThreshold)
    .def_readwrite("ignoreInterfragInteractions", &nvMolKit::MMFFProperties::ignoreInterfragInteractions)
    .def_readwrite("bondTerm", &nvMolKit::MMFFProperties::bondTerm)
    .def_readwrite("angleTerm", &nvMolKit::MMFFProperties::angleTerm)
    .def_readwrite("stretchBendTerm", &nvMolKit::MMFFProperties::stretchBendTerm)
    .def_readwrite("oopTerm", &nvMolKit::MMFFProperties::oopTerm)
    .def_readwrite("torsionTerm", &nvMolKit::MMFFProperties::torsionTerm)
    .def_readwrite("vdwTerm", &nvMolKit::MMFFProperties::vdwTerm)
    .def_readwrite("eleTerm", &nvMolKit::MMFFProperties::eleTerm);

  bp::class_<NativeMMFFBatchedForcefield, boost::noncopyable>(
    "NativeMMFFBatchedForcefield",
    bp::init<const bp::list&, const bp::list&, const bp::list&>())
    .def("computeEnergy", &NativeMMFFBatchedForcefield::computeEnergy)
    .def("computeGradients", &NativeMMFFBatchedForcefield::computeGradients);
}
