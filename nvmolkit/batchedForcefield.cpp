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
#include <memory>
#include <string>
#include <vector>

#include "boost_python_utils.h"
#include "device_vector.h"
#include "ff_utils.h"
#include "forcefield_constraints.h"
#include "mmff_batched_forcefield.h"
#include "mmff_flattened_builder.h"
#include "mmff_properties.h"
#include "mmff_python_utils.h"

namespace bp = boost::python;

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

template <typename Spec, typename Parser>
static std::vector<std::vector<Spec>> extractConstraintLists(const bp::list&    outerList,
                                                             const int          expectedSize,
                                                             const Parser&      parser,
                                                             const std::string& name) {
  if (bp::len(outerList) != expectedSize) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " entries for " + name + ", got " +
                                std::to_string(bp::len(outerList)));
  }
  std::vector<std::vector<Spec>> allSpecs(expectedSize);
  for (int molIdx = 0; molIdx < expectedSize; ++molIdx) {
    const bp::list innerList = bp::extract<bp::list>(bp::object(outerList[molIdx]));
    auto&          specs     = allSpecs[molIdx];
    specs.reserve(bp::len(innerList));
    for (int j = 0; j < bp::len(innerList); ++j) {
      specs.push_back(parser(bp::extract<bp::tuple>(bp::object(innerList[j]))));
    }
  }
  return allSpecs;
}

static nvMolKit::ForceFieldConstraints::DistanceConstraintSpec parseDistanceConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 6) {
    throw std::invalid_argument("Distance constraint tuples must have 6 elements");
  }
  return {bp::extract<int>(value[0]),
          bp::extract<int>(value[1]),
          bp::extract<bool>(value[2]),
          bp::extract<double>(value[3]),
          bp::extract<double>(value[4]),
          bp::extract<double>(value[5])};
}

static nvMolKit::ForceFieldConstraints::PositionConstraintSpec parsePositionConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 3) {
    throw std::invalid_argument("Position constraint tuples must have 3 elements");
  }
  return {bp::extract<int>(value[0]), bp::extract<double>(value[1]), bp::extract<double>(value[2])};
}

static nvMolKit::ForceFieldConstraints::AngleConstraintSpec parseAngleConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 7) {
    throw std::invalid_argument("Angle constraint tuples must have 7 elements");
  }
  return {bp::extract<int>(value[0]),
          bp::extract<int>(value[1]),
          bp::extract<int>(value[2]),
          bp::extract<bool>(value[3]),
          bp::extract<double>(value[4]),
          bp::extract<double>(value[5]),
          bp::extract<double>(value[6])};
}

static nvMolKit::ForceFieldConstraints::TorsionConstraintSpec parseTorsionConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 8) {
    throw std::invalid_argument("Torsion constraint tuples must have 8 elements");
  }
  return {bp::extract<int>(value[0]),
          bp::extract<int>(value[1]),
          bp::extract<int>(value[2]),
          bp::extract<int>(value[3]),
          bp::extract<bool>(value[4]),
          bp::extract<double>(value[5]),
          bp::extract<double>(value[6]),
          bp::extract<double>(value[7])};
}

class NativeMMFFBatchedForcefield {
 public:
  NativeMMFFBatchedForcefield(const bp::list& molecules,
                              const bp::list& properties,
                              const bp::list& confIds,
                              const bp::list& distanceConstraints,
                              const bp::list& positionConstraints,
                              const bp::list& angleConstraints,
                              const bp::list& torsionConstraints) {
    const auto mols     = nvMolKit::extractMolecules(molecules);
    const int  numMols  = static_cast<int>(mols.size());
    const auto props    = nvMolKit::extractMMFFPropertiesList(properties, numMols);
    const auto confList = extractIntList(confIds, numMols, "conf_id");
    const auto distanceConstraintLists =
      extractConstraintLists<nvMolKit::ForceFieldConstraints::DistanceConstraintSpec>(distanceConstraints,
                                                                                      numMols,
                                                                                      parseDistanceConstraintTuple,
                                                                                      "distance constraints");
    const auto positionConstraintLists =
      extractConstraintLists<nvMolKit::ForceFieldConstraints::PositionConstraintSpec>(positionConstraints,
                                                                                      numMols,
                                                                                      parsePositionConstraintTuple,
                                                                                      "position constraints");
    const auto angleConstraintLists =
      extractConstraintLists<nvMolKit::ForceFieldConstraints::AngleConstraintSpec>(angleConstraints,
                                                                                   numMols,
                                                                                   parseAngleConstraintTuple,
                                                                                   "angle constraints");
    const auto torsionConstraintLists =
      extractConstraintLists<nvMolKit::ForceFieldConstraints::TorsionConstraintSpec>(torsionConstraints,
                                                                                     numMols,
                                                                                     parseTorsionConstraintTuple,
                                                                                     "torsion constraints");

    nvMolKit::MMFF::BatchedMolecularSystemHost systemHost;
    nvMolKit::BatchedForcefieldMetadata        metadata;
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      std::vector<double> positions;
      nvMolKit::confPosToVect(*mols[molIdx], positions, confList[molIdx]);

      auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(*mols[molIdx], props[molIdx], confList[molIdx]);
      for (const auto& spec : distanceConstraintLists[molIdx]) {
        nvMolKit::ForceFieldConstraints::appendDistanceConstraint(ffParams, positions, spec);
      }
      for (const auto& spec : positionConstraintLists[molIdx]) {
        nvMolKit::ForceFieldConstraints::appendPositionConstraint(ffParams, positions, spec);
      }
      for (const auto& spec : angleConstraintLists[molIdx]) {
        nvMolKit::ForceFieldConstraints::appendAngleConstraint(ffParams, positions, spec);
      }
      for (const auto& spec : torsionConstraintLists[molIdx]) {
        nvMolKit::ForceFieldConstraints::appendTorsionConstraint(ffParams, positions, spec);
      }
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
    return nvMolKit::vectorToList(copyDeviceVector(energyOutsDevice_));
  }

  bp::list computeGradients() {
    gradDevice_.zero();
    throwIfCudaError(forcefield_->computeGradients(gradDevice_.data(), positionsDevice_.data()), "computeGradients");
    return nvMolKit::vectorOfVectorsToList(
      splitGradients(copyDeviceVector(gradDevice_), forcefield_->atomStartsHost(), 3));
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

  bp::class_<NativeMMFFBatchedForcefield, boost::noncopyable>("NativeMMFFBatchedForcefield",
                                                              bp::init<const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&>())
    .def("computeEnergy", &NativeMMFFBatchedForcefield::computeEnergy)
    .def("computeGradients", &NativeMMFFBatchedForcefield::computeGradients);
}
