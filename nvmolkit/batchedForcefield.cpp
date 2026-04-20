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

#include <GraphMol/Conformer.h>

#include <boost/python.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "bfgs_minimize.h"
#include "bfgs_types.h"
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

struct ConformerEntry {
  RDKit::ROMol*     mol;
  int               molIdx;
  RDKit::Conformer* conformer;
  uint32_t          atomStart;
};

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

bp::list reshapeStatusesToNested(const std::vector<int16_t>& statuses, const std::vector<int>& numConformersPerMol) {
  bp::list outer;
  size_t   idx = 0;
  for (const int nConfs : numConformersPerMol) {
    bp::list inner;
    for (int j = 0; j < nConfs; ++j) {
      inner.append(statuses[idx++] == 0);
    }
    outer.append(inner);
  }
  return outer;
}

bp::list reshapeToNested(const std::vector<double>& flat, const std::vector<int>& numConformersPerMol) {
  bp::list outer;
  size_t   idx = 0;
  for (const int nConfs : numConformersPerMol) {
    bp::list inner;
    for (int j = 0; j < nConfs; ++j) {
      inner.append(flat[idx++]);
    }
    outer.append(inner);
  }
  return outer;
}

bp::list reshapeGradientsToNested(const std::vector<std::vector<double>>& perSystem,
                                  const std::vector<int>&                 numConformersPerMol) {
  bp::list outer;
  size_t   idx = 0;
  for (const int nConfs : numConformersPerMol) {
    bp::list inner;
    for (int j = 0; j < nConfs; ++j) {
      inner.append(nvMolKit::vectorToList(perSystem[idx++]));
    }
    outer.append(inner);
  }
  return outer;
}

void writeBackPositions(const std::vector<ConformerEntry>& entries, const std::vector<double>& hostPositions) {
  for (const auto& entry : entries) {
    const uint32_t numAtoms = entry.mol->getNumAtoms();
    for (uint32_t j = 0; j < numAtoms; ++j) {
      entry.conformer->setAtomPos(j,
                                  RDGeom::Point3D(hostPositions[3 * (entry.atomStart + j) + 0],
                                                  hostPositions[3 * (entry.atomStart + j) + 1],
                                                  hostPositions[3 * (entry.atomStart + j) + 2]));
    }
  }
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

using DistanceSpecs = std::vector<nvMolKit::ForceFieldConstraints::DistanceConstraintSpec>;
using PositionSpecs = std::vector<nvMolKit::ForceFieldConstraints::PositionConstraintSpec>;
using AngleSpecs    = std::vector<nvMolKit::ForceFieldConstraints::AngleConstraintSpec>;
using TorsionSpecs  = std::vector<nvMolKit::ForceFieldConstraints::TorsionConstraintSpec>;

struct PerMolConstraints {
  DistanceSpecs distance;
  PositionSpecs position;
  AngleSpecs    angle;
  TorsionSpecs  torsion;

  bool empty() const { return distance.empty() && position.empty() && angle.empty() && torsion.empty(); }

  template <typename Contribs> void applyTo(Contribs& contribs, const std::vector<double>& positions) const {
    for (const auto& s : distance) {
      nvMolKit::ForceFieldConstraints::appendDistanceConstraint(contribs, positions, s);
    }
    for (const auto& s : position) {
      nvMolKit::ForceFieldConstraints::appendPositionConstraint(contribs, positions, s);
    }
    for (const auto& s : angle) {
      nvMolKit::ForceFieldConstraints::appendAngleConstraint(contribs, positions, s);
    }
    for (const auto& s : torsion) {
      nvMolKit::ForceFieldConstraints::appendTorsionConstraint(contribs, positions, s);
    }
  }
};

static std::vector<PerMolConstraints> extractAllConstraints(const bp::list& distanceConstraints,
                                                            const bp::list& positionConstraints,
                                                            const bp::list& angleConstraints,
                                                            const bp::list& torsionConstraints,
                                                            int             numMols) {
  const auto distLists =
    extractConstraintLists<nvMolKit::ForceFieldConstraints::DistanceConstraintSpec>(distanceConstraints,
                                                                                    numMols,
                                                                                    parseDistanceConstraintTuple,
                                                                                    "distance constraints");
  const auto posLists =
    extractConstraintLists<nvMolKit::ForceFieldConstraints::PositionConstraintSpec>(positionConstraints,
                                                                                    numMols,
                                                                                    parsePositionConstraintTuple,
                                                                                    "position constraints");
  const auto angleLists =
    extractConstraintLists<nvMolKit::ForceFieldConstraints::AngleConstraintSpec>(angleConstraints,
                                                                                 numMols,
                                                                                 parseAngleConstraintTuple,
                                                                                 "angle constraints");
  const auto torsionLists =
    extractConstraintLists<nvMolKit::ForceFieldConstraints::TorsionConstraintSpec>(torsionConstraints,
                                                                                   numMols,
                                                                                   parseTorsionConstraintTuple,
                                                                                   "torsion constraints");

  std::vector<PerMolConstraints> result(numMols);
  for (int i = 0; i < numMols; ++i) {
    result[i] = {distLists[i], posLists[i], angleLists[i], torsionLists[i]};
  }
  return result;
}

class NativeMMFFBatchedForcefield {
 public:
  NativeMMFFBatchedForcefield(const bp::list& molecules,
                              const bp::list& properties,
                              const bp::list& distanceConstraints,
                              const bp::list& positionConstraints,
                              const bp::list& angleConstraints,
                              const bp::list& torsionConstraints) {
    const auto mols    = nvMolKit::extractMolecules(molecules);
    const int  numMols = static_cast<int>(mols.size());
    const auto props   = nvMolKit::extractMMFFPropertiesList(properties, numMols);
    const auto constraints =
      extractAllConstraints(distanceConstraints, positionConstraints, angleConstraints, torsionConstraints, numMols);

    nvMolKit::MMFF::BatchedMolecularSystemHost systemHost;
    nvMolKit::BatchedForcefieldMetadata        metadata;
    numConformersPerMol_.resize(numMols);
    uint32_t currentAtomOffset = 0;

    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      auto* mol          = mols[molIdx];
      auto  baseContribs = nvMolKit::MMFF::constructForcefieldContribs(*mol, props[molIdx]);

      int confIdx = 0;
      for (auto confIter = mol->beginConformers(); confIter != mol->endConformers(); ++confIter, ++confIdx) {
        auto&               conf = **confIter;
        std::vector<double> positions;
        nvMolKit::confPosToVect(conf, positions);

        auto contribs = baseContribs;
        constraints[molIdx].applyTo(contribs, positions);

        nvMolKit::MMFF::addMoleculeToBatch(contribs, positions, systemHost, &metadata, molIdx, confIdx);

        conformerEntries_.push_back({mol, molIdx, &conf, currentAtomOffset});
        currentAtomOffset += mol->getNumAtoms();
      }
      numConformersPerMol_[molIdx] = confIdx;
    }

    forcefield_ = std::make_unique<nvMolKit::MMFFBatchedForcefield>(systemHost, metadata);
    positionsDevice_.setFromVector(systemHost.positions);
    gradDevice_.resize(forcefield_->totalPositions());
    energyOutsDevice_.resize(forcefield_->numMolecules());
  }

  bp::list computeEnergy() {
    energyOutsDevice_.zero();
    throwIfCudaError(forcefield_->computeEnergy(energyOutsDevice_.data(), positionsDevice_.data()), "computeEnergy");
    return reshapeToNested(copyDeviceVector(energyOutsDevice_), numConformersPerMol_);
  }

  bp::list computeGradients() {
    gradDevice_.zero();
    throwIfCudaError(forcefield_->computeGradients(gradDevice_.data(), positionsDevice_.data()), "computeGradients");
    auto perSystem = splitGradients(copyDeviceVector(gradDevice_), forcefield_->atomStartsHost(), 3);
    return reshapeGradientsToNested(perSystem, numConformersPerMol_);
  }

  bp::tuple minimize(int maxIters, double gradTol) {
    gradDevice_.zero();
    energyOutsDevice_.zero();

    nvMolKit::BfgsBatchMinimizer bfgsMinimizer(
      /*dataDim=*/3,
      nvMolKit::DebugLevel::NONE,
      true,
      nullptr,
      nvMolKit::BfgsBackend::BATCHED);
    bfgsMinimizer.minimize(maxIters, gradTol, *forcefield_, positionsDevice_, gradDevice_, energyOutsDevice_);

    auto hostPositions = copyDeviceVector(positionsDevice_);
    writeBackPositions(conformerEntries_, hostPositions);

    bp::list energies  = reshapeToNested(copyDeviceVector(energyOutsDevice_), numConformersPerMol_);
    bp::list converged = reshapeStatusesToNested(copyDeviceVector(bfgsMinimizer.statuses_), numConformersPerMol_);
    return bp::make_tuple(energies, converged);
  }

 private:
  std::unique_ptr<nvMolKit::MMFFBatchedForcefield> forcefield_;
  nvMolKit::AsyncDeviceVector<double>              positionsDevice_;
  nvMolKit::AsyncDeviceVector<double>              gradDevice_;
  nvMolKit::AsyncDeviceVector<double>              energyOutsDevice_;
  std::vector<ConformerEntry>                      conformerEntries_;
  std::vector<int>                                 numConformersPerMol_;
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
    bp::init<const bp::list&, const bp::list&, const bp::list&, const bp::list&, const bp::list&, const bp::list&>())
    .def("computeEnergy", &NativeMMFFBatchedForcefield::computeEnergy)
    .def("computeGradients", &NativeMMFFBatchedForcefield::computeGradients)
    .def("minimize", &NativeMMFFBatchedForcefield::minimize);
}
