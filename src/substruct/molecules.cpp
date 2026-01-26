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

#include "molecules.h"

#include <GraphMol/MolOps.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "nvtx.h"
#include "packed_bonds.h"
#include "substruct_types.h"

namespace nvMolKit {

namespace {

/**
 * @brief Populate non-bond-related atom properties into packed format.
 *
 * Extracts scalar atom properties from RDKit atom. Bond-related properties
 * (ring bond count, heteroatom neighbors, bond type counts) are populated
 * separately during the fused bond iteration.
 */
void populateAtomScalars(const RDKit::Atom* atom, AtomDataPacked& packed, const RDKit::RingInfo* ringInfo) {
  packed.setAtomicNum(atom->getAtomicNum());
  packed.setChiralTag(atom->getChiralTag());
  packed.setNumExplicitHs(atom->getTotalNumHs());
#if RDKIT_VERSION_NUM >= 0x20240300
  packed.setExplicitValence(atom->getValence(RDKit::Atom::ValenceType::EXPLICIT));
  packed.setImplicitValence(atom->getValence(RDKit::Atom::ValenceType::IMPLICIT));
#else
  packed.setExplicitValence(atom->getExplicitValence());
  packed.setImplicitValence(atom->getImplicitValence());
#endif
  packed.setTotalValence(atom->getTotalValence());
  packed.setFormalCharge(atom->getFormalCharge());
  packed.setHybridization(atom->getHybridization());
  packed.setIsAromatic(atom->getIsAromatic());
  packed.setNumRadicalElectrons(atom->getNumRadicalElectrons());

  const int idx      = atom->getIdx();
  const int numRings = ringInfo->numAtomRings(idx);
  if (numRings > AtomDataPacked::kMax4BitValue) {
    throw std::runtime_error("Atom ring count " + std::to_string(numRings) + " exceeds maximum storable value of " +
                             std::to_string(AtomDataPacked::kMax4BitValue));
  }
  packed.setNumRings(numRings);
  packed.setMinRingSize(ringInfo->minAtomRingSize(idx));
  packed.setIsInRing(numRings > 0);

  const unsigned int numImplicitHs = atom->getNumImplicitHs();
  if (numImplicitHs > AtomDataPacked::kMax4BitValue) {
    throw std::runtime_error("Implicit H count " + std::to_string(numImplicitHs) +
                             " exceeds maximum storable value of " + std::to_string(AtomDataPacked::kMax4BitValue));
  }
  packed.setNumImplicitHs(numImplicitHs);

  const unsigned int isotope = atom->getIsotope();
  if (isotope > 255) {
    throw std::runtime_error("Atom isotope " + std::to_string(isotope) + " exceeds maximum supported value of 255");
  }
  packed.setIsotope(static_cast<uint8_t>(isotope));

  packed.setDegree(atom->getDegree());
  packed.setTotalConnectivity(atom->getTotalDegree());
}

/**
 * @brief Increment bond type count based on RDKit bond type.
 */
void incrementBondTypeCount(BondTypeCounts& counts, int bondType) {
  switch (bondType) {
    case 1:
      ++counts.single;
      break;
    case 2:
      ++counts.double_;
      break;
    case 3:
      ++counts.triple;
      break;
    case 7:
    case 12:
      ++counts.aromatic;
      break;
    default:
      throw std::runtime_error("Unsupported bond type " + std::to_string(bondType) +
                               " in target molecule. Only single, double, triple, and aromatic bonds are supported.");
  }
}

/**
 * @brief Process all atoms and bonds for a target molecule in a single fused pass.
 */
void populateTargetMolecule(const RDKit::ROMol* mol, MoleculesHost& batch, const RDKit::RingInfo* ringInfo) {
  auto& atomDataPackedVec  = batch.atomDataPacked;
  auto& bondTypeCountsVec  = batch.bondTypeCounts;
  auto& targetAtomBondsVec = batch.targetAtomBonds;

  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& packed     = atomDataPackedVec.emplace_back();
    auto& bondCounts = bondTypeCountsVec.emplace_back();
    auto& tab        = targetAtomBondsVec.emplace_back();

    populateAtomScalars(atom, packed, ringInfo);

    const unsigned int atomIdx            = atom->getIdx();
    int                ringBondCount      = 0;
    int                numHeteroNeighbors = 0;
    int                totalBonds         = 0;
    tab.degree                            = 0;

    auto [beg, bondEnd] = mol->getAtomBonds(atom);
    while (beg != bondEnd) {
      const auto*        bond        = (*mol)[*beg];
      const unsigned int bondIdx     = bond->getIdx();
      const int          bondType    = bond->getBondType();
      const int          otherAtomId = bond->getOtherAtomIdx(atomIdx);
      const bool         isInRing    = ringInfo->numBondRings(bondIdx) > 0;

      incrementBondTypeCount(bondCounts, bondType);

      ringBondCount += isInRing;

      const int neighborAtomicNum = mol->getAtomWithIdx(otherAtomId)->getAtomicNum();
      numHeteroNeighbors += (neighborAtomicNum != 6 && neighborAtomicNum != 1);

      if (tab.degree < kMaxBondsPerAtom) {
        tab.neighborIdx[tab.degree] = static_cast<uint8_t>(otherAtomId);
        tab.bondInfo[tab.degree]    = packTargetBondInfo(bondType, isInRing);
        ++tab.degree;
      }
      ++totalBonds;
      ++beg;
    }

    if (totalBonds > kMaxBondsPerAtom) {
      throw std::runtime_error("Atom has more than " + std::to_string(kMaxBondsPerAtom) + " bonds");
    }

    if (ringBondCount > AtomDataPacked::kMax4BitValue) {
      throw std::runtime_error("Ring bond count " + std::to_string(ringBondCount) +
                               " exceeds maximum storable value of " + std::to_string(AtomDataPacked::kMax4BitValue));
    }
    packed.setRingBondCount(ringBondCount);

    if (numHeteroNeighbors > AtomDataPacked::kMax4BitValue) {
      throw std::runtime_error("Heteroatom neighbor count " + std::to_string(numHeteroNeighbors) +
                               " exceeds maximum storable value of " + std::to_string(AtomDataPacked::kMax4BitValue));
    }
    packed.setNumHeteroatomNeighbors(numHeteroNeighbors);
  }
}

}  // namespace

MoleculesHost::MoleculesHost() {
  batchAtomStarts.push_back(0);
}

void MoleculesHost::reserve(size_t numMols, size_t numAtoms) {
  batchAtomStarts.reserve(numMols + 1);

  atomDataPacked.reserve(numAtoms);
  bondTypeCounts.reserve(numAtoms);
  targetAtomBonds.reserve(numAtoms);
  queryAtomBonds.reserve(numAtoms);

  atomQueryMasks.reserve(numAtoms);
  atomQueryTrees.reserve(numAtoms);
  atomInstrStarts.reserve(numAtoms);
  atomLeafMaskStarts.reserve(numAtoms);
  recursivePatterns.reserve(numMols);
}

void MoleculesHost::clear() {
  batchAtomStarts.clear();
  batchAtomStarts.push_back(0);

  atomDataPacked.clear();
  bondTypeCounts.clear();
  targetAtomBonds.clear();
  queryAtomBonds.clear();

  atomQueryMasks.clear();
  atomQueryTrees.clear();
  queryInstructions.clear();
  queryLeafMasks.clear();
  queryLeafBondCounts.clear();
  atomInstrStarts.clear();
  atomLeafMaskStarts.clear();
  recursivePatterns.clear();
}

void MoleculesDevice::setStream(cudaStream_t stream) {
  stream_ = stream;
  batchAtomStarts_.setStream(stream);
  atomDataPacked_.setStream(stream);
  atomQueryMasks_.setStream(stream);
  bondTypeCounts_.setStream(stream);
  targetAtomBonds_.setStream(stream);
  queryAtomBonds_.setStream(stream);
  atomQueryTrees_.setStream(stream);
  queryInstructions_.setStream(stream);
  queryLeafMasks_.setStream(stream);
  queryLeafBondCounts_.setStream(stream);
  atomInstrStarts_.setStream(stream);
  atomLeafMaskStarts_.setStream(stream);
}

namespace {

template <typename T> void setFromVectorGrowOnly(AsyncDeviceVector<T>& dest, const std::vector<T>& src) {
  if (src.empty()) {
    return;
  }
  if (src.size() > dest.size()) {
    dest.resize(static_cast<size_t>(src.size() * 1.5));
  }
  dest.copyFromHost(src, src.size());
}

}  // namespace

void MoleculesDevice::copyFromHost(const MoleculesHost& host, cudaStream_t stream) {
  if (host.numMolecules() == 0) {
    throw std::invalid_argument("Cannot copy empty MoleculesHost to device");
  }

  setStream(stream);
  numMolecules_ = static_cast<int>(host.numMolecules());

  setFromVectorGrowOnly(batchAtomStarts_, host.batchAtomStarts);

  // Copy GPU-optimized packed data
  setFromVectorGrowOnly(atomDataPacked_, host.atomDataPacked);
  if (!host.atomQueryMasks.empty()) {
    setFromVectorGrowOnly(atomQueryMasks_, host.atomQueryMasks);
  }
  if (!host.bondTypeCounts.empty()) {
    setFromVectorGrowOnly(bondTypeCounts_, host.bondTypeCounts);
  }
  if (!host.targetAtomBonds.empty()) {
    setFromVectorGrowOnly(targetAtomBonds_, host.targetAtomBonds);
  }
  if (!host.queryAtomBonds.empty()) {
    setFromVectorGrowOnly(queryAtomBonds_, host.queryAtomBonds);
  }

  // Copy boolean expression tree data for compound queries
  if (!host.atomQueryTrees.empty()) {
    setFromVectorGrowOnly(atomQueryTrees_, host.atomQueryTrees);
    setFromVectorGrowOnly(queryInstructions_, host.queryInstructions);
    setFromVectorGrowOnly(queryLeafMasks_, host.queryLeafMasks);
    setFromVectorGrowOnly(queryLeafBondCounts_, host.queryLeafBondCounts);
    setFromVectorGrowOnly(atomInstrStarts_, host.atomInstrStarts);
    setFromVectorGrowOnly(atomLeafMaskStarts_, host.atomLeafMaskStarts);
  }
}

MoleculesDeviceView MoleculesDevice::view() const {
  MoleculesDeviceView v;
  v.batchAtomStarts     = batchAtomStarts_.data();
  v.numMolecules        = numMolecules_;
  v.atomDataPacked      = atomDataPacked_.data();
  v.atomQueryMasks      = atomQueryMasks_.data();
  v.bondTypeCounts      = bondTypeCounts_.data();
  v.targetAtomBonds     = targetAtomBonds_.data();
  v.queryAtomBonds      = queryAtomBonds_.data();
  v.atomQueryTrees      = atomQueryTrees_.data();
  v.queryInstructions   = queryInstructions_.data();
  v.queryLeafMasks      = queryLeafMasks_.data();
  v.queryLeafBondCounts = queryLeafBondCounts_.data();
  v.atomInstrStarts     = atomInstrStarts_.data();
  v.atomLeafMaskStarts  = atomLeafMaskStarts_.data();
  return v;
}

void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  ScopedNvtxRange range("addToBatch");
  if (mol->getNumAtoms() > kMaxTargetAtoms) {
    throw std::runtime_error("Target molecule has " + std::to_string(mol->getNumAtoms()) +
                             " atoms, which exceeds the maximum of " + std::to_string(kMaxTargetAtoms));
  }

  const auto* ringInfo = mol->getRingInfo();
  populateTargetMolecule(mol, batch, ringInfo);

  batch.batchAtomStarts.push_back(static_cast<int>(batch.atomDataPacked.size()));
}

}  // namespace nvMolKit
