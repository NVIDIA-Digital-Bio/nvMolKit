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

#include "tfd_common.h"

#include <Geometry/point.h>
#include <GraphMol/Fingerprints/FingerprintGenerator.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nvtx.h"

namespace nvMolKit {

namespace {

//! Get heavy atom neighbors of an atom, optionally excluding one atom
std::vector<const RDKit::Atom*> getHeavyAtomNeighbors(const RDKit::Atom* atom, int excludeIdx = -1) {
  std::vector<const RDKit::Atom*> neighbors;
  for (const auto* neighbor : atom->getOwningMol().atomNeighbors(atom)) {
    if (neighbor->getAtomicNum() != 1 && neighbor->getIdx() != static_cast<unsigned int>(excludeIdx)) {
      neighbors.push_back(neighbor);
    }
  }
  return neighbors;
}

//! Check if all atoms have the same invariant
bool doMatch(const std::vector<int>& inv, const std::vector<const RDKit::Atom*>& atoms) {
  if (atoms.size() < 2) {
    return true;
  }
  int firstInv = inv[atoms[0]->getIdx()];
  for (size_t i = 1; i < atoms.size(); ++i) {
    if (inv[atoms[i]->getIdx()] != firstInv) {
      return false;
    }
  }
  return true;
}

//! Find the atom that is different when two atoms match (for 3 atoms)
const RDKit::Atom* doMatchExcept1(const std::vector<int>& inv, const std::vector<const RDKit::Atom*>& atoms) {
  if (atoms.size() != 3) {
    return nullptr;
  }
  int a1 = atoms[0]->getIdx();
  int a2 = atoms[1]->getIdx();
  int a3 = atoms[2]->getIdx();

  if (inv[a1] == inv[a2] && inv[a1] != inv[a3] && inv[a2] != inv[a3]) {
    return atoms[2];
  } else if (inv[a1] != inv[a2] && inv[a1] == inv[a3] && inv[a2] != inv[a3]) {
    return atoms[1];
  } else if (inv[a1] != inv[a2] && inv[a1] != inv[a3] && inv[a2] == inv[a3]) {
    return atoms[0];
  }
  return nullptr;
}

//! Get atom invariants using Morgan fingerprints at given radius
std::vector<int> getAtomInvariantsWithRadius(const RDKit::ROMol& mol, int radius) {
  std::vector<int> inv(mol.getNumAtoms(), 0);

  auto fpGen = RDKit::MorganFingerprint::getMorganGenerator<std::uint32_t>(radius,
                                                                           true /* countSimulation */,
                                                                           false /* includeChirality */,
                                                                           false /* useBondTypes */,
                                                                           false /* onlyNonzeroInvariants */,
                                                                           true /* includeRedundantEnvironments */);

  RDKit::AdditionalOutput ao;
  ao.allocateBitInfoMap();

  // Call to populate ao.bitInfoMap; result is unused
  (void)fpGen->getSparseCountFingerprint(mol, nullptr, nullptr, -1, &ao);
  const auto& bitInfo = ao.bitInfoMap;

  if (bitInfo) {
    for (const auto& [bitId, atomRadiusPairs] : *bitInfo) {
      for (const auto& [atomIdx, r] : atomRadiusPairs) {
        if (r == static_cast<unsigned int>(radius)) {
          inv[atomIdx] = static_cast<int>(bitId);
        }
      }
    }
  }

  return inv;
}

//! Get reference atoms for torsion based on neighbor symmetry
std::vector<const RDKit::Atom*> getIndexForTorsion(const std::vector<const RDKit::Atom*>& neighbors,
                                                   const std::vector<int>&                inv) {
  if (neighbors.size() == 1) {
    return neighbors;
  } else if (doMatch(inv, neighbors)) {
    // All symmetric neighbors - return all
    return neighbors;
  } else if (neighbors.size() == 3) {
    const RDKit::Atom* different = doMatchExcept1(inv, neighbors);
    if (different) {
      return {different};
    }
  }
  // Fallback: sort and take first
  auto sorted = neighbors;
  std::sort(sorted.begin(), sorted.end(), [&inv](const RDKit::Atom* a, const RDKit::Atom* b) {
    return inv[a->getIdx()] < inv[b->getIdx()];
  });
  return {sorted[0]};
}

//! Information about a rotatable bond for torsion calculation
struct BondInfo {
  int                             a1;   // First central atom
  int                             a2;   // Second central atom
  std::vector<const RDKit::Atom*> nb1;  // Heavy atom neighbors of a1 (excluding a2)
  std::vector<const RDKit::Atom*> nb2;  // Heavy atom neighbors of a2 (excluding a1)
};

//! Get bonds for which torsions should be calculated
std::vector<BondInfo> getBondsForTorsions(const RDKit::ROMol& mol, bool ignoreColinearBonds) {
  // Flag atoms that cannot be middle atoms of torsion (triple bonds, allenes)
  std::vector<int> atomFlags(mol.getNumAtoms(), 0);

  // Pattern for triple bonds
  auto triplePattern = RDKit::SmartsToMol("*#*");
  if (triplePattern) {
    std::vector<RDKit::MatchVectType> matches;
    RDKit::SubstructMatch(mol, *triplePattern, matches);
    for (const auto& match : matches) {
      for (const auto& [_, atomIdx] : match) {
        atomFlags[atomIdx] = 1;
      }
    }
  }

  // Pattern for allenes
  auto allenePattern = RDKit::SmartsToMol("[$([C](=*)=*)]");
  if (allenePattern) {
    std::vector<RDKit::MatchVectType> matches;
    RDKit::SubstructMatch(mol, *allenePattern, matches);
    for (const auto& match : matches) {
      for (const auto& [_, atomIdx] : match) {
        atomFlags[atomIdx] = 1;
      }
    }
  }

  std::vector<BondInfo> bonds;
  std::vector<int>      doneBonds(mol.getNumBonds(), 0);

  const auto* ringInfo = mol.getRingInfo();
  for (const auto* bond : mol.bonds()) {
    if (ringInfo->numBondRings(bond->getIdx()) > 0) {
      continue;
    }

    int a1 = bond->getBeginAtomIdx();
    int a2 = bond->getEndAtomIdx();

    auto nb1 = getHeavyAtomNeighbors(bond->getBeginAtom(), a2);
    auto nb2 = getHeavyAtomNeighbors(bond->getEndAtom(), a1);

    if (!doneBonds[bond->getIdx()] && !nb1.empty() && !nb2.empty()) {
      doneBonds[bond->getIdx()] = 1;

      // Check if atoms cannot be middle atoms
      if (atomFlags[a1] || atomFlags[a2]) {
        if (!ignoreColinearBonds) {
          // Search for alternative atoms (following the Python logic)
          while (nb1.size() == 1 && atomFlags[a1]) {
            int a1old = a1;
            a1        = nb1[0]->getIdx();
            auto* b   = mol.getBondBetweenAtoms(a1old, a1);
            if (b) {
              if (b->getEndAtomIdx() == static_cast<unsigned int>(a1old)) {
                nb1 = getHeavyAtomNeighbors(b->getBeginAtom(), a1old);
              } else {
                nb1 = getHeavyAtomNeighbors(b->getEndAtom(), a1old);
              }
              doneBonds[b->getIdx()] = 1;
            } else {
              break;
            }
          }
          while (nb2.size() == 1 && atomFlags[a2]) {
            int a2old = a2;
            a2        = nb2[0]->getIdx();
            auto* b   = mol.getBondBetweenAtoms(a2old, a2);
            if (b) {
              if (b->getBeginAtomIdx() == static_cast<unsigned int>(a2old)) {
                nb2 = getHeavyAtomNeighbors(b->getEndAtom(), a2old);
              } else {
                nb2 = getHeavyAtomNeighbors(b->getBeginAtom(), a2old);
              }
              doneBonds[b->getIdx()] = 1;
            } else {
              break;
            }
          }
          if (!nb1.empty() && !nb2.empty()) {
            bonds.push_back({a1, a2, nb1, nb2});
          }
        }
        // If ignoreColinearBonds is true, we skip this bond
      } else {
        bonds.push_back({a1, a2, nb1, nb2});
      }
    }
  }

  return bonds;
}

//! Find the most central bond in a molecule
//! Returns {-1, -1} if no central bond can be found (e.g., methane, linear molecules)
std::pair<int, int> findCentralBond(const RDKit::ROMol& mol, const double* distMat) {
  int numAtoms = mol.getNumAtoms();

  // Calculate STD of distances for each non-terminal atom
  std::vector<std::pair<double, int>> stds;
  for (int i = 0; i < numAtoms; ++i) {
    auto neighbors = getHeavyAtomNeighbors(mol.getAtomWithIdx(i));
    if (neighbors.size() < 2) {
      continue;  // Skip terminal atoms
    }

    // Calculate STD of distances
    double sum   = 0.0;
    double sumSq = 0.0;
    int    count = 0;
    for (int j = 0; j < numAtoms; ++j) {
      if (j != i) {
        double d = distMat[i * numAtoms + j];
        sum += d;
        sumSq += d * d;
        count++;
      }
    }
    double mean     = sum / count;
    double variance = (sumSq / count) - (mean * mean);
    double stdDev   = std::sqrt(std::max(0.0, variance));
    stds.emplace_back(stdDev, i);
  }

  if (stds.empty()) {
    return {-1, -1};  // No non-terminal atoms found
  }

  std::sort(stds.begin(), stds.end());
  int aid1 = stds[0].second;

  // Find second most central atom that is bonded to aid1
  for (size_t i = 1; i < stds.size(); ++i) {
    if (mol.getBondBetweenAtoms(aid1, stds[i].second) != nullptr) {
      return {aid1, stds[i].second};
    }
  }

  return {-1, -1};  // Could not find central bond
}

//! Calculate beta parameter for weight calculation
double calculateBeta(const RDKit::ROMol& mol, const double* distMat, int aid1) {
  int numAtoms = mol.getNumAtoms();

  // Get all non-terminal bonds
  // NOTE: RDKit has a typo in _calculateBeta (TorsionFingerprints.py ~line 391):
  //   `if len(nb2) > 1 and len(nb2) > 1` checks nb2 twice instead of nb1 and nb2.
  //   This includes bonds where only the end atom is non-terminal, inflating dmax.
  //   We replicate this behavior for RDKit compatibility.
  // TODO: Fix once RDKit corrects this, or add a flag for "correct" behavior.
  double dmax = 0.0;
  for (const auto* bond : mol.bonds()) {
    auto nb2 = getHeavyAtomNeighbors(bond->getEndAtom());
    if (nb2.size() > 1 && nb2.size() > 1) {
      int    bid1 = bond->getBeginAtomIdx();
      int    bid2 = bond->getEndAtomIdx();
      double d    = std::max(distMat[aid1 * numAtoms + bid1], distMat[aid1 * numAtoms + bid2]);
      dmax        = std::max(dmax, d);
    }
  }

  double dmax2 = dmax / 2.0;
  if (dmax2 < 1e-6) {
    dmax2 = 1.0;  // Avoid division by zero
  }
  double beta = -std::log(0.1) / (dmax2 * dmax2);
  return beta;
}

}  // namespace

TorsionList extractTorsionList(const RDKit::ROMol& mol,
                               TFDMaxDevMode       maxDevMode,
                               int                 symmRadius,
                               bool                ignoreColinearBonds) {
  TorsionList result;

  // Get bonds for torsions
  auto bonds = getBondsForTorsions(mol, ignoreColinearBonds);

  // Get atom invariants
  std::vector<int> inv;
  if (symmRadius > 0) {
    inv = getAtomInvariantsWithRadius(mol, symmRadius);
  } else {
    // Use connectivity invariants as fallback
    inv.resize(mol.getNumAtoms());
    for (unsigned int i = 0; i < mol.getNumAtoms(); ++i) {
      inv[i] = mol.getAtomWithIdx(i)->getDegree();
    }
  }

  // Process each bond to create torsions
  for (const auto& bond : bonds) {
    auto d1 = getIndexForTorsion(bond.nb1, inv);
    auto d2 = getIndexForTorsion(bond.nb2, inv);

    TorsionDef torsion;

    if (d1.size() == 1 && d2.size() == 1) {
      // Case 1, 2, 4, 5, 7, 10, 16, 12, 17, 19 - single torsion
      torsion.atomQuartets.push_back(
        {static_cast<int>(d1[0]->getIdx()), bond.a1, bond.a2, static_cast<int>(d2[0]->getIdx())});
      torsion.maxDev = 180.0f;
    } else if (d1.size() == 1) {
      // Case 3, 6, 8, 13, 20 - multiple torsions from d2
      for (const auto* nb : d2) {
        torsion.atomQuartets.push_back(
          {static_cast<int>(d1[0]->getIdx()), bond.a1, bond.a2, static_cast<int>(nb->getIdx())});
      }
      torsion.maxDev = (bond.nb2.size() == 2) ? 90.0f : 60.0f;
    } else if (d2.size() == 1) {
      // Case 3, 6, 8, 13, 20 - multiple torsions from d1
      for (const auto* nb : d1) {
        torsion.atomQuartets.push_back(
          {static_cast<int>(nb->getIdx()), bond.a1, bond.a2, static_cast<int>(d2[0]->getIdx())});
      }
      torsion.maxDev = (bond.nb1.size() == 2) ? 90.0f : 60.0f;
    } else {
      // Both symmetric - all combinations
      for (const auto* n1 : d1) {
        for (const auto* n2 : d2) {
          torsion.atomQuartets.push_back(
            {static_cast<int>(n1->getIdx()), bond.a1, bond.a2, static_cast<int>(n2->getIdx())});
        }
      }
      if (bond.nb1.size() == 2 && bond.nb2.size() == 2) {
        torsion.maxDev = 90.0f;
      } else if (bond.nb1.size() == 3 && bond.nb2.size() == 3) {
        torsion.maxDev = 60.0f;
      } else {
        torsion.maxDev = 30.0f;
      }
    }

    // Override with equal mode if requested
    if (maxDevMode == TFDMaxDevMode::Equal) {
      torsion.maxDev = 180.0f;
    }

    result.nonRingTorsions.push_back(std::move(torsion));
  }

  // Process rings
  auto rings = mol.getRingInfo()->atomRings();
  for (const auto& ring : rings) {
    TorsionDef torsion;
    int        num = static_cast<int>(ring.size());

    // Calculate max deviation for ring
    float maxdev;
    if (num >= 14) {
      maxdev = 180.0f;
    } else {
      maxdev = 180.0f * std::exp(-0.025f * (num - 14) * (num - 14));
    }

    // Create torsions for ring (consecutive 4 atoms)
    for (int i = 0; i < num; ++i) {
      torsion.atomQuartets.push_back({ring[i], ring[(i + 1) % num], ring[(i + 2) % num], ring[(i + 3) % num]});
    }
    torsion.maxDev = maxdev;

    result.ringTorsions.push_back(std::move(torsion));
  }

  return result;
}

std::vector<float> computeTorsionWeights(const RDKit::ROMol& mol,
                                         const TorsionList&  torsionList,
                                         bool                ignoreColinearBonds) {
  std::vector<float> weights;

  // If no torsions, return empty weights
  size_t totalTorsions = torsionList.totalCount();
  if (totalTorsions == 0) {
    return weights;
  }

  // Get distance matrix (returns raw pointer, stored in mol's dictionary)
  const double* distMat  = RDKit::MolOps::getDistanceMat(mol);
  int           numAtoms = mol.getNumAtoms();

  // Find central bond
  auto [aid1, aid2] = findCentralBond(mol, distMat);

  // If no central bond found, return uniform weights
  if (aid1 < 0 || aid2 < 0) {
    weights.resize(totalTorsions, 1.0f);
    return weights;
  }

  // Calculate beta
  double beta = calculateBeta(mol, distMat, aid1);

  // Get bonds for torsions (same as used in extractTorsionList)
  auto bonds = getBondsForTorsions(mol, ignoreColinearBonds);

  // Calculate weights for non-ring torsions
  for (size_t i = 0; i < bonds.size(); ++i) {
    const auto& bond = bonds[i];
    double      d;

    if ((bond.a1 == aid1 && bond.a2 == aid2) || (bond.a1 == aid2 && bond.a2 == aid1)) {
      d = 0.0;  // Central bond itself
    } else {
      // Shortest distance to central bond atoms + 1
      d = std::min({distMat[aid1 * numAtoms + bond.a1],
                    distMat[aid1 * numAtoms + bond.a2],
                    distMat[aid2 * numAtoms + bond.a1],
                    distMat[aid2 * numAtoms + bond.a2]}) +
          1.0;
    }

    float w = static_cast<float>(std::exp(-beta * d * d));
    weights.push_back(w);
  }

  // Calculate weights for ring torsions
  auto ringInfo  = mol.getRingInfo();
  auto bondRings = ringInfo->bondRings();

  for (const auto& bondRing : bondRings) {
    int    num  = static_cast<int>(bondRing.size());
    double sumD = 0.0;

    for (int bidx : bondRing) {
      const auto* bond = mol.getBondWithIdx(bidx);
      int         bid1 = bond->getBeginAtomIdx();
      int         bid2 = bond->getEndAtomIdx();
      double      d    = std::min({distMat[aid1 * numAtoms + bid1],
                                   distMat[aid1 * numAtoms + bid2],
                                   distMat[aid2 * numAtoms + bid1],
                                   distMat[aid2 * numAtoms + bid2]}) +
                 1.0;
      sumD += d;
    }

    double avgD = sumD / num;
    float  w    = static_cast<float>(std::exp(-beta * avgD * avgD) * (num / 2.0));
    weights.push_back(w);
  }

  return weights;
}

// Sequential single-molecule builder (used as the building block for parallel batch builds)
static TFDSystemHost buildTFDSystemImpl(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  TFDSystemHost system;

  int numConformers = mol.getNumConformers();
  int numAtoms      = mol.getNumAtoms();

  if (numConformers == 0) {
    throw std::runtime_error("Molecule has no conformers");
  }

  // Extract torsion list
  TorsionList torsionList =
    extractTorsionList(mol, options.maxDevMode, options.symmRadius, options.ignoreColinearBonds);

  // Extract weights if needed
  std::vector<float> weights;
  if (options.useWeights) {
    weights = computeTorsionWeights(mol, torsionList, options.ignoreColinearBonds);
  }

  // Capture CSR starts before pushing (for building flattened work items below)
  int confStart = system.molConformerStarts.back();
  int torsStart = system.molTorsionStarts.back();

  // Update CSR indices
  system.molConformerStarts.push_back(confStart + numConformers);

  int numTorsions = static_cast<int>(torsionList.totalCount());
  system.molTorsionStarts.push_back(torsStart + numTorsions);

  // Remember quartet start for this molecule (before adding quartets)
  int quartetStartForMol = system.totalQuartets();

  int numTFDOutputs = numConformers * (numConformers - 1) / 2;
  system.tfdOutputStarts.push_back(system.tfdOutputStarts.back() + numTFDOutputs);

  // Extract coordinates (tightly packed, no padding)
  for (auto confIt = mol.beginConformers(); confIt != mol.endConformers(); ++confIt) {
    system.confPositionStarts.push_back(static_cast<int>(system.positions.size()));
    const auto& conf = **confIt;
    for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
      const auto& pos = conf.getAtomPos(atomIdx);
      system.positions.push_back(static_cast<float>(pos.x));
      system.positions.push_back(static_cast<float>(pos.y));
      system.positions.push_back(static_cast<float>(pos.z));
    }
  }

  // Add torsion definitions (store ALL quartets, classify type)
  int torsionIdx = 0;

  // Non-ring torsions
  for (const auto& torsion : torsionList.nonRingTorsions) {
    if (torsion.atomQuartets.empty()) {
      torsionIdx++;
      continue;
    }

    for (const auto& q : torsion.atomQuartets) {
      system.torsionAtoms.push_back(q);
    }
    system.quartetStarts.push_back(static_cast<int>(system.torsionAtoms.size()));

    if (torsion.atomQuartets.size() > 1) {
      system.torsionTypes.push_back(TorsionType::Symmetric);
      system.hasMultiQuartet = true;
    } else {
      system.torsionTypes.push_back(TorsionType::Single);
    }

    system.torsionMaxDevs.push_back(torsion.maxDev);

    if (options.useWeights && torsionIdx < static_cast<int>(weights.size())) {
      system.torsionWeights.push_back(weights[torsionIdx]);
    } else {
      system.torsionWeights.push_back(1.0f);
    }
    torsionIdx++;
  }

  // Ring torsions
  for (const auto& torsion : torsionList.ringTorsions) {
    if (torsion.atomQuartets.empty()) {
      torsionIdx++;
      continue;
    }

    for (const auto& q : torsion.atomQuartets) {
      system.torsionAtoms.push_back(q);
    }
    system.quartetStarts.push_back(static_cast<int>(system.torsionAtoms.size()));

    if (torsion.atomQuartets.size() > 1) {
      system.torsionTypes.push_back(TorsionType::Ring);
      system.hasMultiQuartet = true;
    } else {
      system.torsionTypes.push_back(TorsionType::Single);
    }

    system.torsionMaxDevs.push_back(torsion.maxDev);

    if (options.useWeights && torsionIdx < static_cast<int>(weights.size())) {
      system.torsionWeights.push_back(weights[torsionIdx]);
    } else {
      system.torsionWeights.push_back(1.0f);
    }
    torsionIdx++;
  }

  // Dihedral storage: numConformers * totalQuartetsForMol per molecule
  int quartetEndForMol    = system.totalQuartets();
  int totalQuartetsForMol = quartetEndForMol - quartetStartForMol;
  int dihedStart          = system.molDihedralStarts.back();
  int numDihedrals        = numConformers * totalQuartetsForMol;
  system.molDihedralStarts.push_back(dihedStart + numDihedrals);

  // Build flattened dihedral work items for this molecule
  // One work item per (conformer, quartet) pair
  for (int c = 0; c < numConformers; ++c) {
    for (int q = 0; q < totalQuartetsForMol; ++q) {
      system.dihedralConfIdx.push_back(confStart + c);
      system.dihedralTorsIdx.push_back(quartetStartForMol + q);
      system.dihedralOutIdx.push_back(dihedStart + c * totalQuartetsForMol + q);
    }
  }

  // Build flattened TFD pair work items for this molecule
  // One work item per conformer pair (i > j), lower triangular order
  // tfdAnglesI/J point to the start of the quartet-angle block for each conformer
  int outBase = system.tfdOutputStarts[system.tfdOutputStarts.size() - 2];
  for (int i = 1; i < numConformers; ++i) {
    for (int j = 0; j < i; ++j) {
      system.tfdAnglesI.push_back(dihedStart + i * totalQuartetsForMol);
      system.tfdAnglesJ.push_back(dihedStart + j * totalQuartetsForMol);
      system.tfdTorsStart.push_back(torsStart);
      system.tfdNumTorsions.push_back(numTorsions);
      system.tfdOutIdx.push_back(outBase + i * (i - 1) / 2 + j);
    }
  }

  return system;
}

TFDSystemHost buildTFDSystem(const std::vector<const RDKit::ROMol*>& mols, const TFDComputeOptions& options) {
  ScopedNvtxRange range("buildTFDSystem (" + std::to_string(mols.size()) + " mols)", NvtxColor::kCyan);

  if (mols.empty()) {
    return {};
  }
  if (mols.size() == 1) {
    return buildTFDSystemImpl(*mols[0], options);
  }

  // Build per-molecule systems in parallel (RDKit extraction — the expensive part)
  std::vector<TFDSystemHost> perMol(mols.size());

  {
    ScopedNvtxRange buildRange("Parallel RDKit extraction", NvtxColor::kCyan);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < mols.size(); ++i) {
      perMol[i] = buildTFDSystemImpl(*mols[i], options);
    }
  }

  // Merge into single batched system (index arithmetic only — fast)
  ScopedNvtxRange mergeRange("mergeTFDSystems", NvtxColor::kCyan);
  return mergeTFDSystems(perMol);
}

TFDSystemHost buildTFDSystem(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  return buildTFDSystemImpl(mol, options);
}

TFDSystemHost mergeTFDSystems(std::vector<TFDSystemHost>& systems) {
  int N = static_cast<int>(systems.size());
  if (N == 0) {
    return {};
  }
  if (N == 1) {
    return std::move(systems[0]);
  }

  // Compute cumulative offsets across all per-molecule systems
  std::vector<int> confOffset(N);
  std::vector<int> torsOffset(N);
  std::vector<int> quartetOffset(N);
  std::vector<int> dihedOffset(N);
  std::vector<int> tfdOutOffset(N);
  std::vector<int> posOffset(N);

  int totalConfs             = 0;
  int totalTors              = 0;
  int totalQuartets          = 0;
  int totalDiheds            = 0;
  int totalTfdOuts           = 0;
  int totalPos               = 0;
  int totalDihedralWorkItems = 0;
  int totalTFDWorkItems      = 0;
  int totalConfPositions     = 0;

  for (int i = 0; i < N; ++i) {
    confOffset[i]    = totalConfs;
    torsOffset[i]    = totalTors;
    quartetOffset[i] = totalQuartets;
    dihedOffset[i]   = totalDiheds;
    tfdOutOffset[i]  = totalTfdOuts;
    posOffset[i]     = totalPos;

    totalConfs += systems[i].totalConformers();
    totalTors += systems[i].totalTorsions();
    totalQuartets += systems[i].totalQuartets();
    totalDiheds += systems[i].totalDihedrals();
    totalTfdOuts += systems[i].totalTFDOutputs();
    totalPos += static_cast<int>(systems[i].positions.size());
    totalDihedralWorkItems += systems[i].totalDihedralWorkItems();
    totalTFDWorkItems += systems[i].totalTFDWorkItems();
    totalConfPositions += static_cast<int>(systems[i].confPositionStarts.size());
  }

  TFDSystemHost merged;

  // Reserve space to avoid reallocations
  merged.molConformerStarts.reserve(N + 1);
  merged.molTorsionStarts.reserve(N + 1);
  merged.molDihedralStarts.reserve(N + 1);
  merged.tfdOutputStarts.reserve(N + 1);
  merged.quartetStarts.reserve(totalTors + 1);
  merged.positions.reserve(totalPos);
  merged.confPositionStarts.reserve(totalConfPositions);
  merged.torsionAtoms.reserve(totalQuartets);
  merged.torsionWeights.reserve(totalTors);
  merged.torsionMaxDevs.reserve(totalTors);
  merged.torsionTypes.reserve(totalTors);
  merged.dihedralConfIdx.reserve(totalDihedralWorkItems);
  merged.dihedralTorsIdx.reserve(totalDihedralWorkItems);
  merged.dihedralOutIdx.reserve(totalDihedralWorkItems);
  merged.tfdAnglesI.reserve(totalTFDWorkItems);
  merged.tfdAnglesJ.reserve(totalTFDWorkItems);
  merged.tfdTorsStart.reserve(totalTFDWorkItems);
  merged.tfdNumTorsions.reserve(totalTFDWorkItems);
  merged.tfdOutIdx.reserve(totalTFDWorkItems);

  for (int i = 0; i < N; ++i) {
    auto& s = systems[i];

    if (s.numMolecules() == 0) {
      continue;
    }

    // CSR arrays: append final value from each single-molecule system with offset
    merged.molConformerStarts.push_back(confOffset[i] + s.molConformerStarts.back());
    merged.molTorsionStarts.push_back(torsOffset[i] + s.molTorsionStarts.back());
    merged.molDihedralStarts.push_back(dihedOffset[i] + s.molDihedralStarts.back());
    merged.tfdOutputStarts.push_back(tfdOutOffset[i] + s.tfdOutputStarts.back());

    // quartetStarts CSR: skip leading 0, add quartet offset
    for (size_t j = 1; j < s.quartetStarts.size(); ++j) {
      merged.quartetStarts.push_back(quartetOffset[i] + s.quartetStarts[j]);
    }

    // Data arrays: concatenate directly (no offset needed)
    merged.positions.insert(merged.positions.end(), s.positions.begin(), s.positions.end());
    merged.torsionAtoms.insert(merged.torsionAtoms.end(), s.torsionAtoms.begin(), s.torsionAtoms.end());
    merged.torsionWeights.insert(merged.torsionWeights.end(), s.torsionWeights.begin(), s.torsionWeights.end());
    merged.torsionMaxDevs.insert(merged.torsionMaxDevs.end(), s.torsionMaxDevs.begin(), s.torsionMaxDevs.end());
    merged.torsionTypes.insert(merged.torsionTypes.end(), s.torsionTypes.begin(), s.torsionTypes.end());

    // confPositionStarts: add position offset
    for (int idx : s.confPositionStarts) {
      merged.confPositionStarts.push_back(posOffset[i] + idx);
    }

    // Dihedral work items: add respective offsets
    for (int idx : s.dihedralConfIdx) {
      merged.dihedralConfIdx.push_back(confOffset[i] + idx);
    }
    for (int idx : s.dihedralTorsIdx) {
      merged.dihedralTorsIdx.push_back(quartetOffset[i] + idx);
    }
    for (int idx : s.dihedralOutIdx) {
      merged.dihedralOutIdx.push_back(dihedOffset[i] + idx);
    }

    // TFD pair work items: add respective offsets
    for (int idx : s.tfdAnglesI) {
      merged.tfdAnglesI.push_back(dihedOffset[i] + idx);
    }
    for (int idx : s.tfdAnglesJ) {
      merged.tfdAnglesJ.push_back(dihedOffset[i] + idx);
    }
    for (int idx : s.tfdTorsStart) {
      merged.tfdTorsStart.push_back(torsOffset[i] + idx);
    }
    // tfdNumTorsions: counts, no offset needed
    merged.tfdNumTorsions.insert(merged.tfdNumTorsions.end(), s.tfdNumTorsions.begin(), s.tfdNumTorsions.end());
    for (int idx : s.tfdOutIdx) {
      merged.tfdOutIdx.push_back(tfdOutOffset[i] + idx);
    }

    merged.hasMultiQuartet = merged.hasMultiQuartet || s.hasMultiQuartet;
  }

  return merged;
}

}  // namespace nvMolKit
