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

#include <cassert>
#include <vector>

#include "device_vector.h"
#include "dist_geom.h"
#include "dist_geom_kernels.h"
#include "dist_geom_kernels_device.cuh"
#include "kernel_utils.cuh"
#include "nvtx.h"

namespace nvMolKit {
namespace DistGeom {

void addMoleculeToContext(int                  dimension,
                          int                  numAtoms,
                          int&                 nTotalSystems,
                          std::vector<int>&    ctxAtomStarts,
                          std::vector<double>& ctxPositions) {
  nTotalSystems++;
  ctxAtomStarts.push_back(ctxAtomStarts.back() + numAtoms);
  ctxPositions.insert(ctxPositions.end(), numAtoms * dimension, 0.0);
}

void addMoleculeToContextWithPositions(const std::vector<double>& positions,
                                       int                        dimension,
                                       std::vector<int>&          ctxAtomStarts,
                                       std::vector<double>&       ctxPositions) {
  const int numAtoms = positions.size() / dimension;
  ctxPositions.insert(ctxPositions.end(), positions.begin(), positions.end());
  ctxAtomStarts.push_back(ctxAtomStarts.back() + numAtoms);
}

void preallocateEstimatedBatch(const EnergyForceContribsHost& templateContribs,
                               BatchedMolecularSystemHost&    molSystem,
                               const int                      estimatedBatchSize) {
  const size_t bufferMultiplier = static_cast<size_t>(estimatedBatchSize * 1.2);

  auto& contribHolder = molSystem.contribs;

  // Preallocate DistViolation terms
  const size_t numDistTerms = templateContribs.distTerms.idx1.size();
  contribHolder.distTerms.idx1.reserve(numDistTerms * bufferMultiplier);
  contribHolder.distTerms.idx2.reserve(numDistTerms * bufferMultiplier);
  contribHolder.distTerms.lb2.reserve(numDistTerms * bufferMultiplier);
  contribHolder.distTerms.ub2.reserve(numDistTerms * bufferMultiplier);
  contribHolder.distTerms.weight.reserve(numDistTerms * bufferMultiplier);

  // Preallocate ChiralViolation terms
  const size_t numChiralTerms = templateContribs.chiralTerms.idx1.size();
  contribHolder.chiralTerms.idx1.reserve(numChiralTerms * bufferMultiplier);
  contribHolder.chiralTerms.idx2.reserve(numChiralTerms * bufferMultiplier);
  contribHolder.chiralTerms.idx3.reserve(numChiralTerms * bufferMultiplier);
  contribHolder.chiralTerms.idx4.reserve(numChiralTerms * bufferMultiplier);
  contribHolder.chiralTerms.volLower.reserve(numChiralTerms * bufferMultiplier);
  contribHolder.chiralTerms.volUpper.reserve(numChiralTerms * bufferMultiplier);

  // Preallocate FourthDim terms
  const size_t numFourthTerms = templateContribs.fourthTerms.idx.size();
  contribHolder.fourthTerms.idx.reserve(numFourthTerms * bufferMultiplier);

  // Preallocate index vectors
  auto& indexHolder = molSystem.indices;
  indexHolder.distTermStarts.reserve(estimatedBatchSize + 1);
  indexHolder.chiralTermStarts.reserve(estimatedBatchSize + 1);
  indexHolder.fourthTermStarts.reserve(estimatedBatchSize + 1);
  indexHolder.energyBufferStarts.reserve(estimatedBatchSize + 1);
}

void preallocateEstimatedBatch3D(const Energy3DForceContribsHost& templateContribs,
                                 BatchedMolecularSystem3DHost&    molSystem,
                                 const int                        estimatedBatchSize) {
  const size_t bufferMultiplier = static_cast<size_t>(estimatedBatchSize * 1.2);

  auto& contribHolder = molSystem.contribs;

  // Preallocate experimental torsion terms
  const size_t numExpTorsionTerms = templateContribs.experimentalTorsionTerms.idx1.size();
  contribHolder.experimentalTorsionTerms.idx1.reserve(numExpTorsionTerms * bufferMultiplier);
  contribHolder.experimentalTorsionTerms.idx2.reserve(numExpTorsionTerms * bufferMultiplier);
  contribHolder.experimentalTorsionTerms.idx3.reserve(numExpTorsionTerms * bufferMultiplier);
  contribHolder.experimentalTorsionTerms.idx4.reserve(numExpTorsionTerms * bufferMultiplier);
  contribHolder.experimentalTorsionTerms.forceConstants.reserve(numExpTorsionTerms * 6 * bufferMultiplier);
  contribHolder.experimentalTorsionTerms.signs.reserve(numExpTorsionTerms * 6 * bufferMultiplier);

  // Preallocate improper torsion terms
  const size_t numImproperTerms = templateContribs.improperTorsionTerms.idx1.size();
  contribHolder.improperTorsionTerms.idx1.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.idx2.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.idx3.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.idx4.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.at2AtomicNum.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.isCBoundToO.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.C0.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.C1.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.C2.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.forceConstant.reserve(numImproperTerms * bufferMultiplier);
  contribHolder.improperTorsionTerms.numImpropers.reserve(estimatedBatchSize);

  // Preallocate 1-2 distance terms
  const size_t numDist12Terms = templateContribs.dist12Terms.idx1.size();
  contribHolder.dist12Terms.idx1.reserve(numDist12Terms * bufferMultiplier);
  contribHolder.dist12Terms.idx2.reserve(numDist12Terms * bufferMultiplier);
  contribHolder.dist12Terms.minLen.reserve(numDist12Terms * bufferMultiplier);
  contribHolder.dist12Terms.maxLen.reserve(numDist12Terms * bufferMultiplier);
  contribHolder.dist12Terms.forceConstant.reserve(numDist12Terms * bufferMultiplier);

  // Preallocate 1-3 distance terms
  const size_t numDist13Terms = templateContribs.dist13Terms.idx1.size();
  contribHolder.dist13Terms.idx1.reserve(numDist13Terms * bufferMultiplier);
  contribHolder.dist13Terms.idx2.reserve(numDist13Terms * bufferMultiplier);
  contribHolder.dist13Terms.minLen.reserve(numDist13Terms * bufferMultiplier);
  contribHolder.dist13Terms.maxLen.reserve(numDist13Terms * bufferMultiplier);
  contribHolder.dist13Terms.forceConstant.reserve(numDist13Terms * bufferMultiplier);
  contribHolder.dist13Terms.isImproperConstrained.reserve(numDist13Terms * bufferMultiplier);

  // Preallocate 1-3 angle terms
  const size_t numAngle13Terms = templateContribs.angle13Terms.idx1.size();
  contribHolder.angle13Terms.idx1.reserve(numAngle13Terms * bufferMultiplier);
  contribHolder.angle13Terms.idx2.reserve(numAngle13Terms * bufferMultiplier);
  contribHolder.angle13Terms.idx3.reserve(numAngle13Terms * bufferMultiplier);
  contribHolder.angle13Terms.minAngle.reserve(numAngle13Terms * bufferMultiplier);
  contribHolder.angle13Terms.maxAngle.reserve(numAngle13Terms * bufferMultiplier);

  // Preallocate long range distance terms
  const size_t numLongRangeTerms = templateContribs.longRangeDistTerms.idx1.size();
  contribHolder.longRangeDistTerms.idx1.reserve(numLongRangeTerms * bufferMultiplier);
  contribHolder.longRangeDistTerms.idx2.reserve(numLongRangeTerms * bufferMultiplier);
  contribHolder.longRangeDistTerms.minLen.reserve(numLongRangeTerms * bufferMultiplier);
  contribHolder.longRangeDistTerms.maxLen.reserve(numLongRangeTerms * bufferMultiplier);
  contribHolder.longRangeDistTerms.forceConstant.reserve(numLongRangeTerms * bufferMultiplier);

  // Preallocate index vectors
  auto& indexHolder = molSystem.indices;
  indexHolder.experimentalTorsionTermStarts.reserve(estimatedBatchSize + 1);
  indexHolder.improperTorsionTermStarts.reserve(estimatedBatchSize + 1);
  indexHolder.dist12TermStarts.reserve(estimatedBatchSize + 1);
  indexHolder.dist13TermStarts.reserve(estimatedBatchSize + 1);
  indexHolder.angle13TermStarts.reserve(estimatedBatchSize + 1);
  indexHolder.longRangeDistTermStarts.reserve(estimatedBatchSize + 1);
  indexHolder.energyBufferStarts.reserve(estimatedBatchSize + 1);
}

void addMoleculeToMolecularSystem(const EnergyForceContribsHost& contribs,
                                  const int                      numAtoms,
                                  const int                      dimension,
                                  const std::vector<int>&        ctxAtomStarts,
                                  BatchedMolecularSystemHost&    molSystem) {
  // Use distTermStarts.size() - 1 to get the current batch index
  const int batchIdx              = molSystem.indices.distTermStarts.size() - 1;
  // Get the previous last atom index from ctxAtomStarts using the current batch index
  const int previousLastAtomIndex = ctxAtomStarts[batchIdx];

  auto& indexHolder   = molSystem.indices;
  auto& contribHolder = molSystem.contribs;

  // Update max number of atoms
  molSystem.maxNumAtoms = std::max(molSystem.maxNumAtoms, numAtoms);
  // Set dimension if this is the first molecule
  if (batchIdx == 0) {
    molSystem.dimension = dimension;
  } else {
    // Ensure all molecules have the same dimension
    assert(molSystem.dimension == dimension);
  }

  // Resize atomIdxToBatchIdx using the next atom start index
  indexHolder.atomIdxToBatchIdx.resize(ctxAtomStarts[batchIdx + 1], batchIdx);

  // Update term starts
  indexHolder.distTermStarts.push_back(indexHolder.distTermStarts.back() + contribs.distTerms.idx1.size());
  indexHolder.chiralTermStarts.push_back(indexHolder.chiralTermStarts.back() + contribs.chiralTerms.idx1.size());
  indexHolder.fourthTermStarts.push_back(indexHolder.fourthTermStarts.back() + contribs.fourthTerms.idx.size());

  // Calculate number of blocks needed
  int maxNumContribs = 0;
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.distTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.chiralTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.fourthTerms.idx.size());

  const int numBlocksNeeded  = std::max(1, (maxNumContribs + 127) / nvMolKit::FFKernelUtils::blockSizeEnergyReduction);
  const int numThreadsNeeded = numBlocksNeeded * nvMolKit::FFKernelUtils::blockSizeEnergyReduction;

  indexHolder.energyBufferStarts.push_back(indexHolder.energyBufferStarts.back() + numThreadsNeeded);
  for (int i = 0; i < numBlocksNeeded; i++) {
    indexHolder.energyBufferBlockIdxToBatchIdx.push_back(batchIdx);
  }

  // Update contributions
  // DistViolation term
  const size_t numDistTerms = contribs.distTerms.idx1.size();
  contribHolder.distTerms.idx1.reserve(contribHolder.distTerms.idx1.size() + numDistTerms);
  contribHolder.distTerms.idx2.reserve(contribHolder.distTerms.idx2.size() + numDistTerms);
  contribHolder.distTerms.lb2.reserve(contribHolder.distTerms.lb2.size() + numDistTerms);
  contribHolder.distTerms.ub2.reserve(contribHolder.distTerms.ub2.size() + numDistTerms);
  contribHolder.distTerms.weight.reserve(contribHolder.distTerms.weight.size() + numDistTerms);

  for (size_t i = 0; i < numDistTerms; i++) {
    contribHolder.distTerms.idx1.push_back(contribs.distTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.distTerms.idx2.push_back(contribs.distTerms.idx2[i] + previousLastAtomIndex);
  }
  contribHolder.distTerms.lb2.insert(contribHolder.distTerms.lb2.end(),
                                     contribs.distTerms.lb2.begin(),
                                     contribs.distTerms.lb2.end());
  contribHolder.distTerms.ub2.insert(contribHolder.distTerms.ub2.end(),
                                     contribs.distTerms.ub2.begin(),
                                     contribs.distTerms.ub2.end());
  contribHolder.distTerms.weight.insert(contribHolder.distTerms.weight.end(),
                                        contribs.distTerms.weight.begin(),
                                        contribs.distTerms.weight.end());

  // ChiralViolation term
  const size_t numChiralTerms = contribs.chiralTerms.idx1.size();
  contribHolder.chiralTerms.idx1.reserve(contribHolder.chiralTerms.idx1.size() + numChiralTerms);
  contribHolder.chiralTerms.idx2.reserve(contribHolder.chiralTerms.idx2.size() + numChiralTerms);
  contribHolder.chiralTerms.idx3.reserve(contribHolder.chiralTerms.idx3.size() + numChiralTerms);
  contribHolder.chiralTerms.idx4.reserve(contribHolder.chiralTerms.idx4.size() + numChiralTerms);
  contribHolder.chiralTerms.volLower.reserve(contribHolder.chiralTerms.volLower.size() + numChiralTerms);
  contribHolder.chiralTerms.volUpper.reserve(contribHolder.chiralTerms.volUpper.size() + numChiralTerms);

  for (size_t i = 0; i < numChiralTerms; i++) {
    contribHolder.chiralTerms.idx1.push_back(contribs.chiralTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.chiralTerms.idx2.push_back(contribs.chiralTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.chiralTerms.idx3.push_back(contribs.chiralTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.chiralTerms.idx4.push_back(contribs.chiralTerms.idx4[i] + previousLastAtomIndex);
  }
  contribHolder.chiralTerms.volLower.insert(contribHolder.chiralTerms.volLower.end(),
                                            contribs.chiralTerms.volLower.begin(),
                                            contribs.chiralTerms.volLower.end());
  contribHolder.chiralTerms.volUpper.insert(contribHolder.chiralTerms.volUpper.end(),
                                            contribs.chiralTerms.volUpper.begin(),
                                            contribs.chiralTerms.volUpper.end());
  // Note: weight no longer stored in structure

  // FourthDim term
  const size_t numFourthTerms = contribs.fourthTerms.idx.size();
  contribHolder.fourthTerms.idx.reserve(contribHolder.fourthTerms.idx.size() + numFourthTerms);

  for (size_t i = 0; i < numFourthTerms; i++) {
    contribHolder.fourthTerms.idx.push_back(contribs.fourthTerms.idx[i] + previousLastAtomIndex);
  }
  // Note: weight no longer stored in structure
}

void addMoleculeToMolecularSystem3D(const Energy3DForceContribsHost& contribs,
                                    const std::vector<int>&          ctxAtomStarts,
                                    BatchedMolecularSystem3DHost&    molSystem) {
  // Use distTermStarts.size() - 1 to get the current batch index
  const int batchIdx              = molSystem.indices.experimentalTorsionTermStarts.size() - 1;
  // Get the previous last atom index from ctxAtomStarts using the current batch index
  const int previousLastAtomIndex = ctxAtomStarts[batchIdx];

  auto& indexHolder   = molSystem.indices;
  auto& contribHolder = molSystem.contribs;

  // Resize atomIdxToBatchIdx using the next atom start index
  indexHolder.atomIdxToBatchIdx.resize(ctxAtomStarts[batchIdx + 1], batchIdx);

  // Update term starts
  indexHolder.experimentalTorsionTermStarts.push_back(indexHolder.experimentalTorsionTermStarts.back() +
                                                      contribs.experimentalTorsionTerms.idx1.size());
  indexHolder.improperTorsionTermStarts.push_back(indexHolder.improperTorsionTermStarts.back() +
                                                  contribs.improperTorsionTerms.idx1.size());
  indexHolder.dist12TermStarts.push_back(indexHolder.dist12TermStarts.back() + contribs.dist12Terms.idx1.size());
  indexHolder.dist13TermStarts.push_back(indexHolder.dist13TermStarts.back() + contribs.dist13Terms.idx1.size());
  indexHolder.angle13TermStarts.push_back(indexHolder.angle13TermStarts.back() + contribs.angle13Terms.idx1.size());
  indexHolder.longRangeDistTermStarts.push_back(indexHolder.longRangeDistTermStarts.back() +
                                                contribs.longRangeDistTerms.idx1.size());

  // Calculate number of blocks needed
  int maxNumContribs = 0;
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.experimentalTorsionTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.improperTorsionTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.dist12Terms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.dist13Terms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.angle13Terms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.longRangeDistTerms.idx1.size());

  const int numBlocksNeeded  = std::max(1, (maxNumContribs + 127) / nvMolKit::FFKernelUtils::blockSizeEnergyReduction);
  const int numThreadsNeeded = numBlocksNeeded * nvMolKit::FFKernelUtils::blockSizeEnergyReduction;

  indexHolder.energyBufferStarts.push_back(indexHolder.energyBufferStarts.back() + numThreadsNeeded);
  for (int i = 0; i < numBlocksNeeded; i++) {
    indexHolder.energyBufferBlockIdxToBatchIdx.push_back(batchIdx);
  }

  // Update contributions
  // Experimental torsion terms
  const size_t numExpTorsionTerms = contribs.experimentalTorsionTerms.idx1.size();
  contribHolder.experimentalTorsionTerms.idx1.reserve(contribHolder.experimentalTorsionTerms.idx1.size() +
                                                      numExpTorsionTerms);
  contribHolder.experimentalTorsionTerms.idx2.reserve(contribHolder.experimentalTorsionTerms.idx2.size() +
                                                      numExpTorsionTerms);
  contribHolder.experimentalTorsionTerms.idx3.reserve(contribHolder.experimentalTorsionTerms.idx3.size() +
                                                      numExpTorsionTerms);
  contribHolder.experimentalTorsionTerms.idx4.reserve(contribHolder.experimentalTorsionTerms.idx4.size() +
                                                      numExpTorsionTerms);
  contribHolder.experimentalTorsionTerms.forceConstants.reserve(
    contribHolder.experimentalTorsionTerms.forceConstants.size() + numExpTorsionTerms * 6);
  contribHolder.experimentalTorsionTerms.signs.reserve(contribHolder.experimentalTorsionTerms.signs.size() +
                                                       numExpTorsionTerms * 6);

  for (size_t i = 0; i < numExpTorsionTerms; i++) {
    contribHolder.experimentalTorsionTerms.idx1.push_back(contribs.experimentalTorsionTerms.idx1[i] +
                                                          previousLastAtomIndex);
    contribHolder.experimentalTorsionTerms.idx2.push_back(contribs.experimentalTorsionTerms.idx2[i] +
                                                          previousLastAtomIndex);
    contribHolder.experimentalTorsionTerms.idx3.push_back(contribs.experimentalTorsionTerms.idx3[i] +
                                                          previousLastAtomIndex);
    contribHolder.experimentalTorsionTerms.idx4.push_back(contribs.experimentalTorsionTerms.idx4[i] +
                                                          previousLastAtomIndex);
  }
  contribHolder.experimentalTorsionTerms.forceConstants.insert(
    contribHolder.experimentalTorsionTerms.forceConstants.end(),
    contribs.experimentalTorsionTerms.forceConstants.begin(),
    contribs.experimentalTorsionTerms.forceConstants.end());
  contribHolder.experimentalTorsionTerms.signs.insert(contribHolder.experimentalTorsionTerms.signs.end(),
                                                      contribs.experimentalTorsionTerms.signs.begin(),
                                                      contribs.experimentalTorsionTerms.signs.end());

  // Improper torsion terms
  const size_t numImproperTerms = contribs.improperTorsionTerms.idx1.size();
  contribHolder.improperTorsionTerms.idx1.reserve(contribHolder.improperTorsionTerms.idx1.size() + numImproperTerms);
  contribHolder.improperTorsionTerms.idx2.reserve(contribHolder.improperTorsionTerms.idx2.size() + numImproperTerms);
  contribHolder.improperTorsionTerms.idx3.reserve(contribHolder.improperTorsionTerms.idx3.size() + numImproperTerms);
  contribHolder.improperTorsionTerms.idx4.reserve(contribHolder.improperTorsionTerms.idx4.size() + numImproperTerms);
  contribHolder.improperTorsionTerms.at2AtomicNum.reserve(contribHolder.improperTorsionTerms.at2AtomicNum.size() +
                                                          numImproperTerms);
  contribHolder.improperTorsionTerms.isCBoundToO.reserve(contribHolder.improperTorsionTerms.isCBoundToO.size() +
                                                         numImproperTerms);
  contribHolder.improperTorsionTerms.C0.reserve(contribHolder.improperTorsionTerms.C0.size() + numImproperTerms);
  contribHolder.improperTorsionTerms.C1.reserve(contribHolder.improperTorsionTerms.C1.size() + numImproperTerms);
  contribHolder.improperTorsionTerms.C2.reserve(contribHolder.improperTorsionTerms.C2.size() + numImproperTerms);
  contribHolder.improperTorsionTerms.forceConstant.reserve(contribHolder.improperTorsionTerms.forceConstant.size() +
                                                           numImproperTerms);

  for (size_t i = 0; i < numImproperTerms; i++) {
    contribHolder.improperTorsionTerms.idx1.push_back(contribs.improperTorsionTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.improperTorsionTerms.idx2.push_back(contribs.improperTorsionTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.improperTorsionTerms.idx3.push_back(contribs.improperTorsionTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.improperTorsionTerms.idx4.push_back(contribs.improperTorsionTerms.idx4[i] + previousLastAtomIndex);
  }
  contribHolder.improperTorsionTerms.at2AtomicNum.insert(contribHolder.improperTorsionTerms.at2AtomicNum.end(),
                                                         contribs.improperTorsionTerms.at2AtomicNum.begin(),
                                                         contribs.improperTorsionTerms.at2AtomicNum.end());
  contribHolder.improperTorsionTerms.isCBoundToO.insert(contribHolder.improperTorsionTerms.isCBoundToO.end(),
                                                        contribs.improperTorsionTerms.isCBoundToO.begin(),
                                                        contribs.improperTorsionTerms.isCBoundToO.end());
  contribHolder.improperTorsionTerms.C0.insert(contribHolder.improperTorsionTerms.C0.end(),
                                               contribs.improperTorsionTerms.C0.begin(),
                                               contribs.improperTorsionTerms.C0.end());
  contribHolder.improperTorsionTerms.C1.insert(contribHolder.improperTorsionTerms.C1.end(),
                                               contribs.improperTorsionTerms.C1.begin(),
                                               contribs.improperTorsionTerms.C1.end());
  contribHolder.improperTorsionTerms.C2.insert(contribHolder.improperTorsionTerms.C2.end(),
                                               contribs.improperTorsionTerms.C2.begin(),
                                               contribs.improperTorsionTerms.C2.end());
  contribHolder.improperTorsionTerms.forceConstant.insert(contribHolder.improperTorsionTerms.forceConstant.end(),
                                                          contribs.improperTorsionTerms.forceConstant.begin(),
                                                          contribs.improperTorsionTerms.forceConstant.end());

  // Add numImpropers count (0 if no improper torsions, otherwise the count from the source)
  if (contribs.improperTorsionTerms.numImpropers.empty()) {
    contribHolder.improperTorsionTerms.numImpropers.push_back(0);
  } else {
    contribHolder.improperTorsionTerms.numImpropers.push_back(contribs.improperTorsionTerms.numImpropers[0]);
  }

  // 1-2 distance terms
  const size_t numDist12Terms = contribs.dist12Terms.idx1.size();
  contribHolder.dist12Terms.idx1.reserve(contribHolder.dist12Terms.idx1.size() + numDist12Terms);
  contribHolder.dist12Terms.idx2.reserve(contribHolder.dist12Terms.idx2.size() + numDist12Terms);
  contribHolder.dist12Terms.minLen.reserve(contribHolder.dist12Terms.minLen.size() + numDist12Terms);
  contribHolder.dist12Terms.maxLen.reserve(contribHolder.dist12Terms.maxLen.size() + numDist12Terms);
  contribHolder.dist12Terms.forceConstant.reserve(contribHolder.dist12Terms.forceConstant.size() + numDist12Terms);

  for (size_t i = 0; i < numDist12Terms; i++) {
    contribHolder.dist12Terms.idx1.push_back(contribs.dist12Terms.idx1[i] + previousLastAtomIndex);
    contribHolder.dist12Terms.idx2.push_back(contribs.dist12Terms.idx2[i] + previousLastAtomIndex);
  }
  contribHolder.dist12Terms.minLen.insert(contribHolder.dist12Terms.minLen.end(),
                                          contribs.dist12Terms.minLen.begin(),
                                          contribs.dist12Terms.minLen.end());
  contribHolder.dist12Terms.maxLen.insert(contribHolder.dist12Terms.maxLen.end(),
                                          contribs.dist12Terms.maxLen.begin(),
                                          contribs.dist12Terms.maxLen.end());
  contribHolder.dist12Terms.forceConstant.insert(contribHolder.dist12Terms.forceConstant.end(),
                                                 contribs.dist12Terms.forceConstant.begin(),
                                                 contribs.dist12Terms.forceConstant.end());

  // 1-3 distance terms
  const size_t numDist13Terms = contribs.dist13Terms.idx1.size();
  contribHolder.dist13Terms.idx1.reserve(contribHolder.dist13Terms.idx1.size() + numDist13Terms);
  contribHolder.dist13Terms.idx2.reserve(contribHolder.dist13Terms.idx2.size() + numDist13Terms);
  contribHolder.dist13Terms.minLen.reserve(contribHolder.dist13Terms.minLen.size() + numDist13Terms);
  contribHolder.dist13Terms.maxLen.reserve(contribHolder.dist13Terms.maxLen.size() + numDist13Terms);
  contribHolder.dist13Terms.forceConstant.reserve(contribHolder.dist13Terms.forceConstant.size() + numDist13Terms);
  contribHolder.dist13Terms.isImproperConstrained.reserve(contribHolder.dist13Terms.isImproperConstrained.size() +
                                                          numDist13Terms);

  for (size_t i = 0; i < numDist13Terms; i++) {
    contribHolder.dist13Terms.idx1.push_back(contribs.dist13Terms.idx1[i] + previousLastAtomIndex);
    contribHolder.dist13Terms.idx2.push_back(contribs.dist13Terms.idx2[i] + previousLastAtomIndex);
  }
  contribHolder.dist13Terms.minLen.insert(contribHolder.dist13Terms.minLen.end(),
                                          contribs.dist13Terms.minLen.begin(),
                                          contribs.dist13Terms.minLen.end());
  contribHolder.dist13Terms.maxLen.insert(contribHolder.dist13Terms.maxLen.end(),
                                          contribs.dist13Terms.maxLen.begin(),
                                          contribs.dist13Terms.maxLen.end());
  contribHolder.dist13Terms.forceConstant.insert(contribHolder.dist13Terms.forceConstant.end(),
                                                 contribs.dist13Terms.forceConstant.begin(),
                                                 contribs.dist13Terms.forceConstant.end());
  // Note this is only done here, not for 1-2 or LR
  contribHolder.dist13Terms.isImproperConstrained.insert(contribHolder.dist13Terms.isImproperConstrained.end(),
                                                         contribs.dist13Terms.isImproperConstrained.begin(),
                                                         contribs.dist13Terms.isImproperConstrained.end());

  // 1-3 angle terms
  const size_t numAngle13Terms = contribs.angle13Terms.idx1.size();
  contribHolder.angle13Terms.idx1.reserve(contribHolder.angle13Terms.idx1.size() + numAngle13Terms);
  contribHolder.angle13Terms.idx2.reserve(contribHolder.angle13Terms.idx2.size() + numAngle13Terms);
  contribHolder.angle13Terms.idx3.reserve(contribHolder.angle13Terms.idx3.size() + numAngle13Terms);
  contribHolder.angle13Terms.minAngle.reserve(contribHolder.angle13Terms.minAngle.size() + numAngle13Terms);
  contribHolder.angle13Terms.maxAngle.reserve(contribHolder.angle13Terms.maxAngle.size() + numAngle13Terms);

  for (size_t i = 0; i < numAngle13Terms; i++) {
    contribHolder.angle13Terms.idx1.push_back(contribs.angle13Terms.idx1[i] + previousLastAtomIndex);
    contribHolder.angle13Terms.idx2.push_back(contribs.angle13Terms.idx2[i] + previousLastAtomIndex);
    contribHolder.angle13Terms.idx3.push_back(contribs.angle13Terms.idx3[i] + previousLastAtomIndex);
  }
  contribHolder.angle13Terms.minAngle.insert(contribHolder.angle13Terms.minAngle.end(),
                                             contribs.angle13Terms.minAngle.begin(),
                                             contribs.angle13Terms.minAngle.end());
  contribHolder.angle13Terms.maxAngle.insert(contribHolder.angle13Terms.maxAngle.end(),
                                             contribs.angle13Terms.maxAngle.begin(),
                                             contribs.angle13Terms.maxAngle.end());

  // Long range distance terms
  const size_t numLongRangeDistTerms = contribs.longRangeDistTerms.idx1.size();
  contribHolder.longRangeDistTerms.idx1.reserve(contribHolder.longRangeDistTerms.idx1.size() + numLongRangeDistTerms);
  contribHolder.longRangeDistTerms.idx2.reserve(contribHolder.longRangeDistTerms.idx2.size() + numLongRangeDistTerms);
  contribHolder.longRangeDistTerms.minLen.reserve(contribHolder.longRangeDistTerms.minLen.size() +
                                                  numLongRangeDistTerms);
  contribHolder.longRangeDistTerms.maxLen.reserve(contribHolder.longRangeDistTerms.maxLen.size() +
                                                  numLongRangeDistTerms);
  contribHolder.longRangeDistTerms.forceConstant.reserve(contribHolder.longRangeDistTerms.forceConstant.size() +
                                                         numLongRangeDistTerms);

  for (size_t i = 0; i < numLongRangeDistTerms; i++) {
    contribHolder.longRangeDistTerms.idx1.push_back(contribs.longRangeDistTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.longRangeDistTerms.idx2.push_back(contribs.longRangeDistTerms.idx2[i] + previousLastAtomIndex);
  }
  contribHolder.longRangeDistTerms.minLen.insert(contribHolder.longRangeDistTerms.minLen.end(),
                                                 contribs.longRangeDistTerms.minLen.begin(),
                                                 contribs.longRangeDistTerms.minLen.end());
  contribHolder.longRangeDistTerms.maxLen.insert(contribHolder.longRangeDistTerms.maxLen.end(),
                                                 contribs.longRangeDistTerms.maxLen.begin(),
                                                 contribs.longRangeDistTerms.maxLen.end());
  contribHolder.longRangeDistTerms.forceConstant.insert(contribHolder.longRangeDistTerms.forceConstant.end(),
                                                        contribs.longRangeDistTerms.forceConstant.begin(),
                                                        contribs.longRangeDistTerms.forceConstant.end());
}

void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem,
                        const int                      dimension,
                        std::vector<int>&              ctxAtomStarts,
                        std::vector<double>&           ctxPositions) {
  // First update context data
  addMoleculeToContextWithPositions(positions, dimension, ctxAtomStarts, ctxPositions);

  // Then update the molecular system
  addMoleculeToMolecularSystem(contribs, positions.size() / dimension, dimension, ctxAtomStarts, molSystem);
}

void addMoleculeToBatch3D(const Energy3DForceContribsHost& contribs,
                          const std::vector<double>&       positions,
                          BatchedMolecularSystem3DHost&    molSystem,
                          std::vector<int>&                ctxAtomStarts,
                          std::vector<double>&             ctxPositions) {
  // First update context data
  addMoleculeToContextWithPositions(positions, 3, ctxAtomStarts, ctxPositions);

  // Then update the molecular system
  addMoleculeToMolecularSystem3D(contribs, ctxAtomStarts, molSystem);
}

void sendContribsAndIndicesToDevice(const BatchedMolecularSystemHost& molSystemHost,
                                    BatchedMolecularDeviceBuffers&    molSystemDevice) {
  auto&       deviceContribs = molSystemDevice.contribs;
  const auto& hostContribs   = molSystemHost.contribs;

  // DistViolation term
  deviceContribs.distTerms.idx1.setFromVector(hostContribs.distTerms.idx1);
  deviceContribs.distTerms.idx2.setFromVector(hostContribs.distTerms.idx2);
  deviceContribs.distTerms.lb2.setFromVector(hostContribs.distTerms.lb2);
  deviceContribs.distTerms.ub2.setFromVector(hostContribs.distTerms.ub2);
  deviceContribs.distTerms.weight.setFromVector(hostContribs.distTerms.weight);

  // ChiralViolation term
  deviceContribs.chiralTerms.idx1.setFromVector(hostContribs.chiralTerms.idx1);
  deviceContribs.chiralTerms.idx2.setFromVector(hostContribs.chiralTerms.idx2);
  deviceContribs.chiralTerms.idx3.setFromVector(hostContribs.chiralTerms.idx3);
  deviceContribs.chiralTerms.idx4.setFromVector(hostContribs.chiralTerms.idx4);
  deviceContribs.chiralTerms.volLower.setFromVector(hostContribs.chiralTerms.volLower);
  deviceContribs.chiralTerms.volUpper.setFromVector(hostContribs.chiralTerms.volUpper);
  // Note: weight no longer stored in device structure

  // FourthDim term
  deviceContribs.fourthTerms.idx.setFromVector(hostContribs.fourthTerms.idx);
  // Note: weight no longer stored in device structure

  // Indices
  auto&       deviceIndices = molSystemDevice.indices;
  const auto& hostIndices   = molSystemHost.indices;
  deviceIndices.energyBufferStarts.setFromVector(hostIndices.energyBufferStarts);
  deviceIndices.atomIdxToBatchIdx.setFromVector(hostIndices.atomIdxToBatchIdx);
  deviceIndices.energyBufferBlockIdxToBatchIdx.setFromVector(hostIndices.energyBufferBlockIdxToBatchIdx);
  deviceIndices.distTermStarts.setFromVector(hostIndices.distTermStarts);
  deviceIndices.chiralTermStarts.setFromVector(hostIndices.chiralTermStarts);
  deviceIndices.fourthTermStarts.setFromVector(hostIndices.fourthTermStarts);

  // Copy dimension
  molSystemDevice.dimension = molSystemHost.dimension;
}

void sendContribsAndIndicesToDevice3D(const BatchedMolecularSystem3DHost& molSystemHost,
                                      BatchedMolecular3DDeviceBuffers&    molSystemDevice) {
  auto&       deviceContribs = molSystemDevice.contribs;
  const auto& hostContribs   = molSystemHost.contribs;

  // Experimental torsion terms
  deviceContribs.experimentalTorsionTerms.idx1.setFromVector(hostContribs.experimentalTorsionTerms.idx1);
  deviceContribs.experimentalTorsionTerms.idx2.setFromVector(hostContribs.experimentalTorsionTerms.idx2);
  deviceContribs.experimentalTorsionTerms.idx3.setFromVector(hostContribs.experimentalTorsionTerms.idx3);
  deviceContribs.experimentalTorsionTerms.idx4.setFromVector(hostContribs.experimentalTorsionTerms.idx4);
  deviceContribs.experimentalTorsionTerms.forceConstants.setFromVector(
    hostContribs.experimentalTorsionTerms.forceConstants);
  deviceContribs.experimentalTorsionTerms.signs.setFromVector(hostContribs.experimentalTorsionTerms.signs);

  // Improper torsion terms
  deviceContribs.improperTorsionTerms.idx1.setFromVector(hostContribs.improperTorsionTerms.idx1);
  deviceContribs.improperTorsionTerms.idx2.setFromVector(hostContribs.improperTorsionTerms.idx2);
  deviceContribs.improperTorsionTerms.idx3.setFromVector(hostContribs.improperTorsionTerms.idx3);
  deviceContribs.improperTorsionTerms.idx4.setFromVector(hostContribs.improperTorsionTerms.idx4);
  deviceContribs.improperTorsionTerms.at2AtomicNum.setFromVector(hostContribs.improperTorsionTerms.at2AtomicNum);
  // Convert bool vector to uint8_t vector for device
  std::vector<uint8_t> isCBoundToOInt(hostContribs.improperTorsionTerms.isCBoundToO.begin(),
                                      hostContribs.improperTorsionTerms.isCBoundToO.end());
  deviceContribs.improperTorsionTerms.isCBoundToO.setFromVector(isCBoundToOInt);
  deviceContribs.improperTorsionTerms.C0.setFromVector(hostContribs.improperTorsionTerms.C0);
  deviceContribs.improperTorsionTerms.C1.setFromVector(hostContribs.improperTorsionTerms.C1);
  deviceContribs.improperTorsionTerms.C2.setFromVector(hostContribs.improperTorsionTerms.C2);
  deviceContribs.improperTorsionTerms.forceConstant.setFromVector(hostContribs.improperTorsionTerms.forceConstant);
  deviceContribs.improperTorsionTerms.numImpropers.setFromVector(hostContribs.improperTorsionTerms.numImpropers);

  // 1-2 distance terms
  deviceContribs.dist12Terms.idx1.setFromVector(hostContribs.dist12Terms.idx1);
  deviceContribs.dist12Terms.idx2.setFromVector(hostContribs.dist12Terms.idx2);
  deviceContribs.dist12Terms.minLen.setFromVector(hostContribs.dist12Terms.minLen);
  deviceContribs.dist12Terms.maxLen.setFromVector(hostContribs.dist12Terms.maxLen);
  deviceContribs.dist12Terms.forceConstant.setFromVector(hostContribs.dist12Terms.forceConstant);

  // 1-3 distance terms
  deviceContribs.dist13Terms.idx1.setFromVector(hostContribs.dist13Terms.idx1);
  deviceContribs.dist13Terms.idx2.setFromVector(hostContribs.dist13Terms.idx2);
  deviceContribs.dist13Terms.minLen.setFromVector(hostContribs.dist13Terms.minLen);
  deviceContribs.dist13Terms.maxLen.setFromVector(hostContribs.dist13Terms.maxLen);
  deviceContribs.dist13Terms.forceConstant.setFromVector(hostContribs.dist13Terms.forceConstant);
  deviceContribs.dist13Terms.isImproperConstrained.setFromVector(hostContribs.dist13Terms.isImproperConstrained);

  // 1-3 angle terms
  deviceContribs.angle13Terms.idx1.setFromVector(hostContribs.angle13Terms.idx1);
  deviceContribs.angle13Terms.idx2.setFromVector(hostContribs.angle13Terms.idx2);
  deviceContribs.angle13Terms.idx3.setFromVector(hostContribs.angle13Terms.idx3);
  deviceContribs.angle13Terms.minAngle.setFromVector(hostContribs.angle13Terms.minAngle);
  deviceContribs.angle13Terms.maxAngle.setFromVector(hostContribs.angle13Terms.maxAngle);

  // Long range distance terms
  deviceContribs.longRangeDistTerms.idx1.setFromVector(hostContribs.longRangeDistTerms.idx1);
  deviceContribs.longRangeDistTerms.idx2.setFromVector(hostContribs.longRangeDistTerms.idx2);
  deviceContribs.longRangeDistTerms.minLen.setFromVector(hostContribs.longRangeDistTerms.minLen);
  deviceContribs.longRangeDistTerms.maxLen.setFromVector(hostContribs.longRangeDistTerms.maxLen);
  deviceContribs.longRangeDistTerms.forceConstant.setFromVector(hostContribs.longRangeDistTerms.forceConstant);

  // Indices
  auto&       deviceIndices = molSystemDevice.indices;
  const auto& hostIndices   = molSystemHost.indices;
  deviceIndices.energyBufferStarts.setFromVector(hostIndices.energyBufferStarts);
  deviceIndices.atomIdxToBatchIdx.setFromVector(hostIndices.atomIdxToBatchIdx);
  deviceIndices.energyBufferBlockIdxToBatchIdx.setFromVector(hostIndices.energyBufferBlockIdxToBatchIdx);
  deviceIndices.experimentalTorsionTermStarts.setFromVector(hostIndices.experimentalTorsionTermStarts);
  deviceIndices.improperTorsionTermStarts.setFromVector(hostIndices.improperTorsionTermStarts);
  deviceIndices.dist12TermStarts.setFromVector(hostIndices.dist12TermStarts);
  deviceIndices.dist13TermStarts.setFromVector(hostIndices.dist13TermStarts);
  deviceIndices.angle13TermStarts.setFromVector(hostIndices.angle13TermStarts);
  deviceIndices.longRangeDistTermStarts.setFromVector(hostIndices.longRangeDistTermStarts);
}

//! Set all DeviceVector streams for the batched molecular device buffers.
void setStreams(BatchedMolecularDeviceBuffers& devBuffers, cudaStream_t stream) {
  // First the isolated buffers.
  devBuffers.energyBuffer.setStream(stream);
  devBuffers.grad.setStream(stream);
  devBuffers.energyOuts.setStream(stream);

  // Indices
  devBuffers.indices.atomIdxToBatchIdx.setStream(stream);
  devBuffers.indices.energyBufferStarts.setStream(stream);
  devBuffers.indices.energyBufferBlockIdxToBatchIdx.setStream(stream);
  devBuffers.indices.distTermStarts.setStream(stream);
  devBuffers.indices.chiralTermStarts.setStream(stream);
  devBuffers.indices.fourthTermStarts.setStream(stream);

  // contribs
  devBuffers.contribs.distTerms.idx1.setStream(stream);
  devBuffers.contribs.distTerms.idx2.setStream(stream);
  devBuffers.contribs.distTerms.lb2.setStream(stream);
  devBuffers.contribs.distTerms.ub2.setStream(stream);
  devBuffers.contribs.distTerms.weight.setStream(stream);
  devBuffers.contribs.chiralTerms.idx1.setStream(stream);
  devBuffers.contribs.chiralTerms.idx2.setStream(stream);
  devBuffers.contribs.chiralTerms.idx3.setStream(stream);
  devBuffers.contribs.chiralTerms.idx4.setStream(stream);
  devBuffers.contribs.chiralTerms.volLower.setStream(stream);
  devBuffers.contribs.chiralTerms.volUpper.setStream(stream);
  // Note: weight no longer stored in device structure
  devBuffers.contribs.fourthTerms.idx.setStream(stream);
  // Note: weight no longer stored in device structure
}
//! Set all DeviceVector streams for the batched 3D molecular device buffers.
void setStreams(BatchedMolecular3DDeviceBuffers& devBuffers, cudaStream_t stream) {
  devBuffers.energyBuffer.setStream(stream);
  devBuffers.grad.setStream(stream);
  devBuffers.energyOuts.setStream(stream);

  // Indices
  devBuffers.indices.energyBufferStarts.setStream(stream);
  devBuffers.indices.energyBufferBlockIdxToBatchIdx.setStream(stream);
  devBuffers.indices.atomIdxToBatchIdx.setStream(stream);
  devBuffers.indices.experimentalTorsionTermStarts.setStream(stream);
  devBuffers.indices.improperTorsionTermStarts.setStream(stream);
  devBuffers.indices.dist12TermStarts.setStream(stream);
  devBuffers.indices.dist13TermStarts.setStream(stream);
  devBuffers.indices.angle13TermStarts.setStream(stream);
  devBuffers.indices.longRangeDistTermStarts.setStream(stream);
  // contribs
  devBuffers.contribs.experimentalTorsionTerms.idx1.setStream(stream);
  devBuffers.contribs.experimentalTorsionTerms.idx2.setStream(stream);
  devBuffers.contribs.experimentalTorsionTerms.idx3.setStream(stream);
  devBuffers.contribs.experimentalTorsionTerms.idx4.setStream(stream);
  devBuffers.contribs.experimentalTorsionTerms.forceConstants.setStream(stream);
  devBuffers.contribs.experimentalTorsionTerms.signs.setStream(stream);

  devBuffers.contribs.improperTorsionTerms.idx1.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.idx2.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.idx3.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.idx4.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.at2AtomicNum.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.C0.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.C1.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.C2.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.isCBoundToO.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.forceConstant.setStream(stream);
  devBuffers.contribs.improperTorsionTerms.numImpropers.setStream(stream);

  devBuffers.contribs.angle13Terms.idx1.setStream(stream);
  devBuffers.contribs.angle13Terms.idx2.setStream(stream);
  devBuffers.contribs.angle13Terms.idx3.setStream(stream);
  devBuffers.contribs.angle13Terms.minAngle.setStream(stream);
  devBuffers.contribs.angle13Terms.maxAngle.setStream(stream);

  devBuffers.contribs.dist12Terms.idx1.setStream(stream);
  devBuffers.contribs.dist12Terms.idx2.setStream(stream);
  devBuffers.contribs.dist12Terms.minLen.setStream(stream);
  devBuffers.contribs.dist12Terms.maxLen.setStream(stream);
  devBuffers.contribs.dist12Terms.forceConstant.setStream(stream);
  devBuffers.contribs.dist12Terms.isImproperConstrained.setStream(stream);

  devBuffers.contribs.dist13Terms.idx1.setStream(stream);
  devBuffers.contribs.dist13Terms.idx2.setStream(stream);
  devBuffers.contribs.dist13Terms.minLen.setStream(stream);
  devBuffers.contribs.dist13Terms.maxLen.setStream(stream);
  devBuffers.contribs.dist13Terms.forceConstant.setStream(stream);
  devBuffers.contribs.dist13Terms.isImproperConstrained.setStream(stream);

  devBuffers.contribs.longRangeDistTerms.idx1.setStream(stream);
  devBuffers.contribs.longRangeDistTerms.idx2.setStream(stream);
  devBuffers.contribs.longRangeDistTerms.minLen.setStream(stream);
  devBuffers.contribs.longRangeDistTerms.maxLen.setStream(stream);
  devBuffers.contribs.longRangeDistTerms.forceConstant.setStream(stream);
}

void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice) {
  nvMolKit::FFKernelUtils::allocateIntermediateBuffers(molSystemHost,
                                                       molSystemDevice,
                                                       molSystemHost.indices.distTermStarts.size() - 1);
}

void allocateIntermediateBuffers3D(const BatchedMolecularSystem3DHost& molSystemHost,
                                   BatchedMolecular3DDeviceBuffers&    molSystemDevice) {
  nvMolKit::FFKernelUtils::allocateIntermediateBuffers(molSystemHost,
                                                       molSystemDevice,
                                                       molSystemHost.indices.experimentalTorsionTermStarts.size() - 1);
}

void sendContextToDevice(const std::vector<double>&           ctxPositionsHost,
                         nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                         const std::vector<int>&              ctxAtomStartsHost,
                         nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice) {
  ctxPositionsDevice.setFromVector(ctxPositionsHost);
  ctxAtomStartsDevice.setFromVector(ctxAtomStartsHost);
}

void setupDeviceBuffers(BatchedMolecularSystemHost&    molSystemHost,
                        BatchedMolecularDeviceBuffers& molSystemDevice,
                        const std::vector<double>&     ctxPositionsHost,
                        const int                      numMols) {
  nvMolKit::FFKernelUtils::allocateIntermediateBuffers(molSystemHost, molSystemDevice, numMols);
  molSystemDevice.grad.resize(ctxPositionsHost.size());
  molSystemDevice.grad.zero();
}

void setupDeviceBuffers3D(BatchedMolecularSystem3DHost&    molSystemHost,
                          BatchedMolecular3DDeviceBuffers& molSystemDevice,
                          const std::vector<double>&       ctxPositionsHost,
                          const int                        numMols) {
  nvMolKit::FFKernelUtils::allocateIntermediateBuffers(molSystemHost, molSystemDevice, numMols);
  molSystemDevice.grad.resize(ctxPositionsHost.size());
  molSystemDevice.grad.zero();
}

// TODO: More sophisticated error handling for energy and gradient.
cudaError_t computeEnergy(BatchedMolecularDeviceBuffers&             molSystemDevice,
                          const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                          const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                          const double                               chiralWeight,
                          const double                               fourthDimWeight,
                          const uint8_t*                             activeThisStage,
                          const double*                              positions,
                          cudaStream_t                               stream) {
  // Prechecks - tempstorage allocated, energybuffer allocated
  assert(molSystemDevice.energyBuffer.size() > 0);
  assert(molSystemDevice.energyOuts.data() != nullptr);

  // Use provided positions or fall back to context positions
  const double* posData = positions ? positions : ctxPositionsDevice.data();

  // Dispatch each term if there is a contrib for it.
  const auto& contribs = molSystemDevice.contribs;

  cudaError_t err = cudaSuccess;
  if (contribs.distTerms.idx1.size() > 0) {
    err = launchDistViolationEnergyKernel(contribs.distTerms.idx1.size(),
                                          contribs.distTerms.idx1.data(),
                                          contribs.distTerms.idx2.data(),
                                          contribs.distTerms.lb2.data(),
                                          contribs.distTerms.ub2.data(),
                                          contribs.distTerms.weight.data(),
                                          posData,
                                          molSystemDevice.energyBuffer.data(),
                                          molSystemDevice.indices.energyBufferStarts.data(),
                                          molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                          molSystemDevice.indices.distTermStarts.data(),
                                          ctxAtomStartsDevice.data(),
                                          molSystemDevice.dimension,
                                          activeThisStage,
                                          stream);
  }
  if (err == cudaSuccess && contribs.chiralTerms.idx1.size() > 0) {
    err = launchChiralViolationEnergyKernel(contribs.chiralTerms.idx1.size(),
                                            contribs.chiralTerms.idx1.data(),
                                            contribs.chiralTerms.idx2.data(),
                                            contribs.chiralTerms.idx3.data(),
                                            contribs.chiralTerms.idx4.data(),
                                            contribs.chiralTerms.volLower.data(),
                                            contribs.chiralTerms.volUpper.data(),
                                            chiralWeight,
                                            posData,
                                            molSystemDevice.energyBuffer.data(),
                                            molSystemDevice.indices.energyBufferStarts.data(),
                                            molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                            molSystemDevice.indices.chiralTermStarts.data(),
                                            ctxAtomStartsDevice.data(),
                                            molSystemDevice.dimension,
                                            activeThisStage,
                                            stream);
  }
  if (err == cudaSuccess && contribs.fourthTerms.idx.size() > 0) {
    err = launchFourthDimEnergyKernel(contribs.fourthTerms.idx.size(),
                                      contribs.fourthTerms.idx.data(),
                                      fourthDimWeight,
                                      posData,
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferStarts.data(),
                                      molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                      molSystemDevice.indices.fourthTermStarts.data(),
                                      ctxAtomStartsDevice.data(),
                                      molSystemDevice.dimension,
                                      activeThisStage,
                                      stream);
  }
  if (err == cudaSuccess) {
    // Now reduce the energy buffer
    return launchReduceEnergiesKernel(molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.size(),
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.data(),
                                      molSystemDevice.energyOuts.data(),
                                      activeThisStage,
                                      stream);
  }
  return err;
}

cudaError_t computeGradients(BatchedMolecularDeviceBuffers&             molSystemDevice,
                             const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                             const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                             const double                               chiralWeight,
                             const double                               fourthDimWeight,
                             const uint8_t*                             activeThisStage,
                             cudaStream_t                               stream) {
  // Dispatch each term if there is a contrib for it.
  const auto& contribs = molSystemDevice.contribs;

  cudaError_t err = cudaSuccess;
  if (contribs.distTerms.idx1.size() > 0) {
    err = launchDistViolationGradientKernel(contribs.distTerms.idx1.size(),
                                            contribs.distTerms.idx1.data(),
                                            contribs.distTerms.idx2.data(),
                                            contribs.distTerms.lb2.data(),
                                            contribs.distTerms.ub2.data(),
                                            contribs.distTerms.weight.data(),
                                            ctxPositionsDevice.data(),
                                            molSystemDevice.grad.data(),
                                            molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                            ctxAtomStartsDevice.data(),
                                            molSystemDevice.dimension,
                                            activeThisStage,
                                            stream);
  }
  if (err == cudaSuccess && contribs.chiralTerms.idx1.size() > 0) {
    err = launchChiralViolationGradientKernel(contribs.chiralTerms.idx1.size(),
                                              contribs.chiralTerms.idx1.data(),
                                              contribs.chiralTerms.idx2.data(),
                                              contribs.chiralTerms.idx3.data(),
                                              contribs.chiralTerms.idx4.data(),
                                              contribs.chiralTerms.volLower.data(),
                                              contribs.chiralTerms.volUpper.data(),
                                              chiralWeight,
                                              ctxPositionsDevice.data(),
                                              molSystemDevice.grad.data(),
                                              molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                              ctxAtomStartsDevice.data(),
                                              molSystemDevice.dimension,
                                              activeThisStage,
                                              stream);
  }
  if (err == cudaSuccess && contribs.fourthTerms.idx.size() > 0) {
    err = launchFourthDimGradientKernel(contribs.fourthTerms.idx.size(),
                                        contribs.fourthTerms.idx.data(),
                                        fourthDimWeight,
                                        ctxPositionsDevice.data(),
                                        molSystemDevice.grad.data(),
                                        molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                        ctxAtomStartsDevice.data(),
                                        molSystemDevice.dimension,
                                        activeThisStage,
                                        stream);
  }
  return err;
}

cudaError_t computeEnergyETK(BatchedMolecular3DDeviceBuffers&           molSystemDevice,
                             const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                             const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                             const uint8_t*                             activeThisStage,
                             const double*                              positions,
                             const ETKTerm                              term,
                             cudaStream_t                               stream) {
  // Prechecks - tempstorage allocated, energybuffer allocated
  assert(molSystemDevice.energyBuffer.size() > 0);
  assert(molSystemDevice.energyOuts.data() != nullptr);

  // Use provided positions or fall back to context positions
  const double* posData = positions ? positions : ctxPositionsDevice.data();

  // Dispatch each term if there is a contrib for it.
  const auto& contribs = molSystemDevice.contribs;

  cudaError_t err = cudaSuccess;

  // Experimental torsion terms
  if ((term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::EXPERIMANTAL_TORSION) &&
      contribs.experimentalTorsionTerms.idx1.size() > 0) {
    err = launchTorsionAngleEnergyKernel(contribs.experimentalTorsionTerms.idx1.size(),
                                         contribs.experimentalTorsionTerms.idx1.data(),
                                         contribs.experimentalTorsionTerms.idx2.data(),
                                         contribs.experimentalTorsionTerms.idx3.data(),
                                         contribs.experimentalTorsionTerms.idx4.data(),
                                         contribs.experimentalTorsionTerms.forceConstants.data(),
                                         contribs.experimentalTorsionTerms.signs.data(),
                                         posData,
                                         molSystemDevice.energyBuffer.data(),
                                         molSystemDevice.indices.energyBufferStarts.data(),
                                         molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                         molSystemDevice.indices.experimentalTorsionTermStarts.data(),
                                         ctxAtomStartsDevice.data(),
                                         activeThisStage,
                                         stream);
  }

  // Improper torsion terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::IMPPROPER_TORSION) &&
      contribs.improperTorsionTerms.idx1.size() > 0) {
    err = launchInversionEnergyKernel(contribs.improperTorsionTerms.idx1.size(),
                                      contribs.improperTorsionTerms.idx1.data(),
                                      contribs.improperTorsionTerms.idx2.data(),
                                      contribs.improperTorsionTerms.idx3.data(),
                                      contribs.improperTorsionTerms.idx4.data(),
                                      contribs.improperTorsionTerms.at2AtomicNum.data(),
                                      contribs.improperTorsionTerms.isCBoundToO.data(),
                                      contribs.improperTorsionTerms.C0.data(),
                                      contribs.improperTorsionTerms.C1.data(),
                                      contribs.improperTorsionTerms.C2.data(),
                                      contribs.improperTorsionTerms.forceConstant.data(),
                                      posData,
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferStarts.data(),
                                      molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                      molSystemDevice.indices.improperTorsionTermStarts.data(),
                                      ctxAtomStartsDevice.data(),
                                      activeThisStage,
                                      stream);
  }

  // 1-2 distance terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::DISTANCE_12) &&
      contribs.dist12Terms.idx1.size() > 0) {
    err = launchDistanceConstraintEnergyKernel(contribs.dist12Terms.idx1.size(),
                                               contribs.dist12Terms.idx1.data(),
                                               contribs.dist12Terms.idx2.data(),
                                               contribs.dist12Terms.minLen.data(),
                                               contribs.dist12Terms.maxLen.data(),
                                               contribs.dist12Terms.forceConstant.data(),
                                               posData,
                                               molSystemDevice.energyBuffer.data(),
                                               molSystemDevice.indices.energyBufferStarts.data(),
                                               molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                               molSystemDevice.indices.dist12TermStarts.data(),
                                               ctxAtomStartsDevice.data(),
                                               activeThisStage,
                                               stream);
  }

  // 1-3 distance terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::DISTANCE_13) &&
      contribs.dist13Terms.idx1.size() > 0) {
    err = launchDistanceConstraintEnergyKernel(contribs.dist13Terms.idx1.size(),
                                               contribs.dist13Terms.idx1.data(),
                                               contribs.dist13Terms.idx2.data(),
                                               contribs.dist13Terms.minLen.data(),
                                               contribs.dist13Terms.maxLen.data(),
                                               contribs.dist13Terms.forceConstant.data(),
                                               posData,
                                               molSystemDevice.energyBuffer.data(),
                                               molSystemDevice.indices.energyBufferStarts.data(),
                                               molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                               molSystemDevice.indices.dist13TermStarts.data(),
                                               ctxAtomStartsDevice.data(),
                                               activeThisStage,
                                               stream);
  }

  // 1-3 angle terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::ANGLE_13) &&
      contribs.angle13Terms.idx1.size() > 0) {
    err = launchAngleConstraintEnergyKernel(contribs.angle13Terms.idx1.size(),
                                            contribs.angle13Terms.idx1.data(),
                                            contribs.angle13Terms.idx2.data(),
                                            contribs.angle13Terms.idx3.data(),
                                            contribs.angle13Terms.minAngle.data(),
                                            contribs.angle13Terms.maxAngle.data(),
                                            posData,
                                            molSystemDevice.energyBuffer.data(),
                                            molSystemDevice.indices.energyBufferStarts.data(),
                                            molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                            molSystemDevice.indices.angle13TermStarts.data(),
                                            ctxAtomStartsDevice.data(),
                                            activeThisStage,
                                            defaultAngleForceConstant,
                                            stream);
  }

  // Long range distance terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::LONGDISTANCE) &&
      contribs.longRangeDistTerms.idx1.size() > 0) {
    err = launchDistanceConstraintEnergyKernel(contribs.longRangeDistTerms.idx1.size(),
                                               contribs.longRangeDistTerms.idx1.data(),
                                               contribs.longRangeDistTerms.idx2.data(),
                                               contribs.longRangeDistTerms.minLen.data(),
                                               contribs.longRangeDistTerms.maxLen.data(),
                                               contribs.longRangeDistTerms.forceConstant.data(),
                                               posData,
                                               molSystemDevice.energyBuffer.data(),
                                               molSystemDevice.indices.energyBufferStarts.data(),
                                               molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                               molSystemDevice.indices.longRangeDistTermStarts.data(),
                                               ctxAtomStartsDevice.data(),
                                               activeThisStage,
                                               stream);
  }

  if (err == cudaSuccess) {
    // Now reduce the energy buffer
    return launchReduceEnergiesKernel(molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.size(),
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.data(),
                                      molSystemDevice.energyOuts.data(),
                                      activeThisStage,
                                      stream);
  }
  return err;
}

cudaError_t computeGradientsETK(BatchedMolecular3DDeviceBuffers&           molSystemDevice,
                                const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                                const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                                const uint8_t*                             activeThisStage,
                                const ETKTerm                              term,
                                cudaStream_t                               stream) {
  // Dispatch each term if there is a contrib for it.
  const auto& contribs = molSystemDevice.contribs;

  cudaError_t err = cudaSuccess;

  // Experimental torsion terms
  if ((term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::EXPERIMANTAL_TORSION) &&
      contribs.experimentalTorsionTerms.idx1.size() > 0) {
    err = launchTorsionAngleGradientKernel(contribs.experimentalTorsionTerms.idx1.size(),
                                           contribs.experimentalTorsionTerms.idx1.data(),
                                           contribs.experimentalTorsionTerms.idx2.data(),
                                           contribs.experimentalTorsionTerms.idx3.data(),
                                           contribs.experimentalTorsionTerms.idx4.data(),
                                           contribs.experimentalTorsionTerms.forceConstants.data(),
                                           contribs.experimentalTorsionTerms.signs.data(),
                                           ctxPositionsDevice.data(),
                                           molSystemDevice.grad.data(),
                                           molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                           ctxAtomStartsDevice.data(),
                                           activeThisStage,
                                           stream);
  }

  // Improper torsion terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::IMPPROPER_TORSION) &&
      contribs.improperTorsionTerms.idx1.size() > 0) {
    err = launchInversionGradientKernel(contribs.improperTorsionTerms.idx1.size(),
                                        contribs.improperTorsionTerms.idx1.data(),
                                        contribs.improperTorsionTerms.idx2.data(),
                                        contribs.improperTorsionTerms.idx3.data(),
                                        contribs.improperTorsionTerms.idx4.data(),
                                        contribs.improperTorsionTerms.at2AtomicNum.data(),
                                        contribs.improperTorsionTerms.isCBoundToO.data(),
                                        contribs.improperTorsionTerms.C0.data(),
                                        contribs.improperTorsionTerms.C1.data(),
                                        contribs.improperTorsionTerms.C2.data(),
                                        contribs.improperTorsionTerms.forceConstant.data(),
                                        ctxPositionsDevice.data(),
                                        molSystemDevice.grad.data(),
                                        molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                        ctxAtomStartsDevice.data(),
                                        activeThisStage,
                                        stream);
  }

  // 1-2 distance terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::DISTANCE_12) &&
      contribs.dist12Terms.idx1.size() > 0) {
    err = launchDistanceConstraintGradientKernel(contribs.dist12Terms.idx1.size(),
                                                 contribs.dist12Terms.idx1.data(),
                                                 contribs.dist12Terms.idx2.data(),
                                                 contribs.dist12Terms.minLen.data(),
                                                 contribs.dist12Terms.maxLen.data(),
                                                 contribs.dist12Terms.forceConstant.data(),
                                                 ctxPositionsDevice.data(),
                                                 molSystemDevice.grad.data(),
                                                 molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                                 ctxAtomStartsDevice.data(),
                                                 activeThisStage,
                                                 stream);
  }

  // 1-3 distance terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::DISTANCE_13) &&
      contribs.dist13Terms.idx1.size() > 0) {
    err = launchDistanceConstraintGradientKernel(contribs.dist13Terms.idx1.size(),
                                                 contribs.dist13Terms.idx1.data(),
                                                 contribs.dist13Terms.idx2.data(),
                                                 contribs.dist13Terms.minLen.data(),
                                                 contribs.dist13Terms.maxLen.data(),
                                                 contribs.dist13Terms.forceConstant.data(),
                                                 ctxPositionsDevice.data(),
                                                 molSystemDevice.grad.data(),
                                                 molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                                 ctxAtomStartsDevice.data(),
                                                 activeThisStage,
                                                 stream);
  }

  // 1-3 angle terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::ANGLE_13) &&
      contribs.angle13Terms.idx1.size() > 0) {
    err = launchAngleConstraintGradientKernel(contribs.angle13Terms.idx1.size(),
                                              contribs.angle13Terms.idx1.data(),
                                              contribs.angle13Terms.idx2.data(),
                                              contribs.angle13Terms.idx3.data(),
                                              contribs.angle13Terms.minAngle.data(),
                                              contribs.angle13Terms.maxAngle.data(),
                                              ctxPositionsDevice.data(),
                                              molSystemDevice.grad.data(),
                                              molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                              ctxAtomStartsDevice.data(),
                                              activeThisStage,
                                              defaultAngleForceConstant,
                                              stream);
  }

  // Long range distance terms
  if (err == cudaSuccess && (term == ETKTerm::ALL || term == ETKTerm::PLAIN || term == ETKTerm::LONGDISTANCE) &&
      contribs.longRangeDistTerms.idx1.size() > 0) {
    err = launchDistanceConstraintGradientKernel(contribs.longRangeDistTerms.idx1.size(),
                                                 contribs.longRangeDistTerms.idx1.data(),
                                                 contribs.longRangeDistTerms.idx2.data(),
                                                 contribs.longRangeDistTerms.minLen.data(),
                                                 contribs.longRangeDistTerms.maxLen.data(),
                                                 contribs.longRangeDistTerms.forceConstant.data(),
                                                 ctxPositionsDevice.data(),
                                                 molSystemDevice.grad.data(),
                                                 molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                                 ctxAtomStartsDevice.data(),
                                                 activeThisStage,
                                                 stream);
  }

  return err;
}

cudaError_t computePlanarEnergy(BatchedMolecular3DDeviceBuffers&           molSystemDevice,
                                const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                                const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                                const uint8_t*                             activeThisStage,
                                const double*                              positions,
                                const cudaStream_t                         stream) {
  // Prechecks - tempstorage allocated, energybuffer allocated
  assert(molSystemDevice.energyBuffer.size() > 0);
  assert(molSystemDevice.energyOuts.data() != nullptr);
  molSystemDevice.energyOuts.zero();
  molSystemDevice.energyBuffer.zero();

  // Use provided positions or fall back to context positions
  const double* posData = positions ? positions : ctxPositionsDevice.data();

  // Dispatch each term if there is a contrib for it.
  const auto& contribs = molSystemDevice.contribs;

  cudaError_t err = cudaSuccess;

  if (contribs.angle13Terms.idx1.size() > 0) {
    err = launchAngleConstraintEnergyKernel(contribs.angle13Terms.idx1.size(),
                                            contribs.angle13Terms.idx1.data(),
                                            contribs.angle13Terms.idx2.data(),
                                            contribs.angle13Terms.idx3.data(),
                                            contribs.angle13Terms.minAngle.data(),
                                            contribs.angle13Terms.maxAngle.data(),
                                            posData,
                                            molSystemDevice.energyBuffer.data(),
                                            molSystemDevice.indices.energyBufferStarts.data(),
                                            molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                            molSystemDevice.indices.angle13TermStarts.data(),
                                            ctxAtomStartsDevice.data(),
                                            activeThisStage,
                                            /*forceConstant=*/10.0,
                                            stream);
  }

  // Improper torsion terms
  if (err == cudaSuccess && contribs.improperTorsionTerms.idx1.size() > 0) {
    err = launchInversionEnergyKernel(contribs.improperTorsionTerms.idx1.size(),
                                      contribs.improperTorsionTerms.idx1.data(),
                                      contribs.improperTorsionTerms.idx2.data(),
                                      contribs.improperTorsionTerms.idx3.data(),
                                      contribs.improperTorsionTerms.idx4.data(),
                                      contribs.improperTorsionTerms.at2AtomicNum.data(),
                                      contribs.improperTorsionTerms.isCBoundToO.data(),
                                      contribs.improperTorsionTerms.C0.data(),
                                      contribs.improperTorsionTerms.C1.data(),
                                      contribs.improperTorsionTerms.C2.data(),
                                      contribs.improperTorsionTerms.forceConstant.data(),
                                      posData,
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferStarts.data(),
                                      molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                      molSystemDevice.indices.improperTorsionTermStarts.data(),
                                      ctxAtomStartsDevice.data(),
                                      activeThisStage,
                                      stream);
  }

  if (err == cudaSuccess) {
    // Now reduce the energy buffer
    return launchReduceEnergiesKernel(molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.size(),
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.data(),
                                      molSystemDevice.energyOuts.data(),
                                      activeThisStage,
                                      stream);
  }
  return err;
}

}  // namespace DistGeom
}  // namespace nvMolKit
