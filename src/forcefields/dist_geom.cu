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
#include "kernel_utils.cuh"

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

void addMoleculeToMolecularSystem(const EnergyForceContribsHost& contribs,
                                  const int                      numAtoms,
                                  const int                      dimension,
                                  const std::vector<int>&        ctxAtomStarts,
                                  BatchedMolecularSystemHost&    molSystem,
                                  std::vector<int>*              atomNumbers) {
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

  // Handle atom numbers if provided
  if (atomNumbers) {
    molSystem.atomNumbers.insert(molSystem.atomNumbers.end(), atomNumbers->begin(), atomNumbers->end());
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
  for (size_t i = 0; i < contribs.distTerms.idx1.size(); i++) {
    contribHolder.distTerms.idx1.push_back(contribs.distTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.distTerms.idx2.push_back(contribs.distTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.distTerms.lb2.push_back(contribs.distTerms.lb2[i]);
    contribHolder.distTerms.ub2.push_back(contribs.distTerms.ub2[i]);
    contribHolder.distTerms.weight.push_back(contribs.distTerms.weight[i]);
  }

  // ChiralViolation term
  for (size_t i = 0; i < contribs.chiralTerms.idx1.size(); i++) {
    contribHolder.chiralTerms.idx1.push_back(contribs.chiralTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.chiralTerms.idx2.push_back(contribs.chiralTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.chiralTerms.idx3.push_back(contribs.chiralTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.chiralTerms.idx4.push_back(contribs.chiralTerms.idx4[i] + previousLastAtomIndex);
    contribHolder.chiralTerms.volLower.push_back(contribs.chiralTerms.volLower[i]);
    contribHolder.chiralTerms.volUpper.push_back(contribs.chiralTerms.volUpper[i]);
    contribHolder.chiralTerms.weight.push_back(contribs.chiralTerms.weight[i]);
  }

  // FourthDim term
  for (size_t i = 0; i < contribs.fourthTerms.idx.size(); i++) {
    contribHolder.fourthTerms.idx.push_back(contribs.fourthTerms.idx[i] + previousLastAtomIndex);
    contribHolder.fourthTerms.weight.push_back(contribs.fourthTerms.weight[i]);
  }
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
  for (size_t i = 0; i < contribs.experimentalTorsionTerms.idx1.size(); i++) {
    contribHolder.experimentalTorsionTerms.idx1.push_back(contribs.experimentalTorsionTerms.idx1[i] +
                                                          previousLastAtomIndex);
    contribHolder.experimentalTorsionTerms.idx2.push_back(contribs.experimentalTorsionTerms.idx2[i] +
                                                          previousLastAtomIndex);
    contribHolder.experimentalTorsionTerms.idx3.push_back(contribs.experimentalTorsionTerms.idx3[i] +
                                                          previousLastAtomIndex);
    contribHolder.experimentalTorsionTerms.idx4.push_back(contribs.experimentalTorsionTerms.idx4[i] +
                                                          previousLastAtomIndex);
    // Add all 6 force constants for this term
    for (int j = 0; j < 6; j++) {
      contribHolder.experimentalTorsionTerms.forceConstants.push_back(
        contribs.experimentalTorsionTerms.forceConstants[i * 6 + j]);
      contribHolder.experimentalTorsionTerms.signs.push_back(contribs.experimentalTorsionTerms.signs[i * 6 + j]);
    }
  }

  // Improper torsion terms
  for (size_t i = 0; i < contribs.improperTorsionTerms.idx1.size(); i++) {
    contribHolder.improperTorsionTerms.idx1.push_back(contribs.improperTorsionTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.improperTorsionTerms.idx2.push_back(contribs.improperTorsionTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.improperTorsionTerms.idx3.push_back(contribs.improperTorsionTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.improperTorsionTerms.idx4.push_back(contribs.improperTorsionTerms.idx4[i] + previousLastAtomIndex);
    contribHolder.improperTorsionTerms.at2AtomicNum.push_back(contribs.improperTorsionTerms.at2AtomicNum[i]);
    contribHolder.improperTorsionTerms.isCBoundToO.push_back(contribs.improperTorsionTerms.isCBoundToO[i]);
    contribHolder.improperTorsionTerms.C0.push_back(contribs.improperTorsionTerms.C0[i]);
    contribHolder.improperTorsionTerms.C1.push_back(contribs.improperTorsionTerms.C1[i]);
    contribHolder.improperTorsionTerms.C2.push_back(contribs.improperTorsionTerms.C2[i]);
    contribHolder.improperTorsionTerms.forceConstant.push_back(contribs.improperTorsionTerms.forceConstant[i]);
  }
  contribHolder.improperTorsionTerms.numImpropers.push_back(contribs.improperTorsionTerms.numImpropers[0]);

  // 1-2 distance terms
  for (size_t i = 0; i < contribs.dist12Terms.idx1.size(); i++) {
    contribHolder.dist12Terms.idx1.push_back(contribs.dist12Terms.idx1[i] + previousLastAtomIndex);
    contribHolder.dist12Terms.idx2.push_back(contribs.dist12Terms.idx2[i] + previousLastAtomIndex);
    contribHolder.dist12Terms.minLen.push_back(contribs.dist12Terms.minLen[i]);
    contribHolder.dist12Terms.maxLen.push_back(contribs.dist12Terms.maxLen[i]);
    contribHolder.dist12Terms.forceConstant.push_back(contribs.dist12Terms.forceConstant[i]);
  }

  // 1-3 distance terms
  for (size_t i = 0; i < contribs.dist13Terms.idx1.size(); i++) {
    contribHolder.dist13Terms.idx1.push_back(contribs.dist13Terms.idx1[i] + previousLastAtomIndex);
    contribHolder.dist13Terms.idx2.push_back(contribs.dist13Terms.idx2[i] + previousLastAtomIndex);
    contribHolder.dist13Terms.minLen.push_back(contribs.dist13Terms.minLen[i]);
    contribHolder.dist13Terms.maxLen.push_back(contribs.dist13Terms.maxLen[i]);
    contribHolder.dist13Terms.forceConstant.push_back(contribs.dist13Terms.forceConstant[i]);
    // Note this is only done here, not for 1-2 or LR
    contribHolder.dist13Terms.isImproperConstrained.push_back(contribs.dist13Terms.isImproperConstrained[i]);
  }

  // 1-3 angle terms
  for (size_t i = 0; i < contribs.angle13Terms.idx1.size(); i++) {
    contribHolder.angle13Terms.idx1.push_back(contribs.angle13Terms.idx1[i] + previousLastAtomIndex);
    contribHolder.angle13Terms.idx2.push_back(contribs.angle13Terms.idx2[i] + previousLastAtomIndex);
    contribHolder.angle13Terms.idx3.push_back(contribs.angle13Terms.idx3[i] + previousLastAtomIndex);
    contribHolder.angle13Terms.minAngle.push_back(contribs.angle13Terms.minAngle[i]);
    contribHolder.angle13Terms.maxAngle.push_back(contribs.angle13Terms.maxAngle[i]);
  }

  // Long range distance terms
  for (size_t i = 0; i < contribs.longRangeDistTerms.idx1.size(); i++) {
    contribHolder.longRangeDistTerms.idx1.push_back(contribs.longRangeDistTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.longRangeDistTerms.idx2.push_back(contribs.longRangeDistTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.longRangeDistTerms.minLen.push_back(contribs.longRangeDistTerms.minLen[i]);
    contribHolder.longRangeDistTerms.maxLen.push_back(contribs.longRangeDistTerms.maxLen[i]);
    contribHolder.longRangeDistTerms.forceConstant.push_back(contribs.longRangeDistTerms.forceConstant[i]);
  }
}

void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem,
                        const int                      dimension,
                        std::vector<int>&              ctxAtomStarts,
                        std::vector<double>&           ctxPositions,
                        std::vector<int>*              atomNumbers) {
  // First update context data
  addMoleculeToContextWithPositions(positions, dimension, ctxAtomStarts, ctxPositions);

  // Then update the molecular system
  addMoleculeToMolecularSystem(contribs,
                               positions.size() / dimension,
                               dimension,
                               ctxAtomStarts,
                               molSystem,
                               atomNumbers);
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
  deviceContribs.chiralTerms.weight.setFromVector(hostContribs.chiralTerms.weight);

  // FourthDim term
  deviceContribs.fourthTerms.idx.setFromVector(hostContribs.fourthTerms.idx);
  deviceContribs.fourthTerms.weight.setFromVector(hostContribs.fourthTerms.weight);

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
  devBuffers.atomNumbers.setStream(stream);
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
  devBuffers.contribs.chiralTerms.weight.setStream(stream);
  devBuffers.contribs.fourthTerms.idx.setStream(stream);
  devBuffers.contribs.fourthTerms.weight.setStream(stream);
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

void allocateDim4ConversionBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                   BatchedMolecularDeviceBuffers&    molSystemDevice) {
  nvMolKit::FFKernelUtils::allocateDim4ConversionBuffers(molSystemHost,
                                                         molSystemDevice,
                                                         molSystemHost.indices.distTermStarts.size() - 1);
}

// TODO: More sophisticated error handling for energy and gradient.
cudaError_t computeEnergy(BatchedMolecularDeviceBuffers&             molSystemDevice,
                          const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                          const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
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
                                            contribs.chiralTerms.weight.data(),
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
                                      contribs.fourthTerms.weight.data(),
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
                                              contribs.chiralTerms.weight.data(),
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
                                        contribs.fourthTerms.weight.data(),
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
