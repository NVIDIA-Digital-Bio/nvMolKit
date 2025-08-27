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

#include "kernel_utils.cuh"
#include "mmff.h"
#include "mmff_kernels.h"

namespace nvMolKit {
namespace MMFF {
void setStreams(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream) {
  molSystemDevice.atomNumbers.setStream(stream);
  molSystemDevice.positions.setStream(stream);
  molSystemDevice.grad.setStream(stream);
  molSystemDevice.energyOuts.setStream(stream);
  molSystemDevice.energyBuffer.setStream(stream);

  auto& deviceContribs = molSystemDevice.contribs;
  // Bond terms
  deviceContribs.bondTerms.idx1.setStream(stream);
  deviceContribs.bondTerms.idx2.setStream(stream);
  deviceContribs.bondTerms.r0.setStream(stream);
  deviceContribs.bondTerms.kb.setStream(stream);
  // Angle terms
  deviceContribs.angleTerms.idx1.setStream(stream);
  deviceContribs.angleTerms.idx2.setStream(stream);
  deviceContribs.angleTerms.idx3.setStream(stream);
  deviceContribs.angleTerms.theta0.setStream(stream);
  deviceContribs.angleTerms.ka.setStream(stream);
  deviceContribs.angleTerms.isLinear.setStream(stream);

  // Bend terms
  deviceContribs.bendTerms.idx1.setStream(stream);
  deviceContribs.bendTerms.idx2.setStream(stream);
  deviceContribs.bendTerms.idx3.setStream(stream);
  deviceContribs.bendTerms.theta0.setStream(stream);
  deviceContribs.bendTerms.restLen1.setStream(stream);
  deviceContribs.bendTerms.restLen2.setStream(stream);
  deviceContribs.bendTerms.forceConst1.setStream(stream);
  deviceContribs.bendTerms.forceConst2.setStream(stream);

  // Oop terms
  deviceContribs.oopTerms.idx1.setStream(stream);
  deviceContribs.oopTerms.idx2.setStream(stream);
  deviceContribs.oopTerms.idx3.setStream(stream);
  deviceContribs.oopTerms.idx4.setStream(stream);
  deviceContribs.oopTerms.koop.setStream(stream);

  // Torsion terms
  deviceContribs.torsionTerms.idx1.setStream(stream);
  deviceContribs.torsionTerms.idx2.setStream(stream);
  deviceContribs.torsionTerms.idx3.setStream(stream);
  deviceContribs.torsionTerms.idx4.setStream(stream);
  deviceContribs.torsionTerms.V1.setStream(stream);
  deviceContribs.torsionTerms.V2.setStream(stream);
  deviceContribs.torsionTerms.V3.setStream(stream);

  // Vdw terms
  deviceContribs.vdwTerms.idx1.setStream(stream);
  deviceContribs.vdwTerms.idx2.setStream(stream);
  deviceContribs.vdwTerms.R_ij_star.setStream(stream);
  deviceContribs.vdwTerms.wellDepth.setStream(stream);

  // Ele terms
  deviceContribs.eleTerms.idx1.setStream(stream);
  deviceContribs.eleTerms.idx2.setStream(stream);
  deviceContribs.eleTerms.chargeTerm.setStream(stream);
  deviceContribs.eleTerms.dielModel.setStream(stream);
  deviceContribs.eleTerms.is1_4.setStream(stream);

  // Indices
  molSystemDevice.indices.atomStarts.setStream(stream);
  molSystemDevice.indices.energyBufferStarts.setStream(stream);
  molSystemDevice.indices.atomIdxToBatchIdx.setStream(stream);
  molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.setStream(stream);

  molSystemDevice.indices.bondTermStarts.setStream(stream);
  molSystemDevice.indices.angleTermStarts.setStream(stream);
  molSystemDevice.indices.bendTermStarts.setStream(stream);
  molSystemDevice.indices.oopTermStarts.setStream(stream);
  molSystemDevice.indices.torsionTermStarts.setStream(stream);
  molSystemDevice.indices.vdwTermStarts.setStream(stream);
  molSystemDevice.indices.eleTermStarts.setStream(stream);
}

void sendContribsAndIndicesToDevice(const BatchedMolecularSystemHost& molSystemHost,
                                    BatchedMolecularDeviceBuffers&    molSystemDevice) {
  auto&       deviceContribs = molSystemDevice.contribs;
  const auto& hostContribs   = molSystemHost.contribs;
  // Bond terms
  deviceContribs.bondTerms.idx1.setFromVector(hostContribs.bondTerms.idx1);
  deviceContribs.bondTerms.idx2.setFromVector(hostContribs.bondTerms.idx2);
  deviceContribs.bondTerms.r0.setFromVector(hostContribs.bondTerms.r0);
  deviceContribs.bondTerms.kb.setFromVector(hostContribs.bondTerms.kb);
  // Angle terms
  deviceContribs.angleTerms.idx1.setFromVector(hostContribs.angleTerms.idx1);
  deviceContribs.angleTerms.idx2.setFromVector(hostContribs.angleTerms.idx2);
  deviceContribs.angleTerms.idx3.setFromVector(hostContribs.angleTerms.idx3);
  deviceContribs.angleTerms.theta0.setFromVector(hostContribs.angleTerms.theta0);
  deviceContribs.angleTerms.ka.setFromVector(hostContribs.angleTerms.ka);
  deviceContribs.angleTerms.isLinear.setFromVector(hostContribs.angleTerms.isLinear);

  // Bend terms
  deviceContribs.bendTerms.idx1.setFromVector(hostContribs.bendTerms.idx1);
  deviceContribs.bendTerms.idx2.setFromVector(hostContribs.bendTerms.idx2);
  deviceContribs.bendTerms.idx3.setFromVector(hostContribs.bendTerms.idx3);
  deviceContribs.bendTerms.theta0.setFromVector(hostContribs.bendTerms.theta0);
  deviceContribs.bendTerms.restLen1.setFromVector(hostContribs.bendTerms.restLen1);
  deviceContribs.bendTerms.restLen2.setFromVector(hostContribs.bendTerms.restLen2);
  deviceContribs.bendTerms.forceConst1.setFromVector(hostContribs.bendTerms.forceConst1);
  deviceContribs.bendTerms.forceConst2.setFromVector(hostContribs.bendTerms.forceConst2);

  // Oop terms
  deviceContribs.oopTerms.idx1.setFromVector(hostContribs.oopTerms.idx1);
  deviceContribs.oopTerms.idx2.setFromVector(hostContribs.oopTerms.idx2);
  deviceContribs.oopTerms.idx3.setFromVector(hostContribs.oopTerms.idx3);
  deviceContribs.oopTerms.idx4.setFromVector(hostContribs.oopTerms.idx4);
  deviceContribs.oopTerms.koop.setFromVector(hostContribs.oopTerms.koop);

  // Torsion terms
  deviceContribs.torsionTerms.idx1.setFromVector(hostContribs.torsionTerms.idx1);
  deviceContribs.torsionTerms.idx2.setFromVector(hostContribs.torsionTerms.idx2);
  deviceContribs.torsionTerms.idx3.setFromVector(hostContribs.torsionTerms.idx3);
  deviceContribs.torsionTerms.idx4.setFromVector(hostContribs.torsionTerms.idx4);
  deviceContribs.torsionTerms.V1.setFromVector(hostContribs.torsionTerms.V1);
  deviceContribs.torsionTerms.V2.setFromVector(hostContribs.torsionTerms.V2);
  deviceContribs.torsionTerms.V3.setFromVector(hostContribs.torsionTerms.V3);

  // Vdw terms
  deviceContribs.vdwTerms.idx1.setFromVector(hostContribs.vdwTerms.idx1);
  deviceContribs.vdwTerms.idx2.setFromVector(hostContribs.vdwTerms.idx2);
  deviceContribs.vdwTerms.R_ij_star.setFromVector(hostContribs.vdwTerms.R_ij_star);
  deviceContribs.vdwTerms.wellDepth.setFromVector(hostContribs.vdwTerms.wellDepth);

  // Ele terms
  deviceContribs.eleTerms.idx1.setFromVector(hostContribs.eleTerms.idx1);
  deviceContribs.eleTerms.idx2.setFromVector(hostContribs.eleTerms.idx2);
  deviceContribs.eleTerms.chargeTerm.setFromVector(hostContribs.eleTerms.chargeTerm);
  deviceContribs.eleTerms.dielModel.setFromVector(hostContribs.eleTerms.dielModel);
  deviceContribs.eleTerms.is1_4.setFromVector(hostContribs.eleTerms.is1_4);

  // Indices
  molSystemDevice.indices.atomStarts.setFromVector(molSystemHost.indices.atomStarts);
  molSystemDevice.indices.energyBufferStarts.setFromVector(molSystemHost.indices.energyBufferStarts);
  molSystemDevice.indices.atomIdxToBatchIdx.setFromVector(molSystemHost.indices.atomIdxToBatchIdx);
  molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.setFromVector(
    molSystemHost.indices.energyBufferBlockIdxToBatchIdx);

  molSystemDevice.indices.bondTermStarts.setFromVector(molSystemHost.indices.bondTermStarts);
  molSystemDevice.indices.angleTermStarts.setFromVector(molSystemHost.indices.angleTermStarts);
  molSystemDevice.indices.bendTermStarts.setFromVector(molSystemHost.indices.bendTermStarts);
  molSystemDevice.indices.oopTermStarts.setFromVector(molSystemHost.indices.oopTermStarts);
  molSystemDevice.indices.torsionTermStarts.setFromVector(molSystemHost.indices.torsionTermStarts);
  molSystemDevice.indices.vdwTermStarts.setFromVector(molSystemHost.indices.vdwTermStarts);
  molSystemDevice.indices.eleTermStarts.setFromVector(molSystemHost.indices.eleTermStarts);
}

void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem,
                        std::vector<int>*              atomNumbers) {
  const int previousLastAtomIndex = molSystem.indices.atomStarts.back();
  const int numBatches            = molSystem.indices.atomStarts.size() - 1;
  const int newNumAtoms           = positions.size() / 3;
  molSystem.indices.atomStarts.push_back(molSystem.indices.atomStarts.back() + newNumAtoms);
  molSystem.maxNumAtoms = std::max(molSystem.maxNumAtoms, newNumAtoms);

  if (atomNumbers) {
    molSystem.atomNumbers.insert(molSystem.atomNumbers.end(), atomNumbers->begin(), atomNumbers->end());
  }

  auto& indexHolder   = molSystem.indices;
  auto& contribHolder = molSystem.contribs;

  // First append positions
  molSystem.positions.insert(molSystem.positions.end(), positions.begin(), positions.end());

  // Next handle indices
  indexHolder.atomIdxToBatchIdx.resize(molSystem.positions.size() / 3, numBatches);
  indexHolder.bondTermStarts.push_back(indexHolder.bondTermStarts.back() + contribs.bondTerms.idx1.size());
  indexHolder.angleTermStarts.push_back(indexHolder.angleTermStarts.back() + contribs.angleTerms.idx1.size());
  indexHolder.bendTermStarts.push_back(indexHolder.bendTermStarts.back() + contribs.bendTerms.idx1.size());
  indexHolder.oopTermStarts.push_back(indexHolder.oopTermStarts.back() + contribs.oopTerms.idx1.size());
  indexHolder.torsionTermStarts.push_back(indexHolder.torsionTermStarts.back() + contribs.torsionTerms.idx1.size());
  indexHolder.vdwTermStarts.push_back(indexHolder.vdwTermStarts.back() + contribs.vdwTerms.idx1.size());
  indexHolder.eleTermStarts.push_back(indexHolder.eleTermStarts.back() + contribs.eleTerms.idx1.size());

  int maxNumContribs = 0;
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.bondTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.angleTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.bendTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.oopTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.torsionTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.vdwTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.eleTerms.idx1.size());

  const int numBlocksNeeded  = (maxNumContribs + 127) / nvMolKit::FFKernelUtils::blockSizeEnergyReduction;
  const int numThreadsNeeded = numBlocksNeeded * nvMolKit::FFKernelUtils::blockSizeEnergyReduction;

  indexHolder.energyBufferStarts.push_back(numThreadsNeeded + indexHolder.energyBufferStarts.back());
  for (int i = 0; i < numBlocksNeeded; i++) {
    indexHolder.energyBufferBlockIdxToBatchIdx.push_back(numBatches);
  }

  // Now append the contribs, updating indices.
  // Bond terms
  for (size_t i = 0; i < contribs.bondTerms.idx1.size(); i++) {
    contribHolder.bondTerms.idx1.push_back(contribs.bondTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.bondTerms.idx2.push_back(contribs.bondTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.bondTerms.r0.push_back(contribs.bondTerms.r0[i]);
    contribHolder.bondTerms.kb.push_back(contribs.bondTerms.kb[i]);
  }
  // Angle terms
  for (size_t i = 0; i < contribs.angleTerms.idx1.size(); i++) {
    contribHolder.angleTerms.idx1.push_back(contribs.angleTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.angleTerms.idx2.push_back(contribs.angleTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.angleTerms.idx3.push_back(contribs.angleTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.angleTerms.theta0.push_back(contribs.angleTerms.theta0[i]);
    contribHolder.angleTerms.ka.push_back(contribs.angleTerms.ka[i]);
    contribHolder.angleTerms.isLinear.push_back(contribs.angleTerms.isLinear[i]);
  }
  // Bend terms
  for (size_t i = 0; i < contribs.bendTerms.idx1.size(); i++) {
    contribHolder.bendTerms.idx1.push_back(contribs.bendTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.bendTerms.idx2.push_back(contribs.bendTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.bendTerms.idx3.push_back(contribs.bendTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.bendTerms.theta0.push_back(contribs.bendTerms.theta0[i]);
    contribHolder.bendTerms.restLen1.push_back(contribs.bendTerms.restLen1[i]);
    contribHolder.bendTerms.restLen2.push_back(contribs.bendTerms.restLen2[i]);
    contribHolder.bendTerms.forceConst1.push_back(contribs.bendTerms.forceConst1[i]);
    contribHolder.bendTerms.forceConst2.push_back(contribs.bendTerms.forceConst2[i]);
  }
  // Oop terms
  for (size_t i = 0; i < contribs.oopTerms.idx1.size(); i++) {
    contribHolder.oopTerms.idx1.push_back(contribs.oopTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.oopTerms.idx2.push_back(contribs.oopTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.oopTerms.idx3.push_back(contribs.oopTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.oopTerms.idx4.push_back(contribs.oopTerms.idx4[i] + previousLastAtomIndex);
    contribHolder.oopTerms.koop.push_back(contribs.oopTerms.koop[i]);
  }
  // Torsion terms
  for (size_t i = 0; i < contribs.torsionTerms.idx1.size(); i++) {
    contribHolder.torsionTerms.idx1.push_back(contribs.torsionTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.torsionTerms.idx2.push_back(contribs.torsionTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.torsionTerms.idx3.push_back(contribs.torsionTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.torsionTerms.idx4.push_back(contribs.torsionTerms.idx4[i] + previousLastAtomIndex);
    contribHolder.torsionTerms.V1.push_back(contribs.torsionTerms.V1[i]);
    contribHolder.torsionTerms.V2.push_back(contribs.torsionTerms.V2[i]);
    contribHolder.torsionTerms.V3.push_back(contribs.torsionTerms.V3[i]);
  }
  // Vdw terms
  for (size_t i = 0; i < contribs.vdwTerms.idx1.size(); i++) {
    contribHolder.vdwTerms.idx1.push_back(contribs.vdwTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.vdwTerms.idx2.push_back(contribs.vdwTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.vdwTerms.R_ij_star.push_back(contribs.vdwTerms.R_ij_star[i]);
    contribHolder.vdwTerms.wellDepth.push_back(contribs.vdwTerms.wellDepth[i]);
  }
  // Ele terms
  for (size_t i = 0; i < contribs.eleTerms.idx1.size(); i++) {
    contribHolder.eleTerms.idx1.push_back(contribs.eleTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.eleTerms.idx2.push_back(contribs.eleTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.eleTerms.chargeTerm.push_back(contribs.eleTerms.chargeTerm[i]);
    contribHolder.eleTerms.dielModel.push_back(contribs.eleTerms.dielModel[i]);
    contribHolder.eleTerms.is1_4.push_back(contribs.eleTerms.is1_4[i]);
  }
}

void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice) {
  nvMolKit::FFKernelUtils::allocateIntermediateBuffers(molSystemHost,
                                                       molSystemDevice,
                                                       molSystemHost.indices.atomStarts.size() - 1);
}

void allocateDim4ConversionBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                   BatchedMolecularDeviceBuffers&    molSystemDevice) {
  nvMolKit::FFKernelUtils::allocateDim4ConversionBuffers(molSystemHost,
                                                         molSystemDevice,
                                                         molSystemHost.indices.atomStarts.size() - 1);
}

// TODO: More sophisticated error handling for energy and gradient.
cudaError_t computeEnergy(BatchedMolecularDeviceBuffers& molSystemDevice, const double* coords, cudaStream_t stream) {
  // Prechecks - tempstorage allocated, energybuffer allocated
  assert(molSystemDevice.energyBuffer.size() > 0);
  assert(molSystemDevice.energyOuts.data() != nullptr);

  // Dispatch each term if there is a contrib for it.
  const auto& contribs = molSystemDevice.contribs;

  const double* positions = coords != nullptr ? coords : molSystemDevice.positions.data();

  cudaError_t err = cudaSuccess;
  if (contribs.bondTerms.idx1.size() > 0) {
    err = launchBondStretchEnergyKernel(contribs.bondTerms.idx1.size(),
                                        contribs.bondTerms.idx1.data(),
                                        contribs.bondTerms.idx2.data(),
                                        contribs.bondTerms.r0.data(),
                                        contribs.bondTerms.kb.data(),
                                        positions,
                                        molSystemDevice.energyBuffer.data(),
                                        molSystemDevice.indices.energyBufferStarts.data(),
                                        molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                        molSystemDevice.indices.bondTermStarts.data(),
                                        stream);
  }
  if (err == cudaSuccess && contribs.angleTerms.idx1.size() > 0) {
    err = launchAngleBendEnergyKernel(contribs.angleTerms.idx1.size(),
                                      contribs.angleTerms.idx1.data(),
                                      contribs.angleTerms.idx2.data(),
                                      contribs.angleTerms.idx3.data(),
                                      contribs.angleTerms.theta0.data(),
                                      contribs.angleTerms.ka.data(),
                                      contribs.angleTerms.isLinear.data(),
                                      positions,
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferStarts.data(),
                                      molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                      molSystemDevice.indices.angleTermStarts.data(),
                                      stream);
  }
  if (err == cudaSuccess && contribs.bendTerms.idx1.size() > 0) {
    err = launchBendStretchEnergyKernel(contribs.bendTerms.idx1.size(),
                                        contribs.bendTerms.idx1.data(),
                                        contribs.bendTerms.idx2.data(),
                                        contribs.bendTerms.idx3.data(),
                                        contribs.bendTerms.theta0.data(),
                                        contribs.bendTerms.restLen1.data(),
                                        contribs.bendTerms.restLen2.data(),
                                        contribs.bendTerms.forceConst1.data(),
                                        contribs.bendTerms.forceConst2.data(),
                                        positions,
                                        molSystemDevice.energyBuffer.data(),
                                        molSystemDevice.indices.energyBufferStarts.data(),
                                        molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                        molSystemDevice.indices.bendTermStarts.data(),
                                        stream);
  }
  if (err == cudaSuccess && contribs.oopTerms.idx1.size() > 0) {
    err = launchOopBendEnergyKernel(contribs.oopTerms.idx1.size(),
                                    contribs.oopTerms.idx1.data(),
                                    contribs.oopTerms.idx2.data(),
                                    contribs.oopTerms.idx3.data(),
                                    contribs.oopTerms.idx4.data(),
                                    contribs.oopTerms.koop.data(),
                                    positions,
                                    molSystemDevice.energyBuffer.data(),
                                    molSystemDevice.indices.energyBufferStarts.data(),
                                    molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                    molSystemDevice.indices.oopTermStarts.data(),
                                    stream);
  }
  if (err == cudaSuccess && contribs.torsionTerms.idx1.size() > 0) {
    err = launchTorsionEnergyKernel(contribs.torsionTerms.idx1.size(),
                                    contribs.torsionTerms.idx1.data(),
                                    contribs.torsionTerms.idx2.data(),
                                    contribs.torsionTerms.idx3.data(),
                                    contribs.torsionTerms.idx4.data(),
                                    contribs.torsionTerms.V1.data(),
                                    contribs.torsionTerms.V2.data(),
                                    contribs.torsionTerms.V3.data(),
                                    positions,
                                    molSystemDevice.energyBuffer.data(),
                                    molSystemDevice.indices.energyBufferStarts.data(),
                                    molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                    molSystemDevice.indices.torsionTermStarts.data(),
                                    stream);
  }
  if (err == cudaSuccess && contribs.vdwTerms.idx1.size() > 0) {
    err = launchVdwEnergyKernel(contribs.vdwTerms.idx1.size(),
                                contribs.vdwTerms.idx1.data(),
                                contribs.vdwTerms.idx2.data(),
                                contribs.vdwTerms.R_ij_star.data(),
                                contribs.vdwTerms.wellDepth.data(),
                                positions,
                                molSystemDevice.energyBuffer.data(),
                                molSystemDevice.indices.energyBufferStarts.data(),
                                molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                molSystemDevice.indices.vdwTermStarts.data(),
                                stream);
  }
  if (err == cudaSuccess && contribs.eleTerms.idx1.size() > 0) {
    err = launchEleEnergyKernel(contribs.eleTerms.idx1.size(),
                                contribs.eleTerms.idx1.data(),
                                contribs.eleTerms.idx2.data(),
                                contribs.eleTerms.chargeTerm.data(),
                                contribs.eleTerms.dielModel.data(),
                                contribs.eleTerms.is1_4.data(),
                                positions,
                                molSystemDevice.energyBuffer.data(),
                                molSystemDevice.indices.energyBufferStarts.data(),
                                molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                molSystemDevice.indices.eleTermStarts.data(),
                                stream);
  }

  if (err == cudaSuccess) {
    // Now reduce the energy buffer
    return launchReduceEnergiesKernel(molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.size(),
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.data(),
                                      molSystemDevice.energyOuts.data(),
                                      stream);
  }
  return err;
}

cudaError_t computeGradients(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream) {
  // Dispatch each term if there is a contrib for it.
  const auto& contribs = molSystemDevice.contribs;

  cudaError_t err = cudaSuccess;
  if (contribs.bondTerms.idx1.size() > 0) {
    err = launchBondStretchGradientKernel(contribs.bondTerms.idx1.size(),
                                          contribs.bondTerms.idx1.data(),
                                          contribs.bondTerms.idx2.data(),
                                          contribs.bondTerms.r0.data(),
                                          contribs.bondTerms.kb.data(),
                                          molSystemDevice.positions.data(),
                                          molSystemDevice.grad.data(),
                                          stream);
  }
  if (err == cudaSuccess && contribs.angleTerms.idx1.size() > 0) {
    err = launchAngleBendGradientKernel(contribs.angleTerms.idx1.size(),
                                        contribs.angleTerms.idx1.data(),
                                        contribs.angleTerms.idx2.data(),
                                        contribs.angleTerms.idx3.data(),
                                        contribs.angleTerms.theta0.data(),
                                        contribs.angleTerms.ka.data(),
                                        contribs.angleTerms.isLinear.data(),
                                        molSystemDevice.positions.data(),
                                        molSystemDevice.grad.data(),
                                        stream);
  }
  if (err == cudaSuccess && contribs.bendTerms.idx1.size() > 0) {
    err = launchBendStretchGradientKernel(contribs.bendTerms.idx1.size(),
                                          contribs.bendTerms.idx1.data(),
                                          contribs.bendTerms.idx2.data(),
                                          contribs.bendTerms.idx3.data(),
                                          contribs.bendTerms.theta0.data(),
                                          contribs.bendTerms.restLen1.data(),
                                          contribs.bendTerms.restLen2.data(),
                                          contribs.bendTerms.forceConst1.data(),
                                          contribs.bendTerms.forceConst2.data(),
                                          molSystemDevice.positions.data(),
                                          molSystemDevice.grad.data(),
                                          stream);
  }
  if (err == cudaSuccess && contribs.oopTerms.idx1.size() > 0) {
    err = launchOopBendGradientKernel(contribs.oopTerms.idx1.size(),
                                      contribs.oopTerms.idx1.data(),
                                      contribs.oopTerms.idx2.data(),
                                      contribs.oopTerms.idx3.data(),
                                      contribs.oopTerms.idx4.data(),
                                      contribs.oopTerms.koop.data(),
                                      molSystemDevice.positions.data(),
                                      molSystemDevice.grad.data(),
                                      stream);
  }
  if (err == cudaSuccess && contribs.torsionTerms.idx1.size() > 0) {
    err = launchTorsionGradientKernel(contribs.torsionTerms.idx1.size(),
                                      contribs.torsionTerms.idx1.data(),
                                      contribs.torsionTerms.idx2.data(),
                                      contribs.torsionTerms.idx3.data(),
                                      contribs.torsionTerms.idx4.data(),
                                      contribs.torsionTerms.V1.data(),
                                      contribs.torsionTerms.V2.data(),
                                      contribs.torsionTerms.V3.data(),
                                      molSystemDevice.positions.data(),
                                      molSystemDevice.grad.data(),
                                      stream);
  }
  if (err == cudaSuccess && contribs.vdwTerms.idx1.size() > 0) {
    err = launchVdwGradientKernel(contribs.vdwTerms.idx1.size(),
                                  contribs.vdwTerms.idx1.data(),
                                  contribs.vdwTerms.idx2.data(),
                                  contribs.vdwTerms.R_ij_star.data(),
                                  contribs.vdwTerms.wellDepth.data(),
                                  molSystemDevice.positions.data(),
                                  molSystemDevice.grad.data(),
                                  stream);
  }
  if (err == cudaSuccess && contribs.eleTerms.idx1.size() > 0) {
    err = launchEleGradientKernel(contribs.eleTerms.idx1.size(),
                                  contribs.eleTerms.idx1.data(),
                                  contribs.eleTerms.idx2.data(),
                                  contribs.eleTerms.chargeTerm.data(),
                                  contribs.eleTerms.dielModel.data(),
                                  contribs.eleTerms.is1_4.data(),
                                  molSystemDevice.positions.data(),
                                  molSystemDevice.grad.data(),
                                  stream);
  }
  return err;
}

}  // namespace MMFF
}  // namespace nvMolKit
