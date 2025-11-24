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

#ifndef NVMOLKIT_MMFF_KERNELS_H
#define NVMOLKIT_MMFF_KERNELS_H

#include <cuda_runtime.h>

namespace nvMolKit {
namespace MMFF {

cudaError_t launchBondStretchEnergyKernel(int           numBonds,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const double* r0,
                                          const double* kb,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream = 0);

cudaError_t launchBondStretchGradientKernel(int           numBonds,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const double* r0,
                                            const double* kb,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream = 0);

cudaError_t launchAngleBendEnergyKernel(int            numAngles,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const double*  theta0,
                                        const double*  ka,
                                        const uint8_t* isLinear,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomBatchMap,
                                        const int*     termBatchStarts,
                                        cudaStream_t   stream = 0);

cudaError_t launchAngleBendGradientKernel(int            numAngles,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const double*  theta0,
                                          const double*  ka,
                                          const uint8_t* isLinear,
                                          const double*  pos,
                                          double*        grad,
                                          cudaStream_t   stream = 0);

cudaError_t launchBendStretchEnergyKernel(int           numAngles,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const int*    idx3,
                                          const double* theta0,
                                          const double* restLen1,
                                          const double* restLen2,
                                          const double* forceConst1,
                                          const double* forceConst2,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream = 0);

cudaError_t launchBendStretchGradientKernel(int           numAngles,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const int*    idx3,
                                            const double* theta0,
                                            const double* restLen1,
                                            const double* restLen2,
                                            const double* forceConst1,
                                            const double* forceConst2,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream = 0);

cudaError_t launchOopBendEnergyKernel(int           numOopBends,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const double* koop,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts,
                                      cudaStream_t  stream = 0);

cudaError_t launchOopBendGradientKernel(int           numOopBends,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const double* koop,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream = 0);

cudaError_t launchTorsionEnergyKernel(int           numTorsions,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const float*  V1,
                                      const float*  V2,
                                      const float*  V3,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts,
                                      cudaStream_t  stream = 0);

cudaError_t launchTorsionGradientKernel(int           numTorsions,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const float*  V1,
                                        const float*  V2,
                                        const float*  V3,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream = 0);

cudaError_t launchVdwEnergyKernel(int           numVdws,
                                  const int*    idx1,
                                  const int*    idx2,
                                  const double* R_ij_star,
                                  const double* wellDepth,
                                  const double* pos,
                                  double*       energyBuffer,
                                  const int*    energyBufferStarts,
                                  const int*    atomBatchMap,
                                  const int*    termBatchStarts,
                                  cudaStream_t  stream = 0);

cudaError_t launchVdwGradientKernel(int           numVdws,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const double* R_ij_star,
                                    const double* wellDepth,
                                    const double* pos,
                                    double*       grad,
                                    cudaStream_t  stream = 0);

cudaError_t launchEleEnergyKernel(int            numEles,
                                  const int*     idx1,
                                  const int*     idx2,
                                  const double*  chargeTerm,
                                  const uint8_t* dielModel,
                                  const uint8_t* is1_4,
                                  const double*  pos,
                                  double*        energyBuffer,
                                  const int*     energyBufferStarts,
                                  const int*     atomBatchMap,
                                  const int*     termBatchStarts,
                                  cudaStream_t   stream = 0);

cudaError_t launchEleGradientKernel(int            numEles,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const double*  chargeTerm,
                                    const uint8_t* dielModel,
                                    const uint8_t* is1_4,
                                    const double*  pos,
                                    double*        grad,
                                    cudaStream_t   stream = 0);

//! Reduce the energy buffer to the output energies.
//!
//! Energies written to energyBuffer are accumulated in outs, with one term in outs corresponding to each molecule in
//! the batch.
//!
//! \param numBlocks Number of blocks
//! \param energyBuffer Energy terms, guaranteed each block is only assigned to one molecule.
//! \param energyBufferBlockIdxToBatchIdx Matching of blocks to output indices.
//! \param outs Output energies
//! \param stream
//! \return
cudaError_t launchReduceEnergiesKernel(int           numBlocks,
                                       const double* energyBuffer,
                                       const int*    energyBufferBlockIdxToBatchIdx,
                                       double*       outs,
                                       cudaStream_t  stream = 0);
}  // namespace MMFF
}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_KERNELS_H
