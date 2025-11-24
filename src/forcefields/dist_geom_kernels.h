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

#ifndef NVMOLKIT_DISTGEOM_KERNELS_H
#define NVMOLKIT_DISTGEOM_KERNELS_H

#include <cuda_runtime.h>

namespace nvMolKit {
namespace DistGeom {
cudaError_t launchDistViolationEnergyKernel(int      numDist,
                                            const int*     idx1,
                                            const int*     idx2,
                                            const double*  lb2,
                                            const double*  ub2,
                                            const double*  weight,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     distTermStarts,
                                            const int*     atomStarts,
                                            const int      dimension,
                                            const uint8_t* activeThisStage = nullptr,
                                            cudaStream_t   stream          = 0);

cudaError_t launchDistViolationGradientKernel(int      numDist,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const double*  lb2,
                                              const double*  ub2,
                                              const double*  weight,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const int      dimension,
                                              const uint8_t* activeThisStage = nullptr,
                                              cudaStream_t   stream          = 0);

cudaError_t launchChiralViolationEnergyKernel(int      numChiral,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const int*     idx3,
                                              const int*     idx4,
                                              const double*  volLower,
                                              const double*  volUpper,
                                              double   weight,
                                              const double*  pos,
                                              double*        energyBuffer,
                                              const int*     energyBufferStarts,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     chiralTermStarts,
                                              const int*     atomStarts,
                                              int      dimension,
                                              const uint8_t* activeThisStage = nullptr,
                                              cudaStream_t   stream          = 0);

cudaError_t launchChiralViolationGradientKernel(int      numChiral,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const int*     idx4,
                                                const double*  volLower,
                                                const double*  volUpper,
                                                double   weight,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                int      dimension,
                                                const uint8_t* activeThisStage = nullptr,
                                                cudaStream_t   stream          = 0);

cudaError_t launchFourthDimEnergyKernel(int      numFD,
                                        const int*     idx,
                                        double   weight,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     fourthTermStarts,
                                        const int*     atomStarts,
                                        int      dimension,
                                        const uint8_t* activeThisStage = nullptr,
                                        cudaStream_t   stream          = 0);

cudaError_t launchFourthDimGradientKernel(int      numFD,
                                          const int*     idx,
                                          double   weight,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          int      dimension,
                                          const uint8_t* activeThisStage = nullptr,
                                          cudaStream_t   stream          = 0);

// Experimental torsion angle contribution kernels
cudaError_t launchTorsionAngleEnergyKernel(int      numTorsion,
                                           const int*     idx1,
                                           const int*     idx2,
                                           const int*     idx3,
                                           const int*     idx4,
                                           const double*  forceConstant,
                                           const int*     signs,
                                           const double*  pos,
                                           double*        energyBuffer,
                                           const int*     energyBufferStarts,
                                           const int*     atomIdxToBatchIdx,
                                           const int*     torsionTermStarts,
                                           const int*     atomStarts,
                                           const uint8_t* activeThisStage = nullptr,
                                           cudaStream_t   stream          = 0);

cudaError_t launchTorsionAngleGradientKernel(int      numTorsion,
                                             const int*     idx1,
                                             const int*     idx2,
                                             const int*     idx3,
                                             const int*     idx4,
                                             const double*  forceConstant,
                                             const int*     signs,
                                             const double*  pos,
                                             double*        grad,
                                             const int*     atomIdxToBatchIdx,
                                             const int*     atomStarts,
                                             const uint8_t* activeThisStage = nullptr,
                                             cudaStream_t   stream          = 0);

// Improper torsion (inversion) contribution kernels
cudaError_t launchInversionEnergyKernel(int      numInversion,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const int*     idx4,
                                        const int*     at2AtomicNum,
                                        const uint8_t* isCBoundToO,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  forceConstant,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     inversionTermStarts,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage = nullptr,
                                        cudaStream_t   stream          = 0);

cudaError_t launchInversionGradientKernel(int      numInversion,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const int*     idx4,
                                          const int*     at2AtomicNum,
                                          const uint8_t* isCBoundToO,
                                          const double*  C0,
                                          const double*  C1,
                                          const double*  C2,
                                          const double*  forceConstant,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          const uint8_t* activeThisStage = nullptr,
                                          cudaStream_t   stream          = 0);

// Distance constraint contribution kernels
cudaError_t launchDistanceConstraintEnergyKernel(int      numDist,
                                                 const int*     idx1,
                                                 const int*     idx2,
                                                 const double*  minLen,
                                                 const double*  maxLen,
                                                 const double*  forceConstant,
                                                 const double*  pos,
                                                 double*        energyBuffer,
                                                 const int*     energyBufferStarts,
                                                 const int*     atomIdxToBatchIdx,
                                                 const int*     distTermStarts,
                                                 const int*     atomStarts,
                                                 const uint8_t* activeThisStage = nullptr,
                                                 cudaStream_t   stream          = 0);

cudaError_t launchDistanceConstraintGradientKernel(int      numDist,
                                                   const int*     idx1,
                                                   const int*     idx2,
                                                   const double*  minLen,
                                                   const double*  maxLen,
                                                   const double*  forceConstant,
                                                   const double*  pos,
                                                   double*        grad,
                                                   const int*     atomIdxToBatchIdx,
                                                   const int*     atomStarts,
                                                   const uint8_t* activeThisStage = nullptr,
                                                   cudaStream_t   stream          = 0);

// Angle constraint contribution kernels
constexpr double defaultAngleForceConstant = 1.0;
cudaError_t      launchAngleConstraintEnergyKernel(int      numAngle,
                                                   const int*     idx1,
                                                   const int*     idx2,
                                                   const int*     idx3,
                                                   const double*  minAngle,
                                                   const double*  maxAngle,
                                                   const double*  pos,
                                                   double*        energyBuffer,
                                                   const int*     energyBufferStarts,
                                                   const int*     atomIdxToBatchIdx,
                                                   const int*     angleTermStarts,
                                                   const int*     atomStarts,
                                                   const uint8_t* activeThisStage = nullptr,
                                                   double         forceConstant   = defaultAngleForceConstant,
                                                   cudaStream_t   stream          = 0);

cudaError_t launchAngleConstraintGradientKernel(int      numAngle,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const double*  minAngle,
                                                const double*  maxAngle,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                const uint8_t* activeThisStage = nullptr,
                                                double         forceConstant   = defaultAngleForceConstant,
                                                cudaStream_t   stream          = 0);

cudaError_t launchReduceEnergiesKernel(int            numBlocks,
                                       const double*  energyBuffer,
                                       const int*     energyBufferBlockIdxToBatchIdx,
                                       double*        outs,
                                       const uint8_t* activeThisStage = nullptr,
                                       cudaStream_t   stream          = 0);
}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_KERNELS_H
