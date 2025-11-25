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

#include <cub/cub.cuh>

#include "dist_geom_kernels.h"
#include "dist_geom_kernels_device.cuh"
#include "kernel_utils.cuh"

using namespace nvMolKit::FFKernelUtils;

namespace nvMolKit {
namespace DistGeom {

__global__ void DistViolationEnergyKernel(const int      numDist,
                                          const int*     idx1s,
                                          const int*     idx2s,
                                          const double*  lb2s,
                                          const double*  ub2s,
                                          const double*  weights,
                                          const double*  pos,
                                          double*        energyBuffer,
                                          const int*     energyBufferStarts,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     distTermStarts,
                                          const int*     atomStarts,
                                          const int      dimension,
                                          const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2   = idx2s[idx];
      const double lb2    = lb2s[idx];
      const double ub2    = ub2s[idx];
      const double weight = weights[idx];

      const double energy = distViolationEnergy(pos, idx1, idx2, lb2, ub2, weight, dimension);
      if (energy > 0.0) {
        const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, distTermStarts);
        energyBuffer[outputIdx] += energy;
      }
    }
  }
}

template <int dimension>
__global__ void DistViolationGradientKernel(const int      numDist,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const double*  lb2s,
                                            const double*  ub2s,
                                            const double*  weights,
                                            const double*  pos,
                                            double*        grad,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     atomStarts,
                                            const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2   = idx2s[idx];
      const double lb2    = lb2s[idx];
      const double ub2    = ub2s[idx];
      const double weight = weights[idx];

      distViolationGrad(pos, idx1, idx2, lb2, ub2, weight, dimension, grad);
    }
  }
}

__global__ void ChiralViolationEnergyKernel(const int      numChiral,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const int*     idx3s,
                                            const int*     idx4s,
                                            const double*  volLower,
                                            const double*  volUpper,
                                            const double   weight,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     chiralTermStarts,
                                            const int*     atomStarts,
                                            const int      dimension,
                                            const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numChiral) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2    = idx2s[idx];
      const int    idx3    = idx3s[idx];
      const int    idx4    = idx4s[idx];
      const double lb      = volLower[idx];
      const double ub      = volUpper[idx];
      const int    posIdx1 = idx1 * dimension;
      const int    posIdx2 = idx2 * dimension;
      const int    posIdx3 = idx3 * dimension;
      const int    posIdx4 = idx4 * dimension;

      double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
      double vol =
        calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

      const double energy    = chiralViolationEnergy(pos, idx1, idx2, idx3, idx4, lb, ub, weight, dimension);
      const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, chiralTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void ChiralViolationGradientKernel(const int      numChiral,
                                              const int*     idx1s,
                                              const int*     idx2s,
                                              const int*     idx3s,
                                              const int*     idx4s,
                                              const double*  volLower,
                                              const double*  volUpper,
                                              const double   weight,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const int      dimension,
                                              const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numChiral) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2    = idx2s[idx];
      const int    idx3    = idx3s[idx];
      const int    idx4    = idx4s[idx];
      const double lb      = volLower[idx];
      const double ub      = volUpper[idx];
      const int    posIdx1 = idx1 * dimension;
      const int    posIdx2 = idx2 * dimension;
      const int    posIdx3 = idx3 * dimension;
      const int    posIdx4 = idx4 * dimension;

      double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
      double vol =
        calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

      if (vol < lb || vol > ub) {
        double preFactor;
        if (vol < lb) {
          preFactor = weight * (vol - lb);
        } else {  // guaranteed != with outer conditional.
          preFactor = weight * (vol - ub);
        }

        atomicAdd(&grad[posIdx1 + 0], preFactor * (v2y * v3z - v2z * v3y));
        atomicAdd(&grad[posIdx1 + 1], preFactor * (v2z * v3x - v2x * v3z));
        atomicAdd(&grad[posIdx1 + 2], preFactor * (v2x * v3y - v2y * v3x));

        atomicAdd(&grad[posIdx2 + 0], preFactor * (v3y * v1z - v3z * v1y));
        atomicAdd(&grad[posIdx2 + 1], preFactor * (v3z * v1x - v3x * v1z));
        atomicAdd(&grad[posIdx2 + 2], preFactor * (v3x * v1y - v3y * v1x));

        atomicAdd(&grad[posIdx3 + 0], preFactor * (v2z * v1y - v2y * v1z));
        atomicAdd(&grad[posIdx3 + 1], preFactor * (v2x * v1z - v2z * v1x));
        atomicAdd(&grad[posIdx3 + 2], preFactor * (v2y * v1x - v2x * v1y));

        double x1 = pos[posIdx1 + 0];
        double y1 = pos[posIdx1 + 1];
        double z1 = pos[posIdx1 + 2];
        double x2 = pos[posIdx2 + 0];
        double y2 = pos[posIdx2 + 1];
        double z2 = pos[posIdx2 + 2];
        double x3 = pos[posIdx3 + 0];
        double y3 = pos[posIdx3 + 1];
        double z3 = pos[posIdx3 + 2];
        atomicAdd(&grad[posIdx4 + 0], preFactor * (z1 * (y2 - y3) + z2 * (y3 - y1) + z3 * (y1 - y2)));
        atomicAdd(&grad[posIdx4 + 1], preFactor * (x1 * (z2 - z3) + x2 * (z3 - z1) + x3 * (z1 - z2)));
        atomicAdd(&grad[posIdx4 + 2], preFactor * (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)));
      }
    }
  }
}

__global__ void fourthDimEnergyKernel(const int      numFD,
                                      const int*     idxs,
                                      const double   weight,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomIdxToBatchIdx,
                                      const int*     fourthTermStarts,
                                      const int*     atomStarts,
                                      const int      dimension,
                                      const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numFD) {
    const int idx1     = idxs[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      unsigned  pid       = idx1 * dimension + 3;
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, fourthTermStarts);
      energyBuffer[outputIdx] += weight * pos[pid] * pos[pid];
    }
  }
}

__global__ void fourthDimGradientKernel(const int      numFD,
                                        const int*     idxs,
                                        const double   weight,
                                        const double*  pos,
                                        double*        grad,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     atomStarts,
                                        const int      dimension,
                                        const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numFD) {
    const int idx1     = idxs[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      int pid = idx1 * dimension + 3;
      grad[pid] += weight * pos[pid];
    }
  }
}

__global__ void TorsionAngleEnergyKernel(const int      numTorsion,
                                         const int*     idx1s,
                                         const int*     idx2s,
                                         const int*     idx3s,
                                         const int*     idx4s,
                                         const double*  forceConstants,
                                         const int*     signs,
                                         const double*  pos,
                                         double*        energyBuffer,
                                         const int*     energyBufferStarts,
                                         const int*     atomIdxToBatchIdx,
                                         const int*     torsionTermStarts,
                                         const int*     atomStarts,
                                         const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int idx2 = idx2s[idx];
      const int idx3 = idx3s[idx];
      const int idx4 = idx4s[idx];

      const double* fc = &forceConstants[idx * 6];
      const int*    s  = &signs[idx * 6];

      const double energy = torsionAngleEnergy(pos, idx1, idx2, idx3, idx4, fc, s);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, torsionTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void TorsionAngleGradientKernel(const int      numTorsion,
                                           const int*     idx1s,
                                           const int*     idx2s,
                                           const int*     idx3s,
                                           const int*     idx4s,
                                           const double*  forceConstants,
                                           const int*     signs,
                                           const double*  pos,
                                           double*        grad,
                                           const int*     atomIdxToBatchIdx,
                                           const int*     atomStarts,
                                           const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      torsionAngleGrad(pos, idx1, idx2s[idx], idx3s[idx], idx4s[idx], &forceConstants[idx * 6], &signs[idx * 6], grad);
    }
  }
}

__global__ void InversionEnergyKernel(const int      numInversion,
                                      const int*     idx1s,
                                      const int*     idx2s,
                                      const int*     idx3s,
                                      const int*     idx4s,
                                      const int*     at2AtomicNum,
                                      const uint8_t* isCBoundToO,
                                      const double*  C0,
                                      const double*  C1,
                                      const double*  C2,
                                      const double*  forceConstants,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomIdxToBatchIdx,
                                      const int*     inversionTermStarts,
                                      const int*     atomStarts,
                                      const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int idx2 = idx2s[idx];
      const int idx3 = idx3s[idx];
      const int idx4 = idx4s[idx];

      const double energy =
        inversionEnergy(pos, idx1, idx2, idx3, idx4, C0[idx], C1[idx], C2[idx], forceConstants[idx]);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, inversionTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void InversionGradientKernel(const int      numInversion,
                                        const int*     idx1s,
                                        const int*     idx2s,
                                        const int*     idx3s,
                                        const int*     idx4s,
                                        const int*     at2AtomicNum,
                                        const uint8_t* isCBoundToO,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  forceConstants,
                                        const double*  pos,
                                        double*        grad,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      inversionGrad(pos,
                    idx1,
                    idx2s[idx],
                    idx3s[idx],
                    idx4s[idx],
                    C0[idx],
                    C1[idx],
                    C2[idx],
                    forceConstants[idx],
                    grad);
    }
  }
}

__global__ void DistanceConstraintEnergyKernel(const int      numDist,
                                               const int*     idx1s,
                                               const int*     idx2s,
                                               const double*  minLen,
                                               const double*  maxLen,
                                               const double*  forceConstants,
                                               const double*  pos,
                                               double*        energyBuffer,
                                               const int*     energyBufferStarts,
                                               const int*     atomIdxToBatchIdx,
                                               const int*     distTermStarts,
                                               const int*     atomStarts,
                                               const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2          = idx2s[idx];
      const double forceConstant = forceConstants[idx];

      const double energy = distanceConstraintEnergy(pos, idx1, idx2, minLen[idx], maxLen[idx], forceConstant);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, distTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void DistanceConstraintGradientKernel(const int      numDist,
                                                 const int*     idx1s,
                                                 const int*     idx2s,
                                                 const double*  minLen,
                                                 const double*  maxLen,
                                                 const double*  forceConstants,
                                                 const double*  pos,
                                                 double*        grad,
                                                 const int*     atomIdxToBatchIdx,
                                                 const int*     atomStarts,
                                                 const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      distanceConstraintGrad(pos, idx1, idx2s[idx], minLen[idx], maxLen[idx], forceConstants[idx], grad);
    }
  }
}

__global__ void AngleConstraintEnergyKernel(const int      numAngle,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const int*     idx3s,
                                            const double*  minAngle,
                                            const double*  maxAngle,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     angleTermStarts,
                                            const int*     atomStarts,
                                            const uint8_t* activeThisStage,
                                            const double   forceConstant) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngle) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2   = idx2s[idx];
      const int    idx3   = idx3s[idx];
      const double minAng = minAngle[idx];
      const double maxAng = maxAngle[idx];

      const double energy = angleConstraintEnergy(pos, idx1, idx2, idx3, minAng, maxAng, forceConstant);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, angleTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void AngleConstraintGradientKernel(const int      numAngle,
                                              const int*     idx1s,
                                              const int*     idx2s,
                                              const int*     idx3s,
                                              const double*  minAngle,
                                              const double*  maxAngle,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const uint8_t* activeThisStage,
                                              const double   forceConstant) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngle) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      angleConstraintGrad(pos, idx1, idx2s[idx], idx3s[idx], minAngle[idx], maxAngle[idx], forceConstant, grad);
    }
  }
}

cudaError_t launchDistViolationEnergyKernel(const int      numDist,
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
                                            const uint8_t* activeThisStage,
                                            cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  DistViolationEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                 idx1,
                                                                 idx2,
                                                                 lb2,
                                                                 ub2,
                                                                 weight,
                                                                 pos,
                                                                 energyBuffer,
                                                                 energyBufferStarts,
                                                                 atomIdxToBatchIdx,
                                                                 distTermStarts,
                                                                 atomStarts,
                                                                 dimension,
                                                                 activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchDistViolationGradientKernel(const int      numDist,
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
                                              const uint8_t* activeThisStage,
                                              cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  if (dimension == 3) {
    DistViolationGradientKernel<3><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                        idx1,
                                                                        idx2,
                                                                        lb2,
                                                                        ub2,
                                                                        weight,
                                                                        pos,
                                                                        grad,
                                                                        atomIdxToBatchIdx,
                                                                        atomStarts,
                                                                        activeThisStage);
  } else if (dimension == 4) {
    DistViolationGradientKernel<4><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                        idx1,
                                                                        idx2,
                                                                        lb2,
                                                                        ub2,
                                                                        weight,
                                                                        pos,
                                                                        grad,
                                                                        atomIdxToBatchIdx,
                                                                        atomStarts,
                                                                        activeThisStage);
  } else {
    throw std::runtime_error("Unsupported dimension for DistViolationGradientKernel: " + std::to_string(dimension));
  }

  return cudaGetLastError();
}

cudaError_t launchChiralViolationEnergyKernel(const int      numChiral,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const int*     idx3,
                                              const int*     idx4,
                                              const double*  volLower,
                                              const double*  volUpper,
                                              double         weight,
                                              const double*  pos,
                                              double*        energyBuffer,
                                              const int*     energyBufferStarts,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     chiralTermStarts,
                                              const int*     atomStarts,
                                              const int      dimension,
                                              const uint8_t* activeThisStage,
                                              cudaStream_t   stream) {
  if (numChiral == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numChiral + blockSize - 1) / blockSize;
  ChiralViolationEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                   idx1,
                                                                   idx2,
                                                                   idx3,
                                                                   idx4,
                                                                   volLower,
                                                                   volUpper,
                                                                   weight,
                                                                   pos,
                                                                   energyBuffer,
                                                                   energyBufferStarts,
                                                                   atomIdxToBatchIdx,
                                                                   chiralTermStarts,
                                                                   atomStarts,
                                                                   dimension,
                                                                   activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchChiralViolationGradientKernel(const int      numChiral,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const int*     idx4,
                                                const double*  volLower,
                                                const double*  volUpper,
                                                double         weight,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                const int      dimension,
                                                const uint8_t* activeThisStage,
                                                cudaStream_t   stream) {
  if (numChiral == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numChiral + blockSize - 1) / blockSize;
  ChiralViolationGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                     idx1,
                                                                     idx2,
                                                                     idx3,
                                                                     idx4,
                                                                     volLower,
                                                                     volUpper,
                                                                     weight,
                                                                     pos,
                                                                     grad,
                                                                     atomIdxToBatchIdx,
                                                                     atomStarts,
                                                                     dimension,
                                                                     activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchFourthDimEnergyKernel(const int      numFD,
                                        const int*     idx,
                                        double         weight,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     fourthTermStarts,
                                        const int*     atomStarts,
                                        const int      dimension,
                                        const uint8_t* activeThisStage,
                                        cudaStream_t   stream) {
  if (numFD == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numFD + blockSize - 1) / blockSize;
  fourthDimEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                             idx,
                                                             weight,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomIdxToBatchIdx,
                                                             fourthTermStarts,
                                                             atomStarts,
                                                             dimension,
                                                             activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchFourthDimGradientKernel(const int      numFD,
                                          const int*     idx,
                                          double         weight,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          const int      dimension,
                                          const uint8_t* activeThisStage,
                                          cudaStream_t   stream) {
  if (numFD == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numFD + blockSize - 1) / blockSize;
  fourthDimGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                               idx,
                                                               weight,
                                                               pos,
                                                               grad,
                                                               atomIdxToBatchIdx,
                                                               atomStarts,
                                                               dimension,
                                                               activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchTorsionAngleEnergyKernel(const int      numTorsion,
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
                                           const uint8_t* activeThisStage,
                                           cudaStream_t   stream) {
  if (numTorsion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsion + blockSize - 1) / blockSize;
  TorsionAngleEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsion,
                                                                idx1,
                                                                idx2,
                                                                idx3,
                                                                idx4,
                                                                forceConstant,
                                                                signs,
                                                                pos,
                                                                energyBuffer,
                                                                energyBufferStarts,
                                                                atomIdxToBatchIdx,
                                                                torsionTermStarts,
                                                                atomStarts,
                                                                activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchTorsionAngleGradientKernel(const int      numTorsion,
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
                                             const uint8_t* activeThisStage,
                                             cudaStream_t   stream) {
  if (numTorsion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsion + blockSize - 1) / blockSize;
  TorsionAngleGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsion,
                                                                  idx1,
                                                                  idx2,
                                                                  idx3,
                                                                  idx4,
                                                                  forceConstant,
                                                                  signs,
                                                                  pos,
                                                                  grad,
                                                                  atomIdxToBatchIdx,
                                                                  atomStarts,
                                                                  activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchInversionEnergyKernel(const int      numInversion,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const int*     idx4,
                                        const int*     at2AtomicNum,
                                        const uint8_t* isCBoundToO,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  forceConstants,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     inversionTermStarts,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage,
                                        cudaStream_t   stream) {
  if (numInversion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversion + blockSize - 1) / blockSize;
  InversionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numInversion,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             idx4,
                                                             at2AtomicNum,
                                                             isCBoundToO,
                                                             C0,
                                                             C1,
                                                             C2,
                                                             forceConstants,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomIdxToBatchIdx,
                                                             inversionTermStarts,
                                                             atomStarts,
                                                             activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchInversionGradientKernel(const int      numInversion,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const int*     idx4,
                                          const int*     at2AtomicNum,
                                          const uint8_t* isCBoundToO,
                                          const double*  C0,
                                          const double*  C1,
                                          const double*  C2,
                                          const double*  forceConstants,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          const uint8_t* activeThisStage,
                                          cudaStream_t   stream) {
  if (numInversion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversion + blockSize - 1) / blockSize;
  InversionGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numInversion,
                                                               idx1,
                                                               idx2,
                                                               idx3,
                                                               idx4,
                                                               at2AtomicNum,
                                                               isCBoundToO,
                                                               C0,
                                                               C1,
                                                               C2,
                                                               forceConstants,
                                                               pos,
                                                               grad,
                                                               atomIdxToBatchIdx,
                                                               atomStarts,
                                                               activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchDistanceConstraintEnergyKernel(const int      numDist,
                                                 const int*     idx1,
                                                 const int*     idx2,
                                                 const double*  minLen,
                                                 const double*  maxLen,
                                                 const double*  forceConstants,
                                                 const double*  pos,
                                                 double*        energyBuffer,
                                                 const int*     energyBufferStarts,
                                                 const int*     atomIdxToBatchIdx,
                                                 const int*     distTermStarts,
                                                 const int*     atomStarts,
                                                 const uint8_t* activeThisStage,
                                                 cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  DistanceConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                      idx1,
                                                                      idx2,
                                                                      minLen,
                                                                      maxLen,
                                                                      forceConstants,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomIdxToBatchIdx,
                                                                      distTermStarts,
                                                                      atomStarts,
                                                                      activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchDistanceConstraintGradientKernel(const int      numDist,
                                                   const int*     idx1s,
                                                   const int*     idx2s,
                                                   const double*  minLen,
                                                   const double*  maxLen,
                                                   const double*  forceConstants,
                                                   const double*  pos,
                                                   double*        grad,
                                                   const int*     atomIdxToBatchIdx,
                                                   const int*     atomStarts,
                                                   const uint8_t* activeThisStage,
                                                   cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  DistanceConstraintGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                        idx1s,
                                                                        idx2s,
                                                                        minLen,
                                                                        maxLen,
                                                                        forceConstants,
                                                                        pos,
                                                                        grad,
                                                                        atomIdxToBatchIdx,
                                                                        atomStarts,
                                                                        activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintEnergyKernel(const int      numAngle,
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
                                              const uint8_t* activeThisStage,
                                              const double   forceConstant,
                                              cudaStream_t   stream) {
  if (numAngle == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngle + blockSize - 1) / blockSize;
  AngleConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngle,
                                                                   idx1,
                                                                   idx2,
                                                                   idx3,
                                                                   minAngle,
                                                                   maxAngle,
                                                                   pos,
                                                                   energyBuffer,
                                                                   energyBufferStarts,
                                                                   atomIdxToBatchIdx,
                                                                   angleTermStarts,
                                                                   atomStarts,
                                                                   activeThisStage,
                                                                   forceConstant);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintGradientKernel(const int      numAngle,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const double*  minAngle,
                                                const double*  maxAngle,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                const uint8_t* activeThisStage,
                                                const double   forceConstant,
                                                cudaStream_t   stream) {
  if (numAngle == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngle + blockSize - 1) / blockSize;
  AngleConstraintGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numAngle,
                                                                     idx1,
                                                                     idx2,
                                                                     idx3,
                                                                     minAngle,
                                                                     maxAngle,
                                                                     pos,
                                                                     grad,
                                                                     atomIdxToBatchIdx,
                                                                     atomStarts,
                                                                     activeThisStage,
                                                                     forceConstant);
  return cudaGetLastError();
}

cudaError_t launchReduceEnergiesKernel(const int      numBlocks,
                                       const double*  energyBuffer,
                                       const int*     energyBufferBlockIdxToBatchIdx,
                                       double*        outs,
                                       const uint8_t* activeThisStage,
                                       cudaStream_t   stream) {
  reduceEnergiesKernel<<<numBlocks, blockSizeEnergyReduction, 0, stream>>>(energyBuffer,
                                                                           energyBufferBlockIdxToBatchIdx,
                                                                           outs,
                                                                           activeThisStage);
  return cudaGetLastError();
}
}  // namespace DistGeom
}  // namespace nvMolKit
