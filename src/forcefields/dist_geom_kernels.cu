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
#include "kernel_utils.cuh"

using namespace nvMolKit::FFKernelUtils;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
constexpr double RAD2DEG = 180.0 / M_PI;

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
      const int    idx2    = idx2s[idx];
      const double lb2     = lb2s[idx];
      const double ub2     = ub2s[idx];
      const double weight  = weights[idx];
      const int    posIdx1 = idx1 * dimension;
      const int    posIdx2 = idx2 * dimension;

      const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, dimension);
      double       val       = 0.0;
      if (distance2 > ub2) {
        val = (distance2 / ub2) - 1.0;
      } else if (distance2 < lb2) {
        val = ((2 * lb2) / (lb2 + distance2)) - 1.0;
      }
      if (val > 0.0) {
        const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, distTermStarts);
        energyBuffer[outputIdx] += weight * val * val;
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
      const int   idx2    = idx2s[idx];
      const float lb2     = lb2s[idx];
      const float ub2     = ub2s[idx];
      const float weight  = weights[idx];
      const int   posIdx1 = idx1 * dimension;
      const int   posIdx2 = idx2 * dimension;

      const float distance2 = distanceSquaredPosIdx<dimension, float>(pos, posIdx1, posIdx2);
      float       preFactor = 0.0;
      if (distance2 > ub2) {
        preFactor = 4.f * ((distance2 / ub2) - 1.0f) / ub2;
      } else if (distance2 < lb2) {
        const float l2d2 = distance2 + lb2;
        preFactor        = 8.f * lb2 * (1.f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2);
      } else {
        return;
      }
      const float dGradx = weight * preFactor * (pos[posIdx1 + 0] - pos[posIdx2 + 0]);
      const float dGrady = weight * preFactor * (pos[posIdx1 + 1] - pos[posIdx2 + 1]);
      const float dGradz = weight * preFactor * (pos[posIdx1 + 2] - pos[posIdx2 + 2]);
      float       dGradw;
      if constexpr (dimension == 4) {
        dGradw = weight * preFactor * (pos[posIdx1 + 3] - pos[posIdx2 + 3]);
      }
      atomicAdd(&grad[posIdx1 + 0], dGradx);
      atomicAdd(&grad[posIdx1 + 1], dGrady);
      atomicAdd(&grad[posIdx1 + 2], dGradz);
      atomicAdd(&grad[posIdx2 + 0], -dGradx);
      atomicAdd(&grad[posIdx2 + 1], -dGrady);
      atomicAdd(&grad[posIdx2 + 2], -dGradz);
      if constexpr (dimension == 4) {
        atomicAdd(&grad[posIdx1 + 3], dGradw);
        atomicAdd(&grad[posIdx2 + 3], -dGradw);
      }
    }
  }
}

__device__ __forceinline__ double calcChiralVolume(const int&    posIdx1,
                                                   const int&    posIdx2,
                                                   const int&    posIdx3,
                                                   const int&    posIdx4,
                                                   const double* pos,
                                                   double&       v1x,
                                                   double&       v1y,
                                                   double&       v1z,
                                                   double&       v2x,
                                                   double&       v2y,
                                                   double&       v2z,
                                                   double&       v3x,
                                                   double&       v3y,
                                                   double&       v3z) {
  // even if we are minimizing in higher dimension the chiral volume is
  // calculated using only the first 3 dimensions

  v1x = pos[posIdx1 + 0] - pos[posIdx4 + 0];
  v1y = pos[posIdx1 + 1] - pos[posIdx4 + 1];
  v1z = pos[posIdx1 + 2] - pos[posIdx4 + 2];

  v2x = pos[posIdx2 + 0] - pos[posIdx4 + 0];
  v2y = pos[posIdx2 + 1] - pos[posIdx4 + 1];
  v2z = pos[posIdx2 + 2] - pos[posIdx4 + 2];

  v3x = pos[posIdx3 + 0] - pos[posIdx4 + 0];
  v3y = pos[posIdx3 + 1] - pos[posIdx4 + 1];
  v3z = pos[posIdx3 + 2] - pos[posIdx4 + 2];

  double v2v3x, v2v3y, v2v3z;
  crossProduct(v2x, v2y, v2z, v3x, v3y, v3z, v2v3x, v2v3y, v2v3z);
  double vol = dotProduct(v1x, v1y, v1z, v2v3x, v2v3y, v2v3z);
  return vol;
}

__global__ void ChiralViolationEnergyKernel(const int      numChiral,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const int*     idx3s,
                                            const int*     idx4s,
                                            const double*  volLower,
                                            const double*  volUpper,
                                            const double*  weights,
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
      const double weight  = weights[idx];
      const int    posIdx1 = idx1 * dimension;
      const int    posIdx2 = idx2 * dimension;
      const int    posIdx3 = idx3 * dimension;
      const int    posIdx4 = idx4 * dimension;

      double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
      double vol =
        calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, chiralTermStarts);
      if (vol < lb) {
        energyBuffer[outputIdx] += weight * (vol - lb) * (vol - lb);
      } else if (vol > ub) {
        energyBuffer[outputIdx] += weight * (vol - ub) * (vol - ub);
      }
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
                                              const double*  weights,
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
      const double weight  = weights[idx];
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
                                      const double*  weights,
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
      const double weight    = weights[idx];
      unsigned     pid       = idx1 * dimension + 3;
      const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, fourthTermStarts);
      energyBuffer[outputIdx] += weight * pos[pid] * pos[pid];
    }
  }
}

__global__ void fourthDimGradientKernel(const int      numFD,
                                        const int*     idxs,
                                        const double*  weights,
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
      const double weight = weights[idx];
      int          pid    = idx1 * dimension + 3;
      grad[pid] += weight * pos[pid];
    }
  }
}

__device__ __forceinline__ double calcTorsionEnergyM6(const double* forceConstants,
                                                      const int*    signs,
                                                      const double  cosPhi) {
  const double cosPhi2 = cosPhi * cosPhi;
  const double cosPhi3 = cosPhi * cosPhi2;
  const double cosPhi4 = cosPhi * cosPhi3;
  const double cosPhi5 = cosPhi * cosPhi4;
  const double cosPhi6 = cosPhi * cosPhi5;

  const double cos2Phi = 2.0 * cosPhi2 - 1.0;
  const double cos3Phi = 4.0 * cosPhi3 - 3.0 * cosPhi;
  const double cos4Phi = 8.0 * cosPhi4 - 8.0 * cosPhi2 + 1.0;
  const double cos5Phi = 16.0 * cosPhi5 - 20.0 * cosPhi3 + 5.0 * cosPhi;
  const double cos6Phi = 32.0 * cosPhi6 - 48.0 * cosPhi4 + 18.0 * cosPhi2 - 1.0;

  return (forceConstants[0] * (1.0 + signs[0] * cosPhi) + forceConstants[1] * (1.0 + signs[1] * cos2Phi) +
          forceConstants[2] * (1.0 + signs[2] * cos3Phi) + forceConstants[3] * (1.0 + signs[3] * cos4Phi) +
          forceConstants[4] * (1.0 + signs[4] * cos5Phi) + forceConstants[5] * (1.0 + signs[5] * cos6Phi));
}

__device__ __forceinline__ double calcTorsionCosPhi(const double* pos,
                                                    const int     posIdx1,
                                                    const int     posIdx2,
                                                    const int     posIdx3,
                                                    const int     posIdx4) {
  // Calculate vectors
  double r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  double r3x = pos[posIdx2 + 0] - pos[posIdx3 + 0];
  double r3y = pos[posIdx2 + 1] - pos[posIdx3 + 1];
  double r3z = pos[posIdx2 + 2] - pos[posIdx3 + 2];

  double r4x = pos[posIdx4 + 0] - pos[posIdx3 + 0];
  double r4y = pos[posIdx4 + 1] - pos[posIdx3 + 1];
  double r4z = pos[posIdx4 + 2] - pos[posIdx3 + 2];

  // Calculate cross products
  double t1x, t1y, t1z;
  crossProduct(r1x, r1y, r1z, r2x, r2y, r2z, t1x, t1y, t1z);

  double t2x, t2y, t2z;
  crossProduct(r3x, r3y, r3z, r4x, r4y, r4z, t2x, t2y, t2z);

  // Calculate lengths
  double t1_len = sqrt(t1x * t1x + t1y * t1y + t1z * t1z);
  double t2_len = sqrt(t2x * t2x + t2y * t2y + t2z * t2z);

  if (isDoubleZero(t1_len) || isDoubleZero(t2_len)) {
    return 0.0;
  }

  // Calculate cosine of torsion angle
  double cosPhi = dotProduct(t1x, t1y, t1z, t2x, t2y, t2z) / (t1_len * t2_len);
  clipToOne(cosPhi);

  return cosPhi;
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

      // Get positions for all four atoms
      const int posIdx1 = idx1 * 4;
      const int posIdx2 = idx2 * 4;
      const int posIdx3 = idx3 * 4;
      const int posIdx4 = idx4 * 4;

      // Calculate cosine of torsion angle
      double cosPhi = calcTorsionCosPhi(pos, posIdx1, posIdx2, posIdx3, posIdx4);

      // Calculate energy using the M6 formula
      const double* fc     = &forceConstants[idx * 6];  // 6 force constants per torsion
      const int*    s      = &signs[idx * 6];           // 6 signs per torsion
      double        energy = calcTorsionEnergyM6(fc, s, cosPhi);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, torsionTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__device__ __forceinline__ void calcTorsionGrad(const double* r,  // 4 vectors of 3 components each
                                                const double* t,  // 2 vectors of 3 components each
                                                const double* d,  // 2 lengths
                                                double*       g,  // 4 gradient vectors of 3 components each
                                                const double  sinTerm,
                                                const double  cosPhi) {
  // Calculate dCos_dT
  double dCos_dT[6];
  dCos_dT[0] = 1.0 / d[0] * (t[3] - cosPhi * t[0]);  // t[1].x - cosPhi * t[0].x
  dCos_dT[1] = 1.0 / d[0] * (t[4] - cosPhi * t[1]);  // t[1].y - cosPhi * t[0].y
  dCos_dT[2] = 1.0 / d[0] * (t[5] - cosPhi * t[2]);  // t[1].z - cosPhi * t[0].z
  dCos_dT[3] = 1.0 / d[1] * (t[0] - cosPhi * t[3]);  // t[0].x - cosPhi * t[1].x
  dCos_dT[4] = 1.0 / d[1] * (t[1] - cosPhi * t[4]);  // t[0].y - cosPhi * t[1].y
  dCos_dT[5] = 1.0 / d[1] * (t[2] - cosPhi * t[5]);  // t[0].z - cosPhi * t[1].z

  // Calculate gradients for each atom
  // Atom 1 (i)
  g[0] = sinTerm * (dCos_dT[2] * r[4] - dCos_dT[1] * r[5]);  // x
  g[1] = sinTerm * (dCos_dT[0] * r[5] - dCos_dT[2] * r[3]);  // y
  g[2] = sinTerm * (dCos_dT[1] * r[3] - dCos_dT[0] * r[4]);  // z

  // Atom 2 (j)
  g[3] = sinTerm *
         (dCos_dT[1] * (r[5] - r[2]) + dCos_dT[2] * (r[1] - r[4]) + dCos_dT[4] * (-r[11]) + dCos_dT[5] * (r[10]));  // x
  g[4] = sinTerm *
         (dCos_dT[0] * (r[2] - r[5]) + dCos_dT[2] * (r[3] - r[0]) + dCos_dT[3] * (r[11]) + dCos_dT[5] * (-r[9]));  // y
  g[5] = sinTerm *
         (dCos_dT[0] * (r[4] - r[1]) + dCos_dT[1] * (r[0] - r[3]) + dCos_dT[3] * (-r[10]) + dCos_dT[4] * (r[9]));  // z

  // Atom 3 (k)
  g[6] = sinTerm *
         (dCos_dT[1] * (r[2]) + dCos_dT[2] * (-r[1]) + dCos_dT[4] * (r[11] - r[8]) + dCos_dT[5] * (r[7] - r[10]));  // x
  g[7] = sinTerm *
         (dCos_dT[0] * (-r[2]) + dCos_dT[2] * (r[0]) + dCos_dT[3] * (r[8] - r[11]) + dCos_dT[5] * (r[9] - r[6]));  // y
  g[8] = sinTerm *
         (dCos_dT[0] * (r[1]) + dCos_dT[1] * (-r[0]) + dCos_dT[3] * (r[10] - r[7]) + dCos_dT[4] * (r[6] - r[9]));  // z

  // Atom 4 (l)
  g[9]  = sinTerm * (dCos_dT[4] * r[8] - dCos_dT[5] * r[7]);  // x
  g[10] = sinTerm * (dCos_dT[5] * r[6] - dCos_dT[3] * r[8]);  // y
  g[11] = sinTerm * (dCos_dT[3] * r[7] - dCos_dT[4] * r[6]);  // z
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
      const int idx2 = idx2s[idx];
      const int idx3 = idx3s[idx];
      const int idx4 = idx4s[idx];

      // Get positions for all four atoms
      const int posIdx1 = idx1 * 4;
      const int posIdx2 = idx2 * 4;
      const int posIdx3 = idx3 * 4;
      const int posIdx4 = idx4 * 4;

      // Calculate vectors
      double r[12];                                 // 4 vectors of 3 components each
      r[0]  = pos[posIdx1 + 0] - pos[posIdx2 + 0];  // r1.x
      r[1]  = pos[posIdx1 + 1] - pos[posIdx2 + 1];  // r1.y
      r[2]  = pos[posIdx1 + 2] - pos[posIdx2 + 2];  // r1.z
      r[3]  = pos[posIdx3 + 0] - pos[posIdx2 + 0];  // r2.x
      r[4]  = pos[posIdx3 + 1] - pos[posIdx2 + 1];  // r2.y
      r[5]  = pos[posIdx3 + 2] - pos[posIdx2 + 2];  // r2.z
      r[6]  = pos[posIdx2 + 0] - pos[posIdx3 + 0];  // r3.x
      r[7]  = pos[posIdx2 + 1] - pos[posIdx3 + 1];  // r3.y
      r[8]  = pos[posIdx2 + 2] - pos[posIdx3 + 2];  // r3.z
      r[9]  = pos[posIdx4 + 0] - pos[posIdx3 + 0];  // r4.x
      r[10] = pos[posIdx4 + 1] - pos[posIdx3 + 1];  // r4.y
      r[11] = pos[posIdx4 + 2] - pos[posIdx3 + 2];  // r4.z

      // Calculate cross products
      double t[6];  // 2 vectors of 3 components each
      crossProduct(r[0], r[1], r[2], r[3], r[4], r[5], t[0], t[1], t[2]);
      crossProduct(r[6], r[7], r[8], r[9], r[10], r[11], t[3], t[4], t[5]);

      // Calculate lengths
      double d[2];
      d[0] = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
      d[1] = sqrt(t[3] * t[3] + t[4] * t[4] + t[5] * t[5]);

      if (isDoubleZero(d[0]) || isDoubleZero(d[1])) {
        return;
      }

      // Normalize vectors
      t[0] /= d[0];
      t[1] /= d[0];
      t[2] /= d[0];
      t[3] /= d[1];
      t[4] /= d[1];
      t[5] /= d[1];

      // Calculate cosine of torsion angle
      double cosPhi = dotProduct(t[0], t[1], t[2], t[3], t[4], t[5]);
      clipToOne(cosPhi);

      // Calculate sinPhi
      const double sinPhiSq = 1.0 - cosPhi * cosPhi;
      const double sinPhi   = ((sinPhiSq > 0.0) ? sqrt(sinPhiSq) : 0.0);

      // Calculate derivatives
      const double cosPhi2 = cosPhi * cosPhi;
      const double cosPhi3 = cosPhi * cosPhi2;
      const double cosPhi4 = cosPhi * cosPhi3;
      const double cosPhi5 = cosPhi * cosPhi4;

      // Calculate dE/dPhi with corrected indices
      const double dE_dPhi =
        (-forceConstants[idx * 6 + 0] * signs[idx * 6 + 0] * sinPhi -
         2.0 * forceConstants[idx * 6 + 1] * signs[idx * 6 + 1] * (2.0 * cosPhi * sinPhi) -
         3.0 * forceConstants[idx * 6 + 2] * signs[idx * 6 + 2] * (4.0 * cosPhi2 * sinPhi - sinPhi) -
         4.0 * forceConstants[idx * 6 + 3] * signs[idx * 6 + 3] * (8.0 * cosPhi3 * sinPhi - 4.0 * cosPhi * sinPhi) -
         5.0 * forceConstants[idx * 6 + 4] * signs[idx * 6 + 4] *
           (16.0 * cosPhi4 * sinPhi - 12.0 * cosPhi2 * sinPhi + sinPhi) -
         6.0 * forceConstants[idx * 6 + 4] * signs[idx * 6 + 4] *
           (32.0 * cosPhi5 * sinPhi - 32.0 * cosPhi3 * sinPhi + 6.0 * sinPhi));

      // Calculate sinTerm
      double sinTerm = -dE_dPhi * (isDoubleZero(sinPhi) ? (1.0 / cosPhi) : (1.0 / sinPhi));

      // Calculate gradients
      double g[12];  // 4 gradient vectors of 3 components each
      for (int i = 0; i < 12; ++i) {
        g[i] = 0.0;
      }
      calcTorsionGrad(r, t, d, g, sinTerm, cosPhi);

      // Add gradients to global gradient array
      atomicAdd(&grad[posIdx1 + 0], g[0]);
      atomicAdd(&grad[posIdx1 + 1], g[1]);
      atomicAdd(&grad[posIdx1 + 2], g[2]);
      atomicAdd(&grad[posIdx2 + 0], g[3]);
      atomicAdd(&grad[posIdx2 + 1], g[4]);
      atomicAdd(&grad[posIdx2 + 2], g[5]);
      atomicAdd(&grad[posIdx3 + 0], g[6]);
      atomicAdd(&grad[posIdx3 + 1], g[7]);
      atomicAdd(&grad[posIdx3 + 2], g[8]);
      atomicAdd(&grad[posIdx4 + 0], g[9]);
      atomicAdd(&grad[posIdx4 + 1], g[10]);
      atomicAdd(&grad[posIdx4 + 2], g[11]);
    }
  }
}

__device__ __forceinline__ double calcInversionCosY(const double* pos,
                                                    const int     posIdx1,
                                                    const int     posIdx2,
                                                    const int     posIdx3,
                                                    const int     posIdx4) {
  constexpr double inversionZeroTol = 1.0e-16;  // Match original code's zero tolerance

  // Calculate vectors
  double rJIx = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double rJIy = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double rJIz = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double rJKx = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double rJKy = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double rJKz = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  double rJLx = pos[posIdx4 + 0] - pos[posIdx2 + 0];
  double rJLy = pos[posIdx4 + 1] - pos[posIdx2 + 1];
  double rJLz = pos[posIdx4 + 2] - pos[posIdx2 + 2];

  // Calculate lengths squared
  double l2JI = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  double l2JK = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  double l2JL = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;

  // Check for zero lengths using inversion-specific tolerance
  if (l2JI < inversionZeroTol || l2JK < inversionZeroTol || l2JL < inversionZeroTol) {
    return 0.0;
  }

  // Calculate cross product n = rJI × rJK
  double nx, ny, nz;
  crossProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz, nx, ny, nz);

  // Normalize n by sqrt(l2JI) * sqrt(l2JK) as in original code
  double norm_factor = sqrt(l2JI) * sqrt(l2JK);
  nx /= norm_factor;
  ny /= norm_factor;
  nz /= norm_factor;

  // Calculate length squared of normalized n
  double l2n = nx * nx + ny * ny + nz * nz;
  if (l2n < inversionZeroTol) {
    return 0.0;
  }

  // Calculate dot product and normalize by sqrt(l2JL) * sqrt(l2n)
  return dotProduct(nx, ny, nz, rJLx, rJLy, rJLz) / (sqrt(l2JL) * sqrt(l2n));
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

      // Get positions for all four atoms
      const int posIdx1 = idx1 * 4;
      const int posIdx2 = idx2 * 4;
      const int posIdx3 = idx3 * 4;
      const int posIdx4 = idx4 * 4;

      // Calculate cosine of inversion angle
      double cosY = calcInversionCosY(pos, posIdx1, posIdx2, posIdx3, posIdx4);

      // Calculate sinY
      const double sinYSq = 1.0 - cosY * cosY;
      const double sinY   = ((sinYSq > 0.0) ? sqrt(sinYSq) : 0.0);

      // Calculate cos(2W)
      const double cos2W = 2.0 * sinY * sinY - 1.0;

      // Calculate energy
      double energy = forceConstants[idx] * (C0[idx] + C1[idx] * sinY + C2[idx] * cos2W);

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
      const int idx2 = idx2s[idx];
      const int idx3 = idx3s[idx];
      const int idx4 = idx4s[idx];

      // Get positions for all four atoms
      const int posIdx1 = idx1 * 4;
      const int posIdx2 = idx2 * 4;
      const int posIdx3 = idx3 * 4;
      const int posIdx4 = idx4 * 4;

      // Calculate vectors
      double rJIx = pos[posIdx1 + 0] - pos[posIdx2 + 0];
      double rJIy = pos[posIdx1 + 1] - pos[posIdx2 + 1];
      double rJIz = pos[posIdx1 + 2] - pos[posIdx2 + 2];

      double rJKx = pos[posIdx3 + 0] - pos[posIdx2 + 0];
      double rJKy = pos[posIdx3 + 1] - pos[posIdx2 + 1];
      double rJKz = pos[posIdx3 + 2] - pos[posIdx2 + 2];

      double rJLx = pos[posIdx4 + 0] - pos[posIdx2 + 0];
      double rJLy = pos[posIdx4 + 1] - pos[posIdx2 + 1];
      double rJLz = pos[posIdx4 + 2] - pos[posIdx2 + 2];

      // Calculate lengths
      double dJI = sqrt(rJIx * rJIx + rJIy * rJIy + rJIz * rJIz);
      double dJK = sqrt(rJKx * rJKx + rJKy * rJKy + rJKz * rJKz);
      double dJL = sqrt(rJLx * rJLx + rJLy * rJLy + rJLz * rJLz);

      // Check for zero lengths
      if (isDoubleZero(dJI) || isDoubleZero(dJK) || isDoubleZero(dJL)) {
        return;
      }

      // Normalize vectors
      rJIx /= dJI;
      rJIy /= dJI;
      rJIz /= dJI;
      rJKx /= dJK;
      rJKy /= dJK;
      rJKz /= dJK;
      rJLx /= dJL;
      rJLy /= dJL;
      rJLz /= dJL;

      // Calculate n = (-rJI) × rJK
      double nx, ny, nz;
      crossProduct(-rJIx, -rJIy, -rJIz, rJKx, rJKy, rJKz, nx, ny, nz);

      // Normalize n
      double n_len = sqrt(nx * nx + ny * ny + nz * nz);
      nx /= n_len;
      ny /= n_len;
      nz /= n_len;

      // Calculate cosY and clamp
      double cosY = dotProduct(nx, ny, nz, rJLx, rJLy, rJLz);
      clipToOne(cosY);

      // Calculate sinY
      const double sinYSq = 1.0 - cosY * cosY;
      const double sinY   = fmax(sqrt(sinYSq), 1.0e-8);

      // Calculate cosTheta and clamp
      double cosTheta = dotProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz);
      clipToOne(cosTheta);

      // Calculate sinTheta
      const double sinThetaSq = 1.0 - cosTheta * cosTheta;
      const double sinTheta   = fmax(sqrt(sinThetaSq), 1.0e-8);

      // Calculate dE_dW
      const double dE_dW = -forceConstants[idx] * (C1[idx] * cosY - 4.0 * C2[idx] * cosY * sinY);

      // Calculate cross products for gradient terms
      double t1x, t1y, t1z;  // rJL × rJK
      crossProduct(rJLx, rJLy, rJLz, rJKx, rJKy, rJKz, t1x, t1y, t1z);

      double t2x, t2y, t2z;  // rJI × rJL
      crossProduct(rJIx, rJIy, rJIz, rJLx, rJLy, rJLz, t2x, t2y, t2z);

      double t3x, t3y, t3z;  // rJK × rJI
      crossProduct(rJKx, rJKy, rJKz, rJIx, rJIy, rJIz, t3x, t3y, t3z);

      // Calculate terms for gradient
      const double term1 = sinY * sinTheta;
      const double term2 = cosY / (sinY * sinThetaSq);

      // Calculate gradient components for each atom
      double tg1[3] = {(t1x / term1 - (rJIx - rJKx * cosTheta) * term2) / dJI,
                       (t1y / term1 - (rJIy - rJKy * cosTheta) * term2) / dJI,
                       (t1z / term1 - (rJIz - rJKz * cosTheta) * term2) / dJI};

      double tg3[3] = {(t2x / term1 - (rJKx - rJIx * cosTheta) * term2) / dJK,
                       (t2y / term1 - (rJKy - rJIy * cosTheta) * term2) / dJK,
                       (t2z / term1 - (rJKz - rJIz * cosTheta) * term2) / dJK};

      double tg4[3] = {(t3x / term1 - rJLx * cosY / sinY) / dJL,
                       (t3y / term1 - rJLy * cosY / sinY) / dJL,
                       (t3z / term1 - rJLz * cosY / sinY) / dJL};

      // Add gradients to global gradient array
      atomicAdd(&grad[posIdx1 + 0], dE_dW * tg1[0]);
      atomicAdd(&grad[posIdx1 + 1], dE_dW * tg1[1]);
      atomicAdd(&grad[posIdx1 + 2], dE_dW * tg1[2]);

      atomicAdd(&grad[posIdx2 + 0], -dE_dW * (tg1[0] + tg3[0] + tg4[0]));
      atomicAdd(&grad[posIdx2 + 1], -dE_dW * (tg1[1] + tg3[1] + tg4[1]));
      atomicAdd(&grad[posIdx2 + 2], -dE_dW * (tg1[2] + tg3[2] + tg4[2]));

      atomicAdd(&grad[posIdx3 + 0], dE_dW * tg3[0]);
      atomicAdd(&grad[posIdx3 + 1], dE_dW * tg3[1]);
      atomicAdd(&grad[posIdx3 + 2], dE_dW * tg3[2]);

      atomicAdd(&grad[posIdx4 + 0], dE_dW * tg4[0]);
      atomicAdd(&grad[posIdx4 + 1], dE_dW * tg4[1]);
      atomicAdd(&grad[posIdx4 + 2], dE_dW * tg4[2]);
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
      const double minLen2       = minLen[idx] * minLen[idx];  // Square min length
      const double maxLen2       = maxLen[idx] * maxLen[idx];  // Square max length
      const double forceConstant = forceConstants[idx];
      const int    posIdx1       = idx1 * 4;
      const int    posIdx2       = idx2 * 4;

      // Calculate squared distance - always first 3 dimensions.
      const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, 3);

      // Check if distance is outside bounds
      double difference = 0.0;
      if (distance2 < minLen2) {
        difference = minLen[idx] - sqrt(distance2);
      } else if (distance2 > maxLen2) {
        difference = sqrt(distance2) - maxLen[idx];
      } else {
        return;  // Distance within bounds, no energy contribution
      }

      // Calculate energy contribution
      const double energy = 0.5 * forceConstant * difference * difference;

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
      const int    idx2          = idx2s[idx];
      const double minLen2       = minLen[idx] * minLen[idx];  // Square min length
      const double maxLen2       = maxLen[idx] * maxLen[idx];  // Square max length
      const double forceConstant = forceConstants[idx];
      const int    posIdx1       = idx1 * 4;
      const int    posIdx2       = idx2 * 4;

      // Calculate squared distance
      const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, 3);

      // Check if distance is outside bounds
      double preFactor = 0.0;
      double distance  = 0.0;
      if (distance2 < minLen2) {
        distance  = sqrt(distance2);
        preFactor = distance - minLen[idx];
      } else if (distance2 > maxLen2) {
        distance  = sqrt(distance2);
        preFactor = distance - maxLen[idx];
      } else {
        return;  // Distance within bounds, no gradient contribution
      }

      // Calculate final preFactor
      preFactor *= forceConstant;
      preFactor /= fmax(1.0e-8, distance);

      // Calculate and accumulate gradients for each component
      for (int i = 0; i < 3; i++) {
        const double dGrad = preFactor * (pos[posIdx1 + i] - pos[posIdx2 + i]);
        atomicAdd(&grad[posIdx1 + i], dGrad);
        atomicAdd(&grad[posIdx2 + i], -dGrad);
      }
    }
  }
}

__device__ __forceinline__ double computeAngleTerm(const double angle, const double minAngle, const double maxAngle) {
  double angleTerm = 0.0;
  if (angle < minAngle) {
    angleTerm = angle - minAngle;
  } else if (angle > maxAngle) {
    angleTerm = angle - maxAngle;
  }
  return angleTerm;
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

      // Get positions for all three atoms
      const int posIdx1 = idx1 * 4;
      const int posIdx2 = idx2 * 4;
      const int posIdx3 = idx3 * 4;

      // Calculate vectors r1 = p1 - p2 and r2 = p3 - p2
      double r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
      double r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
      double r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

      double r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
      double r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
      double r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

      // Calculate squared lengths and take max with 1.0e-5 as in RDKit
      const double r1LengthSq = fmax(1.0e-5, r1x * r1x + r1y * r1y + r1z * r1z);
      const double r2LengthSq = fmax(1.0e-5, r2x * r2x + r2y * r2y + r2z * r2z);

      // Calculate cosine of angle using dot product
      double cosTheta = dotProduct(r1x, r1y, r1z, r2x, r2y, r2z) / sqrt(r1LengthSq * r2LengthSq);

      // Clamp cosTheta to [-1, 1]
      clipToOne(cosTheta);

      // Convert to degrees using RDKit's RAD2DEG constant
      const double angle = RAD2DEG * acos(cosTheta);

      // Calculate angle term using the separate device function
      const double angleTerm = computeAngleTerm(angle, minAng, maxAng);

      const double energy = forceConstant * angleTerm * angleTerm;

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
      const int    idx2   = idx2s[idx];
      const int    idx3   = idx3s[idx];
      const double minAng = minAngle[idx];
      const double maxAng = maxAngle[idx];

      // Get positions for all three atoms
      const int posIdx1 = idx1 * 4;
      const int posIdx2 = idx2 * 4;
      const int posIdx3 = idx3 * 4;

      // Calculate vectors r1 = p1 - p2 and r2 = p3 - p2
      double r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
      double r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
      double r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

      double r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
      double r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
      double r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

      // Calculate squared lengths and take max with 1.0e-5 as in RDKit
      const double r1LengthSq = fmax(1.0e-5, r1x * r1x + r1y * r1y + r1z * r1z);
      const double r2LengthSq = fmax(1.0e-5, r2x * r2x + r2y * r2y + r2z * r2z);

      // Calculate cosine of angle using dot product
      double cosTheta = dotProduct(r1x, r1y, r1z, r2x, r2y, r2z) / sqrt(r1LengthSq * r2LengthSq);

      // Clamp cosTheta to [-1, 1]
      clipToOne(cosTheta);

      // Convert to degrees using RDKit's RAD2DEG constant
      const double angle = RAD2DEG * acos(cosTheta);

      // Calculate angle term using the separate device function
      const double angleTerm = computeAngleTerm(angle, minAng, maxAng);

      // Calculate dE_dTheta
      const double dE_dTheta = 2.0 * RAD2DEG * forceConstant * angleTerm;

      // Calculate cross product rp = r2 × r1
      double rpx, rpy, rpz;
      crossProduct(r2x, r2y, r2z, r1x, r1y, r1z, rpx, rpy, rpz);

      // Calculate length of rp and prefactor
      const double rpLengthSq = rpx * rpx + rpy * rpy + rpz * rpz;
      const double rpLength   = sqrt(rpLengthSq);
      const double prefactor  = dE_dTheta / fmax(1.0e-5, rpLength);

      // Calculate t factors
      const double t1 = -prefactor / r1LengthSq;
      const double t2 = prefactor / r2LengthSq;

      // Calculate cross products for gradients
      double dedp1x, dedp1y, dedp1z;  // r1 × rp
      crossProduct(r1x, r1y, r1z, rpx, rpy, rpz, dedp1x, dedp1y, dedp1z);

      double dedp3x, dedp3y, dedp3z;  // r2 × rp
      crossProduct(r2x, r2y, r2z, rpx, rpy, rpz, dedp3x, dedp3y, dedp3z);

      // Scale the cross products by t factors
      dedp1x *= t1;
      dedp1y *= t1;
      dedp1z *= t1;

      dedp3x *= t2;
      dedp3y *= t2;
      dedp3z *= t2;

      // Calculate middle point gradient as negative sum of other two
      const double dedp2x = -(dedp1x + dedp3x);
      const double dedp2y = -(dedp1y + dedp3y);
      const double dedp2z = -(dedp1z + dedp3z);

      // Accumulate gradients using atomic operations
      atomicAdd(&grad[posIdx1 + 0], dedp1x);
      atomicAdd(&grad[posIdx1 + 1], dedp1y);
      atomicAdd(&grad[posIdx1 + 2], dedp1z);

      atomicAdd(&grad[posIdx2 + 0], dedp2x);
      atomicAdd(&grad[posIdx2 + 1], dedp2y);
      atomicAdd(&grad[posIdx2 + 2], dedp2z);

      atomicAdd(&grad[posIdx3 + 0], dedp3x);
      atomicAdd(&grad[posIdx3 + 1], dedp3y);
      atomicAdd(&grad[posIdx3 + 2], dedp3z);
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
                                              const double*  weight,
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
                                                const double*  weight,
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
                                        const double*  weight,
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
                                          const double*  weight,
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
