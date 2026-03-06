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

#include "conformer_rmsd.h"
#include "cuda_error_check.h"

#include <cmath>
#include <cuda/std/span>

namespace nvMolKit {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/// Compute eigenvalues of a 3x3 symmetric matrix using Cardano's analytical formula.
/// Input: upper triangle {a00, a01, a02, a11, a12, a22}.
/// Output: three eigenvalues in descending order.
__device__ __forceinline__ void symmetricEigenvalues3x3(const double a00, const double a01, const double a02,
                                                        const double a11, const double a12, const double a22,
                                                        double& e0, double& e1, double& e2) {
  // Characteristic polynomial:  λ³ - p λ² + q λ - r = 0
  const double p = a00 + a11 + a22;                                           // trace
  const double q = a00 * a11 + a00 * a22 + a11 * a22 - a01 * a01 - a02 * a02 - a12 * a12;
  const double r = a00 * a11 * a22 + 2.0 * a01 * a02 * a12 - a00 * a12 * a12 - a11 * a02 * a02 - a22 * a01 * a01;

  // Shift to depressed cubic:  t³ + pt' + q' = 0  where λ = t + p/3
  const double p3   = p / 3.0;
  const double pp   = (p * p - 3.0 * q) / 9.0;  // -p'/3
  const double qq   = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0;
  const double disc = pp * pp * pp - qq * qq;

  if (disc >= 0.0) {
    // Three real roots (always the case for symmetric matrices)
    const double sqrtPP = sqrt(fmax(pp, 0.0));
    const double theta  = acos(fmin(fmax(qq / fmax(sqrtPP * sqrtPP * sqrtPP, 1e-30), -1.0), 1.0)) / 3.0;
    e0                  = 2.0 * sqrtPP * cos(theta) + p3;
    e1                  = 2.0 * sqrtPP * cos(theta - 2.0 * M_PI / 3.0) + p3;
    e2                  = 2.0 * sqrtPP * cos(theta - 4.0 * M_PI / 3.0) + p3;
  } else {
    // Fallback (should not happen for real symmetric matrices)
    e0 = e1 = e2 = p3;
  }

  // Sort descending
  if (e1 > e0) { double t = e0; e0 = e1; e1 = t; }
  if (e2 > e0) { double t = e0; e0 = e2; e2 = t; }
  if (e2 > e1) { double t = e1; e1 = e2; e2 = t; }
}

/// Compute determinant of a 3x3 matrix stored row-major.
__device__ __forceinline__ double det3x3(const double* H) {
  return H[0] * (H[4] * H[8] - H[5] * H[7]) - H[1] * (H[3] * H[8] - H[5] * H[6]) +
         H[2] * (H[3] * H[7] - H[4] * H[6]);
}

// ---------------------------------------------------------------------------
// Kernel: one thread-block per conformer pair
//
// Each block computes:
//   1. Centroids of both conformers            (parallel reduce)
//   2. Centered inner products  Sp, Sq         (parallel reduce)
//   3. Cross-covariance matrix  H = Pc^T Qc    (parallel reduce, 9 values)
//   4. Singular values of H via eigenvalues of H^T H  (thread 0, analytical)
//   5. RMSD with optional Kabsch alignment      (thread 0)
// ---------------------------------------------------------------------------

constexpr int kRmsdBlockSize = 128;

__global__ void conformerRmsdKernel(const double* __restrict__ coords,
                                    double* __restrict__ rmsdOut,
                                    const int numConformers,
                                    const int numAtoms,
                                    const bool prealigned) {
  // Map blockIdx to pair (ci, cj) with ci > cj using lower-triangle indexing.
  const int pairIdx = blockIdx.x;
  // Inverse of pairIdx = ci*(ci-1)/2 + cj:  ci = floor((1 + sqrt(1 + 8*pairIdx)) / 2)
  const int ci = static_cast<int>(floor((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(pairIdx))) / 2.0));
  const int cj = pairIdx - ci * (ci - 1) / 2;

  // Safety check (floating point edge cases in the inverse formula)
  if (ci >= numConformers || cj >= ci) return;

  const int tid       = threadIdx.x;
  const int stride    = numAtoms * 3;
  const double* coordI = coords + ci * stride;
  const double* coordJ = coords + cj * stride;

  // Shared memory for reductions.
  // Layout: [3 centroidI, 3 centroidJ, 1 Sp, 1 Sq, 9 H] = 17 doubles per thread
  // We use a two-pass approach with shared memory for block-wide sums.
  __shared__ double sCentI[3];  // centroid of conformer i
  __shared__ double sCentJ[3];  // centroid of conformer j
  __shared__ double sAccum[11]; // Sp, Sq, H[0..8]

  // ---- Pass 1: Compute centroids ----
  double sumIx = 0.0, sumIy = 0.0, sumIz = 0.0;
  double sumJx = 0.0, sumJy = 0.0, sumJz = 0.0;

  for (int a = tid; a < numAtoms; a += kRmsdBlockSize) {
    sumIx += coordI[a * 3 + 0];
    sumIy += coordI[a * 3 + 1];
    sumIz += coordI[a * 3 + 2];
    sumJx += coordJ[a * 3 + 0];
    sumJy += coordJ[a * 3 + 1];
    sumJz += coordJ[a * 3 + 2];
  }

  // Warp-level reduce then block-level via shared memory
  // Use simple shared-memory reduction for clarity and correctness.
  __shared__ double sReduce[kRmsdBlockSize];

  // Helper lambda-like pattern using a macro for 6 reductions
  #define BLOCK_REDUCE_SUM(val, result_ptr) \
    sReduce[tid] = val; \
    __syncthreads(); \
    for (int s = kRmsdBlockSize / 2; s > 0; s >>= 1) { \
      if (tid < s) sReduce[tid] += sReduce[tid + s]; \
      __syncthreads(); \
    } \
    if (tid == 0) *result_ptr = sReduce[0]; \
    __syncthreads();

  BLOCK_REDUCE_SUM(sumIx, &sCentI[0])
  BLOCK_REDUCE_SUM(sumIy, &sCentI[1])
  BLOCK_REDUCE_SUM(sumIz, &sCentI[2])
  BLOCK_REDUCE_SUM(sumJx, &sCentJ[0])
  BLOCK_REDUCE_SUM(sumJy, &sCentJ[1])
  BLOCK_REDUCE_SUM(sumJz, &sCentJ[2])

  if (tid == 0) {
    const double invN = 1.0 / static_cast<double>(numAtoms);
    sCentI[0] *= invN;
    sCentI[1] *= invN;
    sCentI[2] *= invN;
    sCentJ[0] *= invN;
    sCentJ[1] *= invN;
    sCentJ[2] *= invN;
  }
  __syncthreads();

  const double cIx = sCentI[0], cIy = sCentI[1], cIz = sCentI[2];
  const double cJx = sCentJ[0], cJy = sCentJ[1], cJz = sCentJ[2];

  if (prealigned) {
    // ---- Simple RMSD without alignment ----
    double sumSqDiff = 0.0;
    for (int a = tid; a < numAtoms; a += kRmsdBlockSize) {
      const double dx = (coordI[a * 3 + 0] - cIx) - (coordJ[a * 3 + 0] - cJx);
      const double dy = (coordI[a * 3 + 1] - cIy) - (coordJ[a * 3 + 1] - cJy);
      const double dz = (coordI[a * 3 + 2] - cIz) - (coordJ[a * 3 + 2] - cJz);
      sumSqDiff += dx * dx + dy * dy + dz * dz;
    }
    BLOCK_REDUCE_SUM(sumSqDiff, &sAccum[0])
    if (tid == 0) {
      rmsdOut[pairIdx] = sqrt(sAccum[0] / static_cast<double>(numAtoms));
    }
    return;
  }

  // ---- Pass 2: Compute Sp, Sq, and cross-covariance H (Kabsch alignment) ----
  // Sp = sum ||pi - centI||^2,  Sq = sum ||qj - centJ||^2
  // H[r][c] = sum (pi[r] - centI[r]) * (qj[c] - centJ[c])
  double localSp = 0.0, localSq = 0.0;
  double localH[9] = {0.0};

  for (int a = tid; a < numAtoms; a += kRmsdBlockSize) {
    const double px = coordI[a * 3 + 0] - cIx;
    const double py = coordI[a * 3 + 1] - cIy;
    const double pz = coordI[a * 3 + 2] - cIz;
    const double qx = coordJ[a * 3 + 0] - cJx;
    const double qy = coordJ[a * 3 + 1] - cJy;
    const double qz = coordJ[a * 3 + 2] - cJz;

    localSp += px * px + py * py + pz * pz;
    localSq += qx * qx + qy * qy + qz * qz;

    // H = P^T Q  (sum of outer products)
    localH[0] += px * qx;  localH[1] += px * qy;  localH[2] += px * qz;
    localH[3] += py * qx;  localH[4] += py * qy;  localH[5] += py * qz;
    localH[6] += pz * qx;  localH[7] += pz * qy;  localH[8] += pz * qz;
  }

  // Reduce Sp, Sq, H[0..8]  (11 values total)
  BLOCK_REDUCE_SUM(localSp, &sAccum[0])
  BLOCK_REDUCE_SUM(localSq, &sAccum[1])
  BLOCK_REDUCE_SUM(localH[0], &sAccum[2])
  BLOCK_REDUCE_SUM(localH[1], &sAccum[3])
  BLOCK_REDUCE_SUM(localH[2], &sAccum[4])
  BLOCK_REDUCE_SUM(localH[3], &sAccum[5])
  BLOCK_REDUCE_SUM(localH[4], &sAccum[6])
  BLOCK_REDUCE_SUM(localH[5], &sAccum[7])
  BLOCK_REDUCE_SUM(localH[6], &sAccum[8])
  BLOCK_REDUCE_SUM(localH[7], &sAccum[9])
  BLOCK_REDUCE_SUM(localH[8], &sAccum[10])

  #undef BLOCK_REDUCE_SUM

  // ---- Thread 0: compute RMSD from Sp, Sq, singular values of H ----
  if (tid == 0) {
    const double Sp = sAccum[0];
    const double Sq = sAccum[1];
    const double* H = &sAccum[2];

    // G = H^T H  (3x3 symmetric positive semi-definite)
    const double g00 = H[0] * H[0] + H[3] * H[3] + H[6] * H[6];
    const double g01 = H[0] * H[1] + H[3] * H[4] + H[6] * H[7];
    const double g02 = H[0] * H[2] + H[3] * H[5] + H[6] * H[8];
    const double g11 = H[1] * H[1] + H[4] * H[4] + H[7] * H[7];
    const double g12 = H[1] * H[2] + H[4] * H[5] + H[7] * H[8];
    const double g22 = H[2] * H[2] + H[5] * H[5] + H[8] * H[8];

    // Eigenvalues of G = squared singular values of H
    double ev0, ev1, ev2;
    symmetricEigenvalues3x3(g00, g01, g02, g11, g12, g22, ev0, ev1, ev2);

    // Singular values (clamp negatives from numerical noise)
    const double s0 = sqrt(fmax(ev0, 0.0));
    const double s1 = sqrt(fmax(ev1, 0.0));
    double       s2 = sqrt(fmax(ev2, 0.0));

    // Handle reflection: if det(H) < 0, negate the smallest singular value
    if (det3x3(H) < 0.0) {
      s2 = -s2;
    }

    // RMSD^2 = (Sp + Sq - 2*(s0 + s1 + s2)) / N
    const double invN    = 1.0 / static_cast<double>(numAtoms);
    const double rmsdSq  = fmax((Sp + Sq - 2.0 * (s0 + s1 + s2)) * invN, 0.0);
    rmsdOut[pairIdx]     = sqrt(rmsdSq);
  }
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

void conformerRmsdMatrixGpu(cuda::std::span<const double> coords,
                            cuda::std::span<double>       rmsdOut,
                            const int                     numConformers,
                            const int                     numAtoms,
                            const bool                    prealigned,
                            cudaStream_t                  stream) {
  if (numConformers <= 1) return;

  const int numPairs = numConformers * (numConformers - 1) / 2;
  conformerRmsdKernel<<<numPairs, kRmsdBlockSize, 0, stream>>>(
      coords.data(), rmsdOut.data(), numConformers, numAtoms, prealigned);
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit
