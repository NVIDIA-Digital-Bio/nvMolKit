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

#include "cuda_error_check.h"
#include "tfd_detail.h"
#include "tfd_kernels.h"

namespace nvMolKit {

using detail::circularDifference;
using detail::computeDihedralAngle;

namespace {

//! Kernel to compute dihedral angles for all conformers
//! One thread per (conformer, torsion) work item using flattened indexing
__global__ void dihedralKernel(const int    totalWorkItems,
                               const float* positions,
                               const int*   confPositionStarts,
                               const int*   torsionAtoms,
                               const int*   dihedralConfIdx,
                               const int*   dihedralTorsIdx,
                               const int*   dihedralOutIdx,
                               float*       dihedralAngles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalWorkItems) {
    return;
  }

  // Direct index lookup from flattened work items
  int confIdx = dihedralConfIdx[idx];
  int torsIdx = dihedralTorsIdx[idx];
  int outIdx  = dihedralOutIdx[idx];

  // Get torsion atom indices
  int a = torsionAtoms[torsIdx * 4 + 0];
  int b = torsionAtoms[torsIdx * 4 + 1];
  int c = torsionAtoms[torsIdx * 4 + 2];
  int d = torsionAtoms[torsIdx * 4 + 3];

  // Get positions base for this conformer (tightly packed, no padding)
  const float* posBase = positions + confPositionStarts[confIdx];

  float angle = computeDihedralAngle(posBase + a * 3, posBase + b * 3, posBase + c * 3, posBase + d * 3);

  dihedralAngles[outIdx] = angle;
}

//! Kernel to compute TFD matrix values
//! Handles Single, Ring (averaged abs), and Symmetric (cross-product min) torsion types.
//! One thread per conformer pair using flattened indexing.
__global__ void tfdMatrixKernel(const int      totalWorkItems,
                                const float*   dihedralAngles,
                                const float*   torsionWeights,
                                const float*   torsionMaxDevs,
                                const int*     quartetStarts,
                                const uint8_t* torsionTypes,
                                const int*     tfdAnglesI,
                                const int*     tfdAnglesJ,
                                const int*     tfdTorsStart,
                                const int*     tfdNumTorsions,
                                const int*     tfdOutIdx,
                                float*         tfdOutput) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalWorkItems) {
    return;
  }

  int aI      = tfdAnglesI[idx];
  int aJ      = tfdAnglesJ[idx];
  int ts      = tfdTorsStart[idx];
  int numTors = tfdNumTorsions[idx];
  int outIdx  = tfdOutIdx[idx];

  // Base quartet offset for this molecule's torsions
  int qBase = quartetStarts[ts];

  float sumWeightedDev = 0.0f;
  float sumWeights     = 0.0f;

  for (int t = 0; t < numTors; ++t) {
    int     globalT     = ts + t;
    int     qLocalStart = quartetStarts[globalT] - qBase;
    int     numQ        = quartetStarts[globalT + 1] - quartetStarts[globalT];
    uint8_t type        = torsionTypes[globalT];

    float deviation;
    if (type == 0) {  // Single
      deviation = circularDifference(dihedralAngles[aI + qLocalStart], dihedralAngles[aJ + qLocalStart]) /
                  torsionMaxDevs[globalT];
    } else if (type == 1) {  // Ring
      float avgI = 0.0f;
      float avgJ = 0.0f;
      for (int q = 0; q < numQ; ++q) {
        float ai = dihedralAngles[aI + qLocalStart + q];
        float aj = dihedralAngles[aJ + qLocalStart + q];
        avgI += fminf(ai, 360.0f - ai);
        avgJ += fminf(aj, 360.0f - aj);
      }
      avgI /= numQ;
      avgJ /= numQ;
      deviation = fabsf(avgI - avgJ) / torsionMaxDevs[globalT];
    } else {  // Symmetric
      float minDiff = 180.0f;
      for (int qi = 0; qi < numQ; ++qi) {
        for (int qj = 0; qj < numQ; ++qj) {
          float d = circularDifference(dihedralAngles[aI + qLocalStart + qi], dihedralAngles[aJ + qLocalStart + qj]);
          minDiff = fminf(minDiff, d);
        }
      }
      deviation = minDiff / torsionMaxDevs[globalT];
    }

    float weight = torsionWeights[globalT];
    sumWeightedDev += deviation * weight;
    sumWeights += weight;
  }

  tfdOutput[outIdx] = (sumWeights > 1e-10f) ? (sumWeightedDev / sumWeights) : 0.0f;
}

}  // namespace

void launchDihedralKernel(int          totalWorkItems,
                          const float* positions,
                          const int*   confPositionStarts,
                          const int*   torsionAtoms,
                          const int*   dihedralConfIdx,
                          const int*   dihedralTorsIdx,
                          const int*   dihedralOutIdx,
                          float*       dihedralAngles,
                          cudaStream_t stream) {
  if (totalWorkItems == 0) {
    return;
  }

  int gridSize = (totalWorkItems + kTFDBlockSize - 1) / kTFDBlockSize;

  dihedralKernel<<<gridSize, kTFDBlockSize, 0, stream>>>(totalWorkItems,
                                                         positions,
                                                         confPositionStarts,
                                                         torsionAtoms,
                                                         dihedralConfIdx,
                                                         dihedralTorsIdx,
                                                         dihedralOutIdx,
                                                         dihedralAngles);

  cudaCheckError(cudaGetLastError());
}

void launchTFDMatrixKernel(int            totalWorkItems,
                           const float*   dihedralAngles,
                           const float*   torsionWeights,
                           const float*   torsionMaxDevs,
                           const int*     quartetStarts,
                           const uint8_t* torsionTypes,
                           const int*     tfdAnglesI,
                           const int*     tfdAnglesJ,
                           const int*     tfdTorsStart,
                           const int*     tfdNumTorsions,
                           const int*     tfdOutIdx,
                           float*         tfdOutput,
                           cudaStream_t   stream) {
  if (totalWorkItems == 0) {
    return;
  }

  int gridSize = (totalWorkItems + kTFDBlockSize - 1) / kTFDBlockSize;

  tfdMatrixKernel<<<gridSize, kTFDBlockSize, 0, stream>>>(totalWorkItems,
                                                          dihedralAngles,
                                                          torsionWeights,
                                                          torsionMaxDevs,
                                                          quartetStarts,
                                                          torsionTypes,
                                                          tfdAnglesI,
                                                          tfdAnglesJ,
                                                          tfdTorsStart,
                                                          tfdNumTorsions,
                                                          tfdOutIdx,
                                                          tfdOutput);

  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit
