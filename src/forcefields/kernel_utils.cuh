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

#ifndef NVMOLKIT_FF_KERNEL_UTILS_H
#define NVMOLKIT_FF_KERNEL_UTILS_H

#include <cuda_runtime.h>

#include <cstdint>

#include "device_vector.h"

namespace nvMolKit {
namespace FFKernelUtils {

__device__ __forceinline__ double distanceSquared(const double* pos,
                                                  const int     idx1,
                                                  const int     idx2,
                                                  const int     dim = 3) {
  const double dx   = pos[dim * idx1 + 0] - pos[dim * idx2 + 0];
  const double dy   = pos[dim * idx1 + 1] - pos[dim * idx2 + 1];
  const double dz   = pos[dim * idx1 + 2] - pos[dim * idx2 + 2];
  double       dist = dx * dx + dy * dy + dz * dz;
  if (dim == 4) {
    const double dw = pos[dim * idx1 + 3] - pos[dim * idx2 + 3];
    dist += dw * dw;
  }
  return dist;
}

__device__ __forceinline__ double distanceSquaredPosIdx(const double* pos,
                                                        const int     posIdx1,
                                                        const int     posIdx2,
                                                        const int     dim) {
  const double dx   = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  const double dy   = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  const double dz   = pos[posIdx1 + 2] - pos[posIdx2 + 2];
  double       dist = dx * dx + dy * dy + dz * dz;
  if (dim == 4) {
    const double dw = pos[posIdx1 + 3] - pos[posIdx2 + 3];
    dist += dw * dw;
  }
  return dist;
}

template <int fixedDimension, typename floatType = double>
__device__ __forceinline__ floatType distanceSquaredPosIdx(const double* pos, const int posIdx1, const int posIdx2) {
  const floatType dx   = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  const floatType dy   = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  const floatType dz   = pos[posIdx1 + 2] - pos[posIdx2 + 2];
  floatType       dist = dx * dx + dy * dy + dz * dz;
  if constexpr (fixedDimension == 4) {
    const floatType dw = pos[posIdx1 + 3] - pos[posIdx2 + 3];
    dist += dw * dw;
  }
  return dist;
}

template <typename floatTypeIn = double, typename floatTypeOut = double>
__device__ __forceinline__ double distanceSquaredWithComponents(const floatTypeIn* pos,
                                                                const int          idx1,
                                                                const int          idx2,
                                                                floatTypeOut&      dx,
                                                                floatTypeOut&      dy,
                                                                floatTypeOut&      dz) {
  dx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  dy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  dz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ double clamp(const double val, const double minVal, const double maxVal) {
  return fmax(minVal, fmin(maxVal, val));
}

__device__ __forceinline__ float clamp(const float val, const float minVal, const float maxVal) {
  return fmaxf(minVal, fminf(maxVal, val));
}

template <typename TIn, typename TOut>
__device__ __forceinline__ void crossProduct(const TIn& x1,
                                             const TIn& y1,
                                             const TIn& z1,
                                             const TIn& x2,
                                             const TIn& y2,
                                             const TIn& z2,
                                             TOut&      x,
                                             TOut&      y,
                                             TOut&      z) {
  x = y1 * z2 - z1 * y2;
  y = z1 * x2 - x1 * z2;
  z = x1 * y2 - y1 * x2;
}

template <typename T>
__device__ __forceinline__ T dotProduct(const T& x1, const T& y1, const T& z1, const T& x2, const T& y2, const T& z2) {
  return x1 * x2 + y1 * y2 + z1 * z2;
}

__device__ __forceinline__ void clipToOne(double& x) {
  x = fmax(-1.0, fmin(1.0, x));
}

__device__ __forceinline__ bool isDoubleZero(const double val) {
  return ((val < 1.0e-10) && (val > -1.0e-10));
}

__device__ __forceinline__ bool isFloatZero(const float val) {
  return ((val < 1.0e-10f) && (val > -1.0e-10f));
}

__device__ __forceinline__ int getEnergyAccumulatorIndex(const int  absoluteIdx,
                                                         const int  batchIdx,
                                                         const int* energyBufferStarts,
                                                         const int* termBatchStarts) {
  const int energyBufferStart = energyBufferStarts[batchIdx];
  const int termStart         = termBatchStarts[batchIdx];
  const int termRelativeIdx   = absoluteIdx - termStart;
  return termRelativeIdx + energyBufferStart;
}

__device__ __forceinline__ void normalizeVector(double& x, double& y, double& z) {
  const double norm = sqrt(x * x + y * y + z * z);
  x /= norm;
  y /= norm;
  z /= norm;
}

__global__ void reduceEnergiesKernel(const double*  energyBuffer,
                                     const int*     energyBufferBlockIdxToBatchIdx,
                                     double*        outs,
                                     const uint8_t* activeThisStage = nullptr);

__global__ void paddedToUnpaddedWriteBackKernel(const int     totalNumTerms,
                                                const int*    writeBackIndices,
                                                const double* paddedInput,
                                                double*       unpaddedOutput);

constexpr int blockSizeEnergyReduction = 128;

template <typename BatchedMolecularSystemHost, typename BatchedMolecularDeviceBuffers>
void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice,
                                 const int                         numMols) {
  const int totalNumEnergyTerms = molSystemHost.indices.energyBufferStarts.back();
  molSystemDevice.energyBuffer.resize(totalNumEnergyTerms);
  molSystemDevice.energyBuffer.zero();
  molSystemDevice.energyOuts.resize(numMols);
  molSystemDevice.energyOuts.zero();
}

template <typename BatchedMolecularSystemHost, typename BatchedMolecularDeviceBuffers>
void allocateDim4ConversionBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                   BatchedMolecularDeviceBuffers&    molSystemDevice,
                                   const int                         numMolecules) {
  // Compute maximum system size
  const int maxSystemNumAtoms = molSystemHost.maxNumAtoms;

  const int dim4PaddedSize = 4 * maxSystemNumAtoms * numMolecules;
  const int dim3PaddedSize = 3 * maxSystemNumAtoms * numMolecules;
  molSystemDevice.dataFormatInterchangeBuffers.gradD3Padded.resize(dim3PaddedSize);
  molSystemDevice.dataFormatInterchangeBuffers.gradD3Padded.zero();
  molSystemDevice.dataFormatInterchangeBuffers.positionsD4Padded.resize(dim4PaddedSize);
  molSystemDevice.dataFormatInterchangeBuffers.positionsD4Padded.zero();

  molSystemDevice.dataFormatInterchangeBuffers.writeBackIndices.resize(dim4PaddedSize);
  // TODO: kernel for setting nonzero int values
  std::vector<int> copyBuffer(dim4PaddedSize, -1);
  molSystemDevice.dataFormatInterchangeBuffers.writeBackIndices.setFromVector(copyBuffer);

  if (molSystemHost.atomNumbers.size() > 0) {
    molSystemDevice.dataFormatInterchangeBuffers.atomNumbers.resize(maxSystemNumAtoms * numMolecules);
    molSystemDevice.dataFormatInterchangeBuffers.atomNumbers.zero();
  }
}

template <typename T, int fromDim, int toDim>
cudaError_t launchUnpaddedToPaddedKernel(const int    numAtomsTotal,
                                         const int    maxNumAtomsPerMolecule,
                                         const int*   atomStarts,
                                         const int*   atomIndexToMoleculeIndex,
                                         const T*     unpaddedInput,
                                         T*           paddedOutput,
                                         int*         writeBackIndices = nullptr,
                                         cudaStream_t stream           = nullptr);

// Explicit template instantiations
extern template cudaError_t launchUnpaddedToPaddedKernel<double, 3, 3>(const int,
                                                                       const int,
                                                                       const int*,
                                                                       const int*,
                                                                       const double*,
                                                                       double*,
                                                                       int*,
                                                                       cudaStream_t);
extern template cudaError_t launchUnpaddedToPaddedKernel<double, 3, 4>(const int,
                                                                       const int,
                                                                       const int*,
                                                                       const int*,
                                                                       const double*,
                                                                       double*,
                                                                       int*,
                                                                       cudaStream_t);
extern template cudaError_t launchUnpaddedToPaddedKernel<int, 1, 1>(const int,
                                                                    const int,
                                                                    const int*,
                                                                    const int*,
                                                                    const int*,
                                                                    int*,
                                                                    int*,
                                                                    cudaStream_t);

//! Converts a 3D dense per-atom batched array to a 3D array with each molecule padded out to numAtomsPerMolecule.
//! \param numAtomsTotal              Sum of atoms over all molecules
//! \param maxNumAtomsPerMolecule     Padding amount
//! \param atomStarts                 Start atom index of each molecule
//! \param atomIndexToMoleculeIndex   Mapping of atom index to molecule ID
//! \param unpaddedInput              num_atoms_total * 3 array
//! \param paddedOutput               num_molecules * max_num_atoms_per_molecule * 4 array
//! \param stream                     Optional CUDA stream.
//! \return cudaSuccess or CUDA launch error.
cudaError_t launchUnpaddedDim3ToPaddedDim3Kernel(const int     numAtomsTotal,
                                                 const int     maxNumAtomsPerMolecule,
                                                 const int*    atomStarts,
                                                 const int*    atomIndexToMoleculeIndex,
                                                 const double* unpaddedInput,
                                                 double*       paddedOutput,
                                                 cudaStream_t  stream = 0);

//! Converts a 3D dense per-atom batched array to a 4D array with each molecule padded out to numAtomsPerMolecule.
//! Optionally populates a per-double writebackIndices array for the backwards conversion. If used, must be the same
//! size as paddedOutput
//! \param numAtomsTotal              Sum of atoms over all molecules
//! \param maxNumAtomsPerMolecule     Padding amount
//! \param atomStarts                 Start atom index of each molecule
//! \param atomIndexToMoleculeIndex   Mapping of atom index to molecule ID
//! \param unpaddedInput              num_atoms_total * 3 array
//! \param paddedOutput               num_molecules * max_num_atoms_per_molecule * 4 array
//! \param writeBackIndices           If not null, writes per-thread writeback indices for the backwards conversion.
//! \param stream                     Optional CUDA stream.
//! \return cudaSuccess or CUDA launch error.
cudaError_t launchUnpaddedDim3ToPaddedDim4Kernel(const int     numAtomsTotal,
                                                 const int     maxNumAtomsPerMolecule,
                                                 const int*    atomStarts,
                                                 const int*    atomIndexToMoleculeIndex,
                                                 const double* unpaddedInput,
                                                 double*       paddedOutput,
                                                 int*          writeBackIndices,
                                                 cudaStream_t  stream = 0);

//! Converts a 4D padded per-molecule array to a 3D dense array. Uses info gathered from the 3D->4D conversion for
//! indexing, cannot be used without the data from the forward pass via writeBackIndices. \param numMolecules Batch size
//! \param maxNumAtomsPerMolecule  Current padding in num_atoms of the 4D array
//! \param writeBackIndices        4 * num_atoms * maxAtomsPerMolecule
//! \param paddedInput             4 * num_atoms * maxAtomsPerMolecule
//! \param unpaddedOutput          totalNumAtoms * 3
//! \param stream                  Optional stream
//! \return cudaSuccess or CUDA launch error.
cudaError_t launchPaddedDim4ToUnpaddedDim3Kernel(const int     numMolecules,
                                                 const int     maxNumAtomsPerMolecule,
                                                 const int*    writeBackIndices,
                                                 const double* paddedInput,
                                                 double*       unpaddedOutput,
                                                 cudaStream_t  stream = 0);

//! Converts per-atom dense atom number arrays to padded arrays.
cudaError_t launchPadAtomNumbersKernel(const int    numAtomsTotal,
                                       const int    maxNumAtomsPerMolecule,
                                       const int*   atomStarts,
                                       const int*   atomIndexToMoleculeIndex,
                                       const int*   unpaddedInput,
                                       int*         paddedOutput,
                                       cudaStream_t stream = 0);

}  // namespace FFKernelUtils
}  // namespace nvMolKit

#endif  // NVMOLKIT_FF_KERNEL_UTILS_H
