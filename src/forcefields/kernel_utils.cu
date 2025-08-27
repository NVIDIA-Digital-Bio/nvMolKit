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

#include "kernel_utils.cuh"

namespace nvMolKit {
namespace FFKernelUtils {
__global__ void reduceEnergiesKernel(const double*  energyBuffer,
                                     const int*     energyBufferBlockIdxToBatchIdx,
                                     double*        outs,
                                     const uint8_t* activeThisStage) {
  assert(blockDim.x == nvMolKit::FFKernelUtils::blockSizeEnergyReduction);

  const int outIdx = energyBufferBlockIdxToBatchIdx[blockIdx.x];

  using BlockReduce = cub::BlockReduce<double, nvMolKit::FFKernelUtils::blockSizeEnergyReduction>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
  if (activeThisStage == nullptr || activeThisStage[outIdx] == 1) {
    const int    termIdx    = threadIdx.x + blockIdx.x * blockDim.x;
    const double energyTerm = energyBuffer[termIdx];
    const double blockSum   = BlockReduce(temp_storage).Sum(energyTerm);

    if (threadIdx.x == 0) {
      atomicAdd(&outs[outIdx], blockSum);
    }
  }
}

__global__ void paddedToUnpaddedWriteBackKernel(const int     totalNumTerms,
                                                const int*    writeBackIndices,
                                                const double* paddedInput,
                                                double*       unpaddedOutput) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalNumTerms) {
    const int outputIdx = writeBackIndices[idx];
    if (outputIdx >= 0) {
      unpaddedOutput[writeBackIndices[idx]] = paddedInput[idx];
    }
  }
}

template <typename T, int fromDim, int toDim>
__global__ void unpaddedToPaddedKernel(const int  totalNumTerms,
                                       const int  maxNumAtomsPerMolecule,
                                       const int* atomStarts,
                                       const int* atomIndexToMoleculeIndex,
                                       const T*   unpaddedInput,
                                       T*         paddedOutput,
                                       int*       writeBackIndices) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < totalNumTerms) {
    const int atomIdx         = idx / fromDim;
    const int dimOffset       = idx % fromDim;
    const int moleculeIndex   = atomIndexToMoleculeIndex[atomIdx];
    const int atomIndexOffset = atomIdx - atomStarts[moleculeIndex];
    const int moleculeOffset  = moleculeIndex * maxNumAtomsPerMolecule * toDim;

    const int outputIdx     = moleculeOffset + toDim * atomIndexOffset + dimOffset;
    paddedOutput[outputIdx] = unpaddedInput[idx];

    if (writeBackIndices != nullptr) {
      writeBackIndices[outputIdx] = idx;
    }
  }
}

template <typename T, int fromDim, int toDim>
cudaError_t launchUnpaddedToPaddedKernel(const int    numAtomsTotal,
                                         const int    maxNumAtomsPerMolecule,
                                         const int*   atomStarts,
                                         const int*   atomIndexToMoleculeIndex,
                                         const T*     unpaddedInput,
                                         T*           paddedOutput,
                                         int*         writeBackIndices,
                                         cudaStream_t stream) {
  const int     numThreads = numAtomsTotal * fromDim;
  constexpr int blockSize  = 256;
  const int     numBlocks  = (numThreads + blockSize - 1) / blockSize;
  unpaddedToPaddedKernel<T, fromDim, toDim><<<numBlocks, blockSize, 0, stream>>>(numThreads,
                                                                                 maxNumAtomsPerMolecule,
                                                                                 atomStarts,
                                                                                 atomIndexToMoleculeIndex,
                                                                                 unpaddedInput,
                                                                                 paddedOutput,
                                                                                 writeBackIndices);
  return cudaGetLastError();
}

// Explicit template instantiations
template cudaError_t launchUnpaddedToPaddedKernel<double, 3, 3>(const int,
                                                                const int,
                                                                const int*,
                                                                const int*,
                                                                const double*,
                                                                double*,
                                                                int*,
                                                                cudaStream_t);
template cudaError_t launchUnpaddedToPaddedKernel<double, 3, 4>(const int,
                                                                const int,
                                                                const int*,
                                                                const int*,
                                                                const double*,
                                                                double*,
                                                                int*,
                                                                cudaStream_t);
template cudaError_t launchUnpaddedToPaddedKernel<int, 1, 1>(const int,
                                                             const int,
                                                             const int*,
                                                             const int*,
                                                             const int*,
                                                             int*,
                                                             int*,
                                                             cudaStream_t);

cudaError_t launchUnpaddedDim3ToPaddedDim3Kernel(const int     numAtomsTotal,
                                                 const int     maxNumAtomsPerMolecule,
                                                 const int*    atomStarts,
                                                 const int*    atomIndexToMoleculeIndex,
                                                 const double* unpaddedInput,
                                                 double*       paddedOutput,
                                                 cudaStream_t  stream) {
  return launchUnpaddedToPaddedKernel<double, 3, 3>(numAtomsTotal,
                                                    maxNumAtomsPerMolecule,
                                                    atomStarts,
                                                    atomIndexToMoleculeIndex,
                                                    unpaddedInput,
                                                    paddedOutput,
                                                    /*writeBackIndices=*/nullptr,
                                                    stream);
}

cudaError_t launchUnpaddedDim3ToPaddedDim4Kernel(const int     numAtomsTotal,
                                                 const int     maxNumAtomsPerMolecule,
                                                 const int*    atomStarts,
                                                 const int*    atomIndexToMoleculeIndex,
                                                 const double* unpaddedInput,
                                                 double*       paddedOutput,
                                                 int*          writeBackIndices,
                                                 cudaStream_t  stream) {
  return launchUnpaddedToPaddedKernel<double, 3, 4>(numAtomsTotal,
                                                    maxNumAtomsPerMolecule,
                                                    atomStarts,
                                                    atomIndexToMoleculeIndex,
                                                    unpaddedInput,
                                                    paddedOutput,
                                                    writeBackIndices,
                                                    stream);
}

cudaError_t launchPaddedDim4ToUnpaddedDim3Kernel(const int     numMolecules,
                                                 const int     maxNumAtomsPerMolecule,
                                                 const int*    writeBackIndices,
                                                 const double* paddedInput,
                                                 double*       unpaddedOutput,
                                                 cudaStream_t  stream) {
  const int     numThreads = numMolecules * maxNumAtomsPerMolecule * 4;
  constexpr int blockSize  = 256;
  const int     numBlocks  = (numThreads + blockSize - 1) / blockSize;
  paddedToUnpaddedWriteBackKernel<<<numBlocks, blockSize, 0, stream>>>(numThreads,
                                                                       writeBackIndices,
                                                                       paddedInput,
                                                                       unpaddedOutput);
  return cudaGetLastError();
}

cudaError_t launchPadAtomNumbersKernel(const int    numAtomsTotal,
                                       const int    maxNumAtomsPerMolecule,
                                       const int*   atomStarts,
                                       const int*   atomIndexToMoleculeIndex,
                                       const int*   unpaddedInput,
                                       int*         paddedOutput,
                                       cudaStream_t stream) {
  return launchUnpaddedToPaddedKernel<int, 1, 1>(numAtomsTotal,
                                                 maxNumAtomsPerMolecule,
                                                 atomStarts,
                                                 atomIndexToMoleculeIndex,
                                                 unpaddedInput,
                                                 paddedOutput,
                                                 /*writeBackIndices=*/nullptr,
                                                 stream);
}

}  // namespace FFKernelUtils
}  // namespace nvMolKit
