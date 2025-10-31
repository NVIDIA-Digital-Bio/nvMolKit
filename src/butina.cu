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

#include "butina.h"
#include "host_vector.h"
#include "nvtx.h"

/**
 * TODO: Future optimizations
 * - Keep a live list of active indices and only dispatch counts for those.
 * - Parallelize singlet/doublet assignment (low priority since these only run once)
 * - Use neighborlists instead of hit matrix. This could be implemented in similarity code too.
 * - Use ArgMax from CUB instead of custom kernel. CUB API changed somewhere between 12.5 and 12.9, so we'd need a
 * compatibility layer.
 * - Use CUDA Graphs for inner loop and exit criteria.
 */
namespace nvMolKit {

namespace {
constexpr int blockSizeCount               = 256;
constexpr int kAssignedAsSingletonSentinel = std::numeric_limits<int>::max() - 1;
constexpr int kMinLoopSizeForAssignment    = 3;

//! Kernel to count the size of each cluster around each point
//! Assigns singleton clusters to a sentinel value for later processing.
//! Looks up and skips finished clusters.
__global__ void butinaKernelCountClusterSize(const cuda::std::span<const uint8_t> hitMatrix,
                                             const cuda::std::span<int>           clusters,
                                             const cuda::std::span<int>           clusterSizes) {
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);
  const auto numPoints = static_cast<int>(clusters.size());

  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const cuda::std::span<const uint8_t> hits       = hitMatrix.subspan(pointIdx * numPoints, numPoints);
  int                                  localCount = 0;
  for (int i = tid; i < numPoints; i += blockSizeCount) {
    const bool isNeighbor = hits[i];
    if (isNeighbor) {
      const int cluster = clusters[i];
      if (cluster < 0) {
        localCount++;
      }
    }
  }

  __shared__ cub::BlockReduce<int, blockSizeCount>::TempStorage tempStorage;
  const int totalCount = cub::BlockReduce<int, blockSizeCount>(tempStorage).Sum(localCount);
  if (tid == 0) {
    clusterSizes[pointIdx] = totalCount;
    if (totalCount < 2) {
      // Note that this would be a data race between writing this cluster[] and another thread reading it[]. However,
      // the hit check should preclude any singleton entry from being read by anything other than its own
      // thread.
      clusters[pointIdx] = kAssignedAsSingletonSentinel;
    }
  }
}

//! Kernel to write the cluster assignment for the largest cluster found
__global__ void butinaWriteClusterValue(const cuda::std::span<const uint8_t> hitMatrix,
                                        const cuda::std::span<int>           clusters,
                                        const int*                           centralIdx,
                                        const int*                           clusterIdx,
                                        const int*                           maxClusterSize) {
  const auto numPoints = static_cast<int>(clusters.size());
  const auto tid       = static_cast<int>(threadIdx.x + (blockIdx.x * blockDim.x));
  const int  clusterSz = *maxClusterSize;
  if (clusterSz < kMinLoopSizeForAssignment) {
    return;
  }
  const int pointIdx = *centralIdx;
  if (pointIdx < 0) {
    return;
  }
  const int clusterVal = *clusterIdx;

  const cuda::std::span<const uint8_t> hits = hitMatrix.subspan(pointIdx * numPoints, numPoints);
  if (tid < numPoints) {
    if (hits[tid]) {
      if (clusters[tid] < 0) {
        clusters[tid] = clusterVal;
      }
    }
  }
}

//! Kernel to bump the cluster index if the last cluster assigned was large enough. The edge case is for if we hit the <
//! 3 criteria.
__global__ void bumpClusterIdxKernel(int* clusterIdx, const int* lastClusterSize) {
  if (const auto tid = static_cast<int>(threadIdx.x); tid == 0) {
    if (*lastClusterSize >= kMinLoopSizeForAssignment) {
      clusterIdx[tid] += 1;
    }
  }
}

//! Identify all pairs that must be in a cluster together at the end of the Butina Loop.
//! Mark them with a sentinel value to be assigned in a secondary pass.
__global__ void pairDoubletKernels(const cuda::std::span<const uint8_t> hitMatrix,
                                   const cuda::std::span<int>           clusters,
                                   const cuda::std::span<const int>     clusterSizes) {
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);
  const auto numPoints = static_cast<int>(clusters.size());

  if (clusterSizes[pointIdx] != 2) {
    return;
  }

  const cuda::std::span<const uint8_t> hits = hitMatrix.subspan(pointIdx * numPoints, numPoints);
  // Loop up to point IDX so that only one of the pairs does the write. The followup kernel will set both values
  for (int i = tid; i < pointIdx; i += blockSizeCount) {
    const bool isNeighbor = hits[i];
    if (i != pointIdx && isNeighbor && clusterSizes[i] == 2) {
      clusters[pointIdx] = kAssignedAsSingletonSentinel - 1 - i;  // Mark as paired with i
      break;
    }
  }
}

//! Assign cluster IDs to doublets marked in the previous kernel
__global__ void assignDoubletIdsKernel(const cuda::std::span<int> clusters, int* nextClusterIdx) {
  int       clusterIdx           = *nextClusterIdx;
  const int expectedDoubletRange = kAssignedAsSingletonSentinel - clusters.size();
  for (int i = static_cast<int>(clusters.size()) - 1; i >= 0; i--) {
    const int clustId = clusters[i];
    if (clustId >= expectedDoubletRange && clustId < kAssignedAsSingletonSentinel) {
      int otherIdx       = kAssignedAsSingletonSentinel - 1 - clustId;
      clusters[i]        = clusterIdx;
      clusters[otherIdx] = clusterIdx;
      clusterIdx++;
    }
  }
  *nextClusterIdx = clusterIdx;
}

//! Assign all remaining singleton clusters their own cluster IDs. These were identified in the counts kernel.
__global__ void assignSingletonIdsKernel(const cuda::std::span<int> clusters, const int* nextClusterIdx) {
  int clusterIdx = *nextClusterIdx;
  for (int i = static_cast<int>(clusters.size()) - 1; i >= 0; i--) {
    const int clustId = clusters[i];
    if (clustId < 0 || clustId == kAssignedAsSingletonSentinel) {
      clusters[i] = clusterIdx;
      clusterIdx++;
    }
  }
}

constexpr int argMaxBlockSize = 512;

//! Custom ArgMax kernel that returns the largest value and index.
__global__ void lastArgMax(const cuda::std::span<const int> values, int* outVal, int* outIdx) {
  int            maxVal = cuda::std::numeric_limits<int>::min();
  int            maxID  = -1;
  __shared__ int foundMaxVal[argMaxBlockSize];
  __shared__ int foundMaxIds[argMaxBlockSize];
  const auto     tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < values.size(); i += argMaxBlockSize) {
    if (const int val = values[i]; val >= maxVal) {
      maxID  = i;
      maxVal = val;
    }
  }
  foundMaxVal[tid] = maxVal;
  foundMaxIds[tid] = maxID;

  __shared__ cub::BlockReduce<int, argMaxBlockSize>::TempStorage storage;
  const int actualMaxVal = cub::BlockReduce<int, argMaxBlockSize>(storage).Reduce(maxVal, cub::Max());
  __syncthreads();  // For shared memory write of maxVal and maxID
  if (tid == 0) {
    *outVal = actualMaxVal;
    for (int i = argMaxBlockSize - 1; i >= 0; i--) {
      if (foundMaxVal[i] == actualMaxVal) {
        *outIdx = foundMaxIds[i];
        break;
      }
    }
  }
}

// TODO - consolidate this to device vector code.
template <typename T> __global__ void setAllKernel(const size_t numElements, T value, T* dst) {
  const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}
template <typename T> void setAll(const cuda::std::span<T>& vec, const T& value, cudaStream_t stream) {
  const size_t numElements = vec.size();
  if (numElements == 0) {
    return;
  }
  constexpr int blockSize = 128;
  const size_t  numBlocks = (numElements + blockSize - 1) / blockSize;
  setAllKernel<<<numBlocks, blockSize, 0, stream>>>(numElements, value, vec.data());
  cudaCheckError(cudaGetLastError());
}

void innerButinaLoop(const int                            numPoints,
                     const cuda::std::span<const uint8_t> hitMatrix,
                     const cuda::std::span<int>           clusters,
                     const cuda::std::span<int>           clusterSizesSpan,
                     const AsyncDevicePtr<int>&           maxIndex,
                     const AsyncDevicePtr<int>&           maxValue,
                     const AsyncDevicePtr<int>&           clusterIdx,
                     PinnedHostVector<int>&               maxCluster,
                     cudaStream_t                         stream) {
  const int numBlocksFlat = ((static_cast<int>(clusterSizesSpan.size()) - 1) / blockSizeCount) + 1;

  butinaKernelCountClusterSize<<<numPoints, blockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizesSpan);
  cudaCheckError(cudaGetLastError());
  lastArgMax<<<1, argMaxBlockSize, 0, stream>>>(clusterSizesSpan, maxValue.data(), maxIndex.data());
  cudaCheckError(cudaGetLastError());
  butinaWriteClusterValue<<<numBlocksFlat, blockSizeCount, 0, stream>>>(hitMatrix,
                                                                        clusters,
                                                                        maxIndex.data(),
                                                                        clusterIdx.data(),
                                                                        maxValue.data());
  cudaCheckError(cudaGetLastError());
  bumpClusterIdxKernel<<<1, 1, 0, stream>>>(clusterIdx.data(), maxValue.data());
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaMemcpyAsync(maxCluster.data(), maxValue.data(), sizeof(int), cudaMemcpyDefault, stream));
  cudaStreamSynchronize(stream);
}

}  // namespace

void butinaGpu(const cuda::std::span<const uint8_t> hitMatrix,
               const cuda::std::span<int>           clusters,
               cudaStream_t                         stream) {
  ScopedNvtxRange setupRange("Butina Setup");
  const size_t    numPoints = clusters.size();
  setAll(clusters, -1, stream);
  if (const size_t matSize = hitMatrix.size(); numPoints * numPoints != matSize) {
    throw std::runtime_error("Butina size mismatch" + std::to_string(numPoints) +
                             " points^2 != " + std::to_string(matSize) + " neighbor matrix size");
  }
  AsyncDeviceVector<int> clusterSizes(clusters.size(), stream);
  clusterSizes.zero();

  const AsyncDevicePtr<int> maxIndex(-1, stream);
  const AsyncDevicePtr<int> maxValue(std::numeric_limits<int>::max(), stream);
  const AsyncDevicePtr<int> clusterIdx(0, stream);
  PinnedHostVector<int>     maxCluster(1);
  maxCluster[0] = std::numeric_limits<int>::max();
  setupRange.pop();
  const auto clusterSizesSpan = toSpan(clusterSizes);

  while (maxCluster[0] >= kMinLoopSizeForAssignment) {
    ScopedNvtxRange loopRange("Butina Loop");
    innerButinaLoop(numPoints,
                    hitMatrix,
                    clusters,
                    clusterSizesSpan,
                    maxIndex,
                    maxValue,
                    clusterIdx,
                    maxCluster,
                    stream);
  }
  pairDoubletKernels<<<numPoints, blockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizesSpan);
  assignDoubletIdsKernel<<<1, 1, 0, stream>>>(clusters, clusterIdx.data());
  assignSingletonIdsKernel<<<1, 1, 0, stream>>>(clusters, clusterIdx.data());
  cudaStreamSynchronize(stream);
}

namespace {

//! CUB Op to threshold a distance matrix into a hit matrix
struct ThresholdOp {
  const double* matrix;
  uint8_t*      hits;
  double        cutoff;
  ThresholdOp(const double* m, uint8_t* h, double c) : matrix(m), hits(h), cutoff(c) {}

  __device__ void operator()(const std::size_t idx) const { hits[idx] = (matrix[idx] < cutoff); }
};

}  // namespace

void butinaGpu(const cuda::std::span<const double> distanceMatrix,
               const cuda::std::span<int>          clusters,
               const double                        cutoff,
               cudaStream_t                        stream) {
  AsyncDeviceVector<uint8_t> hitMatrix(distanceMatrix.size(), stream);
  std::size_t                tempStorageBytes = 0;
  AsyncDeviceVector<uint8_t> tempStorage(0, stream);

  ThresholdOp op{distanceMatrix.data(), hitMatrix.data(), cutoff};
  cub::DeviceFor::Bulk(nullptr,
                       tempStorageBytes,

                       distanceMatrix.size(),
                       op,
                       stream);
  tempStorage.resize(tempStorageBytes);
  cub::DeviceFor::Bulk(tempStorage.data(),
                       tempStorageBytes,

                       distanceMatrix.size(),
                       op,
                       stream);
  butinaGpu(toSpan(hitMatrix), clusters, stream);
}

}  // namespace nvMolKit