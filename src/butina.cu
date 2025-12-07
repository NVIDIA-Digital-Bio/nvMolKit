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

#include <cooperative_groups.h>

#include <cub/cub.cuh>

#include "butina.h"
#include "cub_helpers.cuh"
#include "host_vector.h"
#include "nvtx.h"

/**
 * TODO: Future optimizations
 * - Keep a live list of active indices and only dispatch counts for those.
 * - Parallelize singlet/doublet assignment (low priority since these only run once)
 * - Use ArgMax from CUB instead of custom kernel. CUB API changed somewhere between 12.5 and 12.9, so we'd need a
 * compatibility layer.
 * - Use CUDA Graphs for inner loop and exit criteria.
 */
namespace nvMolKit {

namespace {
constexpr int blockSizeCount = 256;
constexpr int kSubTileSize   = 8;

using detail::kMinLoopSizeForAssignment;

__device__ __forceinline__ void sumCountsAndStoreClusterSize(const int                  tid,
                                                             const int                  pointIdx,
                                                             const cuda::std::span<int> clusterSizes,
                                                             const int                  localCount) {
  __shared__ cub::BlockReduce<int, blockSizeCount>::TempStorage tempStorage;
  const int totalCount = cub::BlockReduce<int, blockSizeCount>(tempStorage).Sum(localCount);
  if (tid == 0) {
    clusterSizes[pointIdx] = totalCount;
  }
}

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
    if (hits[i]) {
      const int cluster = clusters[i];
      if (cluster < 0) {
        localCount++;
      }
    }
  }

  sumCountsAndStoreClusterSize(tid, pointIdx, clusterSizes, localCount);
}

//! Kernel to count the size of each cluster around each point, assigning a neighborlist for later use.
//! IMPORTANT: This assumes that the maximum cluster size is small enough to fit in the neighborlist, so should only
//! be called when that is known to be true.
template <int NeighborlistMaxSize>
__global__ void butinaKernelCountClusterSizeWithNeighborlist(const cuda::std::span<const uint8_t> hitMatrix,
                                                             const cuda::std::span<int>           clusters,
                                                             const cuda::std::span<int>           clusterSizes,
                                                             const cuda::std::span<int>           neighborList) {
  static_assert(NeighborlistMaxSize % kSubTileSize == 0, "NeighborlistMaxSize must be multiple of kSubTileSize");
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);
  const auto numPoints = static_cast<int>(clusters.size());

  __shared__ int neighborlistIndex;
  __shared__ int sharedNeighborlist[NeighborlistMaxSize];

  if (tid == 0) {
    neighborlistIndex = 0;
  }
  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const cuda::std::span<const uint8_t> hits       = hitMatrix.subspan(pointIdx * numPoints, numPoints);
  int                                  localCount = 0;
  __syncthreads();  // for neighborlistIndex init
  for (int i = tid; i < numPoints; i += blockSizeCount) {
    if (hits[i]) {
      const int cluster = clusters[i];
      if (cluster < 0) {
        localCount++;
        const int index           = atomicAdd(&neighborlistIndex, 1);
        sharedNeighborlist[index] = i;
      }
    }
  }

  // Coalesced write of neighborlist using loop for variable sizes
  __syncthreads();  // for sharedNeighborlist final value
  for (int i = tid; i < NeighborlistMaxSize; i += blockSizeCount) {
    neighborList[pointIdx * NeighborlistMaxSize + i] = (i < neighborlistIndex) ? sharedNeighborlist[i] : -1;
  }

  sumCountsAndStoreClusterSize(tid, pointIdx, clusterSizes, localCount);
}

namespace cg = cooperative_groups;

constexpr int blockSizeAssign      = 128;
constexpr int kTilesPerBlockAssign = blockSizeAssign / kSubTileSize;

template <int NeighborlistMaxSize, bool StrictIndexing>
__global__ void attemptAssignClustersFromNeighborlist(const cuda::std::span<int>       clusters,
                                                      const cuda::std::span<const int> clusterSizes,
                                                      const cuda::std::span<const int> neighborList,
                                                      const int*                       maxClusterSize,
                                                      const int*                       designatedMaxIdx,
                                                      int*                             nextClusterIdx) {
  static_assert(NeighborlistMaxSize % kSubTileSize == 0, "NeighborlistMaxSize must be multiple of kSubTileSize");

  const auto     tile8       = cg::tiled_partition<kSubTileSize>(cg::this_thread_block());
  const int      rankInBlock = tile8.meta_group_rank();
  const int      tid         = tile8.thread_rank();
  __shared__ int candidateNeighborsBlock[kTilesPerBlockAssign][NeighborlistMaxSize];
  __shared__ int foundIssueBlock[kTilesPerBlockAssign];

  int* sharedFoundIssue         = &foundIssueBlock[rankInBlock];
  int* sharedCandidateNeighbors = &candidateNeighborsBlock[rankInBlock][0];

  if (tid == 0) {
    foundIssueBlock[rankInBlock] = 0;
  }

  // For global tile index across the grid:
  constexpr int tilesPerBlock = blockSizeAssign / kSubTileSize;
  const int     pointIdx      = blockIdx.x * tilesPerBlock + rankInBlock;
  if (pointIdx >= clusters.size()) {
    return;
  }

  const int clustId = clusters[pointIdx];
  if (clustId >= 0) {
    return;
  }
  const int clusterSize     = clusterSizes[pointIdx];
  const int isDesignatedMax = (pointIdx == *designatedMaxIdx);

  if constexpr (StrictIndexing) {
    if (clusterSize != *maxClusterSize && !isDesignatedMax) {
      return;
    }
  }

  // Load neighborlist into shared memory using loop for variable sizes
  for (int i = tid; i < NeighborlistMaxSize; i += kSubTileSize) {
    sharedCandidateNeighbors[i] = neighborList[pointIdx * NeighborlistMaxSize + i];
  }
  tile8.sync();

  for (int i = 0; i < clusterSize; i++) {
    const int candidateNeighbor            = sharedCandidateNeighbors[i];
    const int candidateNeighborClusterSize = clusterSizes[candidateNeighbor];

    // If neighbor has larger cluster, they should be processed instead
    if (candidateNeighborClusterSize > clusterSize) {
      return;
    }

    // If neighbor has SAME cluster size and lower index, defer to them for consistency
    // Also defer if neighbor is the designated max (guarantees only designated max assigns among ties)
    // Designated max itself skips this check to guarantee forward progress
    if (!isDesignatedMax && candidateNeighborClusterSize == clusterSize &&
        (candidateNeighbor < pointIdx || candidateNeighbor == *designatedMaxIdx)) {
      return;
    }

    // If neighbor has smaller cluster size, we're the better centroid - continue

    // Now we verify that all of these neighbors have the same neighbors we do. Each thread checks 1 candidate at a
    // time. This will rule out our neighbors being connected to a larger cluster.
    for (int oidx = tid; oidx < candidateNeighborClusterSize; oidx += kSubTileSize) {
      const int otherNeighbor = neighborList[candidateNeighbor * NeighborlistMaxSize + oidx];
      bool      foundMatch    = false;
      // One of the neighbors will be ourselves, by definition.
      if (otherNeighbor == pointIdx) {
        foundMatch = true;
      } else {
        for (int j = 0; j < clusterSize; j++) {
          if (otherNeighbor == sharedCandidateNeighbors[j]) {
            foundMatch = true;
            break;
          }
        }
      }
      if (!foundMatch) {
        // We might still be ok if that neighbor is a smaller cluster.
        // Designated max only bails on strictly larger (which can't happen for the true max).
        if (clusterSizes[otherNeighbor] > clusterSize ||
            (clusterSizes[otherNeighbor] == clusterSize && !isDesignatedMax)) {
          atomicExch(sharedFoundIssue, 1);
        }
      }
    }
    tile8.sync();
    if (*sharedFoundIssue) {
      return;
    }
  }

  // At this point, we have a valid cluster. Assign it.
  int clusterVal;
  if (tid == 0) {
    clusterVal         = atomicAdd(nextClusterIdx, 1);
    clusters[pointIdx] = clusterVal;
    // if (tid == 0) {
    //   printf("Assigning NL cluster %d of size %d around point %d\n  Neighbors: %d %d %d %d %d %d %d %d\n",
    //   clusterVal, clusterSize, pointIdx, sharedCandidateNeighbors[0], sharedCandidateNeighbors[1],
    //   sharedCandidateNeighbors[2], sharedCandidateNeighbors[3], sharedCandidateNeighbors[4],
    //   sharedCandidateNeighbors[5], sharedCandidateNeighbors[6], sharedCandidateNeighbors[7]);
    // }
  }
  tile8.sync();
  clusterVal = tile8.shfl(clusterVal, 0);
  // Assign neighbors using loop for variable sizes
  for (int i = tid; i < clusterSize; i += kSubTileSize) {
    const int assignIdx = sharedCandidateNeighbors[i];
    if (clusters[assignIdx] < 0) {
      clusters[assignIdx] = clusterVal;
    }  // else {
    //   printf("GUARD: Skipping point %d (already cluster %d) for NL cluster %d\n",
    //          assignIdx, clusters[assignIdx], clusterVal);
    // }
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
  // if (tid == 0) {
  //   printf("Assigning hit cluster %d of size %d around point %d\n", *clusterIdx, clusterSz, pointIdx);
  // }
  const int                            clusterVal = *clusterIdx;
  const cuda::std::span<const uint8_t> hits       = hitMatrix.subspan(pointIdx * numPoints, numPoints);
  if (tid < numPoints) {
    if (hits[tid]) {
      if (clusters[tid] < 0) {
        // printf("Assigning hit cluster %d to point %d\n", clusterVal, tid);
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
      clusterIdx[0] += 1;
    }
  }
}

constexpr int kSingletonBlockSize = 512;

//! Assign all remaining unassigned points their own singleton cluster IDs.
__global__ void assignSingletonIdsKernel(const cuda::std::span<int> clusters, int* nextClusterIdx) {
  __shared__ int sharedClusterIdx;
  const int      tid       = threadIdx.x;
  const int      numPoints = static_cast<int>(clusters.size());

  if (tid == 0) {
    sharedClusterIdx = *nextClusterIdx;
  }
  __syncthreads();

  for (int i = tid; i < numPoints; i += kSingletonBlockSize) {
    if (clusters[i] < 0) {
      const int myClusterIdx = atomicAdd(&sharedClusterIdx, 1);
      clusters[i]            = myClusterIdx;
    }
  }
}

}  // namespace

constexpr int argMaxBlockSize = 512;

/**
 * @brief Prune neighborlists by removing assigned neighbors, using CUB WarpMergeSort.
 */
template <int NeighborlistMaxSize>
__global__ void pruneNeighborlistKernel(const cuda::std::span<int> clusters,
                                        const cuda::std::span<int> clusterSizes,
                                        const cuda::std::span<int> neighborList) {
  constexpr int kWarpSize       = 32;
  constexpr int kItemsPerThread = (NeighborlistMaxSize + kWarpSize - 1) / kWarpSize;
  static_assert(NeighborlistMaxSize <= 128, "NeighborlistMaxSize must be <= 128");
  static_assert(NeighborlistMaxSize % 8 == 0, "NeighborlistMaxSize must be multiple of 8");

  using WarpMergeSort = cub::WarpMergeSort<int, kItemsPerThread, kWarpSize, int>;
  using WarpReduce    = cub::WarpReduce<int>;

  constexpr int                                  kWarpsPerBlock = 4;
  __shared__ typename WarpMergeSort::TempStorage sortStorage[kWarpsPerBlock];
  __shared__ typename WarpReduce::TempStorage    reduceStorage[kWarpsPerBlock];

  const auto tile     = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  const int  tid      = tile.thread_rank();
  const int  warpId   = tile.meta_group_rank();
  const int  pointIdx = blockIdx.x * kWarpsPerBlock + warpId;

  if (pointIdx >= static_cast<int>(clusters.size())) {
    return;
  }

  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const int currentSize = clusterSizes[pointIdx];
  const int baseOffset  = pointIdx * NeighborlistMaxSize;

  // Each thread loads kItemsPerThread neighbors in blocked arrangement
  int keys[kItemsPerThread];
  int values[kItemsPerThread];

  for (int item = 0; item < kItemsPerThread; ++item) {
    const int globalIdx = tid * kItemsPerThread + item;
    if (globalIdx < NeighborlistMaxSize) {
      values[item]     = neighborList[baseOffset + globalIdx];
      const bool valid = (globalIdx < currentSize) && (values[item] >= 0) && (clusters[values[item]] < 0);
      keys[item]       = valid ? 0 : 1;  // 0 = valid (sort first), 1 = invalid (sort last)
    } else {
      values[item] = -1;
      keys[item]   = 1;
    }
  }

  // Sort by key ascending: valid neighbors (key=0) come first
  WarpMergeSort(sortStorage[warpId]).Sort(keys, values, cubLess{});

  // Count valid entries across all items in this thread
  int localValidCount = 0;
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int globalIdx = tid * kItemsPerThread + item;
    if (globalIdx < NeighborlistMaxSize && keys[item] == 0) {
      ++localValidCount;
    }
  }

  // Reduce across warp using CUB (result only valid in lane 0, broadcast to all)
  int newCount = WarpReduce(reduceStorage[warpId]).Sum(localValidCount);
  newCount     = tile.shfl(newCount, 0);

  if (tid == 0) {
    clusterSizes[pointIdx] = newCount;
  }

  // Write sorted neighborlist back in blocked arrangement
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int globalIdx = tid * kItemsPerThread + item;
    if (globalIdx < NeighborlistMaxSize) {
      neighborList[baseOffset + globalIdx] = (globalIdx < newCount) ? values[item] : -1;
    }
  }
}

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
  const int actualMaxVal = cub::BlockReduce<int, argMaxBlockSize>(storage).Reduce(maxVal, cubMax());
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

namespace {

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

/**
 * @brief Build the initial neighborlist and cluster sizes from the hit matrix.
 *
 * This is called once before entering the pruning loop.
 */
template <int NeighborlistMaxSize>
void buildInitialNeighborlist(const int                            numPoints,
                              const cuda::std::span<const uint8_t> hitMatrix,
                              const cuda::std::span<int>           clusters,
                              const cuda::std::span<int>           clusterSizesSpan,
                              const cuda::std::span<int>           neighborList,
                              const AsyncDevicePtr<int>&           maxValue,
                              PinnedHostVector<int>&               maxCluster,
                              cudaStream_t                         stream) {
  const ScopedNvtxRange range("Build initial neighborlist");
  butinaKernelCountClusterSizeWithNeighborlist<NeighborlistMaxSize>
    <<<numPoints, blockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizesSpan, neighborList);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaStreamSynchronize(stream));
}

/**
 * @brief Inner loop iteration that attempts assignment then prunes neighborlists.
 *
 * Unlike the rebuild approach, this modifies neighborlists in-place by removing
 * assigned neighbors and compacting valid ones to the front.
 */
template <int NeighborlistMaxSize>
void innerButinaLoopWithPruning(const int                  numPoints,
                                const cuda::std::span<int> clusters,
                                const cuda::std::span<int> clusterSizesSpan,
                                const AsyncDevicePtr<int>& maxIndex,
                                const AsyncDevicePtr<int>& maxValue,
                                const AsyncDevicePtr<int>& clusterIdx,
                                PinnedHostVector<int>&     maxCluster,
                                const cuda::std::span<int> neighborList,
                                const bool                 enforceStrictIndexing,
                                cudaStream_t               stream) {
  // Uses maxValue/maxIndex from previous iteration's argmax (or initial argmax before loop)
  const int numBlocksAssign = (numPoints + kTilesPerBlockAssign - 1) / kTilesPerBlockAssign;
  if (enforceStrictIndexing) {
    attemptAssignClustersFromNeighborlist<NeighborlistMaxSize, true>
      <<<numBlocksAssign, blockSizeAssign, 0, stream>>>(clusters,
                                                        clusterSizesSpan,
                                                        neighborList,
                                                        maxValue.data(),
                                                        maxIndex.data(),
                                                        clusterIdx.data());
  } else {
    attemptAssignClustersFromNeighborlist<NeighborlistMaxSize, false>
      <<<numBlocksAssign, blockSizeAssign, 0, stream>>>(clusters,
                                                        clusterSizesSpan,
                                                        neighborList,
                                                        maxValue.data(),
                                                        maxIndex.data(),
                                                        clusterIdx.data());
  }
  cudaCheckError(cudaGetLastError());

  // Prune assigned neighbors from all neighborlists and update counts
  constexpr int kWarpsPerBlock  = 4;
  constexpr int kPruneBlockSize = kWarpsPerBlock * 32;
  const int     numBlocksPrune  = (numPoints + kWarpsPerBlock - 1) / kWarpsPerBlock;
  pruneNeighborlistKernel<NeighborlistMaxSize>
    <<<numBlocksPrune, kPruneBlockSize, 0, stream>>>(clusters, clusterSizesSpan, neighborList);
  cudaCheckError(cudaGetLastError());

  // Compute argmax for next iteration, copy to host before sync
  lastArgMax<<<1, argMaxBlockSize, 0, stream>>>(clusterSizesSpan, maxValue.data(), maxIndex.data());
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaMemcpyAsync(maxCluster.data(), maxValue.data(), sizeof(int), cudaMemcpyDefault, stream));

  cudaStreamSynchronize(stream);
}

template <int NeighborlistMaxSize>
void butinaGpuImpl(const cuda::std::span<const uint8_t> hitMatrix,
                   const cuda::std::span<int>           clusters,
                   const bool                           enforceStrictIndexing,
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
  AsyncDeviceVector<int> neighborList(NeighborlistMaxSize * numPoints, stream);
  const auto             neighborListSpan = toSpan(neighborList);

  const AsyncDevicePtr<int> maxIndex(-1, stream);
  const AsyncDevicePtr<int> maxValue(std::numeric_limits<int>::max(), stream);
  const AsyncDevicePtr<int> clusterIdx(0, stream);
  PinnedHostVector<int>     maxCluster(1);
  maxCluster[0] = std::numeric_limits<int>::max();
  setupRange.pop();
  const auto clusterSizesSpan = toSpan(clusterSizes);

  // If a neighborlist is up to N, then the cluster is up to N+1 (including the central point).
  constexpr int clusterSizeWithMaxNeighborlist = NeighborlistMaxSize + 1;
  while (maxCluster[0] >= clusterSizeWithMaxNeighborlist) {
    const std::string     maxClusterSize = std::to_string(maxCluster[0]);
    const ScopedNvtxRange loopRange("Large cluster Butina Loop, max cluster: " + maxClusterSize);
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

  // Build neighborlist once, then prune dynamically instead of rebuilding each iteration
  if (maxCluster[0] >= kMinLoopSizeForAssignment) {
    buildInitialNeighborlist<NeighborlistMaxSize>(numPoints,
                                                  hitMatrix,
                                                  clusters,
                                                  clusterSizesSpan,
                                                  neighborListSpan,
                                                  maxValue,
                                                  maxCluster,
                                                  stream);

    // Initial argmax to prime the loop (buildInitialNeighborlist already synced)
    lastArgMax<<<1, argMaxBlockSize, 0, stream>>>(clusterSizesSpan, maxValue.data(), maxIndex.data());
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaMemcpyAsync(maxCluster.data(), maxValue.data(), sizeof(int), cudaMemcpyDefault, stream));
    cudaStreamSynchronize(stream);

    while (maxCluster[0] >= kMinLoopSizeForAssignment) {
      const std::string     maxClusterSize = std::to_string(maxCluster[0]);
      const ScopedNvtxRange loopRange("Small cluster Butina Loop with pruning, max cluster: " + maxClusterSize);
      innerButinaLoopWithPruning<NeighborlistMaxSize>(numPoints,
                                                      clusters,
                                                      clusterSizesSpan,
                                                      maxIndex,
                                                      maxValue,
                                                      clusterIdx,
                                                      maxCluster,
                                                      neighborListSpan,
                                                      enforceStrictIndexing,
                                                      stream);
    }
  }

  assignSingletonIdsKernel<<<1, kSingletonBlockSize, 0, stream>>>(clusters, clusterIdx.data());
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaStreamSynchronize(stream));
}

}  // namespace

void butinaGpu(const cuda::std::span<const uint8_t> hitMatrix,
               const cuda::std::span<int>           clusters,
               const bool                           enforceStrictIndexing,
               const int                            neighborlistMaxSize,
               cudaStream_t                         stream) {
  switch (neighborlistMaxSize) {
    case 8:
      butinaGpuImpl<8>(hitMatrix, clusters, enforceStrictIndexing, stream);
      break;
    case 16:
      butinaGpuImpl<16>(hitMatrix, clusters, enforceStrictIndexing, stream);
      break;
    case 24:
      butinaGpuImpl<24>(hitMatrix, clusters, enforceStrictIndexing, stream);
      break;
    case 32:
      butinaGpuImpl<32>(hitMatrix, clusters, enforceStrictIndexing, stream);
      break;
    case 64:
      butinaGpuImpl<64>(hitMatrix, clusters, enforceStrictIndexing, stream);
      break;
    case 128:
      butinaGpuImpl<128>(hitMatrix, clusters, enforceStrictIndexing, stream);
      break;
    default:
      throw std::invalid_argument("neighborlistMaxSize must be 8, 16, 24, 32, 64, or 128. Got: " +
                                  std::to_string(neighborlistMaxSize));
  }
}

namespace {

__global__ void thresholdDistanceMatrixKernel(const double* __restrict__ matrix,
                                              uint8_t* __restrict__ hits,
                                              const double cutoff,
                                              const size_t numElements) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    hits[idx] = (matrix[idx] <= cutoff);
  }
}

}  // namespace

void butinaGpu(const cuda::std::span<const double> distanceMatrix,
               const cuda::std::span<int>          clusters,
               const double                        cutoff,
               const bool                          enforceStrictIndexing,
               const int                           neighborlistMaxSize,
               cudaStream_t                        stream) {
  AsyncDeviceVector<uint8_t> hitMatrix(distanceMatrix.size(), stream);

  constexpr int blockSize = 256;
  const size_t  numBlocks = (distanceMatrix.size() + blockSize - 1) / blockSize;
  thresholdDistanceMatrixKernel<<<numBlocks, blockSize, 0, stream>>>(distanceMatrix.data(),
                                                                     hitMatrix.data(),
                                                                     cutoff,
                                                                     distanceMatrix.size());
  cudaCheckError(cudaGetLastError());

  butinaGpu(toSpan(hitMatrix), clusters, enforceStrictIndexing, neighborlistMaxSize, stream);
}

namespace detail {

constexpr int kBlockSizeCount = 256;
constexpr int argMaxBlockSize = 512;

template <int NeighborlistMaxSize>
void launchPruneNeighborlistKernel(cuda::std::span<int> clusters,
                                   cuda::std::span<int> clusterSizes,
                                   cuda::std::span<int> neighborList,
                                   const int            numPoints,
                                   cudaStream_t         stream) {
  constexpr int kWarpsPerBlock  = 4;
  constexpr int kPruneBlockSize = kWarpsPerBlock * 32;
  const int     numBlocks       = (numPoints + kWarpsPerBlock - 1) / kWarpsPerBlock;
  pruneNeighborlistKernel<NeighborlistMaxSize>
    <<<numBlocks, kPruneBlockSize, 0, stream>>>(clusters, clusterSizes, neighborList);
  cudaCheckError(cudaGetLastError());
}

template void launchPruneNeighborlistKernel<8>(cuda::std::span<int>,
                                               cuda::std::span<int>,
                                               cuda::std::span<int>,
                                               int,
                                               cudaStream_t);
template void launchPruneNeighborlistKernel<16>(cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                int,
                                                cudaStream_t);
template void launchPruneNeighborlistKernel<24>(cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                int,
                                                cudaStream_t);
template void launchPruneNeighborlistKernel<32>(cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                int,
                                                cudaStream_t);
template void launchPruneNeighborlistKernel<64>(cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                int,
                                                cudaStream_t);
template void launchPruneNeighborlistKernel<128>(cuda::std::span<int>,
                                                 cuda::std::span<int>,
                                                 cuda::std::span<int>,
                                                 int,
                                                 cudaStream_t);

template <int NeighborlistMaxSize>
void launchBuildNeighborlistKernel(cuda::std::span<const uint8_t> hitMatrix,
                                   cuda::std::span<int>           clusters,
                                   cuda::std::span<int>           clusterSizes,
                                   cuda::std::span<int>           neighborList,
                                   const int                      numPoints,
                                   cudaStream_t                   stream) {
  butinaKernelCountClusterSizeWithNeighborlist<NeighborlistMaxSize>
    <<<numPoints, kBlockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizes, neighborList);
  cudaCheckError(cudaGetLastError());
}

template void launchBuildNeighborlistKernel<8>(cuda::std::span<const uint8_t>,
                                               cuda::std::span<int>,
                                               cuda::std::span<int>,
                                               cuda::std::span<int>,
                                               int,
                                               cudaStream_t);
template void launchBuildNeighborlistKernel<16>(cuda::std::span<const uint8_t>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                int,
                                                cudaStream_t);
template void launchBuildNeighborlistKernel<24>(cuda::std::span<const uint8_t>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                int,
                                                cudaStream_t);
template void launchBuildNeighborlistKernel<32>(cuda::std::span<const uint8_t>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                int,
                                                cudaStream_t);
template void launchBuildNeighborlistKernel<64>(cuda::std::span<const uint8_t>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                cuda::std::span<int>,
                                                int,
                                                cudaStream_t);
template void launchBuildNeighborlistKernel<128>(cuda::std::span<const uint8_t>,
                                                 cuda::std::span<int>,
                                                 cuda::std::span<int>,
                                                 cuda::std::span<int>,
                                                 int,
                                                 cudaStream_t);

void launchArgMaxKernel(cuda::std::span<const int> values, int* outVal, int* outIdx, cudaStream_t stream) {
  lastArgMax<<<1, argMaxBlockSize, 0, stream>>>(values, outVal, outIdx);
  cudaCheckError(cudaGetLastError());
}

}  // namespace detail

}  // namespace nvMolKit