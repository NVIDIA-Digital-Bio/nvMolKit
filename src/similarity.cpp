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

#include "similarity.h"

#include "device.h"
#include "device_vector.h"
#include "host_vector.h"
#include "similarity_kernels.h"
#include <omp.h>
namespace nvMolKit {

using internal::kBlockType;

namespace {
constexpr int kBitsPerByte = 8;
}  // namespace

// --------------------------------
// Tanimoto similarity wrapper functions
// --------------------------------

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bits,
                                                           int                                        fpSize) {
  const size_t nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t nFps           = bits.size() / nElementsPerFp;
  auto         similarities_d = AsyncDeviceVector<double>(nFps * nFps);
  launchCrossTanimotoSimilarity(bits, bits, nElementsPerFp, toSpan(similarities_d), 0);
  return similarities_d;
}

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                           const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                           int                                        fpSize) {
  const size_t nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t nFps1          = bitsOneBuffer.size() / nElementsPerFp;
  const size_t nFps2          = bitsTwoBuffer.size() / nElementsPerFp;
  auto         similarities_d = AsyncDeviceVector<double>(nFps1 * nFps2);
  launchCrossTanimotoSimilarity(bitsOneBuffer, bitsTwoBuffer, nElementsPerFp, toSpan(similarities_d), 0);
  return similarities_d;
}

struct SimilaritiesRotBuffers {
  // Declare streams before buffers so buffers are destroyed before streams.
  ScopedStream              streamA;
  ScopedStream              streamB;
  AsyncDeviceVector<double> bufferA;
  AsyncDeviceVector<double> bufferB;

  // 100MB pinned host transfer buffers, double-buffered per thread index
  PinnedHostVector<double> pinnedA0;
  PinnedHostVector<double> pinnedA1;
  PinnedHostVector<double> pinnedB0;
  PinnedHostVector<double> pinnedB1;

  // CUDA events per pinned buffer to signal when D2H completes
  ScopedCudaEvent eventA0;
  ScopedCudaEvent eventA1;
  ScopedCudaEvent eventB0;
  ScopedCudaEvent eventB1;

  [[nodiscard]] cudaStream_t streamForThread(int tid) { return (tid % 2 == 0) ? streamA.stream() : streamB.stream(); }
  AsyncDeviceVector<double>& bufferForThread(int tid) { return (tid % 2 == 0) ? bufferA : bufferB; }
  PinnedHostVector<double>&  pinnedForThreadBuffer(int tid, int bufIdx) {
    const bool isA = (tid % 2 == 0);
    if (isA) {
      return (bufIdx == 0) ? pinnedA0 : pinnedA1;
    } else {
      return (bufIdx == 0) ? pinnedB0 : pinnedB1;
    }
  }
  cudaEvent_t eventForThreadBuffer(int tid, int bufIdx) {
    const bool isA = (tid % 2 == 0);
    if (isA) {
      return (bufIdx == 0) ? eventA0.event() : eventA1.event();
    } else {
      return (bufIdx == 0) ? eventB0.event() : eventB1.event();
    }
  }
};

std::vector<double> crossTanimotoSimilarityCPUResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                     const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                     const int                                  fpSize,
                                                     const CrossSimilarityOptions&              options) {
  const size_t nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t nFps1          = bitsOneBuffer.size() / nElementsPerFp;
  const size_t nFps2          = bitsTwoBuffer.size() / nElementsPerFp;
  const size_t freeBytes      = options.maxDeviceMemoryBytes.value_or(getDeviceFreeMemory());
  if (freeBytes >= nFps1 * nFps2 * sizeof(double)) {
    auto                resGpu = crossTanimotoSimilarityGpuResult(bitsOneBuffer, bitsTwoBuffer, fpSize);
    std::vector<double> res(resGpu.size());
    resGpu.copyToHost(res);
    cudaDeviceSynchronize();
    return res;
  }
  const size_t     freeBytesPerBuffer = freeBytes / 2;  // We need space for two device buffers
  const size_t     maxDoublesInBuffer = (freeBytesPerBuffer / sizeof(double) * 9) / 10;  // Leave ~10% headroom
  constexpr size_t minBatchSizeA      = 32;
  const size_t     blockSizeIncrement = minBatchSizeA * nFps2;
  if (blockSizeIncrement > maxDoublesInBuffer) {
    throw std::runtime_error("Not enough memory to compute cross similarity");
  }
  const size_t batchSizeA = std::max(minBatchSizeA, (maxDoublesInBuffer / blockSizeIncrement) * minBatchSizeA);

  // Prepare per-thread resources
  SimilaritiesRotBuffers rotBuffers;
  rotBuffers.bufferA = AsyncDeviceVector<double>(batchSizeA * nFps2, rotBuffers.streamA.stream());
  rotBuffers.bufferB = AsyncDeviceVector<double>(batchSizeA * nFps2, rotBuffers.streamB.stream());

  // 100 MB pinned buffers per thread (double-buffered)
  constexpr size_t kPinnedBytes              = 100ull * 1024ull * 1024ull;
  constexpr size_t pinnedCapacityDoubles     = kPinnedBytes / sizeof(double);
  rotBuffers.pinnedA0                        = PinnedHostVector<double>(pinnedCapacityDoubles);
  rotBuffers.pinnedA1                        = PinnedHostVector<double>(pinnedCapacityDoubles);
  rotBuffers.pinnedB0                        = PinnedHostVector<double>(pinnedCapacityDoubles);
  rotBuffers.pinnedB1                        = PinnedHostVector<double>(pinnedCapacityDoubles);

  std::vector<double> res(nFps1 * nFps2);

  const size_t nBatches = (nFps1 + batchSizeA - 1) / batchSizeA;

  #pragma omp parallel for num_threads(2) schedule(dynamic)
  for (size_t batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
    const int                  tid               = omp_get_thread_num();
    cudaStream_t               currentStream     = rotBuffers.streamForThread(tid);
    AsyncDeviceVector<double>& currentBuffer     = rotBuffers.bufferForThread(tid);
    // Double-buffered pinned buffers for overlap
    PinnedHostVector<double>&  pinnedBuffer0     = rotBuffers.pinnedForThreadBuffer(tid, 0);
    PinnedHostVector<double>&  pinnedBuffer1     = rotBuffers.pinnedForThreadBuffer(tid, 1);

    const size_t startIdx           = batchIdx * batchSizeA;
    const size_t currentBatchSizeA  = std::min(batchSizeA, nFps1 - startIdx);

    // Launch compute for this batch
    launchCrossTanimotoSimilarity(bitsOneBuffer.subspan(startIdx * nElementsPerFp, currentBatchSizeA * nElementsPerFp),
                                  bitsTwoBuffer,
                                  nElementsPerFp,
                                  toSpan(currentBuffer),
                                  0,
                                  currentStream);

    // Double-buffered chunked D2H into pinned buffers with overlap of CPU memcpy
    const size_t batchElemsTotal = currentBatchSizeA * nFps2;
    int          bufIdx          = 0;  // 0 -> use pinnedBuffer0, 1 -> pinnedBuffer1
    bool         hasPrev         = false;
    size_t       prevOffset      = 0;
    size_t       prevSize        = 0;
    int          prevBufIdx      = 0;
    for (size_t chunkOffset = 0; chunkOffset < batchElemsTotal; chunkOffset += pinnedCapacityDoubles, bufIdx ^= 1) {
      const size_t chunkSize   = std::min(pinnedCapacityDoubles, batchElemsTotal - chunkOffset);
      auto&        pinnedBuf   = (bufIdx == 0) ? pinnedBuffer0 : pinnedBuffer1;
      auto&        prevPinned  = (bufIdx == 0) ? pinnedBuffer1 : pinnedBuffer0;
      cudaEvent_t  bufEvent    = rotBuffers.eventForThreadBuffer(tid, bufIdx);
      cudaEvent_t  prevEvent   = rotBuffers.eventForThreadBuffer(tid, prevBufIdx);

      // Enqueue D2H into the selected pinned buffer and record event
      const size_t deviceOffset = chunkOffset;
      currentBuffer.copyToHost(pinnedBuf.data(), chunkSize, 0, deviceOffset);
      cudaEventRecord(bufEvent, currentStream);

      // While this D2H is running, memcpy from the previous buffer (if not first iteration)
      if (hasPrev) {
        const size_t resOffset  = startIdx * nFps2 + prevOffset;
        cudaEventSynchronize(prevEvent);
        std::memcpy(res.data() + resOffset, prevPinned.data(), sizeof(double) * prevSize);
      }

      // Set current as previous for next iteration
      hasPrev    = true;
      prevOffset = chunkOffset;
      prevSize   = chunkSize;
      prevBufIdx = bufIdx;
    }

    // Copy the last in-flight chunk
    if (hasPrev) {
      auto&       lastPinned = (prevBufIdx == 0) ? pinnedBuffer0 : pinnedBuffer1;
      cudaEvent_t lastEvent  = rotBuffers.eventForThreadBuffer(tid, prevBufIdx);
      cudaEventSynchronize(lastEvent);
      const size_t resOffset = startIdx * nFps2 + prevOffset;
      std::memcpy(res.data() + resOffset, lastPinned.data(), sizeof(double) * prevSize);
    }

    // Ensure all D2H copies are complete before proceeding
    cudaStreamSynchronize(currentStream);
  }
  return res;
}

// --------------------------------
// Cosine similarity wrapper functions
// --------------------------------

AsyncDeviceVector<double> crossCosineSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bits, int fpSize) {
  const size_t              nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t              nFps           = bits.size() / nElementsPerFp;
  AsyncDeviceVector<double> similarities_d = AsyncDeviceVector<double>(nFps * nFps);
  launchCrossCosineSimilarity(bits, bits, nElementsPerFp, toSpan(similarities_d), 0);
  return similarities_d;
}

AsyncDeviceVector<double> crossCosineSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                         const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                         int                                        fpSize) {
  const size_t              nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t              nFps1          = bitsOneBuffer.size() / nElementsPerFp;
  const size_t              nFps2          = bitsTwoBuffer.size() / nElementsPerFp;
  AsyncDeviceVector<double> similarities_d = AsyncDeviceVector<double>(nFps1 * nFps2);
  launchCrossCosineSimilarity(bitsOneBuffer, bitsTwoBuffer, nElementsPerFp, toSpan(similarities_d), 0);
  return similarities_d;
}

std::vector<double> crossCosineSimilarityCPUResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                   const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                   const int                                  fpSize,
                                                   const CrossSimilarityOptions&              options) {
  const size_t nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t nFps1          = bitsOneBuffer.size() / nElementsPerFp;
  const size_t nFps2          = bitsTwoBuffer.size() / nElementsPerFp;
  const size_t freeBytes      = options.maxDeviceMemoryBytes.value_or(getDeviceFreeMemory());
  if (freeBytes >= nFps1 * nFps2 * sizeof(double)) {
    auto                resGpu = crossCosineSimilarityGpuResult(bitsOneBuffer, bitsTwoBuffer, fpSize);
    std::vector<double> res(resGpu.size());
    resGpu.copyToHost(res);
    cudaDeviceSynchronize();
    return res;
  }
  const size_t     freeBytesPerBuffer = freeBytes / 2;  // We need space for two device buffers (two threads)
  const size_t     maxDoublesInBuffer = (freeBytesPerBuffer / sizeof(double) * 9) / 10;  // Leave ~10% headroom
  constexpr size_t minBatchSizeA      = 32;
  const size_t     blockSizeIncrement = minBatchSizeA * nFps2;
  if (blockSizeIncrement > maxDoublesInBuffer) {
    throw std::runtime_error("Not enough memory to compute cross similarity");
  }
  const size_t batchSizeA = std::max(minBatchSizeA, (maxDoublesInBuffer / blockSizeIncrement) * minBatchSizeA);

  SimilaritiesRotBuffers rotBuffers;
  rotBuffers.bufferA = AsyncDeviceVector<double>(batchSizeA * nFps2, rotBuffers.streamA.stream());
  rotBuffers.bufferB = AsyncDeviceVector<double>(batchSizeA * nFps2, rotBuffers.streamB.stream());

  // 100 MB pinned buffers per thread (double-buffered)
  constexpr size_t kPinnedBytes           = 100ull * 1024ull * 1024ull;
  const size_t     pinnedCapacityDoubles  = kPinnedBytes / sizeof(double);
  rotBuffers.pinnedA0                     = PinnedHostVector<double>(pinnedCapacityDoubles);
  rotBuffers.pinnedA1                     = PinnedHostVector<double>(pinnedCapacityDoubles);
  rotBuffers.pinnedB0                     = PinnedHostVector<double>(pinnedCapacityDoubles);
  rotBuffers.pinnedB1                     = PinnedHostVector<double>(pinnedCapacityDoubles);

  std::vector<double> res(nFps1 * nFps2);

  const size_t nBatches = (nFps1 + batchSizeA - 1) / batchSizeA;

  #pragma omp parallel for num_threads(2) schedule(static)
  for (size_t batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
    const int                  tid               = omp_get_thread_num();
    cudaStream_t               currentStream     = rotBuffers.streamForThread(tid);
    AsyncDeviceVector<double>& currentBuffer     = rotBuffers.bufferForThread(tid);
    PinnedHostVector<double>&  pinnedBuffer0     = rotBuffers.pinnedForThreadBuffer(tid, 0);
    PinnedHostVector<double>&  pinnedBuffer1     = rotBuffers.pinnedForThreadBuffer(tid, 1);

    const size_t startIdx           = batchIdx * batchSizeA;
    const size_t currentBatchSizeA  = std::min(batchSizeA, nFps1 - startIdx);
    if (currentBatchSizeA == 0) continue;

    launchCrossCosineSimilarity(bitsOneBuffer.subspan(startIdx * nElementsPerFp, currentBatchSizeA * nElementsPerFp),
                                bitsTwoBuffer,
                                nElementsPerFp,
                                toSpan(currentBuffer),
                                0,
                                currentStream);

    const size_t batchElemsTotal = currentBatchSizeA * nFps2;
    int          bufIdx          = 0;  // 0 -> use pinnedBuffer0, 1 -> pinnedBuffer1
    bool         hasPrev         = false;
    size_t       prevOffset      = 0;
    size_t       prevSize        = 0;
    int          prevBufIdx      = 0;
    for (size_t chunkOffset = 0; chunkOffset < batchElemsTotal; chunkOffset += pinnedCapacityDoubles, bufIdx ^= 1) {
      const size_t chunkSize   = std::min(pinnedCapacityDoubles, batchElemsTotal - chunkOffset);
      auto&        pinnedBuf   = (bufIdx == 0) ? pinnedBuffer0 : pinnedBuffer1;
      auto&        prevPinned  = (bufIdx == 0) ? pinnedBuffer1 : pinnedBuffer0;
      cudaEvent_t  bufEvent    = rotBuffers.eventForThreadBuffer(tid, bufIdx);
      cudaEvent_t  prevEvent   = rotBuffers.eventForThreadBuffer(tid, prevBufIdx);
      const size_t deviceOffset = chunkOffset;
      currentBuffer.copyToHost(pinnedBuf.data(), chunkSize, 0, deviceOffset);
      cudaEventRecord(bufEvent, currentStream);
      if (hasPrev) {
        const size_t resOffset  = startIdx * nFps2 + prevOffset;
        cudaEventSynchronize(prevEvent);
        std::memcpy(res.data() + resOffset, prevPinned.data(), sizeof(double) * prevSize);
      }
      hasPrev    = true;
      prevOffset = chunkOffset;
      prevSize   = chunkSize;
      prevBufIdx = bufIdx;
    }
    if (hasPrev) {
      auto&       lastPinned = (prevBufIdx == 0) ? pinnedBuffer0 : pinnedBuffer1;
      cudaEvent_t lastEvent  = rotBuffers.eventForThreadBuffer(tid, prevBufIdx);
      cudaEventSynchronize(lastEvent);
      const size_t resOffset = startIdx * nFps2 + prevOffset;
      std::memcpy(res.data() + resOffset, lastPinned.data(), sizeof(double) * prevSize);
    }
  }
  return res;
}

}  // namespace nvMolKit
