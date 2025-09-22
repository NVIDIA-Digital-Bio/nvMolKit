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
#include "similarity_kernels.h"
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

  bool                       useA = true;
  [[nodiscard]] cudaStream_t currentStream() const { return useA ? streamA.stream() : streamB.stream(); }
  AsyncDeviceVector<double>& currentBuffer() { return useA ? bufferA : bufferB; }
};
// Send compute A
// Copy A, send compute B
// Await copy A

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
  const size_t     freeBytesPerBuffer = freeBytes / 2;  // We need space for two buffers to rotate
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
  std::vector<double> res(nFps1 * nFps2);

  for (size_t startIdx = 0; startIdx < nFps1; startIdx += batchSizeA) {
    cudaStream_t               currentStream     = rotBuffers.currentStream();
    AsyncDeviceVector<double>& currentBuffer     = rotBuffers.currentBuffer();
    const size_t               currentBatchSizeA = std::min(batchSizeA, nFps1 - startIdx);
    launchCrossTanimotoSimilarity(bitsOneBuffer.subspan(startIdx * nElementsPerFp, currentBatchSizeA * nElementsPerFp),
                                  bitsTwoBuffer,
                                  nElementsPerFp,
                                  toSpan(currentBuffer),
                                  0,
                                  currentStream);
    currentBuffer.copyToHost(res, currentBatchSizeA * nFps2, startIdx * nFps2);
  }
  cudaStreamSynchronize(rotBuffers.streamA.stream());
  cudaStreamSynchronize(rotBuffers.streamB.stream());
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
  const size_t     freeBytesPerBuffer = freeBytes / 2;  // We need space for two buffers to rotate
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
  std::vector<double> res(nFps1 * nFps2);

  for (size_t startIdx = 0; startIdx < nFps1; startIdx += batchSizeA) {
    cudaStream_t               currentStream     = rotBuffers.currentStream();
    AsyncDeviceVector<double>& currentBuffer     = rotBuffers.currentBuffer();
    const size_t               currentBatchSizeA = std::min(batchSizeA, nFps1 - startIdx);
    launchCrossCosineSimilarity(bitsOneBuffer.subspan(startIdx * nElementsPerFp, currentBatchSizeA * nElementsPerFp),
                                bitsTwoBuffer,
                                nElementsPerFp,
                                toSpan(currentBuffer),
                                0,
                                currentStream);
    currentBuffer.copyToHost(res, currentBatchSizeA * nFps2, startIdx * nFps2);
  }
  cudaStreamSynchronize(rotBuffers.streamA.stream());
  cudaStreamSynchronize(rotBuffers.streamB.stream());
  return res;
}

}  // namespace nvMolKit
