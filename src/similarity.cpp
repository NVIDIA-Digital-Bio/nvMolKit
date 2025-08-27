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

#include <DataStructs/ExplicitBitVect.h>

#include <iostream>
#include <unordered_set>

#include "cuda_error_check.h"
#include "device.h"
#include "device_vector.h"
#include "similarity_kernels.h"
namespace nvMolKit {

using internal::kBlockType;

namespace {

//! Restrictions on fingerprint sizes.
//! Supported sizes are powers of 2 up to 2048. 32 bit and smaller types should be handled with thread-per-int, which
//! we have implemented but not hooked up. 4096 and larger will need moderate work to support.
const std::unordered_set<size_t> kSupportedFingerprintSizes = {64, 128, 256, 512, 1024, 2048};
constexpr int                    kBitsPerByte               = 8;

constexpr int    kMaxBitsWith32BitSubdivision = 1024;
constexpr size_t kNBitsInBoostBitSet          = kBitsPerByte * sizeof(kBlockType);

struct BulkFingerprintStreamHandle {
  ScopedStream                  stream;
  AsyncDeviceVector<kBlockType> bits;
  std::vector<kBlockType>       bitsHost;
  AsyncDeviceVector<double>     resultsFragment;
  cudaEvent_t                   checkPreviousEvent = nullptr;

  BulkFingerprintStreamHandle()                                                  = default;
  BulkFingerprintStreamHandle(const BulkFingerprintStreamHandle&)                = delete;
  BulkFingerprintStreamHandle& operator=(const BulkFingerprintStreamHandle&)     = delete;
  BulkFingerprintStreamHandle(BulkFingerprintStreamHandle&&) noexcept            = default;
  BulkFingerprintStreamHandle& operator=(BulkFingerprintStreamHandle&&) noexcept = default;
  ~BulkFingerprintStreamHandle() noexcept {
    if (checkPreviousEvent != nullptr) {
      cudaCheckErrorNoThrow(cudaEventDestroy(checkPreviousEvent));
    }
  }
};

// --------------------------------
// Core function templates
// --------------------------------

enum class SimilarityType {
  Tanimoto = 0,
  Cosine
};

template <SimilarityType similarityType>
std::vector<float> bulkSimilarity(std::uint32_t bitsOne, const std::vector<std::uint32_t>& bitsTwo) {
  if (bitsTwo.empty()) {
    return {};
  }

  const AsyncDevicePtr<std::uint32_t> bitsOne_d(bitsOne);
  AsyncDeviceVector<std::uint32_t>    bitsTwo_d(bitsTwo.size());
  bitsTwo_d.copyFromHost(bitsTwo);
  AsyncDeviceVector<float> similarities_d(bitsTwo.size());

  if constexpr (similarityType == SimilarityType::Tanimoto) {
    launchBulkTanimotoSimilarity(bitsOne_d, bitsTwo_d, similarities_d);
  } else if constexpr (similarityType == SimilarityType::Cosine) {
    launchBulkCosineSimilarity(bitsOne_d, bitsTwo_d, similarities_d);
  }
  std::vector<float> similarities(similarities_d.size());
  cudaCheckError(cudaDeviceSynchronize());
  similarities_d.copyToHost(similarities, similarities.size());
  return similarities;
}

template <SimilarityType similarityType>
std::vector<double> bulkSimilarity(const ExplicitBitVect&                     bitsOne,
                                   const std::vector<const ExplicitBitVect*>& bitsTwo,
                                   const BulkFingerprintOptions&              options) {
  if (bitsTwo.empty()) {
    return {};
  }
  const size_t numBitsPerMolecule = bitsOne.getNumBits();
  if (kSupportedFingerprintSizes.find(numBitsPerMolecule) == kSupportedFingerprintSizes.end()) {
    throw std::runtime_error("Unsupported fingerprint size: " + std::to_string(numBitsPerMolecule) +
                             ", supported sizes are powers of 2 up to 2048");
  }

  std::vector<kBlockType> bitsOneHost(numBitsPerMolecule / kNBitsInBoostBitSet);
  boost::to_block_range(*bitsOne.dp_bits, bitsOneHost.begin());

  // kBlockType* bitsTwoHost = nullptr;
  //  const size_t bitsTwoHostSize = bitsTwo.size() * (numBitsPerMolecule / kNBitsInBoostBitSet);
  // cudaCheckError(cudaMallocHost(&bitsTwoHost, bitsTwoHostSize  * sizeof(kBlockType)));
  AsyncDeviceVector<kBlockType> bitsOneDevice(bitsOneHost.size());
  bitsOneDevice.copyFromHost(bitsOneHost);

  std::vector<double> similarities(bitsTwo.size());

  const size_t dispatchBatchSize = options.batchSize.has_value() ? *options.batchSize : bitsTwo.size();

  // Make sure streams don't mess with thread cache.
  constexpr int                                                   kMaxCacheSize = 128;
  alignas(kMaxCacheSize) std::vector<BulkFingerprintStreamHandle> streamHandles(
    std::min(options.maxStreams, (bitsTwo.size() / dispatchBatchSize) + 1));

  const size_t numBatches = (bitsTwo.size() + dispatchBatchSize - 1) / dispatchBatchSize;

#pragma omp parallel for schedule(static, 1), default(none), \
  shared(streamHandles, numBatches, dispatchBatchSize, bitsTwo, similarities, bitsOneDevice, numBitsPerMolecule)
  for (size_t batchId = 0; batchId < numBatches; ++batchId) {
    const size_t streamID     = batchId % streamHandles.size();
    auto&        streamHandle = streamHandles[streamID];

    // Compute batch-relative indices
    const size_t startIdx                      = batchId * dispatchBatchSize;
    const size_t endIdx                        = std::min(bitsTwo.size(), startIdx + dispatchBatchSize);
    const size_t truncatedDispatchBatchSize    = std::min(dispatchBatchSize, bitsTwo.size() - startIdx);
    const size_t dispatchBatchSizeInBoostUnits = truncatedDispatchBatchSize * numBitsPerMolecule / kNBitsInBoostBitSet;

    // Relying on the fact that the small batch will be at the end if it exists, so we'll
    // never resize to something too small.
    if (streamHandle.bits.size() == 0) {
      streamHandle.bits = AsyncDeviceVector<kBlockType>(dispatchBatchSizeInBoostUnits, streamHandle.stream.stream());
      streamHandle.resultsFragment = AsyncDeviceVector<double>(dispatchBatchSize, streamHandle.stream.stream());
      streamHandle.bitsHost.resize(dispatchBatchSizeInBoostUnits);
      cudaCheckError(cudaEventCreateWithFlags(&streamHandle.checkPreviousEvent, cudaEventDisableTiming));
    } else {
    }

    // If this is the second or more time around for this stream, wait for the previous one to complete before
    // overriding the first buffer
    // TODO we might be able to relax this with a second event tracking just the bit copy.
    if (batchId > streamID) {
      cudaCheckError(cudaStreamWaitEvent(streamHandle.stream.stream(), streamHandles[streamID].checkPreviousEvent, 0));
    }

    // Copy just the batch size in from the boost bit vectors.
    // TODO: look into multithreading.
    for (size_t i = startIdx; i < endIdx; i++) {
      const size_t hostRelativeI = i - startIdx;
      boost::to_block_range(
        *bitsTwo[i]->dp_bits,
        streamHandle.bitsHost.begin() + hostRelativeI * numBitsPerMolecule / kNBitsInBoostBitSet);  // NOLINT
    }

    streamHandle.bits.copyFromHost(streamHandle.bitsHost,
                                   /*size=*/dispatchBatchSizeInBoostUnits,
                                   /*host start=*/0,
                                   /*device start=*/0);
    if (batchId == 0) {
      // wait for bitsOne to be ready.
      cudaCheckErrorNoThrow(cudaStreamSynchronize(nullptr));
    }
    if constexpr (similarityType == SimilarityType::Tanimoto) {
      if (numBitsPerMolecule <= kMaxBitsWith32BitSubdivision) {
        launchBulkTanimotoSimilarity<std::uint32_t>(bitsOneDevice,
                                                    streamHandle.bits,
                                                    numBitsPerMolecule / (kBitsPerByte * sizeof(std::uint32_t)),
                                                    truncatedDispatchBatchSize,
                                                    streamHandle.resultsFragment,
                                                    0,
                                                    streamHandle.stream.stream());
      } else {
        launchBulkTanimotoSimilarity<std::uint64_t>(bitsOneDevice,
                                                    streamHandle.bits,
                                                    numBitsPerMolecule / (kBitsPerByte * sizeof(std::uint64_t)),
                                                    truncatedDispatchBatchSize,
                                                    streamHandle.resultsFragment,
                                                    0,
                                                    streamHandle.stream.stream());
      }
    } else if constexpr (similarityType == SimilarityType::Cosine) {
      if (numBitsPerMolecule <= kMaxBitsWith32BitSubdivision) {
        launchBulkCosineSimilarity<std::uint32_t>(bitsOneDevice,
                                                  streamHandle.bits,
                                                  numBitsPerMolecule / (kBitsPerByte * sizeof(std::uint32_t)),
                                                  truncatedDispatchBatchSize,
                                                  streamHandle.resultsFragment,
                                                  0,
                                                  streamHandle.stream.stream());
      } else {
        launchBulkCosineSimilarity<std::uint64_t>(bitsOneDevice,
                                                  streamHandle.bits,
                                                  numBitsPerMolecule / (kBitsPerByte * sizeof(std::uint64_t)),
                                                  truncatedDispatchBatchSize,
                                                  streamHandle.resultsFragment,
                                                  0,
                                                  streamHandle.stream.stream());
      }
    }
    cudaCheckError(cudaGetLastError());
    streamHandle.resultsFragment.copyToHost(similarities,
                                            /*size=*/truncatedDispatchBatchSize,
                                            /*host start=*/startIdx);
    cudaCheckError(cudaEventRecord(streamHandle.checkPreviousEvent, streamHandle.stream.stream()));
  }
  cudaCheckError(cudaDeviceSynchronize());
  return similarities;
}

template <SimilarityType similarityType>
std::vector<double> crossSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsOne,
                                    const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo,
                                    const CrossSimilarityOptions&                        options) {
  // TODO dynamic
  if (bitsOne.empty() || bitsTwo.empty()) {
    return {};
  }

  const size_t numOne = bitsOne.size();
  const size_t numTwo = bitsTwo.size();

  std::vector<double> similarities(bitsOne.size() * bitsTwo.size());

  const size_t numBitsPerMolecule = bitsOne[0]->getNumBits();
  if (kSupportedFingerprintSizes.find(numBitsPerMolecule) == kSupportedFingerprintSizes.end()) {
    throw std::runtime_error("Unsupported fingerprint size: " + std::to_string(numBitsPerMolecule) +
                             ", supported sizes are powers of 2 up to 2048");
  }

  std::vector<kBlockType> bitsOneHost(bitsOne.size() * numBitsPerMolecule / kNBitsInBoostBitSet);
  for (size_t i = 0; i < bitsOne.size(); ++i) {
    boost::to_block_range(*bitsOne[i]->dp_bits,
                          bitsOneHost.begin() + i * numBitsPerMolecule / kNBitsInBoostBitSet);  // NOLINT
  }

  std::vector<kBlockType> bitsTwoHost(bitsTwo.size() * numBitsPerMolecule / kNBitsInBoostBitSet);
  for (size_t i = 0; i < bitsTwo.size(); ++i) {
    boost::to_block_range(*bitsTwo[i]->dp_bits,
                          bitsTwoHost.begin() + i * numBitsPerMolecule / kNBitsInBoostBitSet);  // NOLINT
  }
  AsyncDeviceVector<kBlockType> bitsTwoDevice(bitsTwoHost.size());
  bitsTwoDevice.copyFromHost(bitsTwoHost);

  const size_t doublesInMemoryAvailable = options.maxDeviceMemoryBytes.has_value() ?
                                          *options.maxDeviceMemoryBytes / sizeof(double) :
                                          getDeviceFreeMemory() / sizeof(double);

  AsyncDeviceVector<double> similarities_d;
  // TODO n division!
  size_t                    numOne_local      = numOne;
  size_t                    doublesToAllocate = roundUpToNearestPowerOfTwo(numOne * bitsTwo.size());
  while (doublesToAllocate >= doublesInMemoryAvailable) {
    numOne_local /= 2;
    if (numOne_local == 0) {
      numOne_local = 1;
      if (numOne_local * numTwo >= doublesInMemoryAvailable) {
        throw std::runtime_error("Out of memory error\n");
      }
      doublesToAllocate = numOne_local * numTwo;
      /// TODO divide n direction too!
      break;
    }
    doublesToAllocate = numOne_local * numTwo;
  }

  similarities_d                         = AsyncDeviceVector<double>(doublesToAllocate);
  const size_t totalNumThreadsToDispatch = (numOne + numOne_local - 1) / numOne_local;

  const size_t                  sizeKBlockTypePerMolecule = numBitsPerMolecule / (sizeof(kBlockType) * 8);
  AsyncDeviceVector<kBlockType> bitsOneDevice(numOne_local * sizeKBlockTypePerMolecule);

  for (size_t offset = 0; offset < totalNumThreadsToDispatch; offset += 1) {
    const size_t cappedNumElementsToCopyA = std::min(numOne_local, numOne - offset * numOne_local);

    bitsOneDevice.copyFromHost(&bitsOneHost[numOne_local * offset * numBitsPerMolecule / kNBitsInBoostBitSet],
                               cappedNumElementsToCopyA * sizeKBlockTypePerMolecule);

    if constexpr (similarityType == SimilarityType::Tanimoto) {
      launchCrossTanimotoSimilarity<std::uint32_t>(bitsOneDevice,
                                                   bitsTwoDevice,
                                                   numBitsPerMolecule / (kBitsPerByte * sizeof(std::uint32_t)),
                                                   similarities_d,
                                                   offset);
    } else if constexpr (similarityType == SimilarityType::Cosine) {
      launchCrossCosineSimilarity<std::uint32_t>(bitsOneDevice,
                                                 bitsTwoDevice,
                                                 numBitsPerMolecule / (kBitsPerByte * sizeof(std::uint32_t)),
                                                 similarities_d,
                                                 offset);
    }
    cudaCheckError(cudaGetLastError());
    const size_t HostOffset              = offset * numOne_local * numTwo;
    const size_t cappedNumElementsToCopy = std::min(doublesToAllocate, similarities.size() - HostOffset);
    similarities_d.copyToHost(similarities, cappedNumElementsToCopy, HostOffset);
    cudaDeviceSynchronize();
  }

  cudaCheckError(cudaDeviceSynchronize());
  return similarities;
}

template <SimilarityType similarityType>
std::vector<double> crossSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bits,
                                    const CrossSimilarityOptions&                        options) {
  // TODO dynamic
  if (bits.empty()) {
    return {};
  }
  std::vector<double> similarities(bits.size() * bits.size());

  const size_t numBitsPerMolecule = bits[0]->getNumBits();
  if (kSupportedFingerprintSizes.find(numBitsPerMolecule) == kSupportedFingerprintSizes.end()) {
    throw std::runtime_error("Unsupported fingerprint size: " + std::to_string(numBitsPerMolecule) +
                             ", supported sizes are powers of 2 up to 2048");
  }

  std::vector<kBlockType> bitsHost(bits.size() * numBitsPerMolecule / kNBitsInBoostBitSet);

  for (size_t i = 0; i < bits.size(); ++i) {
    boost::to_block_range(*bits[i]->dp_bits,
                          bitsHost.begin() + i * numBitsPerMolecule / kNBitsInBoostBitSet);  // NOLINT
  }

  AsyncDeviceVector<kBlockType> bitsTwoDevice(bitsHost.size());
  bitsTwoDevice.copyFromHost(bitsHost);

  const size_t doublesInMemoryAvailable = options.maxDeviceMemoryBytes.has_value() ?
                                          *options.maxDeviceMemoryBytes / sizeof(double) :
                                          getDeviceFreeMemory() / sizeof(double);

  AsyncDeviceVector<double> similarities_d;

  const size_t numOne            = bits.size();
  size_t       numOne_local      = numOne;
  size_t       doublesToAllocate = roundUpToNearestPowerOfTwo(numOne_local * numOne);
  while (doublesToAllocate >= doublesInMemoryAvailable) {
    numOne_local /= 2;
    if (numOne_local == 0) {
      numOne_local = 1;
      if (numOne_local * numOne >= doublesInMemoryAvailable) {
        throw std::runtime_error("Out of memory error\n");
      }
      doublesToAllocate = numOne_local * numOne;
      /// TODO divide n direction too!
      break;
    }
    doublesToAllocate = numOne_local * numOne;
  }

  similarities_d                         = AsyncDeviceVector<double>(doublesToAllocate);
  const size_t totalNumThreadsToDispatch = (numOne + numOne_local - 1) / numOne_local;

  const size_t sizeKBlockTypePerMolecule = numBitsPerMolecule / (sizeof(kBlockType) * 8);

  AsyncDeviceVector<kBlockType> bitsOneDevice(numOne_local * sizeKBlockTypePerMolecule);

  for (size_t offset = 0; offset < totalNumThreadsToDispatch; offset += 1) {
    const size_t cappedNumElementsToCopyA = std::min(numOne_local, numOne - offset * numOne_local);

    bitsOneDevice.copyFromHost(&bitsHost[numOne_local * offset * numBitsPerMolecule / kNBitsInBoostBitSet],
                               cappedNumElementsToCopyA * sizeKBlockTypePerMolecule);

    if constexpr (similarityType == SimilarityType::Tanimoto) {
      launchCrossTanimotoSimilarity<std::uint32_t>(bitsOneDevice,
                                                   bitsTwoDevice,
                                                   numBitsPerMolecule / (kBitsPerByte * sizeof(std::uint32_t)),
                                                   similarities_d,
                                                   offset);
    } else if constexpr (similarityType == SimilarityType::Cosine) {
      launchCrossCosineSimilarity<std::uint32_t>(bitsOneDevice,
                                                 bitsTwoDevice,
                                                 numBitsPerMolecule / (kBitsPerByte * sizeof(std::uint32_t)),
                                                 similarities_d,
                                                 offset);
    }
    cudaCheckError(cudaGetLastError());
    const size_t HostOffset              = offset * numOne_local * numOne;
    const size_t cappedNumElementsToCopy = std::min(doublesToAllocate, similarities.size() - HostOffset);
    similarities_d.copyToHost(similarities, cappedNumElementsToCopy, HostOffset);
    cudaDeviceSynchronize();
  }

  cudaCheckError(cudaDeviceSynchronize());
  return similarities;
}

}  // namespace

// --------------------------------
// Tanimoto similarity wrapper functions
// --------------------------------

std::vector<float> bulkTanimotoSimilarity(std::uint32_t bitsOne, const std::vector<std::uint32_t>& bitsTwo) {
  return bulkSimilarity<SimilarityType::Tanimoto>(bitsOne, bitsTwo);
}

std::vector<double> bulkTanimotoSimilarity(const ExplicitBitVect&                     bitsOne,
                                           const std::vector<const ExplicitBitVect*>& bitsTwo,
                                           const BulkFingerprintOptions&              options) {
  return bulkSimilarity<SimilarityType::Tanimoto>(bitsOne, bitsTwo, options);
}

std::vector<double> bulkTanimotoSimilarity(const ExplicitBitVect&                               bitsOne,
                                           const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo,
                                           const BulkFingerprintOptions&                        options) {
  std::vector<const ExplicitBitVect*> bitsTwoRaw;
  bitsTwoRaw.reserve(bitsTwo.size());
  for (const auto& bit : bitsTwo) {
    bitsTwoRaw.push_back(bit.get());
  }
  return bulkTanimotoSimilarity(bitsOne, bitsTwoRaw, options);
}
std::vector<double> bulkTanimotoSimilarity(const ExplicitBitVect&               bitsOne,
                                           const std::vector<ExplicitBitVect*>& bitsTwo,
                                           const BulkFingerprintOptions&        options) {
  std::vector<const ExplicitBitVect*> bitsTwoRaw;
  bitsTwoRaw.reserve(bitsTwo.size());
  for (const auto& bit : bitsTwo) {
    bitsTwoRaw.push_back(bit);
  }
  return bulkTanimotoSimilarity(bitsOne, bitsTwoRaw, options);
}

std::vector<double> crossTanimotoSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsOne,
                                            const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo,
                                            const CrossSimilarityOptions&                        options) {
  return crossSimilarity<SimilarityType::Tanimoto>(bitsOne, bitsTwo, options);
}

std::vector<double> crossTanimotoSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bits,
                                            const CrossSimilarityOptions&                        options) {
  return crossSimilarity<SimilarityType::Tanimoto>(bits, options);
}

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bits,
                                                           int                                        fpSize) {
  const size_t              nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t              nFps           = bits.size() / nElementsPerFp;
  AsyncDeviceVector<double> similarities_d = AsyncDeviceVector<double>(nFps * nFps);
  launchCrossTanimotoSimilarity(bits, bits, nElementsPerFp, toSpan(similarities_d), 0);
  return similarities_d;
}

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                           const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                           int                                        fpSize) {
  const size_t              nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t              nFps1          = bitsOneBuffer.size() / nElementsPerFp;
  const size_t              nFps2          = bitsTwoBuffer.size() / nElementsPerFp;
  AsyncDeviceVector<double> similarities_d = AsyncDeviceVector<double>(nFps1 * nFps2);
  launchCrossTanimotoSimilarity(bitsOneBuffer, bitsTwoBuffer, nElementsPerFp, toSpan(similarities_d), 0);
  return similarities_d;
}

template <typename blockType>
AsyncDeviceVector<double> bulkTanimotoSimilarityGpuResult(const cuda::std::span<const blockType> bitsOneBuffer,
                                                          const cuda::std::span<const blockType> bitsTwoBuffer,
                                                          int                                    fpSize) {
  const int blocksPerElement = fpSize / (kBitsPerByte * sizeof(blockType));
  const int numElements      = bitsTwoBuffer.size() / blocksPerElement;

  AsyncDeviceVector<double> similarities_d(numElements);
  launchBulkTanimotoSimilarity(bitsOneBuffer, bitsTwoBuffer, blocksPerElement, similarities_d, nullptr);
  cudaDeviceSynchronize();
  return similarities_d;
}

template <typename blockType>
std::vector<double> bulkTanimotoSimilarity(const cuda::std::span<const blockType> bitsOneBuffer,
                                           const cuda::std::span<const blockType> bitsTwoBuffer,
                                           const int                              fpSize) {
  auto                similarities_d = bulkTanimotoSimilarityGpuResult(bitsOneBuffer, bitsTwoBuffer, fpSize);
  std::vector<double> similarities(similarities_d.size());
  cudaDeviceSynchronize();
  similarities_d.copyToHost(similarities);
  return similarities;
}

template std::vector<double> bulkTanimotoSimilarity<std::uint32_t>(
  const cuda::std::span<const std::uint32_t> bitsOneBuffer,
  const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
  const int                                  fpSize);
template std::vector<double> bulkTanimotoSimilarity<std::uint64_t>(
  const cuda::std::span<const std::uint64_t> bitsOneBuffer,
  const cuda::std::span<const std::uint64_t> bitsTwoBuffer,
  const int                                  fpSize);

template AsyncDeviceVector<double> bulkTanimotoSimilarityGpuResult<std::uint32_t>(
  const cuda::std::span<const std::uint32_t> bitsOneBuffer,
  const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
  int                                        fpSize);
template AsyncDeviceVector<double> bulkTanimotoSimilarityGpuResult<std::uint64_t>(
  const cuda::std::span<const std::uint64_t> bitsOneBuffer,
  const cuda::std::span<const std::uint64_t> bitsTwoBuffer,
  int                                        fpSize);

// --------------------------------
// Cosine similarity wrapper functions
// --------------------------------

std::vector<float> bulkCosineSimilarity(std::uint32_t bitsOne, const std::vector<std::uint32_t>& bitsTwo) {
  return bulkSimilarity<SimilarityType::Cosine>(bitsOne, bitsTwo);
}

std::vector<double> bulkCosineSimilarity(const ExplicitBitVect&                     bitsOne,
                                         const std::vector<const ExplicitBitVect*>& bitsTwo,
                                         const BulkFingerprintOptions&              options) {
  return bulkSimilarity<SimilarityType::Cosine>(bitsOne, bitsTwo, options);
}

std::vector<double> bulkCosineSimilarity(const ExplicitBitVect&                               bitsOne,
                                         const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo,
                                         const BulkFingerprintOptions&                        options) {
  std::vector<const ExplicitBitVect*> bitsTwoRaw;
  bitsTwoRaw.reserve(bitsTwo.size());
  for (const auto& bit : bitsTwo) {
    bitsTwoRaw.push_back(bit.get());
  }
  return bulkCosineSimilarity(bitsOne, bitsTwoRaw, options);
}
std::vector<double> bulkCosineSimilarity(const ExplicitBitVect&               bitsOne,
                                         const std::vector<ExplicitBitVect*>& bitsTwo,
                                         const BulkFingerprintOptions&        options) {
  std::vector<const ExplicitBitVect*> bitsTwoRaw;
  bitsTwoRaw.reserve(bitsTwo.size());
  for (const auto& bit : bitsTwo) {
    bitsTwoRaw.push_back(bit);
  }
  return bulkCosineSimilarity(bitsOne, bitsTwoRaw, options);
}

std::vector<double> crossCosineSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsOne,
                                          const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo,
                                          const CrossSimilarityOptions&                        options) {
  return crossSimilarity<SimilarityType::Cosine>(bitsOne, bitsTwo, options);
}

std::vector<double> crossCosineSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bits,
                                          const CrossSimilarityOptions&                        options) {
  return crossSimilarity<SimilarityType::Cosine>(bits, options);
}

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

template <typename blockType>
AsyncDeviceVector<double> bulkCosineSimilarityGpuResult(const cuda::std::span<const blockType> bitsOneBuffer,
                                                        const cuda::std::span<const blockType> bitsTwoBuffer,
                                                        int                                    fpSize) {
  const int blocksPerElement = fpSize / (kBitsPerByte * sizeof(blockType));
  const int numElements      = bitsTwoBuffer.size() / blocksPerElement;

  AsyncDeviceVector<double> similarities_d(numElements);
  launchBulkCosineSimilarity(bitsOneBuffer, bitsTwoBuffer, blocksPerElement, similarities_d, nullptr);
  cudaDeviceSynchronize();
  return similarities_d;
}

template <typename blockType>
std::vector<double> bulkCosineSimilarity(const cuda::std::span<const blockType> bitsOneBuffer,
                                         const cuda::std::span<const blockType> bitsTwoBuffer,
                                         const int                              fpSize) {
  auto                similarities_d = bulkCosineSimilarityGpuResult(bitsOneBuffer, bitsTwoBuffer, fpSize);
  std::vector<double> similarities(similarities_d.size());
  cudaDeviceSynchronize();
  similarities_d.copyToHost(similarities);
  return similarities;
}

template std::vector<double> bulkCosineSimilarity<std::uint32_t>(
  const cuda::std::span<const std::uint32_t> bitsOneBuffer,
  const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
  const int                                  fpSize);
template std::vector<double> bulkCosineSimilarity<std::uint64_t>(
  const cuda::std::span<const std::uint64_t> bitsOneBuffer,
  const cuda::std::span<const std::uint64_t> bitsTwoBuffer,
  const int                                  fpSize);

template AsyncDeviceVector<double> bulkCosineSimilarityGpuResult<std::uint32_t>(
  const cuda::std::span<const std::uint32_t> bitsOneBuffer,
  const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
  int                                        fpSize);
template AsyncDeviceVector<double> bulkCosineSimilarityGpuResult<std::uint64_t>(
  const cuda::std::span<const std::uint64_t> bitsOneBuffer,
  const cuda::std::span<const std::uint64_t> bitsTwoBuffer,
  int                                        fpSize);

}  // namespace nvMolKit
