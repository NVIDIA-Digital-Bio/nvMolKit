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

#ifndef NVMOLKIT_SIMILARITY_H
#define NVMOLKIT_SIMILARITY_H

#include <DataStructs/ExplicitBitVect.h>

#include <cstdint>
#include <cuda/std/span>
#include <optional>
#include <vector>

#include "device_vector.h"

namespace nvMolKit {

//! Compute options for cross-similarity computation
struct CrossSimilarityOptions {
  //! Overrides autodetect of maximum memory on device for chunking compute.
  std::optional<std::size_t> maxDeviceMemoryBytes = std::nullopt;
};

//! Options for bulk fingerprinting
struct BulkFingerprintOptions {
  //! Batch size for bulk fingerprinting. If nullopt, no batching is done.
  std::optional<std::size_t> batchSize  = 8192;
  //! Max number of concurrent streams
  size_t                     maxStreams = 1024;
};

// --------------------------------
// Tanimoto similarity wrapper functions
// --------------------------------

//! Toy example of Tanimoto similarity between two 32 bit fingerprints. Validation of bit math.
std::vector<float> bulkTanimotoSimilarity(std::uint32_t bitsOne, const std::vector<std::uint32_t>& bitsTwo);

//! Tanimoto similarity between a bit vector and a list of bit vectors
//! \param bitsOne The first bit vector
//! \param bitsTwo The list of bit vectors
//! \param options Options for how to compute the similarities
//! \return A vector of similarities between the first bit vector and each of the second bit vectors
std::vector<double> bulkTanimotoSimilarity(const ExplicitBitVect&                               bitsOne,
                                           const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo,
                                           const BulkFingerprintOptions& options = BulkFingerprintOptions());

//! Tanimoto similarity between a bit vector and a list of bit vectors
//! \param bitsOne The first bit vector
//! \param bitsTwo The list of bit vectors
//! \param options Options for how to compute the similarities
//! \return A vector of similarities between the first bit vector and each of the second bit vectors
std::vector<double> bulkTanimotoSimilarity(const ExplicitBitVect&                     bitsOne,
                                           const std::vector<const ExplicitBitVect*>& bitsTwo,
                                           const BulkFingerprintOptions& options = BulkFingerprintOptions());

std::vector<double> bulkTanimotoSimilarity(const ExplicitBitVect&               bitsOne,
                                           const std::vector<ExplicitBitVect*>& bitsTwo,
                                           const BulkFingerprintOptions&        options = BulkFingerprintOptions());

template <typename blockType>
std::vector<double> bulkTanimotoSimilarity(const cuda::std::span<const blockType> bitsOneBuffer,
                                           const cuda::std::span<const blockType> bitsTwoBuffer,
                                           int                                    fpSize);
template <typename blockType>
AsyncDeviceVector<double> bulkTanimotoSimilarityGpuResult(const cuda::std::span<const blockType> bitsOneBuffer,
                                                          const cuda::std::span<const blockType> bitsTwoBuffer,
                                                          int                                    fpSize);

//! Tanimoto similarity between every element in a list of bit vectors
//! \param bits The list of bit vectors
//! \param options Options for how to compute the similarities
//! \return A vector of similarities between each pair of bit vectors. results[i * n + j] is the similarity between
//! bits[i] and bits[j]
std::vector<double> crossTanimotoSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bits,
                                            const CrossSimilarityOptions& options = CrossSimilarityOptions());

//! Tanimoto similarity between every element combination of two lists of bit vectors
//! \param bitsOne The first list of bit vectors
//! \param bitsTwo The second list of bit vectors
//! \param options Options for how to compute the similarities
//! \return A vector of similarities between each pair of bit vectors. results[i * n + j] is the similarity between
//! bits[i] and bits[j]
std::vector<double> crossTanimotoSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsOne,
                                            const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo,
                                            const CrossSimilarityOptions& options = CrossSimilarityOptions());

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bits, int fpSize);

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                           const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                           int                                        fpSize);

// --------------------------------
// Cosine similarity wrapper functions
// --------------------------------

//! Toy example of Cosine similarity between two 32 bit fingerprints. Validation of bit math.
std::vector<float> bulkCosineSimilarity(std::uint32_t bitsOne, const std::vector<std::uint32_t>& bitsTwo);

//! Cosine similarity between a bit vector and a list of bit vectors
//! \param bitsOne The first bit vector
//! \param bitsTwo The list of bit vectors
//! \param options Options for how to compute the similarities
//! \return A vector of similarities between the first bit vector and each of the second bit vectors
std::vector<double> bulkCosineSimilarity(const ExplicitBitVect&                               bitsOne,
                                         const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo,
                                         const BulkFingerprintOptions& options = BulkFingerprintOptions());

//! Cosine similarity between a bit vector and a list of bit vectors
//! \param bitsOne The first bit vector
//! \param bitsTwo The list of bit vectors
//! \param options Options for how to compute the similarities
//! \return A vector of similarities between the first bit vector and each of the second bit vectors
std::vector<double> bulkCosineSimilarity(const ExplicitBitVect&                     bitsOne,
                                         const std::vector<const ExplicitBitVect*>& bitsTwo,
                                         const BulkFingerprintOptions&              options = BulkFingerprintOptions());

std::vector<double> bulkCosineSimilarity(const ExplicitBitVect&               bitsOne,
                                         const std::vector<ExplicitBitVect*>& bitsTwo,
                                         const BulkFingerprintOptions&        options = BulkFingerprintOptions());

template <typename blockType>
std::vector<double> bulkCosineSimilarity(const cuda::std::span<const blockType> bitsOneBuffer,
                                         const cuda::std::span<const blockType> bitsTwoBuffer,
                                         int                                    fpSize);
template <typename blockType>
AsyncDeviceVector<double> bulkCosineSimilarityGpuResult(const cuda::std::span<const blockType> bitsOneBuffer,
                                                        const cuda::std::span<const blockType> bitsTwoBuffer,
                                                        int                                    fpSize);

//! Cosine similarity between every element in a list of bit vectors
//! \param bits The list of bit vectors
//! \param options Options for how to compute the similarities
//! \return A vector of similarities between each pair of bit vectors. results[i * n + j] is the similarity between
//! bits[i] and bits[j]
std::vector<double> crossCosineSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bits,
                                          const CrossSimilarityOptions& options = CrossSimilarityOptions());

//! Cosine similarity between every element combination of two lists of bit vectors
//! \param bitsOne The first list of bit vectors
//! \param bitsTwo The second list of bit vectors
//! \param options Options for how to compute the similarities
//! \return A vector of similarities between each pair of bit vectors. results[i * n + j] is the similarity between
//! bits[i] and bits[j]
std::vector<double> crossCosineSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsOne,
                                          const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo,
                                          const CrossSimilarityOptions& options = CrossSimilarityOptions());

AsyncDeviceVector<double> crossCosineSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bits, int fpSize);

AsyncDeviceVector<double> crossCosineSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                         const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                         int                                        fpSize);
}  // namespace nvMolKit

#endif  // NVMOLKIT_SIMILARITY_H
