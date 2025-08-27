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

#ifndef SIMILARITY_KERNELS_H
#define SIMILARITY_KERNELS_H

#include <boost/dynamic_bitset.hpp>
#include <cstdint>
#include <vector>

#include "device_vector.h"

namespace nvMolKit {
namespace internal {

//! Threads per pair of fps
constexpr int kNumThreadsPerFingerPrintTemporaryFixed = 32;
using kBlockType                                      = boost::dynamic_bitset<>::block_type;

}  // namespace internal

// --------------------------------
// Tanimoto similarity launch functions
// --------------------------------

//! Launches a kernel to compute the Tanimoto similarity between a 32bit fingerprint and a list of 32bit fingerprints.
//! \param bitsOne The first fingerprint
//! \param bitsTwo The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
void launchBulkTanimotoSimilarity(const AsyncDevicePtr<std::uint32_t>&    bitsOne,
                                  const AsyncDeviceVector<std::uint32_t>& bitsTwo,
                                  AsyncDeviceVector<float>&               results);
//! Launches a kernel to compute the Tanimoto similarity between an arbitrary fingerprint and a list of same sized
//! fingerprints. \param bitsOne The first fingerprint \param bitsTwo The list of fingerprints \param results The output
//! similarity, not safe to use until a sync.
template <typename T>
void launchBulkTanimotoSimilarity(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                  const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                  const size_t                                   elementsPerMolecule,
                                  const size_t                                   batchSize,

                                  AsyncDeviceVector<double>& results,
                                  const size_t               offset,
                                  const cudaStream_t         stream);

template <typename blockType>
void launchBulkTanimotoSimilarity(const cuda::std::span<const blockType> bitsOneBuffer,
                                  const cuda::std::span<const blockType> bitsTwoBuffer,
                                  const size_t                           elementsPerMolecule,
                                  AsyncDeviceVector<double>&             results,
                                  const cudaStream_t                     stream);

//! Launches a kernel to compute the all-to-all Tanimoto similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
template <typename kThreadReductionType>
void launchCrossTanimotoSimilarity(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                   const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                   const size_t                                   numBitsPerMolecule,
                                   AsyncDeviceVector<double>&                     results,
                                   const size_t                                   offset);

//! Launches a kernel to compute the all-to-all Tanimoto similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
void launchCrossTanimotoSimilarity(const cuda::std::span<const std::uint32_t> bitsOne,
                                   const cuda::std::span<const std::uint32_t> bitsTwo,
                                   const size_t                               numBitsPerMolecule,
                                   const cuda::std::span<double>              results,
                                   const size_t                               offset);

// --------------------------------
// Tanimoto similarity explicit template instantiations
// --------------------------------

extern template void launchBulkTanimotoSimilarity<std::uint32_t>(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                                                 const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                                                 const size_t elementsPerMolecule,
                                                                 const size_t batchSize,

                                                                 AsyncDeviceVector<double>& results,
                                                                 const size_t               offset,
                                                                 const cudaStream_t         stream);
extern template void launchBulkTanimotoSimilarity<std::uint64_t>(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                                                 const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                                                 const size_t elementsPerMolecule,
                                                                 const size_t batchSize,

                                                                 AsyncDeviceVector<double>& results,
                                                                 const size_t               offset,
                                                                 const cudaStream_t         stream);

extern template void launchCrossTanimotoSimilarity<typename std::uint32_t>(
  const AsyncDeviceVector<internal::kBlockType>& bitsOne,
  const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
  const size_t                                   numBitsPerMolecule,
  AsyncDeviceVector<double>&                     results,
  const size_t                                   offset

);

extern template void launchBulkTanimotoSimilarity<std::uint32_t>(
  const cuda::std::span<const std::uint32_t> bitsOneBuffer,
  const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
  const size_t                               elementsPerMolecule,
  AsyncDeviceVector<double>&                 results,
  const cudaStream_t                         stream);
extern template void launchBulkTanimotoSimilarity<std::uint64_t>(
  const cuda::std::span<const std::uint64_t> bitsOneBuffer,
  const cuda::std::span<const std::uint64_t> bitsTwoBuffer,
  const size_t                               elementsPerMolecule,
  AsyncDeviceVector<double>&                 results,
  const cudaStream_t                         stream);

// --------------------------------
// Cosine similarity launch functions
// --------------------------------

//! Launches a kernel to compute the Cosine similarity between a 32bit fingerprint and a list of 32bit fingerprints.
//! \param bitsOne The first fingerprint
//! \param bitsTwo The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
void launchBulkCosineSimilarity(const AsyncDevicePtr<std::uint32_t>&    bitsOne,
                                const AsyncDeviceVector<std::uint32_t>& bitsTwo,
                                AsyncDeviceVector<float>&               results);
//! Launches a kernel to compute the Cosine similarity between an arbitrary fingerprint and a list of same sized
//! fingerprints. \param bitsOne The first fingerprint \param bitsTwo The list of fingerprints \param results The output
//! similarity, not safe to use until a sync.
template <typename T>
void launchBulkCosineSimilarity(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                const size_t                                   elementsPerMolecule,
                                const size_t                                   batchSize,

                                AsyncDeviceVector<double>& results,
                                const size_t               offset,
                                const cudaStream_t         stream);

template <typename blockType>
void launchBulkCosineSimilarity(const cuda::std::span<const blockType> bitsOneBuffer,
                                const cuda::std::span<const blockType> bitsTwoBuffer,
                                const size_t                           elementsPerMolecule,
                                AsyncDeviceVector<double>&             results,
                                const cudaStream_t                     stream);

//! Launches a kernel to compute the all-to-all Cosine similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
template <typename kThreadReductionType>
void launchCrossCosineSimilarity(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                 const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                 const size_t                                   numBitsPerMolecule,
                                 AsyncDeviceVector<double>&                     results,
                                 const size_t                                   offset);

//! Launches a kernel to compute the all-to-all Cosine similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
void launchCrossCosineSimilarity(const cuda::std::span<const std::uint32_t> bitsOne,
                                 const cuda::std::span<const std::uint32_t> bitsTwo,
                                 const size_t                               numBitsPerMolecule,
                                 const cuda::std::span<double>              results,
                                 const size_t                               offset);

// --------------------------------
// Cosine similarity explicit template instantiations
// --------------------------------

extern template void launchBulkCosineSimilarity<std::uint32_t>(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                                               const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                                               const size_t elementsPerMolecule,
                                                               const size_t batchSize,

                                                               AsyncDeviceVector<double>& results,
                                                               const size_t               offset,
                                                               const cudaStream_t         stream);
extern template void launchBulkCosineSimilarity<std::uint64_t>(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                                               const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                                               const size_t elementsPerMolecule,
                                                               const size_t batchSize,

                                                               AsyncDeviceVector<double>& results,
                                                               const size_t               offset,
                                                               const cudaStream_t         stream);

extern template void launchCrossCosineSimilarity<typename std::uint32_t>(
  const AsyncDeviceVector<internal::kBlockType>& bitsOne,
  const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
  const size_t                                   numBitsPerMolecule,
  AsyncDeviceVector<double>&                     results,
  const size_t                                   offset

);

extern template void launchBulkCosineSimilarity<std::uint32_t>(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                               const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                               const size_t               elementsPerMolecule,
                                                               AsyncDeviceVector<double>& results,
                                                               const cudaStream_t         stream);
extern template void launchBulkCosineSimilarity<std::uint64_t>(const cuda::std::span<const std::uint64_t> bitsOneBuffer,
                                                               const cuda::std::span<const std::uint64_t> bitsTwoBuffer,
                                                               const size_t               elementsPerMolecule,
                                                               AsyncDeviceVector<double>& results,
                                                               const cudaStream_t         stream);
}  // namespace nvMolKit

#endif  // SIMILARITY_KERNELS_H
