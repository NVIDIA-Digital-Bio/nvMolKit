// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_PINNED_BUFFER_POOL_H
#define NVMOLKIT_PINNED_BUFFER_POOL_H

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "pinned_host_allocator.h"
#include "substruct_types.h"
#include "thread_safe_queue.h"

namespace nvMolKit {

/**
 * @brief Compute bytes required for a pinned host buffer.
 */
size_t computePinnedHostBufferBytes(int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth);

/**
 * @brief Host-side pinned buffer for a mini-batch.
 */
struct PinnedHostBuffer {
  PinnedHostView<int>     pairIndices;
  PinnedHostView<int>     miniBatchPairMatchStarts;
  PinnedHostView<int>     matchCounts;
  PinnedHostView<int>     reportedCounts;
  PinnedHostView<int16_t> matchIndices;
  PinnedHostView<uint8_t> overflowFlags;

  std::array<PinnedHostView<int>, kMaxRecursionDepth + 1> matchGlobalPairIndicesHost = {};
  std::array<PinnedHostView<int>, kMaxRecursionDepth + 1> matchBatchLocalIndicesHost = {};
  std::array<PinnedHostView<BatchedPatternEntry>, 2>      patternsAtDepthHost        = {};
};

/**
 * @brief Pool for pinned host buffers used by pipeline workers.
 */
class PinnedHostBufferPool {
 public:
  void initialize(int poolSize, int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth);

  PinnedHostBuffer* acquire();
  void              release(PinnedHostBuffer* buffer);
  void              shutdown();

 private:
  static std::unique_ptr<PinnedHostBuffer> createBuffer(int maxBatchSize,
                                                        int maxMatchIndicesEstimate,
                                                        int maxPatternsPerDepth);

  std::vector<std::unique_ptr<PinnedHostBuffer>>      buffers_;
  std::unique_ptr<ThreadSafeQueue<PinnedHostBuffer*>> available_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_PINNED_BUFFER_POOL_H
