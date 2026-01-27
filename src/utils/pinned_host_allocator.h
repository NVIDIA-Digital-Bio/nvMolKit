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

#ifndef NVMOLKIT_PINNED_HOST_ALLOCATOR_H
#define NVMOLKIT_PINNED_HOST_ALLOCATOR_H

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "host_vector.h"

namespace nvMolKit {

class PinnedHostAllocator {
 public:
  PinnedHostAllocator() = default;
  explicit PinnedHostAllocator(size_t estimatedBytes);

  PinnedHostAllocator(const PinnedHostAllocator&)                = delete;
  PinnedHostAllocator& operator=(const PinnedHostAllocator&)     = delete;
  PinnedHostAllocator(PinnedHostAllocator&&) noexcept            = default;
  PinnedHostAllocator& operator=(PinnedHostAllocator&&) noexcept = default;

  void preallocate(size_t estimatedBytes);

  template <typename T> PinnedHostView<T> allocate(size_t count) {
    if (count == 0) {
      throw std::invalid_argument("PinnedHostAllocator allocate requires non-zero size.");
    }
    const size_t         bytes = count * sizeof(T);
    PinnedHostAllocation alloc = allocateBytes(bytes);
    return PinnedHostView<T>(std::span<T>(reinterpret_cast<T*>(alloc.data), count), std::move(alloc.owner));
  }

 private:
  struct PinnedHostAllocation {
    std::byte*                 data  = nullptr;
    size_t                     bytes = 0;
    std::shared_ptr<std::byte> owner;
  };

  struct BufferEntry {
    std::shared_ptr<PinnedHostVector<std::byte>> buffer;
    size_t                                       offset = 0;
  };

  PinnedHostAllocation allocateBytes(size_t bytes);

  std::vector<BufferEntry> buffers_;
  size_t                   bufferBytes_  = 0;
  bool                     preallocated_ = false;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_PINNED_HOST_ALLOCATOR_H
