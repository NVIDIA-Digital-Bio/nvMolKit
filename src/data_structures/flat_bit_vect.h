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

#ifndef NVMOLKIT_FLAT_BIT_VECT_H
#define NVMOLKIT_FLAT_BIT_VECT_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CPU_ONLY_MEMBER      __host__
#else
#define CUDA_CALLABLE_MEMBER
#define CPU_ONLY_MEMBER
#endif  // __CUDACC__

#include <cassert>
#include <cstring>
#include <functional>

namespace nvMolKit {

namespace detail {

// TODO explore std::memcmp for comparison and std::memcpy for cp/move.
template <std::size_t NBits> struct FlatBitVectStorage {
  using StorageType                          = std::int32_t;
  constexpr static std::size_t kNBits        = NBits;
  constexpr static std::size_t kStorageBytes = sizeof(StorageType);
  constexpr static std::size_t kStorageBits  = kStorageBytes * 8;
  constexpr static std::size_t kStorageCount = (NBits + kStorageBits - 1) / kStorageBits;
  std::uint32_t                bits[kStorageCount];

  static_assert(kNBits > 0, "kNBits must be greater than 0");
  static_assert(kStorageCount > 0, "kStorageCount must be greater than 0");

  // Default constructor does not initialize.
  CUDA_CALLABLE_MEMBER FlatBitVectStorage() = default;  // cppcheck-suppress uninitMemberVar
  CUDA_CALLABLE_MEMBER FlatBitVectStorage(const FlatBitVectStorage& other) {
    for (std::size_t i = 0; i < kStorageCount; ++i) {
      bits[i] = other.bits[i];
    }
  }
  CUDA_CALLABLE_MEMBER FlatBitVectStorage(FlatBitVectStorage&& other) {
    for (std::size_t i = 0; i < kStorageCount; ++i) {
      bits[i] = other.bits[i];
    }
  }
  CUDA_CALLABLE_MEMBER FlatBitVectStorage& operator=(const FlatBitVectStorage& other) {
    for (std::size_t i = 0; i < kStorageCount; ++i) {
      bits[i] = other.bits[i];
    }
    return *this;
  }
  CUDA_CALLABLE_MEMBER FlatBitVectStorage& operator=(FlatBitVectStorage&& other) {
    for (std::size_t i = 0; i < kStorageCount; ++i) {
      bits[i] = other.bits[i];
    }
    return *this;
  }

  CUDA_CALLABLE_MEMBER std::uint32_t& operator[](const std::size_t i) { return bits[i]; }
  CUDA_CALLABLE_MEMBER const std::uint32_t& operator[](const std::size_t i) const { return bits[i]; }

  CUDA_CALLABLE_MEMBER bool operator==(const FlatBitVectStorage& other) const {
    for (std::size_t i = 0; i < kStorageCount; ++i) {
      // Since we don't always default initialize, we need to mask out any unused bits.
      // User is responsible for initializing everything within the buffer.
      if (i == kStorageCount - 1) {
        const std::size_t nBitsInLastStorage = kNBits % kStorageBits;

        const std::uint32_t mask = nBitsInLastStorage == 0 ? 0xFFFFFFFF : (1 << nBitsInLastStorage) - 1;
        if ((bits[i] & mask) != (other.bits[i] & mask)) {
          return false;
        }
      } else if (bits[i] != other.bits[i]) {
        return false;
      }
    }
    return true;
  }
  CUDA_CALLABLE_MEMBER bool operator!=(const FlatBitVectStorage& other) const { return !(*this == other); }
  CUDA_CALLABLE_MEMBER bool operator<(const FlatBitVectStorage& other) = delete;
};

}  // namespace detail

//! In-memory bitvect.
template <std::size_t NBits> class FlatBitVect {
 public:
  using StorageType                          = typename detail::FlatBitVectStorage<NBits>::StorageType;
  constexpr static std::size_t kNBits        = detail::FlatBitVectStorage<NBits>::kNBits;
  constexpr static std::size_t kStorageBytes = detail::FlatBitVectStorage<NBits>::kStorageBytes;
  constexpr static std::size_t kStorageBits  = detail::FlatBitVectStorage<NBits>::kStorageBits;
  constexpr static std::size_t kStorageCount = detail::FlatBitVectStorage<NBits>::kStorageCount;

 private:
  detail::FlatBitVectStorage<NBits> bits_;

  //! Make all sizes of FlatBitVect friends of each other.
  template <size_t otherSize> friend class FlatBitVect;

 public:
  CUDA_CALLABLE_MEMBER FlatBitVect() = default;
  CUDA_CALLABLE_MEMBER explicit FlatBitVect(bool fillValue) {
    const int fillValExpanded = fillValue ? 0xFFFF : 0x0000;
    std::memset(bits_.bits, fillValExpanded, kStorageBytes * kStorageCount);
  }
  CUDA_CALLABLE_MEMBER              FlatBitVect(const FlatBitVect& other) = default;
  CUDA_CALLABLE_MEMBER              FlatBitVect(FlatBitVect&& other)      = default;
  CUDA_CALLABLE_MEMBER FlatBitVect& operator=(const FlatBitVect& other)   = default;
  CUDA_CALLABLE_MEMBER FlatBitVect& operator=(FlatBitVect&& other)        = default;

  CUDA_CALLABLE_MEMBER bool operator[](const std::size_t i) const {
    const std::size_t storageIdx = i / kStorageBits;
    const std::size_t bitIdx     = i % kStorageBits;
    return bits_[storageIdx] & (1U << bitIdx);
  }

  CUDA_CALLABLE_MEMBER void setBit(const std::size_t i, const bool value) {
    const std::size_t storageIdx = i / kStorageBits;
    assert(storageIdx < kStorageCount);
    const std::size_t bitIdx = i % kStorageBits;
    if (value) {
      bits_[storageIdx] |= (1U << bitIdx);
    } else {
      bits_[storageIdx] &= ~(1U << bitIdx);
    }
  }

  CUDA_CALLABLE_MEMBER void clear() {
    for (std::size_t i = 0; i < kStorageCount; ++i) {
      bits_[i] = 0;
    }
  }

  CUDA_CALLABLE_MEMBER bool operator==(const FlatBitVect& other) const { return bits_ == other.bits_; }
  CUDA_CALLABLE_MEMBER bool operator!=(const FlatBitVect& other) const { return bits_ != other.bits_; }

  CUDA_CALLABLE_MEMBER void operator|=(const FlatBitVect& other) {
    for (std::size_t i = 0; i < kStorageCount; ++i) {
      bits_[i] |= other.bits_[i];
    }
  }

  template <size_t otherSize>
  FlatBitVect<otherSize> CUDA_CALLABLE_MEMBER resize(bool checkRange      = false,
                                                     bool fillLarger      = false,
                                                     bool fillLargerValue = false) const {
    static_assert(otherSize != kNBits, "Cannot resize to the same size");
    constexpr int otherNumBlocks = otherSize / kStorageBits;
    if (checkRange) {
      for (std::size_t i = otherNumBlocks; i < kStorageCount; ++i) {
        if (bits_[i] != 0) {
          throw std::runtime_error("Cannot resize to a smaller size with bits set");
        }
      }
    }
    FlatBitVect<otherSize> resized;
    for (std::size_t i = 0; i < std::min(kStorageCount, resized.kStorageCount); ++i) {
      resized.bits_[i] = bits_[i];
    }
    if (fillLarger) {
      for (std::size_t i = kStorageCount; i < resized.kStorageCount; ++i) {
        resized.bits_[i] = fillLargerValue ? 0xFFFFFFFF : 0;
      }
    }
    return resized;
  }

  // Lexicographical <, should be bitwise identical to Boost implementation
  CUDA_CALLABLE_MEMBER bool operator<(const FlatBitVect& other) const {
    // Need a mask for the first entry
    const std::size_t   nBitsInLastStorage = kNBits % kStorageBits;
    const std::uint32_t mask               = nBitsInLastStorage == 0 ? 0xFFFFFFFF : (1 << nBitsInLastStorage) - 1;
    if ((bits_[kStorageCount - 1] & mask) < (other.bits_[kStorageCount - 1] & mask)) {
      return true;
    } else if ((bits_[kStorageCount - 1] & mask) > (other.bits_[kStorageCount - 1] & mask)) {
      return false;
    }

    for (int i = static_cast<int>(kStorageCount) - 2; i >= 0; --i) {
      if (bits_[i] < other.bits_[i]) {
        return true;
      } else if (bits_[i] > other.bits_[i]) {
        return false;
      }
    }
    return false;
  }

  CUDA_CALLABLE_MEMBER std::uint32_t* begin() { return &bits_[0]; }

  CUDA_CALLABLE_MEMBER const std::uint32_t* cbegin() const { return &bits_[0]; }

  CUDA_CALLABLE_MEMBER std::uint32_t* end() { return &bits_[kStorageCount - 1] + 1; }

  CUDA_CALLABLE_MEMBER const std::uint32_t* cend() const { return &bits_[kStorageCount - 1] + 1; }

  CPU_ONLY_MEMBER std::size_t numOnBits() const {
    std::size_t count = 0;
    for (std::size_t i = 0; i < kStorageCount; ++i) {
      if (i == kStorageCount - 1) {
        const std::size_t   nBitsInLastStorage = kNBits % kStorageBits;
        const std::uint32_t mask               = nBitsInLastStorage == 0 ? 0xFFFFFFFF : (1 << nBitsInLastStorage) - 1;
        count += __builtin_popcount(bits_[i] & mask);
      } else {
        count += __builtin_popcount(bits_[i]);
      }
    }
    return count;
  }
};

// Compile time test that we can reason about the bits layout.
static_assert(sizeof(FlatBitVect<32>) == 4, "FlatBitVect<32> should be 4 bytes");
static_assert(sizeof(FlatBitVect<33>) == 8, "FlatBitVect<33> should be 8 bytes");
static_assert(sizeof(FlatBitVect<64>) == 8, "FlatBitVect<64> should be 8 bytes");

}  // namespace nvMolKit

namespace std {
template <std::size_t NBits> struct hash<nvMolKit::FlatBitVect<NBits>> {
  std::size_t operator()(const nvMolKit::FlatBitVect<NBits>& fbv) const noexcept {
    std::size_t           result        = 0;
    constexpr std::size_t kStorageCount = nvMolKit::FlatBitVect<NBits>::kStorageCount;
    constexpr std::size_t kStorageBits  = nvMolKit::FlatBitVect<NBits>::kStorageBits;
    constexpr std::size_t kNBits        = nvMolKit::FlatBitVect<NBits>::kNBits;
    for (std::size_t i = 0; i < kStorageCount; ++i) {
      std::uint32_t word = fbv.cbegin()[i];
      if (i == kStorageCount - 1) {
        std::size_t   nBitsInLast = kNBits % kStorageBits;
        std::uint32_t mask        = nBitsInLast == 0 ? 0xFFFFFFFF : (1U << nBitsInLast) - 1;
        word &= mask;
      }
      result ^= std::hash<std::uint32_t>{}(word) + 0x9e3779b9 + (result << 6) + (result >> 2);
    }
    return result;
  }
};
}  // namespace std

#undef CUDA_CALLABLE_MEMBER
#undef CPU_ONLY_MEMBER
#endif  // NVMOLKIT_FLAT_BIT_VECT_H
