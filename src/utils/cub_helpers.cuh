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

#ifndef NVMOLKIT_CUB_HELPERS_H
#define NVMOLKIT_CUB_HELPERS_H

#include <cub/cub.cuh>

#include "nvtx.h"

// Default to 0 if not defined (no CCCL or unknown version)
// Version encoding: MAJOR * 10000 + MINOR * 100 + PATCH
// Examples: 2.8.0 = 20800, 3.0.0 = 30000
#ifndef NVMOLKIT_CCCL_VERSION
#define NVMOLKIT_CCCL_VERSION 0
#endif

// Feature detection macros for cleaner conditional compilation
// NVMOLKIT_HAS_NEW_ARGMAX_API: CUB's DeviceReduce::ArgMax returns separate value/index (CCCL >= 2.8.0)
#if NVMOLKIT_CCCL_VERSION >= 20800 || (NVMOLKIT_CCCL_VERSION == 0 && CUDART_VERSION >= 12090)
#define NVMOLKIT_HAS_NEW_ARGMAX_API 1
#else
#define NVMOLKIT_HAS_NEW_ARGMAX_API 0
#endif

// Check for modern C++ operators support:
// - CCCL >= 3.0.0 (detected via CMake as numeric version >= 30000), OR
// - CUDA >= 12.9 when CCCL version detection failed (bundled CCCL >= 3.0.0)
#if NVMOLKIT_CCCL_VERSION >= 30000 || (NVMOLKIT_CCCL_VERSION == 0 && CUDART_VERSION >= 12090)
// CCCL >= 3.0.0 provides modern C++ functional operators
using cubMax  = cuda::maximum<>;
using cubMin  = cuda::minimum<>;
using cubSum  = cuda::std::plus<>;
using cubLess = cuda::std::less<>;
#else
// Fall back to CUB operators for older CCCL or bundled CUDA headers
// Suppress deprecation warnings for cub::Max and cub::Sum in CCCL 2.x
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
using cubMax = cub::Max;
using cubMin = cub::Min;
using cubSum = cub::Sum;
struct cubLess {
  template <typename T> __host__ __device__ __forceinline__ bool operator()(const T& a, const T& b) const {
    return a < b;
  }
};
#pragma GCC diagnostic pop
#endif  // NVMOLKIT_CCCL_VERSION >= 30000

namespace nvmolkit {
namespace detail {

#if NVMOLKIT_HAS_NEW_ARGMAX_API

// CCCL >= 2.8.0: Use CUB's DeviceReduce::ArgMax with new API
//! Wrapper for CUB's DeviceReduce::ArgMax (CCCL >= 2.8.0)
//! Uses the new API that returns max value and index separately
template <typename InputIteratorT>
inline cudaError_t DeviceArgMax(void*          d_temp_storage,
                                size_t&        temp_storage_bytes,
                                InputIteratorT d_in,
                                int*           d_max_value_out,
                                int*           d_max_index_out,
                                int            num_items,
                                cudaStream_t   stream = 0) {
  nvMolKit::ScopedNvtxRange range("CUB ArgMax (CCCL >= 2.8.0)");
  return cub::DeviceReduce::ArgMax(d_temp_storage,
                                   temp_storage_bytes,
                                   d_in,
                                   d_max_value_out,
                                   d_max_index_out,
                                   static_cast<int64_t>(num_items),
                                   stream);
}

#endif  // NVMOLKIT_HAS_NEW_ARGMAX_API

}  // namespace detail
}  // namespace nvmolkit

#endif  // NVMOLKIT_CUB_HELPERS_H
