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

// Check for modern C++ operators support (CCCL >= 3.0.0)
// CUB_MAJOR_VERSION is defined in <cub/version.cuh>, included by cub.cuh
#if CUB_MAJOR_VERSION >= 3
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
#endif  // CUB_MAJOR_VERSION >= 3
#endif  // NVMOLKIT_CUB_HELPERS_H