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

#ifdef NVMOLKIT_HAS_CCCL_GE_3
// CCCL >= 3.0.0 provides modern C++ functional operators
using cubMax = cuda::maximum<>;
using cubSum = cuda::std::plus<>;
#else
// Fall back to CUB operators for older CCCL or bundled CUDA headers
// Suppress deprecation warnings for cub::Max and cub::Sum in CCCL 2.x
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
using cubMax = cub::Max;
using cubSum = cub::Sum;
#pragma GCC diagnostic pop
#endif
#endif  // NVMOLKIT_CUB_HELPERS_H
