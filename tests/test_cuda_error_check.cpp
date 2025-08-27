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

#include <gtest/gtest.h>

#include "cuda_error_check.h"

using namespace nvMolKit;

TEST(CudaErrorCheck, NoThrowOnGoodCode) {
  cudaCheckError(cudaSuccess);  // NOLINT
}

TEST(CudaErrorCheck, ThrowOnBadCode) {  // NOLINT(readability-function-cognitive-complexity)
  EXPECT_THROW(
    {
      try {
        cudaCheckError(cudaErrorInvalidValue);
      } catch (const CudaBadReturnCode& e) {
        EXPECT_EQ(e.rc(), cudaErrorInvalidValue);
        throw;
      }
    },
    CudaBadReturnCode);
}

TEST(CudaErrorCheck, NoThrowOnGoodCodeNoThrow) {
  cudaCheckErrorNoThrow(cudaSuccess);
}

TEST(CudaErrorCheck, NoThrowOnBadCodeNoThrow) {
  cudaCheckErrorNoThrow(cudaErrorInvalidValue);
}
