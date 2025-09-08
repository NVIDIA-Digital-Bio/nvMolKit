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

#include "exceptions.h"

#include <cuda_runtime.h>

namespace nvMolKit {

CudaBadReturnCode::CudaBadReturnCode(cudaError_t returnCode) : rc_(returnCode) {
  message_ = "Encountered CUDA error " + std::to_string(rc_) + ": " + cudaGetErrorString(returnCode);
}
const char* CudaBadReturnCode::what() const noexcept {
  return message_.c_str();
}

int CudaBadReturnCode::rc() const {
  return rc_;
}

}  // namespace nvMolKit
