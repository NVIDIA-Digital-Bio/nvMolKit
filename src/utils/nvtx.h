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

#ifndef NVMOLKIT_UTILS_NVTX_H
#define NVMOLKIT_UTILS_NVTX_H

#include <string>

#include "nvtx3/nvToolsExt.h"

namespace nvMolKit {

class ScopedNvtxRange {
 public:
  explicit ScopedNvtxRange(const std::string& name) { nvtxRangePushA(name.c_str()); }
  explicit ScopedNvtxRange(const char* name) { nvtxRangePushA(name); }
  ScopedNvtxRange(const ScopedNvtxRange&)            = delete;
  ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
  ScopedNvtxRange(ScopedNvtxRange&&)                 = delete;
  ScopedNvtxRange& operator=(ScopedNvtxRange&&)      = delete;

  void pop() noexcept {
    if (!popped_) {
      nvtxRangePop();
      popped_ = true;
    }
  }

  ~ScopedNvtxRange() noexcept { pop(); }

 private:
  bool popped_ = false;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_UTILS_NVTX_H
