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
#include "nvtx3/nvToolsExtCudaRt.h"

namespace nvMolKit {

namespace NvtxColor {
constexpr uint32_t kGrey   = 0xFF808080;
constexpr uint32_t kRed    = 0xFFFF0000;
constexpr uint32_t kGreen  = 0xFF00FF00;
constexpr uint32_t kBlue   = 0xFF0000FF;
constexpr uint32_t kYellow = 0xFFFFFF00;
constexpr uint32_t kCyan   = 0xFF00FFFF;
constexpr uint32_t kOrange = 0xFFFFA500;
}  // namespace NvtxColor

class ScopedNvtxRange {
 public:
  explicit ScopedNvtxRange(const std::string& name, uint32_t color = NvtxColor::kGrey) {
    pushWithColor(name.c_str(), color);
  }
  explicit ScopedNvtxRange(const char* name, uint32_t color = NvtxColor::kGrey) { pushWithColor(name, color); }
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
  void pushWithColor(const char* name, uint32_t color) {
    nvtxEventAttributes_t attrib = {0};
    attrib.version               = NVTX_VERSION;
    attrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attrib.colorType             = NVTX_COLOR_ARGB;
    attrib.color                 = color;
    attrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
    attrib.message.ascii         = name;
    nvtxRangePushEx(&attrib);
  }

  bool popped_ = false;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_UTILS_NVTX_H
