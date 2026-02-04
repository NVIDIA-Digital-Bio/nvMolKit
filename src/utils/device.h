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

#ifndef NVMOLKIT_DEVICE_H
#define NVMOLKIT_DEVICE_H

#include <cstddef>
#include <cstdint>

#include "cuda_runtime.h"

namespace nvMolKit {

//! Returns the number of CUDA devices available on the system.
int countCudaDevices();

//! RAII class to set the current device to a specific device, and reset it to the original device when the object goes
//! out of scope.
//! Usage:
//!   {
//!     WithDevice withDevice(1);
//!     // Do stuff on device 1
//!   }
//! // Device is reset to original device
class WithDevice {
 public:
  explicit WithDevice(int device_id);
  WithDevice(const WithDevice&)             = delete;
  WithDevice& operator=(const WithDevice&)  = delete;
  WithDevice(WithDevice&& other)            = delete;
  WithDevice& operator=(WithDevice&& other) = delete;
  ~WithDevice();

 private:
  int original_device_id_ = -1;
};

//! Creates and holds a CUDA stream on the current device
class ScopedStream {
 public:
  explicit ScopedStream(const char* name = nullptr);
  ScopedStream(const ScopedStream&)            = delete;
  ScopedStream& operator=(const ScopedStream&) = delete;
  ~ScopedStream() noexcept;

  ScopedStream(ScopedStream&& other) noexcept;
  ScopedStream& operator=(ScopedStream&& other) noexcept;
  cudaStream_t  stream() const noexcept { return original_stream_; }

 private:
  cudaStream_t original_stream_ = nullptr;
};

/**
 * @brief RAII stream with explicit priority.
 *
 * Creates a non-blocking stream with the specified priority.
 * Lower numerical priority means higher execution priority.
 */
class ScopedStreamWithPriority {
 public:
  explicit ScopedStreamWithPriority(int priority, const char* name = nullptr);
  ScopedStreamWithPriority(const ScopedStreamWithPriority&)            = delete;
  ScopedStreamWithPriority& operator=(const ScopedStreamWithPriority&) = delete;
  ~ScopedStreamWithPriority() noexcept;

  ScopedStreamWithPriority(ScopedStreamWithPriority&& other) noexcept;
  ScopedStreamWithPriority& operator=(ScopedStreamWithPriority&& other) noexcept;
  cudaStream_t              stream() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class ScopedCudaEvent {
 public:
  explicit ScopedCudaEvent();
  ScopedCudaEvent(const ScopedCudaEvent&)            = delete;
  ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;
  ~ScopedCudaEvent() noexcept;

  ScopedCudaEvent(ScopedCudaEvent&& other) noexcept;
  ScopedCudaEvent& operator=(ScopedCudaEvent&& other) noexcept;
  cudaEvent_t      event() const noexcept { return original_event_; }

 private:
  cudaEvent_t original_event_ = nullptr;
};

//! Returns the amount of free memory on the current device.
size_t getDeviceFreeMemory();

//! Rounds a number up to the nearest multiple of two.
constexpr std::uint32_t roundUpToNearestMultipleOfTwo(std::uint32_t num) {
  return (num + 1) & ~1U;
}

//! Rounds a number up to the nearest power of two.
constexpr std::uint32_t roundUpToNearestPowerOfTwo(std::uint32_t num) {
  if (num == 0) {
    return 1;
  }
  --num;
  for (std::uint32_t i = 1; i < sizeof(num) * 8; i *= 2) {
    num |= num >> i;
  }
  return num + 1;
}

}  // namespace nvMolKit
#endif  // NVMOLKIT_DEVICE_H
