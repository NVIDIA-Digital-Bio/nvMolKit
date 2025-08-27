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

#include "device.h"

#include <cuda_runtime.h>

#include "cuda_error_check.h"

namespace nvMolKit {

int countCudaDevices() {
  int device_count = 0;
  cudaCheckError(cudaGetDeviceCount(&device_count));
  return device_count;
}

WithDevice::WithDevice(int device_id) {
  cudaCheckError(cudaGetDevice(&original_device_id_));
  cudaCheckError(cudaSetDevice(device_id));
}

WithDevice::~WithDevice() {
  cudaCheckErrorNoThrow(cudaSetDevice(original_device_id_));
}

size_t getDeviceFreeMemory() {
  size_t free  = 0;
  size_t total = 0;
  cudaCheckError(cudaMemGetInfo(&free, &total));
  return free;
}

ScopedStream::ScopedStream() {
  cudaCheckError(cudaStreamCreateWithFlags(&original_stream_, cudaStreamNonBlocking));
}

ScopedStream::~ScopedStream() noexcept {
  if (original_stream_ == nullptr) {
    return;
  }
  cudaCheckErrorNoThrow(cudaStreamSynchronize(original_stream_));
  cudaCheckErrorNoThrow(cudaStreamDestroy(original_stream_));
}

ScopedStream::ScopedStream(ScopedStream&& other) noexcept : original_stream_(other.original_stream_) {
  other.original_stream_ = nullptr;
}

ScopedCudaEvent::ScopedCudaEvent() {
  cudaCheckError(cudaEventCreateWithFlags(&original_event_, cudaEventDisableTiming));
}

ScopedCudaEvent::~ScopedCudaEvent() noexcept {
  if (original_event_ == nullptr) {
    return;
  }
  cudaCheckErrorNoThrow(cudaEventDestroy(original_event_));
}

ScopedCudaEvent::ScopedCudaEvent(ScopedCudaEvent&& other) noexcept : original_event_(other.original_event_) {
  other.original_event_ = nullptr;
}

ScopedCudaEvent& ScopedCudaEvent::operator=(ScopedCudaEvent&& other) noexcept {
  if (original_event_ != nullptr && original_event_ != other.original_event_) {
    cudaCheckErrorNoThrow(cudaEventDestroy(original_event_));
  }
  original_event_       = other.original_event_;
  other.original_event_ = nullptr;
  return *this;
}

ScopedStream& ScopedStream::operator=(ScopedStream&& other) noexcept {
  if (original_stream_ != nullptr && original_stream_ != other.original_stream_) {
    cudaCheckErrorNoThrow(cudaStreamDestroy(original_stream_));
  }
  original_stream_       = other.original_stream_;
  other.original_stream_ = nullptr;
  return *this;
}

}  // namespace nvMolKit
