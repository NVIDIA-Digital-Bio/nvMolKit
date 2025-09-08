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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cuda_error_check.h"
#include "device.h"

using namespace nvMolKit;

TEST(Device, SetDevice) {
  WithDevice wd(0);
  int        device_id;
  cudaCheckError(cudaGetDevice(&device_id)) EXPECT_EQ(device_id, 0);
}

TEST(Device, SetDeviceSwap) {
  int nDevices;
  cudaCheckError(cudaGetDeviceCount(&nDevices)) if (nDevices < 2) {
    GTEST_SKIP();
  }
  auto wd = std::make_unique<WithDevice>(1);
  int  device_id;
  cudaCheckError(cudaGetDevice(&device_id)) EXPECT_EQ(device_id, 1);
  wd.reset();
  cudaCheckError(cudaGetDevice(&device_id))

    EXPECT_EQ(device_id, 0);
}

TEST(Device, GetDeviceFreeMemory) {
  size_t free = getDeviceFreeMemory();
  EXPECT_GT(free, 0);

  int* toAllocate = nullptr;
  // Allocate a megabyte
  cudaCheckError(cudaMalloc(&toAllocate, 1000000));
  size_t freeAfter = getDeviceFreeMemory();
  EXPECT_LT(freeAfter, free);
  cudaCheckError(cudaFree(toAllocate));
}

TEST(DeviceTest, RoundUpToNearestMultipleOfTwo) {
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(5), 6);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(2), 2);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(15), 16);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(0), 0);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(1), 2);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(1023), 1024);
}

TEST(DeviceTest, RoundUpToNearestPowerOfTwo) {
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(5), 8);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(2), 2);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(15), 16);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(0), 1);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(1), 1);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(1023), 1024);
}

TEST(DeviceTest, ScopedStream) {
  ScopedStream stream;
  cudaStream_t s = stream.stream();
  EXPECT_NE(s, nullptr);
}

TEST(DeviceTest, ScopedStreamMove) {
  ScopedStream stream;
  ScopedStream stream2(std::move(stream));
  EXPECT_EQ(stream.stream(), nullptr);
  EXPECT_NE(stream2.stream(), nullptr);
}
