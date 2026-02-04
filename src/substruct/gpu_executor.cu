// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_error_check.h"
#include "gpu_executor.h"

namespace nvMolKit {

std::pair<int, int> getStreamPriorityRange() {
  int leastPriority    = 0;
  int greatestPriority = 0;
  cudaCheckError(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  return {greatestPriority, leastPriority};
}

GpuExecutor::GpuExecutor(int executorIdx, int gpuDeviceId)
    : computeStream(("executor" + std::to_string(executorIdx) + "_mainStream").c_str()),
      recursiveStream(getStreamPriorityRange().first,
                      ("executor" + std::to_string(executorIdx) + "_priorityRecursiveStream").c_str()),
      postRecursionStream(getStreamPriorityRange().second,
                          ("executor" + std::to_string(executorIdx) + "_postRecursionStream").c_str()),
      recursiveScratch(nullptr),
      deviceId(gpuDeviceId) {}

void GpuExecutor::initializeForStream() {
  cudaStream_t s         = computeStream.stream();
  cudaStream_t recStream = recursiveStream.stream();
  deviceResults.setStream(s);
  pairIndicesDev.setStream(s);
  recursiveScratch.setStream(recStream);
}

void GpuExecutor::applyMiniBatchPlan(MiniBatchPlan&& plan) {
  this->plan = std::move(plan);
}

}  // namespace nvMolKit
