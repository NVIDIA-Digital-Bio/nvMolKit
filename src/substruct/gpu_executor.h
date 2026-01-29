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

#ifndef NVMOLKIT_GPU_EXECUTOR_H
#define NVMOLKIT_GPU_EXECUTOR_H

#include <array>
#include <utility>
#include <vector>

#include "device.h"
#include "device_vector.h"
#include "minibatch_planner.h"
#include "molecules_device.cuh"
#include "recursive_preprocessor.h"
#include "substruct_search_internal.h"
#include "substruct_types.h"

namespace nvMolKit {

/**
 * @brief Owns CUDA resources and device buffers for a worker executor.
 */
struct GpuExecutor {
  MiniBatchPlan plan;

  // Streams and events (declared first so they're destroyed last)
  ScopedStream    computeStream;
  ScopedCudaEvent copyDoneEvent;
  ScopedCudaEvent allocDoneEvent;
  ScopedCudaEvent targetsReadyEvent;

  // Recursive pipeline
  ScopedStreamWithPriority                                   recursiveStream;
  ScopedStreamWithPriority                                   postRecursionStream;
  std::array<ScopedCudaEvent, kMaxRecursionDepth>            depthEvents;
  ScopedCudaEvent                                            recursiveDoneEvent;
  ScopedCudaEvent                                            postRecursionDoneEvent;
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchGlobalPairIndices;
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchMiniBatchLocalIndices;
  RecursiveScratchBuffers                                    recursiveScratch;
  MiniBatchResultsDevice                                     deviceResults;
  AsyncDeviceVector<int>                                     pairIndicesDev;
  MoleculesDevice                                            targetsDevice;

  int deviceId = 0;  ///< GPU device ID this executor is assigned to

  GpuExecutor(int executorIdx, int gpuDeviceId);

  void initializeForStream();

  cudaStream_t stream() const { return computeStream.stream(); }

  void applyMiniBatchPlan(MiniBatchPlan&& plan);
};

std::pair<int, int> getStreamPriorityRange();

}  // namespace nvMolKit

#endif  // NVMOLKIT_GPU_EXECUTOR_H
