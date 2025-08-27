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

#ifndef NV_MOLKIT_HARDWARE_OPTIONS_H
#define NV_MOLKIT_HARDWARE_OPTIONS_H

#include <vector>
namespace nvMolKit {

//! \brief Options for configuring hardware usage in batch processing of molecules such as with ETKDG and MMFF.
//! This struct allows users to specify how many CPU threads to use for parallel preprocessing,
//! how many molecules to process in each batch, how many batches to run concurrently on a GPU,
//! and which GPU devices to use.
struct BatchHardwareOptions {
  //! Number of CPU threads to use for parallel preprocessing. Default to all available threads.
  int              preprocessingThreads = -1;
  //! Number of molecules to process in each batch
  int              batchSize            = 200;
  //! Number of batches to run concurrently on a GPU. Default to distributing all threads evenly across GPUs.
  int              batchesPerGpu        = -1;
  //! GPU device IDs to use. If empty, use all available GPUs (0 to N-1).
  std::vector<int> gpuIds               = {};
};
}  // namespace nvMolKit

inline std::vector<int> getGpuIds(const nvMolKit::BatchHardwareOptions& opts) {
  return opts.gpuIds;
}

#endif  // NV_MOLKIT_HARDWARE_OPTIONS_H
