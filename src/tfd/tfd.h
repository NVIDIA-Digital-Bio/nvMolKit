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

#ifndef NVMOLKIT_TFD_H
#define NVMOLKIT_TFD_H

#include <memory>
#include <vector>

#include "tfd_common.h"
#include "tfd_cpu.h"
#include "tfd_gpu.h"

namespace nvMolKit {

//! Unified TFD generator with automatic backend selection
//!
//! Provides a single interface for TFD computation, automatically
//! selecting between CPU and GPU backends based on the options provided.
//! Backends are lazily initialized on first use.
//!
//! For GPU-resident output, use TFDGpuGenerator::GetTFDMatricesGpuBuffer directly.
class TFDGenerator {
 public:
  TFDGenerator() = default;

  //! Compute TFD matrix for a single molecule
  std::vector<double> GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options = TFDComputeOptions{});

  //! Compute TFD matrices for multiple molecules
  std::vector<std::vector<double>> GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                  const TFDComputeOptions& options = TFDComputeOptions{});

 private:
  void initializeBackendIfNeeded(TFDComputeBackend backend);

  std::unique_ptr<TFDGpuGenerator> gpuGenerator_;
  std::unique_ptr<TFDCpuGenerator> cpuGenerator_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_H
