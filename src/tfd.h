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
//! This class provides a unified interface for TFD computation, automatically
//! selecting between CPU and GPU backends based on the options provided.
//! Backends are lazily initialized on first use.
//!
//! Example usage:
//! @code
//!   TFDGenerator generator;
//!
//!   // Use GPU (default)
//!   auto tfd = generator.GetTFDMatrix(mol);
//!
//!   // Explicitly use CPU
//!   TFDComputeOptions cpuOptions;
//!   cpuOptions.backend = TFDComputeBackend::CPU;
//!   auto tfdCpu = generator.GetTFDMatrix(mol, cpuOptions);
//!
//!   // Batch processing
//!   auto results = generator.GetTFDMatrices(mols);
//!
//!   // Keep results on GPU for further processing
//!   auto gpuResult = generator.GetTFDMatricesGpuBuffer(mols);
//! @endcode
class TFDGenerator {
 public:
  TFDGenerator() = default;

  //! Compute TFD matrix for a single molecule
  //! @param mol Molecule with conformers
  //! @param options Computation options (default: GPU backend)
  //! @return Lower triangular TFD matrix as flat vector [C*(C-1)/2 values]
  std::vector<double> GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options = TFDComputeOptions{});

  //! Compute TFD matrices for multiple molecules
  //! @param mols Vector of molecules
  //! @param options Computation options (default: GPU backend)
  //! @return Vector of TFD matrices, one per molecule
  std::vector<std::vector<double>> GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                  const TFDComputeOptions& options = TFDComputeOptions{});

  //! Compute TFD matrices and keep results on GPU
  //! @param mols Vector of molecules
  //! @param options Computation options (must use GPU backend)
  //! @return TFDGpuResult with GPU-resident data
  //! @throws std::invalid_argument if CPU backend is requested
  TFDGpuResult GetTFDMatricesGpuBuffer(const std::vector<const RDKit::ROMol*>& mols,
                                       const TFDComputeOptions&                options = TFDComputeOptions{});

 private:
  //! Initialize backend if not already created
  void initializeBackendIfNeeded(TFDComputeBackend backend);

  std::unique_ptr<TFDGpuGenerator> gpuGenerator_;
  std::unique_ptr<TFDCpuGenerator> cpuGenerator_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_H
