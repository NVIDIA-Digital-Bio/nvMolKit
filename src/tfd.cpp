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

#include "tfd.h"

#include <stdexcept>

namespace nvMolKit {

void TFDGenerator::initializeBackendIfNeeded(TFDComputeBackend backend) {
  switch (backend) {
    case TFDComputeBackend::GPU:
      if (!gpuGenerator_) {
        gpuGenerator_ = std::make_unique<TFDGpuGenerator>();
      }
      break;
    case TFDComputeBackend::CPU:
      if (!cpuGenerator_) {
        cpuGenerator_ = std::make_unique<TFDCpuGenerator>();
      }
      break;
  }
}

std::vector<double> TFDGenerator::GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  initializeBackendIfNeeded(options.backend);

  switch (options.backend) {
    case TFDComputeBackend::GPU:
      return gpuGenerator_->GetTFDMatrix(mol, options);
    case TFDComputeBackend::CPU:
      return cpuGenerator_->GetTFDMatrix(mol, options);
  }

  // Unreachable, but needed to suppress compiler warning
  return {};
}

std::vector<std::vector<double>> TFDGenerator::GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                              const TFDComputeOptions&                options) {
  initializeBackendIfNeeded(options.backend);

  switch (options.backend) {
    case TFDComputeBackend::GPU:
      return gpuGenerator_->GetTFDMatrices(mols, options);
    case TFDComputeBackend::CPU:
      return cpuGenerator_->GetTFDMatrices(mols, options);
  }

  // Unreachable, but needed to suppress compiler warning
  return {};
}

TFDGpuResult TFDGenerator::GetTFDMatricesGpuBuffer(const std::vector<const RDKit::ROMol*>& mols,
                                                   const TFDComputeOptions&                options) {
  if (options.backend != TFDComputeBackend::GPU) {
    throw std::invalid_argument("GetTFDMatricesGpuBuffer requires GPU backend");
  }

  initializeBackendIfNeeded(TFDComputeBackend::GPU);
  return gpuGenerator_->GetTFDMatricesGpuBuffer(mols, options);
}

}  // namespace nvMolKit
