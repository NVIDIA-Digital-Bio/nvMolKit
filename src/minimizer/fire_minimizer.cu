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

#include "fire_minimizer.h"

#include <cub/cub.cuh>

#include "nvtx.h"

namespace nvMolKit {

FireBatchMinimizer::FireBatchMinimizer() = default;

bool FireBatchMinimizer::minimize(const int                     numIters,
                                  const double                  gradTol,
                                  const std::vector<int>&       atomStartsHost,
                                  const AsyncDeviceVector<int>& atomStarts,
                                  AsyncDeviceVector<double>&    positions,
                                  AsyncDeviceVector<double>&    grad,
                                  AsyncDeviceVector<double>&    energyOuts,
                                  AsyncDeviceVector<double>&    energyBuffer,
                                  EnergyFunctor                 eFunc,
                                  GradFunctor                   gFunc,
                                  const uint8_t*                activeThisStage) {
  (void)numIters;
  (void)gradTol;
  (void)atomStartsHost;
  (void)atomStarts;
  (void)positions;
  (void)grad;
  (void)energyOuts;
  (void)energyBuffer;
  (void)eFunc;
  (void)gFunc;
  (void)activeThisStage;
  return compactAndCountConverged() == 0;
}

int FireBatchMinimizer::compactAndCountConverged() const {
  const ScopedNvtxRange fireCompact("FireBatchMinimizer::compactAndCountConverged");
  size_t                temp_storage_bytes = countTempStorage_.size();

  cudaCheckError(cub::DeviceSelect::Flagged(nullptr,
                                            temp_storage_bytes,
                                            allSystemIndices_.data(),
                                            countTempStorage_.data(),
                                            activeSystemIndices_.data(),
                                            countFinished_.data(),
                                            allSystemIndices_.size(),
                                            nullptr));

  if (temp_storage_bytes > countTempStorage_.size()) {
    countTempStorage_.resize(temp_storage_bytes);
  }

  cudaCheckError(cub::DeviceSelect::Flagged(countTempStorage_.data(),
                                            temp_storage_bytes,
                                            allSystemIndices_.data(),
                                            countTempStorage_.data(),
                                            activeSystemIndices_.data(),
                                            countFinished_.data(),
                                            allSystemIndices_.size(),
                                            nullptr));

  int& unfinishedHost = loopStatusHost_[0];
  countFinished_.get(unfinishedHost);
  cudaStreamSynchronize(nullptr);
  numUnfinishedSystems_ = unfinishedHost;
  return numUnfinishedSystems_;
}

}  // namespace nvMolKit


