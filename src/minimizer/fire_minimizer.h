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

#ifndef NVMOLKIT_FIRE_MINIMIZER_H
#define NVMOLKIT_FIRE_MINIMIZER_H

#include "minimizer_api.h"

namespace nvMolKit {

class FireBatchMinimizer : public BatchMinimizer {
 public:
  FireBatchMinimizer();
  ~FireBatchMinimizer() override = default;

  bool minimize(int                           numIters,
                double                        gradTol,
                const std::vector<int>&       atomStartsHost,
                const AsyncDeviceVector<int>& atomStarts,
                AsyncDeviceVector<double>&    positions,
                AsyncDeviceVector<double>&    grad,
                AsyncDeviceVector<double>&    energyOuts,
                AsyncDeviceVector<double>&    energyBuffer,
                EnergyFunctor                 eFunc,
                GradFunctor                   gFunc,
                const uint8_t*                activeThisStage = nullptr) override;

 private:
  int compactAndCountConverged() const;

  mutable AsyncDeviceVector<uint8_t> countTempStorage_;
  mutable AsyncDevicePtr<int>        countFinished_;
  mutable PinnedHostVector<int>      loopStatusHost_;
  mutable AsyncDeviceVector<int>     activeSystemIndices_;
  AsyncDeviceVector<int>             allSystemIndices_;
  mutable int                        numUnfinishedSystems_ = 0;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_FIRE_MINIMIZER_H


