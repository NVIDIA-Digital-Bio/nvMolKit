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

#include "device_vector.h"
#include "host_vector.h"
#include "minimizer_api.h"

namespace nvMolKit {

//! Algorithm parameters for the FIRE minimizer. Defaults taken from ASE
//! (https://gitlab.com/ase/ase/-/blob/master/ase/optimize/fire.py)
struct FireOptions {
  double dtInit = 0.1;  //!< Initial time step
  double dtMax  = 1.0;  //!< Maximum time step

  double maxStep = 0.2;  //!< Maximum distance an atom can move per step.

  double timeStepIncrement = 0.1;  //!< Factor to increase time step when conditions are met
  double timeStepDecrement = 0.5;  //!< Factor to decrease time step when conditions are not met

  int nMinForIncrease = 5;  //!< Number of steps with positive power before increasing time step

  double alphaInit      = 0.25;  //!< Initial value of alpha
  double alphaDecrement = 0.99;  //!< Factor to decrease alpha when conditions are met
};

class FireBatchMinimizer final : public BatchMinimizer {
 public:
  explicit FireBatchMinimizer(int                dataDim = 3,
                              const FireOptions& options = FireOptions(),
                              cudaStream_t       stream  = nullptr);
  ~FireBatchMinimizer() override = default;

  void initialize(const std::vector<int>& atomStartsHost, const uint8_t* activeSystems = nullptr);

  bool step(double                        gradTol,
            const AsyncDeviceVector<int>& atomStarts,
            AsyncDeviceVector<double>&    positions,
            AsyncDeviceVector<double>&    grad,
            const GradFunctor&            gFunc);

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
  void fireV1(double                        gradTol,
              const AsyncDeviceVector<int>& atomStarts,
              AsyncDeviceVector<double>&    positions,
              AsyncDeviceVector<double>&    grad);
  int  compactAndCountConverged();

  int          dataDim_;
  FireOptions  fireOptions_;
  cudaStream_t stream_;

  // Per atom * dim quantities
  AsyncDeviceVector<double> velocities_;
  AsyncDeviceVector<double> prevVelocities_;

  // Per system quantities.
  AsyncDeviceVector<double>  powers_;
  AsyncDeviceVector<double>  dt_;
  AsyncDeviceVector<double>  alpha_;
  AsyncDeviceVector<int>     numStepsWithPositivePower_;
  AsyncDeviceVector<int>     numStepsWithNegativePower_;
  AsyncDeviceVector<uint8_t> statuses_;

  // Status trackers.
  AsyncDeviceVector<uint8_t> countTempStorage_;
  AsyncDevicePtr<int>        countFinished_;
  PinnedHostVector<int>      loopStatusHost_;
  AsyncDeviceVector<int>     activeSystemIndices_;
  AsyncDeviceVector<int>     allSystemIndices_;
  int                        numUnfinishedSystems_ = 0;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_FIRE_MINIMIZER_H
