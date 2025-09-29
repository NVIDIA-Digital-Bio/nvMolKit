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

#ifndef NVMOLKIT_BFGS_MINIMIZE_H
#define NVMOLKIT_BFGS_MINIMIZE_H

#include <device_vector.h>

#include <functional>
#include <vector>

#include "host_vector.h"

namespace nvMolKit {

//! Compute energies, optionally on an external set of positions. If nullptr, expect to find in internal coordinates.
using EnergyFunctor = std::function<void(const double*)>;
//! Compute gradients on internal positions writing to internal buffer.
using GradFunctor   = std::function<void()>;

enum class DebugLevel {
  NONE     = 0,
  STEPWISE = 1,
};

//! BFGS Batch Minimizer
//!
//! This class implements a BFGS minimizer for batch systems, should be a 1:1 port of the RDKit BFGS minimizer.
//! \param dataDim Dimensionality of positions, default is 3 for 3D systems.
//! \param debugLevel Debug level, default is NONE. STEPWISE will collect stepwise data for debugging.
//! \param scaleGrads Whether to dynamically scale down gradients to match RDKit forcefield calculations, default is
//! true.
//!                   Note that when true, simple systems may not converge as well, but it is necessary for
//!                   compatibility with RDKit forcefield calculations.
//! TODO: Constructor should be parameter struct based, now that we have more parameters.
struct BfgsBatchMinimizer {
  explicit BfgsBatchMinimizer(int          dataDim    = 3,
                              DebugLevel   debugLevel = DebugLevel::NONE,
                              bool         scaleGrads = true,
                              cudaStream_t stream     = nullptr);
  ~BfgsBatchMinimizer();

  //! Run up to numIters iterations of BFGS minimization
  //! Returns 0 if all systems converged, 1 if some systems did not converge.
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
                const uint8_t*                activeThisStage = nullptr);

  // Set up the minimizer for a new system.
  void initialize(const std::vector<int>& atomStartsHost,
                  const int*              atomStarts,
                  double*                 positions,
                  double*                 grad,
                  double*                 energyOuts,
                  const uint8_t*          activeThisStage = nullptr);

  //! Set Initial Hessian
  void setHessianToIdentity();
  //! Determine max steps for each system.
  void setMaxStep();
  //! Set up the line search.
  void doLineSearchSetup(const double* srcEnergies);
  //! In-loop line search pre-energy position update.
  void doLineSearchPerturb();
  //! In-loop line search post-energy lambda calculation
  void doLineSearchPostEnergy(int iter);
  //! Line search cleanup step.
  void doLineSearchPostLoop();
  //! Count the number of finished line search systems.
  int  lineSearchCountFinished() const;
  void setDirection();
  //! Scale down gradients to match RDKit calcGradient in forcefield code.
  void scaleGrad(bool preLoop);
  void updateDGrad();
  int  compactAndCountConverged() const;
  void updateHessian();
  void collectDebugData();

  AsyncDeviceVector<int> allSystemIndices_;
  AsyncDeviceVector<int> activeSystemIndices_;  // Indices of systems that are active in the current iteration.
  mutable int            numUnfinishedSystems_ = 0;

  AsyncDeviceVector<double>  scratchPositions_;
  AsyncDeviceVector<int16_t> statuses_;

  // Intermediate buffers used for linear search
  AsyncDeviceVector<double>  lineSearchDir_;  // xi
  AsyncDeviceVector<int16_t> lineSearchStatus_;
  AsyncDeviceVector<double>  lineSearchLambdaMins_;
  AsyncDeviceVector<double>  lineSearchLambdas_;
  AsyncDeviceVector<double>  lineSearchLambdas2_;
  AsyncDeviceVector<double>  lineSearchSlope_;
  AsyncDeviceVector<double>  lineSearchMaxSteps_;

  AsyncDeviceVector<double> lineSearchStoredEnergy_;
  AsyncDeviceVector<double> lineSearchEnergyScratch_;
  AsyncDeviceVector<double> lineSearchEnergyOut_;

  // Temporary buffers for counting finished systems. Mutable to all
  // for const counting methods.
  mutable AsyncDeviceVector<uint8_t> countTempStorage_;
  mutable AsyncDevicePtr<int>        countFinished_;
  mutable PinnedHostVector<int>      loopStatusHost_;

  AsyncDeviceVector<double> finalEnergies_;

  // Hessian approximation and scratch buffers.
  AsyncDeviceVector<int> hessianStarts_;

  AsyncDeviceVector<double> scratchGrad_;
  AsyncDeviceVector<double> gradScales_;
  AsyncDeviceVector<double> inverseHessian_;
  AsyncDeviceVector<double> hessDGrad_;

  int  dataDim_    = 3;     // Dimensionality of positions.
  bool scaleGrads_ = true;  // Whether to scale gradients to match RDKit forcefield.
  bool hasLargeSystem_ = false;  // Whether any system exceeds shared-memory kernel limit

  // Tracking variables to determine if system needs initializing.
  int numAtomsTotal_ = 0;
  int numSystems_    = 0;

  double gradTol_ = 0.0;

  // The following are non-owning pointers to device, owned by
  // (e.g.) an MMFF system description.
  const int* atomStartsDevice = nullptr;
  double*    positionsDevice  = nullptr;
  double*    gradDevice       = nullptr;
  double*    energyOutsDevice = nullptr;

  DebugLevel                        debugLevel_ = DebugLevel::NONE;
  std::vector<std::vector<int16_t>> stepwiseStatuses;
  std::vector<std::vector<double>>  stepwiseEnergies;

  cudaStream_t stream_ = nullptr;
};

void copyAndInvert(const AsyncDeviceVector<double>& src, AsyncDeviceVector<double>& dst);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BFGS_MINIMIZE_H
