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

#include "bfgs_distgeom.h"

#include <memory>

#include "bfgs_minimize.h"
#include "dist_geom.h"

namespace nvMolKit::DistGeom {

void DistGeomMinimizeBFGS(BatchedMolecularSystemHost&    molSystemHost,
                          BatchedMolecularDeviceBuffers& molSystemDevice,
                          detail::ETKDGContext&          context,
                          const int                      maxIters,
                          const double                   gradTol,
                          const bool                     repeatUntilConverged,
                          cudaStream_t                   stream) {
  // Setup device buffers
  setupDeviceBuffers(molSystemHost,
                     molSystemDevice,
                     context.systemHost.positions,
                     static_cast<int>(context.systemHost.atomStarts.size() - 1));

  const size_t numAtoms = context.systemHost.atomStarts.back();
  const size_t numPos   = context.systemHost.positions.size();
  const int    dim      = (numPos == numAtoms * 3) ? 3 : 4;

  // Create energy and gradient functions
  auto eFunc = [&](const double* positions) {
    computeEnergy(molSystemDevice,
                  context.systemDevice.atomStarts,
                  context.systemDevice.positions,
                  context.activeThisStage.data(),
                  positions,
                  stream);
  };

  auto gFunc = [&]() {
    computeGradients(molSystemDevice,
                     context.systemDevice.atomStarts,
                     context.systemDevice.positions,
                     context.activeThisStage.data(),
                     stream);
  };

  // Create and configure BFGS minimizer
  std::unique_ptr<nvMolKit::BatchMinimizer> minimizer =
    std::make_unique<nvMolKit::BfgsBatchMinimizer>(/*dataDim=*/dim, nvMolKit::DebugLevel::NONE, true, stream);

  // Run minimization
  bool needsMore = minimizer->minimize(maxIters,
                                       gradTol,
                                       context.systemHost.atomStarts,
                                       context.systemDevice.atomStarts,
                                       context.systemDevice.positions,
                                       molSystemDevice.grad,
                                       molSystemDevice.energyOuts,
                                       molSystemDevice.energyBuffer,
                                       eFunc,
                                       gFunc,
                                       context.activeThisStage.data());
  while (needsMore && repeatUntilConverged) {
    needsMore = minimizer->minimize(maxIters,
                                    gradTol,
                                    context.systemHost.atomStarts,
                                    context.systemDevice.atomStarts,
                                    context.systemDevice.positions,
                                    molSystemDevice.grad,
                                    molSystemDevice.energyOuts,
                                    molSystemDevice.energyBuffer,
                                    eFunc,
                                    gFunc,
                                    context.activeThisStage.data());
  }
}

}  // namespace nvMolKit::DistGeom
