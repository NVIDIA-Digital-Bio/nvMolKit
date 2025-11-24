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

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include "dist_geom.h"
#include "dist_geom_flattened_builder.h"
#include "etkdg_impl.h"
#include "etkdg_stage_firstminimization.h"
#include "forcefields/kernel_utils.cuh"
#include "minimizer/bfgs_distgeom.h"

using ::nvMolKit::detail::ETKDGContext;
using ::nvMolKit::detail::ETKDGStage;

namespace nvMolKit {

namespace {
__global__ void checkMinimizedEnergiesKernel(const int     molNum,
                                             const double* energyOuts,
                                             const int*    atomStarts,
                                             uint8_t*      failedThisStage) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= molNum) {
    return;
  }

  const int    numAtoms      = atomStarts[idx + 1] - atomStarts[idx];
  const double energyPerAtom = energyOuts[idx] / numAtoms;

  if (energyPerAtom >= nvMolKit::detail::MAX_MINIMIZED_E_PER_ATOM) {
    failedThisStage[idx] = 1;
  }
}
}  // namespace

namespace detail {
constexpr int kBlockSize = 256;
FirstMinimizeStage::FirstMinimizeStage(const std::vector<const RDKit::ROMol*>&     mols,
                                       const std::vector<EmbedArgs>&               eargs,
                                       const RDKit::DGeomHelpers::EmbedParameters& embedParam,
                                       ETKDGContext&                               ctx,
                                       cudaStream_t                                stream)
    : embedParam_(embedParam),
      stream_(stream) {
  // Check that all vectors have the same size
  if (mols.size() != eargs.size()) {
    throw std::runtime_error("Number of molecules and embed args must be the same");
  }

  // Process each molecule
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto& mol      = mols[i];
    const auto& embedArg = eargs[i];
    const auto& numAtoms = mol->getNumAtoms();
    auto        ffParams = nvMolKit::DistGeom::constructForceFieldContribs(embedArg.dim,
                                                                    *embedArg.mmat,
                                                                    embedArg.chiralCenters,
                                                                    1.0,
                                                                    0.1,
                                                                    nullptr,
                                                                    embedParam.basinThresh);

    // Add to molecular system
    nvMolKit::DistGeom::addMoleculeToMolecularSystem(ffParams,
                                                     numAtoms,
                                                     embedArg.dim,
                                                     ctx.systemHost.atomStarts,
                                                     molSystemHost);
  }
  DistGeom::setStreams(molSystemDevice, stream_);
  nvMolKit::DistGeom::sendContribsAndIndicesToDevice(molSystemHost, molSystemDevice);
  nvMolKit::DistGeom::setupDeviceBuffers(molSystemHost,
                                         molSystemDevice,
                                         ctx.systemHost.positions,
                                         ctx.systemHost.atomStarts.size() - 1);
}

void FirstMinimizeStage::execute(ETKDGContext& ctx) {
  nvMolKit::DistGeom::DistGeomMinimizeBFGS(molSystemHost,
                                           molSystemDevice,
                                           ctx,
                                           400,
                                           embedParam_.optimizerForceTol,
                                           true,
                                           stream_);

  nvMolKit::DistGeom::allocateIntermediateBuffers(molSystemHost, molSystemDevice);
  nvMolKit::DistGeom::computeEnergy(molSystemDevice,
                                    ctx.systemDevice.atomStarts,
                                    ctx.systemDevice.positions,
                                    nullptr,
                                    nullptr,
                                    stream_);
  const int molNum   = molSystemDevice.energyOuts.size();
  const int gridSize = (molNum + kBlockSize - 1) / kBlockSize;

  checkMinimizedEnergiesKernel<<<gridSize, kBlockSize, 0, stream_>>>(molNum,
                                                                     molSystemDevice.energyOuts.data(),
                                                                     ctx.systemDevice.atomStarts.data(),
                                                                     ctx.failedThisStage.data());
}

}  // namespace detail

}  // namespace nvMolKit
