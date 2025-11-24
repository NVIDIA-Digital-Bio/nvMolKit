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
#include "etkdg_stage_distgeom_minimize.h"
#include "forcefields/kernel_utils.cuh"
#include "minimizer/bfgs_distgeom.h"
#include "nvtx.h"

using ::nvMolKit::detail::ETKDGContext;
using ::nvMolKit::detail::ETKDGStage;

namespace nvMolKit {

namespace {
constexpr int kBlockSize = 256;

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

DistGeomMinimizeStage::DistGeomMinimizeStage(
  const std::vector<const RDKit::ROMol*>&                                               mols,
  const std::vector<EmbedArgs>&                                                         eargs,
  const RDKit::DGeomHelpers::EmbedParameters&                                           embedParam,
  ETKDGContext&                                                                         ctx,
  BfgsBatchMinimizer&                                                                   minimizer,
  double                                                                                chiralWeight,
  double                                                                                fourthDimWeight,
  int                                                                                   maxIters,
  bool                                                                                  checkEnergy,
  const std::string&                                                                    stageName,
  cudaStream_t                                                                          stream,
  std::unordered_map<const RDKit::ROMol*, nvMolKit::DistGeom::EnergyForceContribsHost>* cache)
    : embedParam_(embedParam),
      minimizer_(minimizer),
      chiralWeight_(chiralWeight),
      fourthDimWeight_(fourthDimWeight),
      maxIters_(maxIters),
      checkEnergy_(checkEnergy),
      stageName_(stageName),
      stream_(stream) {
  // Check that all vectors have the same size
  if (mols.size() != eargs.size()) {
    throw std::runtime_error("Number of molecules and embed args must be the same");
  }

  // Preallocate memory based on first molecule (if available)
  bool preallocated = false;

  // Process each molecule
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto& mol      = mols[i];
    const auto& embedArg = eargs[i];
    const auto& numAtoms = mol->getNumAtoms();

    // Get or construct force field parameters
    const nvMolKit::DistGeom::EnergyForceContribsHost* ffParams = nullptr;
    nvMolKit::DistGeom::EnergyForceContribsHost        uncachedParams;

    if (cache != nullptr) {
      auto it = cache->find(mol);
      if (it != cache->end()) {
        ffParams = &it->second;
      } else {
        // Construct directly into cache
        auto [fst, snd] = cache->emplace(
          mol,
          DistGeom::constructForceFieldContribs(embedArg.dim,
                                                *embedArg.mmat,
                                                embedArg.chiralCenters,
                                                1.0,  // Default weight (actual weights passed to executeImpl)
                                                0.1,  // Default weight (actual weights passed to executeImpl)
                                                nullptr,
                                                embedParam.basinThresh));
        ffParams = &fst->second;
      }
    } else {
      // No cache, construct locally
      uncachedParams =
        DistGeom::constructForceFieldContribs(embedArg.dim,
                                              *embedArg.mmat,
                                              embedArg.chiralCenters,
                                              1.0,  // Default weight (actual weights passed to executeImpl)
                                              0.1,  // Default weight (actual weights passed to executeImpl)
                                              nullptr,
                                              embedParam.basinThresh);
      ffParams = &uncachedParams;
    }

    // Preallocate once using the first molecule's parameters
    if (!preallocated) {
      nvMolKit::DistGeom::preallocateEstimatedBatch(*ffParams, molSystemHost, static_cast<int>(mols.size()));
      preallocated = true;
    }

    // Add to molecular system
    nvMolKit::DistGeom::addMoleculeToMolecularSystem(*ffParams,
                                                     numAtoms,
                                                     embedArg.dim,
                                                     ctx.systemHost.atomStarts,
                                                     molSystemHost);
  }
  ScopedNvtxRange buffersRange("Setup buffers");
  DistGeom::setStreams(molSystemDevice, stream_);
  nvMolKit::DistGeom::sendContribsAndIndicesToDevice(molSystemHost, molSystemDevice);
}

void DistGeomMinimizeStage::executeImpl(ETKDGContext& ctx,
                                        double        chiralWeight,
                                        double        fourthDimWeight,
                                        int           maxIters,
                                        bool          checkEnergy) {
  // Setup device buffers for minimization
  DistGeom::setupDeviceBuffers(molSystemHost,
                               molSystemDevice,
                               ctx.systemHost.positions,
                               static_cast<int>(ctx.systemHost.atomStarts.size() - 1));

  // Allocate energy buffer (not used for DG but required by interface)
  AsyncDeviceVector<double> energyBuffer(0, stream_);

  // Use shared minimizer with repeat-until-converged
  auto eFunc = [&](const double* positions) {
    DistGeom::computeEnergy(molSystemDevice,
                            ctx.systemDevice.atomStarts,
                            ctx.systemDevice.positions,
                            chiralWeight,
                            fourthDimWeight,
                            ctx.activeThisStage.data(),
                            positions,
                            stream_);
  };

  auto gFunc = [&]() {
    DistGeom::computeGradients(molSystemDevice,
                               ctx.systemDevice.atomStarts,
                               ctx.systemDevice.positions,

                               chiralWeight,
                               fourthDimWeight,
                               ctx.activeThisStage.data(),
                               stream_);
  };
  bool needsMore = minimizer_.minimize(maxIters,
                                       embedParam_.optimizerForceTol,
                                       ctx.systemHost.atomStarts,
                                       ctx.systemDevice.atomStarts,
                                       ctx.systemDevice.positions,
                                       molSystemDevice.grad,
                                       molSystemDevice.energyOuts,
                                       molSystemDevice.energyBuffer,
                                       eFunc,
                                       gFunc,
                                       ctx.activeThisStage.data());

  // Repeat until converged
  while (needsMore) {
    needsMore = minimizer_.minimize(maxIters,
                                    embedParam_.optimizerForceTol,
                                    ctx.systemHost.atomStarts,
                                    ctx.systemDevice.atomStarts,
                                    ctx.systemDevice.positions,
                                    molSystemDevice.grad,
                                    molSystemDevice.energyOuts,
                                    molSystemDevice.energyBuffer,
                                    eFunc,
                                    gFunc,
                                    ctx.activeThisStage.data());
  }

  // Check energy per atom if requested
  if (checkEnergy) {
    nvMolKit::DistGeom::computeEnergy(molSystemDevice,
                                      ctx.systemDevice.atomStarts,
                                      ctx.systemDevice.positions,
                                      chiralWeight,
                                      fourthDimWeight,
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
}

}  // namespace detail

}  // namespace nvMolKit
