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
#include "etkdg_stage_fourthdimminimization.h"
#include "forcefields/kernel_utils.cuh"
#include "minimizer/bfgs_distgeom.h"

using ::nvMolKit::detail::ETKDGContext;
using ::nvMolKit::detail::ETKDGStage;

namespace nvMolKit {

namespace detail {
FourthDimMinimizeStage::FourthDimMinimizeStage(const std::vector<const RDKit::ROMol*>&     mols,
                                               const std::vector<EmbedArgs>&               eargs,
                                               const RDKit::DGeomHelpers::EmbedParameters& embedParam,
                                               ETKDGContext&                               ctx,
                                               const cudaStream_t                          stream)
    : embedParam_(embedParam),
      stream_(stream) {
  if (mols.size() != eargs.size()) {
    throw std::runtime_error("Number of molecules and embed args must be the same");
  }
  setStreams(molSystemDevice, stream_);

  // Process each molecule
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto&      mol      = mols[i];
    const auto&      embedArg = eargs[i];
    const auto&      numAtoms = mol->getNumAtoms();
    auto             ffParams = nvMolKit::DistGeom::constructForceFieldContribs(embedArg.dim,
                                                                    *embedArg.mmat,
                                                                    embedArg.chiralCenters,
                                                                    0.2,
                                                                    1.0,
                                                                    nullptr,
                                                                    embedParam_.basinThresh);
    // Get atom numbers
    std::vector<int> atomNumbers;
    atomNumbers.reserve(numAtoms);
    for (const auto& atom : mol->atoms()) {
      atomNumbers.push_back(atom->getAtomicNum());
    }

    // Add to molecular system
    nvMolKit::DistGeom::addMoleculeToMolecularSystem(ffParams,
                                                     numAtoms,
                                                     embedArg.dim,
                                                     ctx.systemHost.atomStarts,
                                                     molSystemHost,
                                                     &atomNumbers);
  }
  nvMolKit::DistGeom::sendContribsAndIndicesToDevice(molSystemHost, molSystemDevice);
  nvMolKit::DistGeom::setupDeviceBuffers(molSystemHost,
                                         molSystemDevice,
                                         ctx.systemHost.positions,
                                         ctx.systemHost.atomStarts.size() - 1);
}

void FourthDimMinimizeStage::execute([[maybe_unused]] ETKDGContext& ctx) {
  nvMolKit::DistGeom::DistGeomMinimizeBFGS(molSystemHost,
                                           molSystemDevice,
                                           ctx,
                                           200,
                                           embedParam_.optimizerForceTol,
                                           true,
                                           stream_);
}

}  // namespace detail
}  // namespace nvMolKit
