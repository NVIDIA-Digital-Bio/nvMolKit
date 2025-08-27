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

#include <GraphMol/ROMol.h>

#include "etkdg_stage_update_conformers.h"

namespace nvMolKit {
namespace detail {

ETKDGUpdateConformersStage::ETKDGUpdateConformersStage(const std::vector<RDKit::ROMol*>& mols,
                                                       const std::vector<EmbedArgs>&     eargs,
                                                       cudaStream_t                      stream,
                                                       std::mutex*                       conformer_mutex,
                                                       int                               maxConformersPerMol)
    : mols_(mols),
      eargs_(eargs),
      stream_(stream),
      conformer_mutex_(conformer_mutex),
      maxConformersPerMol_(maxConformersPerMol) {}

void ETKDGUpdateConformersStage::execute(ETKDGContext& ctx) {
  // Copy positions from device to host
  std::vector<double> hostPositions;
  hostPositions.resize(ctx.systemDevice.positions.size());
  ctx.systemDevice.positions.copyToHost(hostPositions);

  // Copy active this stage from device to host
  std::vector<uint8_t> hostActiveThisStage;
  hostActiveThisStage.resize(ctx.activeThisStage.size());
  ctx.activeThisStage.copyToHost(hostActiveThisStage);
  cudaStreamSynchronize(stream_);

  // Update each molecule's conformer with the new positions
  for (size_t i = 0; i < mols_.size(); ++i) {
    // Skip if not active this stage
    if (hostActiveThisStage[i] != 1) {
      continue;
    }

    const auto& mol         = mols_[i];
    const int   dim         = eargs_[i].dim;
    const int   startPosIdx = ctx.systemHost.atomStarts[i] * dim;
    const int   nAtoms      = mol->getNumAtoms();

    auto newConf = std::make_unique<RDKit::Conformer>(mol->getNumAtoms());

    for (int j = 0; j < nAtoms; ++j) {
      const int       posIdx = startPosIdx + j * dim;
      RDGeom::Point3D pos(hostPositions[posIdx], hostPositions[posIdx + 1], hostPositions[posIdx + 2]);
      newConf->setAtomPos(j, pos);
    }

    // Thread-safe conformer addition with count checking
    if (conformer_mutex_) {
      std::lock_guard<std::mutex> lock(*conformer_mutex_);
      // Check if molecule already has enough conformers
      if (maxConformersPerMol_ <= 0 || mol->getNumConformers() < static_cast<unsigned int>(maxConformersPerMol_)) {
        mol->addConformer(newConf.release(), /*assignId=*/true);  // true to take ownership
      }
    } else {
      // Without mutex, assume single-threaded and add conformer
      mol->addConformer(newConf.release(), /*assignId=*/true);
    }
    // If conformer wasn't added, it's still a unique_ptr, and will destruct out of scope.
  }
}

}  // namespace detail
}  // namespace nvMolKit
