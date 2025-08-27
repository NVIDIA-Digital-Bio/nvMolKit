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

#include "ff_utils.h"

#include <ForceField/ForceField.h>
#include <GraphMol/ROMol.h>

#include <vector>

namespace nvMolKit {
void confPosToVect(const RDKit::ROMol& mol, std::vector<double>& positions, int confId) {
  const unsigned int      numAtoms = mol.getNumAtoms();
  const RDKit::Conformer& conf     = mol.getConformer(confId);
  positions.resize(3 * numAtoms);

  // Fill positions vector with conformer coordinates
  for (unsigned int i = 0; i < numAtoms; ++i) {
    positions[3 * i]     = conf.getAtomPos(i).x;
    positions[3 * i + 1] = conf.getAtomPos(i).y;
    positions[3 * i + 2] = conf.getAtomPos(i).z;
  }
}

void confPosToVect(const RDKit::Conformer& conf, std::vector<double>& positions) {
  const unsigned int numAtoms = conf.getNumAtoms();
  positions.resize(3 * numAtoms);

  // Fill positions vector with conformer coordinates
  for (unsigned int i = 0; i < numAtoms; ++i) {
    positions[3 * i]     = conf.getAtomPos(i).x;
    positions[3 * i + 1] = conf.getAtomPos(i).y;
    positions[3 * i + 2] = conf.getAtomPos(i).z;
  }
}

void setFFPosFromConf(RDKit::ROMol& mol, ForceFields::ForceField* forcefield, int confId) {
  RDKit::Conformer& conf = mol.getConformer(confId);

  for (unsigned int i = 0; i < mol.getNumAtoms(); ++i) {
    forcefield->positions().push_back(&(conf.getAtomPos(i)));
  }
}
}  // namespace nvMolKit
