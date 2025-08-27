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

#ifndef NVMOLKIT_BFGS_MMFF_H
#define NVMOLKIT_BFGS_MMFF_H

#include <vector>

#include "../hardware_options.h"

namespace RDKit {
class ROMol;
}

namespace nvMolKit::MMFF {

//! \brief Optimize conformers for multiple molecules using MMFF force field
//! \param mols The molecules to optimize
//! \param maxIters The maximum number of iterations to perform
//! \param nonBondedThreshold The radius threshold for non-bonded interactions
//! \param perfOptions Performance tuning options (threading, batching)
//! \return A vector of vectors of energies, where each inner vector contains energies for conformers of one molecule
std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                                int                         maxIters           = 200,
                                                                double                      nonBondedThreshold = 100.0,
                                                                const BatchHardwareOptions& perfOptions        = {});

}  // namespace nvMolKit::MMFF
#endif  // NVMOLKIT_BFGS_MMFF_H
