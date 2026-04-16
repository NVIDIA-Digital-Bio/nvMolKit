// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_BFGS_UFF_H
#define NVMOLKIT_BFGS_UFF_H

#include <vector>

#include "../hardware_options.h"

namespace RDKit {
class ROMol;
}

namespace nvMolKit::UFF {

std::vector<std::vector<double>> UFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                               int                         maxIters,
                                                               const std::vector<double>&  vdwThresholds,
                                                               const std::vector<bool>&    ignoreInterfragInteractions,
                                                               const BatchHardwareOptions& perfOptions = {});

}  // namespace nvMolKit::UFF

#endif  // NVMOLKIT_BFGS_UFF_H
