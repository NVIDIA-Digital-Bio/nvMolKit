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

#ifndef NVMOLKIT_CONFORMER_CHECKERS_H
#define NVMOLKIT_CONFORMER_CHECKERS_H

namespace RDKit {
class ROMol;
}

#include <optional>
#include <vector>

namespace nvMolKit {

//! Checks if all conformers in the provided molecules are completed.
//! One of totalFailuresTolerance or failsPerMoleculeTolerance must be provided.
/*!
 * \param mols Vector of pointers to ROMol objects containing conformers.
 * \param numConfsExpected Expected number of conformers per molecule.
 * \param totalFailuresTolerance Maximum allowed total failures across all molecules.
 * \param failsPerMoleculeTolerance Maximum allowed failures per individual molecule.
 * \param acceptEitherMetricAsPass If true, passing either total or per-molecule metric is sufficient.
 * \param printResults If true, prints the results of the checks.
 * \return True if all checks pass, false otherwise.
 */
bool checkForCompletedConformers(const std::vector<const RDKit::ROMol*>& mols,
                                 int                                     numConfsExpected,
                                 std::optional<int>                      totalFailuresTolerance    = std::nullopt,
                                 std::optional<int>                      failsPerMoleculeTolerance = std::nullopt,
                                 bool                                    acceptEitherMetricAsPass  = false,
                                 bool                                    printResults              = false);

//! \overload
bool checkForCompletedConformers(const std::vector<RDKit::ROMol*>& mols,
                                 int                               numConfsExpected,
                                 std::optional<int>                totalFailuresTolerance    = std::nullopt,
                                 std::optional<int>                failsPerMoleculeTolerance = std::nullopt,
                                 bool                              acceptEitherMetricAsPass  = false,
                                 bool                              printResults              = false);
}  // namespace nvMolKit

#endif  // NVMOLKIT_CONFORMER_CHECKERS_H
