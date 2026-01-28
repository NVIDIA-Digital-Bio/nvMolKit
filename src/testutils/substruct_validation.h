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

#pragma once

#include <cuda_runtime.h>
#include <GraphMol/ROMol.h>

#include <tuple>
#include <vector>

#include "substruct_types.h"

namespace nvMolKit {

/**
 * @brief Compute the expected label matrix using RDKit atom matching.
 *
 * For each (targetAtom, queryAtom) pair, determines if the query atom's
 * constraints are satisfied by the target atom using RDKit's Match() method.
 *
 * @param targetMol Target molecule
 * @param queryMol Query molecule (typically from SMARTS)
 * @return 2D matrix of compatibility flags indexed [targetAtom][queryAtom]
 */
std::vector<std::vector<uint8_t>> computeRDKitLabelMatrix(const RDKit::ROMol& targetMol, const RDKit::ROMol& queryMol);

/**
 * @brief Result from comparing GPU vs RDKit label matrices.
 */
struct LabelMatrixComparisonResult {
  int  numTargetAtoms   = 0;
  int  numQueryAtoms    = 0;
  int  totalComparisons = 0;
  int  falsePositives   = 0;  ///< GPU says match, RDKit says no
  int  falseNegatives   = 0;  ///< GPU says no match, RDKit says yes
  bool allMatch         = false;

  std::vector<std::tuple<int, int, bool, bool>> mismatches;
};

/**
 * @brief Compute a GPU label matrix for a single target/query pair.
 *
 * Handles recursive SMARTS preprocessing if the query contains recursive patterns.
 *
 * @param targetMol Target molecule
 * @param queryMol Query molecule (typically from SMARTS)
 * @param stream CUDA stream to use
 * @return 2D matrix of compatibility flags indexed [targetAtom][queryAtom]
 */
std::vector<std::vector<uint8_t>> computeGpuLabelMatrix(const RDKit::ROMol& targetMol,
                                                        const RDKit::ROMol& queryMol,
                                                        cudaStream_t        stream);

/**
 * @brief Compare GPU and RDKit label matrices for a single target/query pair.
 *
 * @param targetMol Target molecule
 * @param queryMol Query molecule
 * @param stream CUDA stream to use
 * @return Comparison result with mismatch details
 */
LabelMatrixComparisonResult compareLabelMatrices(const RDKit::ROMol& targetMol,
                                                 const RDKit::ROMol& queryMol,
                                                 cudaStream_t        stream);

}  // namespace nvMolKit
