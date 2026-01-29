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

#ifndef NVMOLKIT_SUBSTRUCT_TYPES_H
#define NVMOLKIT_SUBSTRUCT_TYPES_H

namespace nvMolKit {

//! Maximum size for GPU substruct path
constexpr int kMaxTargetAtoms = 128;
constexpr int kMaxQueryAtoms  = 64;

/**
 * @brief Algorithm choice for substructure matching.
 */
enum class SubstructAlgorithm {
  VF2,  ///< VF2 iterative stack-based DFS
  GSI   ///< GSI-inspired BFS level-by-level join
};

/**
 * @brief Partial match for BFS-style algorithms.
 *
 * @tparam MaxQueryAtoms Maximum query atoms for array sizing
 *
 * Represents a partial mapping from query atoms to target atoms.
 * Stored compactly for queue-based BFS exploration.
 *
 * Complete matches are never stored in the queue, so we only need
 * MaxQueryAtoms - 1 slots (matching atoms 0 through numQueryAtoms-2).
 */
template <std::size_t MaxQueryAtoms = kMaxQueryAtoms> struct PartialMatchT {
  static constexpr std::size_t kMaxQueryAtomsValue = MaxQueryAtoms;
  int8_t mapping[MaxQueryAtoms - 1];  ///< mapping[q] = target atom (only [0..nextQueryAtom-1] valid)
  int8_t nextQueryAtom;               ///< Next query atom to extend (also serves as depth)
};

/// Type alias for max-sized partial match (backward compatibility)
using PartialMatch = PartialMatchT<kMaxQueryAtoms>;
static_assert(sizeof(PartialMatch) == kMaxQueryAtoms, "PartialMatch must be kMaxQueryAtoms bytes");

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_TYPES_H
