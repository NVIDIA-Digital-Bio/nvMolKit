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

#ifndef NVMOLKIT_MOLECULES_DEVICE_CUH
#define NVMOLKIT_MOLECULES_DEVICE_CUH

#include "atom_data_packed.h"
#include "boolean_tree.cuh"
#include "molecules.h"
#include "packed_bonds.h"

namespace nvMolKit {

/**
 * @brief Per-molecule view with pre-shifted pointers for device access.
 *
 * This is a lightweight view into a single molecule's data, with all pointers
 * already offset to this molecule's portion of the batch. Constructed via
 * getMolecule() from a MoleculesDeviceView.
 */
struct MoleculeView {
  int numAtoms;

  // GPU-optimized packed data
  const AtomDataPacked* __restrict__ atomDataPacked;    ///< Packed atom properties for GPU matching
  const AtomQueryMask* __restrict__ atomQueryMasks;     ///< Precomputed query masks (query molecules only)
  const BondTypeCounts* __restrict__ bondTypeCounts;    ///< Precomputed bond type counts per atom
  const TargetAtomBonds* __restrict__ targetAtomBonds;  ///< Packed bond adjacency (targets only)
  const QueryAtomBonds* __restrict__ queryAtomBonds;    ///< Packed bond adjacency (queries only)

  // Boolean expression tree data for compound queries
  const AtomQueryTree* __restrict__ atomQueryTrees;        ///< Tree metadata per atom
  const BoolInstruction* __restrict__ queryInstructions;   ///< All instructions (use atomInstrStarts to index)
  const AtomQueryMask* __restrict__ queryLeafMasks;        ///< All leaf masks (use atomLeafMaskStarts)
  const BondTypeCounts* __restrict__ queryLeafBondCounts;  ///< All leaf bond counts
  const int* __restrict__ atomInstrStarts;                 ///< Start index into queryInstructions per atom
  const int* __restrict__ atomLeafMaskStarts;              ///< Start index into queryLeafMasks per atom

  __device__ __forceinline__ int getAtomDegree(int atomIdx) const {
    if (targetAtomBonds) {
      return targetAtomBonds[atomIdx].degree;
    }
    return queryAtomBonds[atomIdx].degree;
  }

  /// Get packed atom data for GPU matching
  __device__ __forceinline__ const AtomDataPacked& getAtomPacked(int atomIdx) const { return atomDataPacked[atomIdx]; }

  /// Get precomputed query mask (only valid for query molecules)
  __device__ __forceinline__ const AtomQueryMask& getQueryMask(int atomIdx) const { return atomQueryMasks[atomIdx]; }

  /// Get precomputed bond type counts
  __device__ __forceinline__ const BondTypeCounts& getBondTypeCounts(int atomIdx) const {
    return bondTypeCounts[atomIdx];
  }

  /// Get packed target atom bonds (only valid for target molecules)
  __device__ __forceinline__ const TargetAtomBonds& getTargetBonds(int atomIdx) const {
    return targetAtomBonds[atomIdx];
  }

  /// Get packed query atom bonds (only valid for query molecules)
  __device__ __forceinline__ const QueryAtomBonds& getQueryBonds(int atomIdx) const { return queryAtomBonds[atomIdx]; }

  /// Get query tree metadata for compound queries (only valid for query molecules)
  __device__ __forceinline__ const AtomQueryTree& getQueryTree(int atomIdx) const { return atomQueryTrees[atomIdx]; }

  /// Get pointer to instructions for a query atom's boolean expression
  __device__ __forceinline__ const BoolInstruction* getQueryInstructions(int atomIdx) const {
    return queryInstructions + atomInstrStarts[atomIdx];
  }

  /// Get pointer to leaf masks for a query atom's boolean expression
  __device__ __forceinline__ const AtomQueryMask* getQueryLeafMasks(int atomIdx) const {
    return queryLeafMasks + atomLeafMaskStarts[atomIdx];
  }

  /// Get pointer to leaf bond counts for a query atom's boolean expression
  __device__ __forceinline__ const BondTypeCounts* getQueryLeafBondCounts(int atomIdx) const {
    return queryLeafBondCounts + atomLeafMaskStarts[atomIdx];
  }

  /// Check if this molecule has boolean query trees populated
  __device__ __forceinline__ bool hasQueryTrees() const { return atomQueryTrees != nullptr; }
};

/**
 * @brief Get a per-molecule view with pre-shifted pointers from a batch view.
 * @param view The batch-level view
 * @param molIdx Index of the molecule in the batch
 * @return MoleculeView with pointers offset to this molecule's data
 */
__device__ __forceinline__ MoleculeView getMolecule(const MoleculesDeviceView& view, int molIdx) {
  MoleculeView mol;
  const int    atomStart = view.batchAtomStarts[molIdx];
  mol.numAtoms           = view.batchAtomStarts[molIdx + 1] - atomStart;

  // GPU-optimized packed data (may be nullptr if not populated)
  mol.atomDataPacked  = view.atomDataPacked ? view.atomDataPacked + atomStart : nullptr;
  mol.atomQueryMasks  = view.atomQueryMasks ? view.atomQueryMasks + atomStart : nullptr;
  mol.bondTypeCounts  = view.bondTypeCounts ? view.bondTypeCounts + atomStart : nullptr;
  mol.targetAtomBonds = view.targetAtomBonds ? view.targetAtomBonds + atomStart : nullptr;
  mol.queryAtomBonds  = view.queryAtomBonds ? view.queryAtomBonds + atomStart : nullptr;

  // Boolean expression tree data (may be nullptr if not populated)
  mol.atomQueryTrees      = view.atomQueryTrees ? view.atomQueryTrees + atomStart : nullptr;
  mol.queryInstructions   = view.queryInstructions;  // global array, indexed via atomInstrStarts
  mol.queryLeafMasks      = view.queryLeafMasks;     // global array, indexed via atomLeafMaskStarts
  mol.queryLeafBondCounts = view.queryLeafBondCounts;
  mol.atomInstrStarts     = view.atomInstrStarts ? view.atomInstrStarts + atomStart : nullptr;
  mol.atomLeafMaskStarts  = view.atomLeafMaskStarts ? view.atomLeafMaskStarts + atomStart : nullptr;
  return mol;
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_MOLECULES_DEVICE_CUH
