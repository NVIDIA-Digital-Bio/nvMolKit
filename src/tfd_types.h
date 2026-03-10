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

#ifndef NVMOLKIT_TFD_TYPES_H
#define NVMOLKIT_TFD_TYPES_H

#include <cstdint>

namespace nvMolKit {

//! Torsion type for multi-quartet handling
enum class TorsionType : uint8_t {
  Single,     //!< 1 quartet: direct circularDifference
  Ring,       //!< N quartets: average abs(signed), compare averages
  Symmetric,  //!< N quartets: min circularDiff across all (qi,qj) pairs
};

//! Per-molecule descriptor for GPU kernel dispatch.
//! Replaces per-work-item index arrays with compact per-molecule metadata.
//! Kernels use binary search on cumulative work-item counts to find the molecule,
//! then compute local indices arithmetically.
struct MolDescriptor {
  int confStart;      //!< First conformer index (into confPositionStarts)
  int numConformers;  //!< Number of conformers for this molecule
  int quartetStart;   //!< First quartet index (into torsionAtoms)
  int numQuartets;    //!< Total quartets for this molecule
  int dihedStart;     //!< Offset into dihedralAngles output
  int torsStart;      //!< First torsion index (into weights/maxDevs/types/quartetStarts)
  int numTorsions;    //!< Number of torsions
  int tfdOutStart;    //!< Offset into tfdOutput
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_TYPES_H
