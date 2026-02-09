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

#ifndef NVMOLKIT_TFD_KERNELS_H
#define NVMOLKIT_TFD_KERNELS_H

#include <cuda_runtime.h>

namespace nvMolKit {

//! Block size for TFD kernels
constexpr int kTFDBlockSize = 256;

//! Parameters for TFD kernel launch (POD type for easy device copy)
struct TFDKernelParams {
  //! Total number of dihedral work items (one per conformer-torsion pair)
  int totalDihedralWorkItems;
  //! Total number of TFD pair work items (one per conformer pair)
  int totalTFDWorkItems;
};

//! Launch kernel to compute dihedral angles for all conformers
//! One thread per (conformer, torsion) work item using flattened indexing
//! @param params Kernel parameters
//! @param positions Tightly packed coordinates (no padding)
//! @param confPositionStarts Position offset per conformer [totalConformers]
//! @param torsionAtoms Torsion atom indices [totalTorsions * 4]
//! @param dihedralConfIdx Global conformer index per work item [totalDihedralWorkItems]
//! @param dihedralTorsIdx Global torsion index per work item [totalDihedralWorkItems]
//! @param dihedralOutIdx Output index per work item [totalDihedralWorkItems]
//! @param dihedralAngles Output: angles in degrees [totalDihedrals]
//! @param stream CUDA stream
void launchDihedralKernel(const TFDKernelParams& params,
                          const float*           positions,
                          const int*             confPositionStarts,
                          const int*             torsionAtoms,
                          const int*             dihedralConfIdx,
                          const int*             dihedralTorsIdx,
                          const int*             dihedralOutIdx,
                          float*                 dihedralAngles,
                          cudaStream_t           stream);

//! Launch kernel to compute TFD matrix for all conformer pairs
//! One thread per conformer pair using flattened indexing
//! @param params Kernel parameters
//! @param dihedralAngles Computed dihedral angles [totalDihedrals]
//! @param torsionWeights Weights per torsion [totalTorsions]
//! @param torsionMaxDevs Max deviation per torsion [totalTorsions]
//! @param tfdAnglesI Offset into dihedralAngles for conformer i [totalTFDWorkItems]
//! @param tfdAnglesJ Offset into dihedralAngles for conformer j [totalTFDWorkItems]
//! @param tfdTorsStart Global torsion start per work item [totalTFDWorkItems]
//! @param tfdNumTorsions Number of torsions per work item [totalTFDWorkItems]
//! @param tfdOutIdx Output index per work item [totalTFDWorkItems]
//! @param tfdOutput Output: TFD matrix values
//! @param stream CUDA stream
void launchTFDMatrixKernel(const TFDKernelParams& params,
                           const float*           dihedralAngles,
                           const float*           torsionWeights,
                           const float*           torsionMaxDevs,
                           const int*             tfdAnglesI,
                           const int*             tfdAnglesJ,
                           const int*             tfdTorsStart,
                           const int*             tfdNumTorsions,
                           const int*             tfdOutIdx,
                           float*                 tfdOutput,
                           cudaStream_t           stream);

}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_KERNELS_H
