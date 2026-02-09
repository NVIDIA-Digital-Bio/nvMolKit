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

#ifndef NVMOLKIT_TFD_CPU_H
#define NVMOLKIT_TFD_CPU_H

#include <vector>

#include "tfd_common.h"

namespace nvMolKit {

//! CPU implementation of TFD computation
class TFDCpuGenerator {
 public:
  TFDCpuGenerator() = default;

  //! Compute TFD matrix for a single molecule
  //! @param mol Molecule with conformers
  //! @param options Computation options
  //! @return Lower triangular TFD matrix as flat vector [C*(C-1)/2 values]
  std::vector<double> GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options = TFDComputeOptions{});

  //! Compute TFD matrices for multiple molecules
  //! @param mols Vector of molecules
  //! @param options Computation options
  //! @return Vector of TFD matrices, one per molecule
  std::vector<std::vector<double>> GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                  const TFDComputeOptions& options = TFDComputeOptions{});

  //! Compute dihedral angles for all conformers of a molecule
  //! @param system Prepared TFD system data
  //! @param molIdx Index of molecule in batch (0 for single molecule)
  //! @return Angles array [numConformers][numTorsions], flattened
  std::vector<float> computeDihedralAngles(const TFDSystemHost& system, int molIdx = 0);

  //! Compute TFD matrix from precomputed angles
  //! @param angles Dihedral angles [numConformers][numTorsions]
  //! @param weights Torsion weights
  //! @param maxDevs Maximum deviations for normalization
  //! @param numConformers Number of conformers
  //! @param numTorsions Number of torsions
  //! @return Lower triangular TFD matrix
  std::vector<double> computeTFDMatrixFromAngles(const std::vector<float>& angles,
                                                 const std::vector<float>& weights,
                                                 const std::vector<float>& maxDevs,
                                                 int                       numConformers,
                                                 int                       numTorsions);

 private:
  //! Compute TFD between two conformers
  //! @param angles1 Angles for conformer 1
  //! @param angles2 Angles for conformer 2
  //! @param weights Torsion weights
  //! @param maxDevs Maximum deviations
  //! @param numTorsions Number of torsions
  //! @return TFD value
  static double computeTFD(const float* angles1,
                           const float* angles2,
                           const float* weights,
                           const float* maxDevs,
                           int          numTorsions);
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_CPU_H
