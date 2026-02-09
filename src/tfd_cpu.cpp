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

#include "tfd_cpu.h"

#include <stdexcept>

#include "tfd_detail.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nvMolKit {

double TFDCpuGenerator::computeTFD(const float* angles1,
                                   const float* angles2,
                                   const float* weights,
                                   const float* maxDevs,
                                   int          numTorsions) {
  double sumWeightedDev = 0.0;
  double sumWeights     = 0.0;

  for (int t = 0; t < numTorsions; ++t) {
    float diff      = detail::circularDifference(angles1[t], angles2[t]);
    float deviation = diff / maxDevs[t];
    float weight    = weights[t];

    sumWeightedDev += deviation * weight;
    sumWeights += weight;
  }

  if (sumWeights < 1e-10) {
    return 0.0;
  }

  return sumWeightedDev / sumWeights;
}

std::vector<float> TFDCpuGenerator::computeDihedralAngles(const TFDSystemHost& system, int molIdx) {
  if (molIdx >= system.numMolecules()) {
    throw std::out_of_range("Molecule index out of range");
  }

  int confStart     = system.molConformerStarts[molIdx];
  int confEnd       = system.molConformerStarts[molIdx + 1];
  int numConformers = confEnd - confStart;

  int torsStart   = system.molTorsionStarts[molIdx];
  int torsEnd     = system.molTorsionStarts[molIdx + 1];
  int numTorsions = torsEnd - torsStart;

  std::vector<float> angles(numConformers * numTorsions);

// Parallel over conformers
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int c = 0; c < numConformers; ++c) {
    int globalConfIdx = confStart + c;

    for (int t = 0; t < numTorsions; ++t) {
      int         globalTorsIdx = torsStart + t;
      const auto& quartet       = system.torsionAtoms[globalTorsIdx];

      // Get position base for this conformer (tightly packed)
      const float* posBase = system.positions.data() + system.confPositionStarts[globalConfIdx];

      float angle                 = detail::computeDihedralAngle(posBase + quartet[0] * 3,
                                                 posBase + quartet[1] * 3,
                                                 posBase + quartet[2] * 3,
                                                 posBase + quartet[3] * 3);
      angles[c * numTorsions + t] = angle;
    }
  }

  return angles;
}

std::vector<double> TFDCpuGenerator::computeTFDMatrixFromAngles(const std::vector<float>& angles,
                                                                const std::vector<float>& weights,
                                                                const std::vector<float>& maxDevs,
                                                                int                       numConformers,
                                                                int                       numTorsions) {
  int                 numPairs = numConformers * (numConformers - 1) / 2;
  std::vector<double> tfdMatrix(numPairs);

// Parallel over conformer pairs
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 1; i < numConformers; ++i) {
    for (int j = 0; j < i; ++j) {
      int pairIdx = i * (i - 1) / 2 + j;  // Lower triangular index

      double tfd = computeTFD(angles.data() + i * numTorsions,
                              angles.data() + j * numTorsions,
                              weights.data(),
                              maxDevs.data(),
                              numTorsions);

      tfdMatrix[pairIdx] = tfd;
    }
  }

  return tfdMatrix;
}

std::vector<double> TFDCpuGenerator::GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  // Build system data
  TFDSystemHost system = buildTFDSystem(mol, options);

  if (system.numMolecules() == 0) {
    return {};
  }

  int numConformers = system.molConformerStarts[1] - system.molConformerStarts[0];
  if (numConformers < 2) {
    return {};  // Need at least 2 conformers for comparison
  }

  int torsStart   = system.molTorsionStarts[0];
  int torsEnd     = system.molTorsionStarts[1];
  int numTorsions = torsEnd - torsStart;

  if (numTorsions == 0) {
    // No torsions - return zeros
    int numPairs = numConformers * (numConformers - 1) / 2;
    return std::vector<double>(numPairs, 0.0);
  }

  // Compute dihedral angles
  std::vector<float> angles = computeDihedralAngles(system, 0);

  // Extract weights and maxDevs for this molecule
  std::vector<float> weights(system.torsionWeights.begin() + torsStart, system.torsionWeights.begin() + torsEnd);
  std::vector<float> maxDevs(system.torsionMaxDevs.begin() + torsStart, system.torsionMaxDevs.begin() + torsEnd);

  // Compute TFD matrix
  return computeTFDMatrixFromAngles(angles, weights, maxDevs, numConformers, numTorsions);
}

std::vector<std::vector<double>> TFDCpuGenerator::GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                                 const TFDComputeOptions&                options) {
  std::vector<std::vector<double>> results(mols.size());

// Parallelize at molecule level for larger batches
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < mols.size(); ++i) {
    results[i] = GetTFDMatrix(*mols[i], options);
  }

  return results;
}

}  // namespace nvMolKit
