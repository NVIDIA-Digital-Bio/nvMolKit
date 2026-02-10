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

double TFDCpuGenerator::computeTFDPair(const float*         anglesI,
                                       const float*         anglesJ,
                                       const TFDSystemHost& system,
                                       int                  molIdx) {
  int torsStart   = system.molTorsionStarts[molIdx];
  int torsEnd     = system.molTorsionStarts[molIdx + 1];
  int numTorsions = torsEnd - torsStart;
  int molQBase    = system.quartetStarts[torsStart];

  double sumWeightedDev = 0.0;
  double sumWeights     = 0.0;

  for (int t = 0; t < numTorsions; ++t) {
    int         globalT = torsStart + t;
    int         qLocal  = system.quartetStarts[globalT] - molQBase;
    int         numQ    = system.quartetStarts[globalT + 1] - system.quartetStarts[globalT];
    TorsionType type    = system.torsionTypes[globalT];

    double deviation;
    if (type == TorsionType::Single) {
      float diff = detail::circularDifference(anglesI[qLocal], anglesJ[qLocal]);
      deviation  = diff / system.torsionMaxDevs[globalT];
    } else if (type == TorsionType::Ring) {
      // Average abs(signed dihedral) for each conformer, then compare averages
      double avgI = 0.0;
      double avgJ = 0.0;
      for (int q = 0; q < numQ; ++q) {
        float ai = anglesI[qLocal + q];
        float aj = anglesJ[qLocal + q];
        // Convert [0,360) to abs(signed) = min(angle, 360 - angle) giving [0,180]
        avgI += std::min(static_cast<double>(ai), 360.0 - ai);
        avgJ += std::min(static_cast<double>(aj), 360.0 - aj);
      }
      avgI /= numQ;
      avgJ /= numQ;
      deviation = std::abs(avgI - avgJ) / system.torsionMaxDevs[globalT];
    } else {
      // Symmetric: minimum circular difference across all (qi, qj) cross-product pairs
      double minDiff = 180.0;
      for (int qi = 0; qi < numQ; ++qi) {
        for (int qj = 0; qj < numQ; ++qj) {
          float diff = detail::circularDifference(anglesI[qLocal + qi], anglesJ[qLocal + qj]);
          minDiff    = std::min(minDiff, static_cast<double>(diff));
        }
      }
      deviation = minDiff / system.torsionMaxDevs[globalT];
    }

    sumWeightedDev += deviation * system.torsionWeights[globalT];
    sumWeights += system.torsionWeights[globalT];
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
  int molQStart   = system.quartetStarts[torsStart];
  int molQEnd     = system.quartetStarts[torsEnd];
  int numQuartets = molQEnd - molQStart;

  std::vector<float> angles(numConformers * numQuartets);

// Parallel over conformers
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int c = 0; c < numConformers; ++c) {
    int globalConfIdx = confStart + c;

    for (int q = 0; q < numQuartets; ++q) {
      const auto& quartet = system.torsionAtoms[molQStart + q];

      // Get position base for this conformer (tightly packed)
      const float* posBase = system.positions.data() + system.confPositionStarts[globalConfIdx];

      float angle                 = detail::computeDihedralAngle(posBase + quartet[0] * 3,
                                                 posBase + quartet[1] * 3,
                                                 posBase + quartet[2] * 3,
                                                 posBase + quartet[3] * 3);
      angles[c * numQuartets + q] = angle;
    }
  }

  return angles;
}

std::vector<double> TFDCpuGenerator::computeTFDMatrixFromAngles(const TFDSystemHost&      system,
                                                                int                       molIdx,
                                                                const std::vector<float>& angles) {
  int numConformers = system.molConformerStarts[molIdx + 1] - system.molConformerStarts[molIdx];
  int torsStart     = system.molTorsionStarts[molIdx];
  int torsEnd       = system.molTorsionStarts[molIdx + 1];
  int molQStart     = system.quartetStarts[torsStart];
  int molQEnd       = system.quartetStarts[torsEnd];
  int numQuartets   = molQEnd - molQStart;

  int                 numPairs = numConformers * (numConformers - 1) / 2;
  std::vector<double> tfdMatrix(numPairs);

// Parallel over conformer pairs
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 1; i < numConformers; ++i) {
    for (int j = 0; j < i; ++j) {
      int pairIdx = i * (i - 1) / 2 + j;  // Lower triangular index

      double tfd = computeTFDPair(angles.data() + i * numQuartets, angles.data() + j * numQuartets, system, molIdx);

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

  // Compute TFD matrix
  return computeTFDMatrixFromAngles(system, 0, angles);
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
