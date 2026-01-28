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

#include <GraphMol/QueryAtom.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include "cuda_error_check.h"
#include "flat_bit_vect.h"
#include "graph_labeler.cuh"
#include "molecules_device.cuh"
#include "substruct_validation.h"

namespace nvMolKit {

using LabelMatrixStorage = FlatBitVect<kMaxTargetAtoms * kMaxQueryAtoms>;
using LabelMatrixView    = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

namespace {
template <std::size_t MaxTarget, std::size_t MaxQuery>
__global__ void populateLabelMatrixKernel(TargetMoleculesDeviceView          targetBatch,
                                          int                                targetMolIdx,
                                          QueryMoleculesDeviceView           queryBatch,
                                          int                                queryMolIdx,
                                          FlatBitVect<MaxTarget * MaxQuery>* matrix,
                                          const uint32_t*                    pairRecursiveBits) {
  TargetMoleculeView                   target = getMolecule(targetBatch, targetMolIdx);
  QueryMoleculeView                    query  = getMolecule(queryBatch, queryMolIdx);
  BitMatrix2DView<MaxTarget, MaxQuery> view(matrix);
  populateLabelMatrix<MaxTarget, MaxQuery>(target, query, view, pairRecursiveBits);
}
}  // namespace

std::vector<std::vector<uint8_t>> computeRDKitLabelMatrix(const RDKit::ROMol& targetMol, const RDKit::ROMol& queryMol) {
  const int numTargetAtoms = static_cast<int>(targetMol.getNumAtoms());
  const int numQueryAtoms  = static_cast<int>(queryMol.getNumAtoms());

  std::vector<std::vector<uint8_t>> result(numTargetAtoms, std::vector<uint8_t>(numQueryAtoms, 0));

  RDKit::MatchVectType            match;
  RDKit::SubstructMatchParameters params;
  params.uniquify   = false;
  params.maxMatches = 0;  // Find all matches

  const auto matches = RDKit::SubstructMatch(targetMol, queryMol, params);

  for (const auto& matchVec : matches) {
    for (const auto& pair : matchVec) {
      const int qa = pair.first;
      const int ta = pair.second;
      if (ta >= 0 && ta < numTargetAtoms && qa >= 0 && qa < numQueryAtoms) {
        result[ta][qa] = 1;
      }
    }
  }

  return result;
}

std::vector<std::vector<uint8_t>> computeGpuLabelMatrix(const RDKit::ROMol& targetMol,
                                                        const RDKit::ROMol& queryMol,
                                                        cudaStream_t        stream) {
  MoleculesHost targetHost;
  MoleculesHost queryHost;
  addToBatch(&targetMol, targetHost);
  addQueryToBatch(&queryMol, queryHost);

  MoleculesDevice targetDevice(stream);
  MoleculesDevice queryDevice(stream);
  targetDevice.copyFromHost(targetHost);
  queryDevice.copyFromHost(queryHost);

  const int numTargetAtoms = static_cast<int>(targetHost.totalAtoms());
  const int numQueryAtoms  = static_cast<int>(queryHost.totalAtoms());

  std::vector<int> queryAtomCounts          = {numQueryAtoms};
  std::vector<int> miniBatchPairMatchStarts = {0, 0};

  RecursivePatternInfo info = extractRecursivePatterns(&queryMol);

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream);
  const LabelMatrixStorage              hostMatrix(false);
  matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

  populateLabelMatrixKernel<kMaxTargetAtoms, kMaxQueryAtoms>
    <<<1, 128, 0, stream>>>(targetDevice.view<MoleculeType::Target>(),
                            0,
                            queryDevice.view<MoleculeType::Query>(),
                            0,
                            matrixDev.data(),
                            /*pairRecursiveBits=*/nullptr);
  cudaCheckError(cudaGetLastError());

  std::vector<LabelMatrixStorage> resultMatrix(1);
  matrixDev.copyToHost(resultMatrix);
  cudaCheckError(cudaStreamSynchronize(stream));

  const LabelMatrixView view(resultMatrix[0]);

  std::vector<std::vector<uint8_t>> result(numTargetAtoms, std::vector<uint8_t>(numQueryAtoms));
  for (int ta = 0; ta < numTargetAtoms; ++ta) {
    for (int qa = 0; qa < numQueryAtoms; ++qa) {
      result[ta][qa] = view.get(ta, qa) ? 1 : 0;
    }
  }

  return result;
}

LabelMatrixComparisonResult compareLabelMatrices(const RDKit::ROMol& targetMol,
                                                 const RDKit::ROMol& queryMol,
                                                 cudaStream_t        stream) {
  auto gpuMatrix   = computeGpuLabelMatrix(targetMol, queryMol, stream);
  auto rdkitMatrix = computeRDKitLabelMatrix(targetMol, queryMol);

  LabelMatrixComparisonResult result;
  result.numTargetAtoms   = static_cast<int>(gpuMatrix.size());
  result.numQueryAtoms    = result.numTargetAtoms > 0 ? static_cast<int>(gpuMatrix[0].size()) : 0;
  result.totalComparisons = result.numTargetAtoms * result.numQueryAtoms;

  for (int ta = 0; ta < result.numTargetAtoms; ++ta) {
    for (int qa = 0; qa < result.numQueryAtoms; ++qa) {
      bool gpuResult   = gpuMatrix[ta][qa] != 0;
      bool rdkitResult = rdkitMatrix[ta][qa] != 0;

      if (gpuResult != rdkitResult) {
        if (gpuResult && !rdkitResult) {
          ++result.falsePositives;
        } else {
          ++result.falseNegatives;
        }
        result.mismatches.emplace_back(ta, qa, gpuResult, rdkitResult);
      }
    }
  }

  result.allMatch = (result.falsePositives == 0 && result.falseNegatives == 0);
  return result;
}

}  // namespace nvMolKit
