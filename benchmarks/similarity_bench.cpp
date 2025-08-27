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

#include <DataStructs/BitOps.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <nanobench.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mol_data.h"
#include "similarity.h"
#include "similarity_cpu.h"

using namespace RDKit;
using namespace nvMolKit;

void doRdkitSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& explicitVects,
                       const std::string&                                   similarityType) {
  for (const auto& bitVec : explicitVects) {
    if (similarityType == "Tanimoto") {
      ankerl::nanobench::doNotOptimizeAway(TanimotoSimilarity(*explicitVects[0], *bitVec));
    } else if (similarityType == "Cosine") {
      ankerl::nanobench::doNotOptimizeAway(CosineSimilarity(*explicitVects[0], *bitVec));
    }
  }
}

void doRdkitParallelSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& explicitVects,
                               const std::string&                                   similarityType) {
  if (similarityType == "Tanimoto") {
    ankerl::nanobench::doNotOptimizeAway(bulkTanimotoSimilarityCpu(*explicitVects[0], explicitVects));
  } else if (similarityType == "Cosine") {
    ankerl::nanobench::doNotOptimizeAway(bulkCosineSimilarityCpu(*explicitVects[0], explicitVects));
  }
}

void doNvMolKitSimilarity(const std::vector<std::unique_ptr<ExplicitBitVect>>& explicitVects,
                          const std::string&                                   similarityType,
                          const BulkFingerprintOptions&                        options) {
  if (similarityType == "Tanimoto") {
    ankerl::nanobench::doNotOptimizeAway(bulkTanimotoSimilarity(*explicitVects[0], explicitVects, options));
  } else if (similarityType == "Cosine") {
    ankerl::nanobench::doNotOptimizeAway(bulkCosineSimilarity(*explicitVects[0], explicitVects, options));
  }
}

std::string parseSimilarityType(const char* arg) {
  std::string similarityType = std::string(arg);
  if (similarityType == "Tanimoto" || similarityType == "tanimoto") {
    return "Tanimoto";
  } else if (similarityType == "Cosine" || similarityType == "cosine") {
    return "Cosine";
  } else {
    throw std::invalid_argument("Unsupported similarity type. Please use Tanimoto or Cosine.");
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    throw std::invalid_argument("Please provide a similarity type as an argument.");
  }
  std::string similarityType = parseSimilarityType(argv[1]);

  constexpr int kNumBits  = 1024;
  const auto [mols, smis] = nvMolKit::testing::loadNChemblMolecules(1000);

  constexpr std::array<int, 5>                  molCounts = {1, 10, 100, 1000, 10000};
  std::vector<std::unique_ptr<ExplicitBitVect>> explicitVectsRef;
  printf("Loading %zu smis\n", smis.size());
  for (const auto& mol : mols) {
    assert(mol != nullptr);
    explicitVectsRef.emplace_back(MorganFingerprints::getFingerprintAsBitVect(*mol,
                                                                              /*radius=*/3,
                                                                              /*nBits=*/kNumBits,
                                                                              /*invariants=*/nullptr,
                                                                              /*fromAtoms=*/nullptr,
                                                                              /*useChirality=*/false,
                                                                              /*useBondTypes=*/true,
                                                                              /*onlyNonzeroInvariants=*/false,
                                                                              /*atomsSettingBits=*/nullptr,
                                                                              /*includeRedundantEnvironments=*/false));
  }

  for (const int count : molCounts) {
    std::vector<std::unique_ptr<ExplicitBitVect>> explicitVects;
    for (int i = 0; i < count; i++) {
      const int   refid   = i % explicitVectsRef.size();
      const auto& refvect = *explicitVectsRef[refid];
      explicitVects.push_back(std::make_unique<ExplicitBitVect>(refvect));
    }
    const std::string rdkitTitlePar =
      "RDKit PARALLEL 1024 bit " + similarityType + " similarity with " + std::to_string(count) + " molecules";

    const std::string rdkitTitle =
      "RDKit 1024 bit " + similarityType + " similarity with " + std::to_string(count) + " molecules";
    const std::string nvMolKitTitle =
      "nvMolKit 1024 bit " + similarityType + " similarity with " + std::to_string(count) + " molecules";
    const std::string nvMolKitTitle2 =
      "nvMolKit GPU ONLY 1024 bit " + similarityType + " similarity with " + std::to_string(count) + " molecules";
    const std::string nvMolKitTitlebatch1 =
      "nvMolKit 1024 bit 1024 batch " + similarityType + " similarity with " + std::to_string(count) + " molecules";
    const std::string nvMolKitTitlebatch2 =
      "nvMolKit 1024 bit 1024*8 batch " + similarityType + " similarity with " + std::to_string(count) + " molecules";
    const std::string nvMolKitTitlebatch3 =
      "nvMolKit 1024 bit 512 batch " + similarityType + " similarity with " + std::to_string(count) + " molecules";

    BulkFingerprintOptions options;

    ankerl::nanobench::Bench().run(rdkitTitle, [&] { doRdkitSimilarity(explicitVects, similarityType); });
    ankerl::nanobench::Bench().run(rdkitTitlePar, [&] { doRdkitParallelSimilarity(explicitVects, similarityType); });

    ankerl::nanobench::Bench().warmup(1).run(nvMolKitTitle,
                                             [&] { doNvMolKitSimilarity(explicitVects, similarityType, options); });
    options.batchSize = 1024;

    for (int batchSizeMultiplier = 1; batchSizeMultiplier <= 64; batchSizeMultiplier *= 2) {
      options.batchSize                      = 1024 * batchSizeMultiplier;
      const std::string nvMolKitRunningTitle = "nvMolKit 1024 bit batch 1024 * " + std::to_string(batchSizeMultiplier) +
                                               " batch " + similarityType + " similarity with " +
                                               std::to_string(count) + " molecules";
      ankerl::nanobench::Bench().warmup(1).run(nvMolKitRunningTitle,
                                               [&] { doNvMolKitSimilarity(explicitVects, similarityType, options); });
    }

    ankerl::nanobench::Bench().warmup(1).run(nvMolKitTitlebatch1,
                                             [&] { doNvMolKitSimilarity(explicitVects, similarityType, options); });
    options.batchSize = 1024 * 8;
    ankerl::nanobench::Bench().warmup(1).run(nvMolKitTitlebatch2,
                                             [&] { doNvMolKitSimilarity(explicitVects, similarityType, options); });
    options.batchSize = 512;
    ankerl::nanobench::Bench().warmup(1).run(nvMolKitTitlebatch3,
                                             [&] { doNvMolKitSimilarity(explicitVects, similarityType, options); });
  }

  return 0;
}
