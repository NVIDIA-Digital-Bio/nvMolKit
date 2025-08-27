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

#include "similarity_cpu.h"

#include <DataStructs/BitOps.h>
#include <DataStructs/ExplicitBitVect.h>

#include <vector>
namespace nvMolKit {

// --------------------------------
// Tanimoto similarity wrapper functions
// --------------------------------

std::vector<double> bulkTanimotoSimilarityCpu(const ExplicitBitVect&                     bitsOne,
                                              const std::vector<const ExplicitBitVect*>& bitsTwo) {
  std::vector<double> similarities(bitsTwo.size());
#pragma omp parallel for default(none) shared(bitsOne, bitsTwo, similarities)
  for (size_t i = 0; i < bitsTwo.size(); ++i) {
    similarities[i] = TanimotoSimilarity(bitsOne, *bitsTwo[i]);
  }
  return similarities;
}

std::vector<double> bulkTanimotoSimilarityCpu(const ExplicitBitVect&                               bitsOne,
                                              const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo) {
  std::vector<const ExplicitBitVect*> bitsTwoRaw;
  bitsTwoRaw.reserve(bitsTwo.size());
  for (const auto& bit : bitsTwo) {
    bitsTwoRaw.push_back(bit.get());
  }
  return bulkTanimotoSimilarityCpu(bitsOne, bitsTwoRaw);
}

std::vector<double> bulkTanimotoSimilarityCpu(const ExplicitBitVect&               bitsOne,
                                              const std::vector<ExplicitBitVect*>& bitsTwo) {
  std::vector<const ExplicitBitVect*> bitsTwoRaw;
  bitsTwoRaw.reserve(bitsTwo.size());
  for (const auto& bit : bitsTwo) {
    bitsTwoRaw.push_back(bit);
  }
  return bulkTanimotoSimilarityCpu(bitsOne, bitsTwoRaw);
}

// --------------------------------
// Cosine similarity wrapper functions
// --------------------------------

std::vector<double> bulkCosineSimilarityCpu(const ExplicitBitVect&                     bitsOne,
                                            const std::vector<const ExplicitBitVect*>& bitsTwo) {
  std::vector<double> similarities(bitsTwo.size());
#pragma omp parallel for default(none) shared(bitsOne, bitsTwo, similarities)
  for (size_t i = 0; i < bitsTwo.size(); ++i) {
    similarities[i] = CosineSimilarity(bitsOne, *bitsTwo[i]);
  }
  return similarities;
}

std::vector<double> bulkCosineSimilarityCpu(const ExplicitBitVect&                               bitsOne,
                                            const std::vector<std::unique_ptr<ExplicitBitVect>>& bitsTwo) {
  std::vector<const ExplicitBitVect*> bitsTwoRaw;
  bitsTwoRaw.reserve(bitsTwo.size());
  for (const auto& bit : bitsTwo) {
    bitsTwoRaw.push_back(bit.get());
  }
  return bulkCosineSimilarityCpu(bitsOne, bitsTwoRaw);
}

std::vector<double> bulkCosineSimilarityCpu(const ExplicitBitVect&               bitsOne,
                                            const std::vector<ExplicitBitVect*>& bitsTwo) {
  std::vector<const ExplicitBitVect*> bitsTwoRaw;
  bitsTwoRaw.reserve(bitsTwo.size());
  for (const auto& bit : bitsTwo) {
    bitsTwoRaw.push_back(bit);
  }
  return bulkCosineSimilarityCpu(bitsOne, bitsTwoRaw);
}
}  // namespace nvMolKit
