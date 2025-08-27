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

#include <gmock/gmock.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <memory>
#include <tuple>

using namespace RDKit;

using fptype = FingerprintGenerator<std::uint32_t>;

class RefBasicMorganFpTest : public ::testing::TestWithParam<std::tuple<std::string, std::array<int, 4>>> {};

// Vals taken from testMorganFP() in testFingerprintGenerators.cpp
TEST_P(RefBasicMorganFpTest, Basics) {
  const std::string        smi             = std::get<0>(GetParam());
  const std::array<int, 4> wantNonZeroSize = std::get<1>(GetParam());
  std::unique_ptr<ROMol>   mol(SmilesToMol(smi));
  for (size_t i = 0; i < wantNonZeroSize.size(); i++) {
    auto      generator   = std::unique_ptr<fptype>(MorganFingerprint::getMorganGenerator<std::uint32_t>(i));
    auto      fingerprint = std::unique_ptr<SparseIntVect<std::uint32_t>>(generator->getSparseCountFingerprint(*mol));
    const int wantSize    = wantNonZeroSize.at(i);
    EXPECT_EQ(fingerprint->getNonzeroElements().size(), wantSize) << "radius: " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
  RefBasicMorganFpTestInstance,
  RefBasicMorganFpTest,
  ::testing::Values(std::make_tuple("CCCCC", std::array<int, 4>{2, 5, 7, 7}),
                    std::make_tuple("O=C(O)CC1CC1", std::array<int, 4>{6, 12, 16, 17}),
                    std::make_tuple("OC(=O)CC1CC1", std::array<int, 4>{6, 12, 16, 17})));  // invariance to atom order

TEST(RefBasicMorganFpTest, Symmetry) {
  std::unique_ptr<ROMol> mol(SmilesToMol("OCCCCO"));
  auto                   generator = std::unique_ptr<fptype>(MorganFingerprint::getMorganGenerator<std::uint32_t>(2));
  auto fingerprint = std::unique_ptr<SparseIntVect<std::uint32_t>>(generator->getSparseCountFingerprint(*mol));
  EXPECT_EQ(fingerprint->getNonzeroElements().size(), 7);
  std::set<std::uint32_t> wantCounts = {2, 4};
  for (const auto& elem : fingerprint->getNonzeroElements()) {
    EXPECT_THAT(wantCounts, ::testing::Contains(elem.second));
  }
}

// TODO
TEST(RefMorganFpTestOptions, Chirality) {}

// TODO
TEST(RefMorganFpTestOptions, FromAtoms) {}

// TODO
TEST(RefMorganFPTestOptions, BitVec) {}

// TODO all the various options
