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

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "tfd_common.h"
#include "tfd_cpu.h"

namespace {

constexpr double kTolerance = 1e-4;

//! Generate conformers for a molecule using RDKit
void generateConformers(RDKit::ROMol& mol, int numConformers, int seed = 42) {
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.randomSeed                           = seed;
  params.numThreads                           = 1;
  RDKit::DGeomHelpers::EmbedMultipleConfs(mol, numConformers, params);
}

}  // namespace

class TFDCpuTest : public ::testing::Test {
 protected:
  nvMolKit::TFDCpuGenerator generator_;
};

// =============================================================================
// extractTorsionList
// =============================================================================

TEST_F(TFDCpuTest, ExtractTorsionListRDKitReference) {
  // Compare extractTorsionList output against RDKit CalculateTorsionLists.
  // Parameters: maxDev='equal', symmRadius=2, ignoreColinearBonds=True

  // --- CCCC: 1 non-ring torsion, 0 ring ---
  {
    SCOPED_TRACE("CCCC");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    ASSERT_EQ(tl.nonRingTorsions.size(), 1u);
    EXPECT_EQ(tl.ringTorsions.size(), 0u);

    ASSERT_EQ(tl.nonRingTorsions[0].atomQuartets.size(), 1u);
    EXPECT_EQ(tl.nonRingTorsions[0].atomQuartets[0], (std::array<int, 4>{0, 1, 2, 3}));
    EXPECT_NEAR(tl.nonRingTorsions[0].maxDev, 180.0f, 0.01f);
  }

  // --- CCCCC: 2 non-ring torsions, 0 ring ---
  {
    SCOPED_TRACE("CCCCC");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    ASSERT_EQ(tl.nonRingTorsions.size(), 2u);
    EXPECT_EQ(tl.ringTorsions.size(), 0u);

    ASSERT_EQ(tl.nonRingTorsions[0].atomQuartets.size(), 1u);
    EXPECT_EQ(tl.nonRingTorsions[0].atomQuartets[0], (std::array<int, 4>{0, 1, 2, 3}));

    ASSERT_EQ(tl.nonRingTorsions[1].atomQuartets.size(), 1u);
    EXPECT_EQ(tl.nonRingTorsions[1].atomQuartets[0], (std::array<int, 4>{1, 2, 3, 4}));
  }

  // --- CCCCCC: 3 non-ring torsions, 0 ring ---
  {
    SCOPED_TRACE("CCCCCC");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCCC"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    ASSERT_EQ(tl.nonRingTorsions.size(), 3u);
    EXPECT_EQ(tl.ringTorsions.size(), 0u);

    EXPECT_EQ(tl.nonRingTorsions[0].atomQuartets[0], (std::array<int, 4>{0, 1, 2, 3}));
    EXPECT_EQ(tl.nonRingTorsions[1].atomQuartets[0], (std::array<int, 4>{1, 2, 3, 4}));
    EXPECT_EQ(tl.nonRingTorsions[2].atomQuartets[0], (std::array<int, 4>{2, 3, 4, 5}));
  }

  // --- c1ccccc1 (benzene): 0 non-ring, 1 ring torsion with 6 quartets ---
  {
    SCOPED_TRACE("c1ccccc1");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("c1ccccc1"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    EXPECT_EQ(tl.nonRingTorsions.size(), 0u);
    ASSERT_EQ(tl.ringTorsions.size(), 1u);

    const auto& ringTorsion = tl.ringTorsions[0];
    ASSERT_EQ(ringTorsion.atomQuartets.size(), 6u);

    // Ring torsion maxDev = 180 * exp(-0.025 * (6-14)^2) ≈ 36.34
    EXPECT_NEAR(ringTorsion.maxDev, 36.34f, 0.1f);

    // Quartets from RDKit (ring order: 0, 5, 4, 3, 2, 1)
    EXPECT_EQ(ringTorsion.atomQuartets[0], (std::array<int, 4>{0, 5, 4, 3}));
    EXPECT_EQ(ringTorsion.atomQuartets[1], (std::array<int, 4>{5, 4, 3, 2}));
    EXPECT_EQ(ringTorsion.atomQuartets[2], (std::array<int, 4>{4, 3, 2, 1}));
    EXPECT_EQ(ringTorsion.atomQuartets[3], (std::array<int, 4>{3, 2, 1, 0}));
    EXPECT_EQ(ringTorsion.atomQuartets[4], (std::array<int, 4>{2, 1, 0, 5}));
    EXPECT_EQ(ringTorsion.atomQuartets[5], (std::array<int, 4>{1, 0, 5, 4}));
  }
}

// =============================================================================
// computeTorsionWeights
// =============================================================================

TEST_F(TFDCpuTest, ComputeTorsionWeightsRDKitReference) {
  // Compare computeTorsionWeights output against RDKit CalculateTorsionWeights.
  // Reference generated with: TorsionFingerprints.CalculateTorsionWeights(mol, ignoreColinearBonds=True)

  struct TestCase {
    const char*        smiles;
    std::vector<float> expectedWeights;
  };

  // clang-format off
  std::vector<TestCase> cases = {
    {"CCCC",   {1.0f}},                     // 1 torsion: central bond
    {"CCCCC",  {1.0f, 0.1f}},               // 2 torsions: central + terminal
    {"CCCCCC", {0.1f, 1.0f, 0.1f}},         // 3 torsions: symmetric around center
  };
  // clang-format on

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.smiles);
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(tc.smiles));
    ASSERT_NE(mol, nullptr);

    auto tl      = nvMolKit::extractTorsionList(*mol);
    auto weights = nvMolKit::computeTorsionWeights(*mol, tl);

    ASSERT_EQ(weights.size(), tc.expectedWeights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
      EXPECT_NEAR(weights[i], tc.expectedWeights[i], 5e-4f) << "Weight[" << i << "] mismatch";
    }
  }
}

// =============================================================================
// buildTFDSystem
// =============================================================================

TEST_F(TFDCpuTest, BuildTFDSystem) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 5);
  ASSERT_EQ(mol->getNumConformers(), 5);

  nvMolKit::TFDComputeOptions options;
  auto                        system = nvMolKit::buildTFDSystem(*mol, options);

  EXPECT_EQ(system.numMolecules(), 1);
  EXPECT_EQ(system.totalConformers(), 5);
  EXPECT_GT(system.totalTorsions(), 0);

  // TFD output size: 5 conformers = 5*4/2 = 10 pairs
  EXPECT_EQ(system.totalTFDOutputs(), 10);
}

// =============================================================================
// computeDihedralAngles
// =============================================================================

TEST_F(TFDCpuTest, KnownDihedralAngle) {
  // Test dihedral angle computation with hand-crafted geometry (RDKit-independent).
  // Create n-butane and manually set conformer coordinates to known dihedral angles.
  //
  // Dihedral angle definition: Looking down the C1-C2 bond (central bond),
  // the angle between C0 and C3 measured clockwise from C0.
  // - Trans (180°): C0 and C3 on opposite sides
  // - Gauche (60°): C0 and C3 at 60° apart

  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);
  ASSERT_EQ(mol->getNumAtoms(), 4);

  // Trans conformer (180°): C0-C1 and C2-C3 vectors are antiparallel
  {
    auto* conf = new RDKit::Conformer(4);
    conf->setId(0);
    conf->setAtomPos(0, RDGeom::Point3D(1.0, 0.0, 0.0));
    conf->setAtomPos(1, RDGeom::Point3D(0.0, 0.0, 0.0));
    conf->setAtomPos(2, RDGeom::Point3D(0.0, 1.5, 0.0));
    conf->setAtomPos(3, RDGeom::Point3D(1.0, 1.5, 0.0));
    mol->addConformer(conf, true);
  }

  // Gauche conformer (60°): C3 rotated to give 60° dihedral
  {
    auto* conf = new RDKit::Conformer(4);
    conf->setId(1);
    conf->setAtomPos(0, RDGeom::Point3D(1.0, 0.0, 0.0));
    conf->setAtomPos(1, RDGeom::Point3D(0.0, 0.0, 0.0));
    conf->setAtomPos(2, RDGeom::Point3D(0.0, 1.5, 0.0));
    conf->setAtomPos(3, RDGeom::Point3D(-0.5, 1.5, 0.866));
    mol->addConformer(conf, true);
  }

  ASSERT_EQ(mol->getNumConformers(), 2);

  nvMolKit::TFDComputeOptions options;
  options.useWeights = false;
  options.maxDevMode = nvMolKit::TFDMaxDevMode::Equal;

  auto system = nvMolKit::buildTFDSystem(*mol, options);
  auto angles = generator_.computeDihedralAngles(system, 0);

  int numTorsions = system.molTorsionStarts[1] - system.molTorsionStarts[0];
  ASSERT_GE(numTorsions, 1);

  EXPECT_NEAR(angles[0], 180.0f, 5.0f) << "Trans conformer should have ~180° dihedral";
  EXPECT_NEAR(angles[numTorsions], 60.0f, 5.0f) << "Gauche conformer should have ~60° dihedral";

  // TFD: difference = |180 - 60| = 120°, normalized by 180° ≈ 0.667
  auto tfdMatrix = generator_.GetTFDMatrix(*mol, options);
  ASSERT_EQ(tfdMatrix.size(), 1u);
  EXPECT_NEAR(tfdMatrix[0], 0.667, 0.05);
}

TEST_F(TFDCpuTest, ComputeDihedralAnglesRDKitReference) {
  // Compare computeDihedralAngles output against RDKit rdMolTransforms.GetDihedralDeg.
  // 4 conformers per molecule, seed=42, ETKDGv3.
  //
  // NOTE: Our dihedral convention is offset by 180° from RDKit's GetDihedralDeg.
  // This does NOT affect TFD (circularDifference is invariant to shared offset).
  // Reference values below are RDKit values + 180° (mod 360).
  //
  // Original RDKit values generated with:
  //   angle = rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)
  //   our_angle = (angle + 180) % 360

  struct TestCase {
    const char*        smiles;
    int                numTorsions;
    std::vector<float> expectedAngles;  // [numConf * numTors], our convention
  };

  // clang-format off
  std::vector<TestCase> cases = {
    {"CCCC", 1, {
      120.0130f,  // conf[0] tors[0]  (RDKit: 300.013)
        0.0000f,  // conf[1] tors[0]  (RDKit: 180.000)
      120.0000f,  // conf[2] tors[0]  (RDKit: 300.000)
        0.0000f   // conf[3] tors[0]  (RDKit: 180.000)
    }},
    {"CCCCC", 2, {
      239.9983f, 240.0002f,   // conf[0]  (RDKit: 60.0, 60.0)
      119.9983f, 240.0001f,   // conf[1]  (RDKit: 300.0, 60.0)
      359.9974f, 240.0025f,   // conf[2]  (RDKit: 180.0, 60.0)
        0.0037f, 120.0134f    // conf[3]  (RDKit: 180.0, 300.0)
    }},
    {"CCCCCC", 3, {
      240.0055f, 359.9997f, 119.9928f,  // conf[0]  (RDKit: 60, 180, 300)
      359.9961f, 239.9959f, 119.9996f,  // conf[1]  (RDKit: 180, 60, 300)
      239.9980f, 359.9978f, 239.9987f,  // conf[2]  (RDKit: 60, 180, 60)
      239.9980f, 119.9980f, 119.9976f   // conf[3]  (RDKit: 60, 300, 300)
    }},
  };
  // clang-format on

  // Use circular distance to handle 0°/360° wrap-around
  constexpr float kAngleTolerance = 0.05f;  // degrees; generous for float vs double

  nvMolKit::TFDComputeOptions options;
  options.useWeights          = true;
  options.maxDevMode          = nvMolKit::TFDMaxDevMode::Equal;
  options.symmRadius          = 2;
  options.ignoreColinearBonds = true;

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.smiles);
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(tc.smiles));
    ASSERT_NE(mol, nullptr);

    generateConformers(*mol, 4, 42);
    ASSERT_EQ(mol->getNumConformers(), 4);

    auto system = nvMolKit::buildTFDSystem(*mol, options);
    auto angles = generator_.computeDihedralAngles(system, 0);

    int numConf = 4;
    int numTors = tc.numTorsions;
    ASSERT_EQ(static_cast<int>(angles.size()), numConf * numTors);
    ASSERT_EQ(static_cast<int>(tc.expectedAngles.size()), numConf * numTors);

    for (size_t i = 0; i < angles.size(); ++i) {
      // Circular distance handles 0°/360° boundary
      float diff = std::abs(angles[i] - tc.expectedAngles[i]);
      if (diff > 180.0f) {
        diff = 360.0f - diff;
      }
      EXPECT_LE(diff, kAngleTolerance) << "Angle[" << i << "] (conf=" << i / numTors << " tors=" << i % numTors
                                       << "): got " << angles[i] << ", expected " << tc.expectedAngles[i];
    }
  }
}

// =============================================================================
// GetTFDMatrix (pipeline)
// =============================================================================

TEST_F(TFDCpuTest, ComputeTFDMatrixSelfComparison) {
  // Identical conformers should produce TFD = 0
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 1);
  ASSERT_EQ(mol->getNumConformers(), 1);

  // Duplicate the conformer
  RDKit::Conformer conf = mol->getConformer(0);
  conf.setId(1);
  mol->addConformer(new RDKit::Conformer(conf), true);
  ASSERT_EQ(mol->getNumConformers(), 2);

  nvMolKit::TFDComputeOptions options;
  auto                        tfdMatrix = generator_.GetTFDMatrix(*mol, options);

  ASSERT_EQ(tfdMatrix.size(), 1u);
  EXPECT_NEAR(tfdMatrix[0], 0.0, kTolerance);
}

TEST_F(TFDCpuTest, NoTorsionsMolecule) {
  // Methane has no rotatable bonds — TFD should be zero
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("C"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 3);

  nvMolKit::TFDComputeOptions options;
  auto                        tfdMatrix = generator_.GetTFDMatrix(*mol, options);

  ASSERT_EQ(tfdMatrix.size(), 3u);  // 3 conformers = 3 pairs
  for (double tfd : tfdMatrix) {
    EXPECT_NEAR(tfd, 0.0, kTolerance);
  }
}

TEST_F(TFDCpuTest, SingleConformer) {
  // Single conformer should return empty matrix (no pairs)
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 1);
  ASSERT_EQ(mol->getNumConformers(), 1);

  nvMolKit::TFDComputeOptions options;
  auto                        tfdMatrix = generator_.GetTFDMatrix(*mol, options);

  EXPECT_TRUE(tfdMatrix.empty());
}

TEST_F(TFDCpuTest, UseWeightsOption) {
  // Weighted and unweighted TFD should differ for multi-torsion molecules
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCCCCC"));  // n-octane
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 4);
  ASSERT_GE(mol->getNumConformers(), 2);

  nvMolKit::TFDComputeOptions optionsWithWeights;
  optionsWithWeights.useWeights = true;

  nvMolKit::TFDComputeOptions optionsNoWeights;
  optionsNoWeights.useWeights = false;

  auto tfdWithWeights = generator_.GetTFDMatrix(*mol, optionsWithWeights);
  auto tfdNoWeights   = generator_.GetTFDMatrix(*mol, optionsNoWeights);

  ASSERT_EQ(tfdWithWeights.size(), tfdNoWeights.size());
  ASSERT_GT(tfdWithWeights.size(), 0u);

  for (size_t i = 0; i < tfdWithWeights.size(); ++i) {
    EXPECT_GE(tfdWithWeights[i], 0.0);
    EXPECT_GE(tfdNoWeights[i], 0.0);
  }

  bool anyDifferent = false;
  for (size_t i = 0; i < tfdWithWeights.size(); ++i) {
    if (std::abs(tfdWithWeights[i] - tfdNoWeights[i]) > 1e-6) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent) << "Weighted and unweighted TFD should differ for n-octane";
}

TEST_F(TFDCpuTest, MaxDevModes) {
  // Both Equal and Spec modes should produce valid finite results
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 3);

  nvMolKit::TFDComputeOptions optionsEqual;
  optionsEqual.maxDevMode = nvMolKit::TFDMaxDevMode::Equal;

  nvMolKit::TFDComputeOptions optionsSpec;
  optionsSpec.maxDevMode = nvMolKit::TFDMaxDevMode::Spec;

  auto tfdEqual = generator_.GetTFDMatrix(*mol, optionsEqual);
  auto tfdSpec  = generator_.GetTFDMatrix(*mol, optionsSpec);

  ASSERT_EQ(tfdEqual.size(), tfdSpec.size());

  for (size_t i = 0; i < tfdEqual.size(); ++i) {
    EXPECT_TRUE(std::isfinite(tfdEqual[i]));
    EXPECT_TRUE(std::isfinite(tfdSpec[i]));
  }
}

TEST_F(TFDCpuTest, CompareWithRDKitReference) {
  // Compare full TFD pipeline against pre-computed RDKit reference values.
  // No AddHs to avoid multi-quartet symmetric torsions (added in a later commit).
  //
  // Reference values generated with RDKit Python:
  //   mol = Chem.MolFromSmiles(smiles)  # No AddHs
  //   params = AllChem.ETKDGv3()
  //   params.randomSeed = 42
  //   AllChem.EmbedMultipleConfs(mol, 4, params)
  //   tfd = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev='equal', symmRadius=2)

  struct TestCase {
    const char*         smiles;
    std::vector<double> reference;
  };

  // clang-format off
  std::vector<TestCase> cases = {
    {"CCCC", {                    // n-butane, 1 torsion
      0.6667389132,  // TFD(1,0)
      0.0000726610,  // TFD(2,0)
      0.6666662521,  // TFD(2,1)
      0.6667387931,  // TFD(3,0)
      0.0000001200,  // TFD(3,1)
      0.6666661321   // TFD(3,2)
    }},
    {"CCCCC", {                   // n-pentane, 2 torsions
      0.6060606631,  // TFD[0]
      0.6060573299,  // TFD[1]
      0.6060662252,  // TFD[2]
      0.6666872992,  // TFD[3]
      0.6666326206,  // TFD[4]
      0.0606323184   // TFD[5]
    }},
    {"CCCCCC", {                  // n-hexane, 3 torsions
      0.6111276139,  // TFD[0]
      0.0555704226,  // TFD[1]
      0.6666744357,  // TFD[2]
      0.5555532106,  // TFD[3]
      0.6111014381,  // TFD[4]
      0.6111123144   // TFD[5]
    }},
    // TODO: Add these back once multi-quartet torsion support is implemented (commit 6):
    //   "CC(C)CC"    - isopentane: 2 quartets due to branching symmetry
    //   "c1ccccc1CC" - ethylbenzene: 2 quartets on non-ring, 6 on ring torsion
  };

  nvMolKit::TFDComputeOptions options;
  options.useWeights          = true;
  options.maxDevMode          = nvMolKit::TFDMaxDevMode::Equal;
  options.symmRadius          = 2;
  options.ignoreColinearBonds = true;

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.smiles);

    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(tc.smiles));
    ASSERT_NE(mol, nullptr);

    generateConformers(*mol, 4, 42);
    auto tfdMatrix = generator_.GetTFDMatrix(*mol, options);

    ASSERT_EQ(tfdMatrix.size(), tc.reference.size());
    // Slightly relaxed tolerance for multi-torsion molecules where floating-point
    // differences in weight computation accumulate (e.g., hexane ~0.00013 diff)
    constexpr double kRDKitTolerance = 5e-4;
    for (size_t i = 0; i < tfdMatrix.size(); ++i) {
      EXPECT_NEAR(tfdMatrix[i], tc.reference[i], kRDKitTolerance)
          << "TFD[" << i << "] mismatch with RDKit reference";
    }
  }
}

// =============================================================================
// GetTFDMatrices (batch)
// =============================================================================

TEST_F(TFDCpuTest, BatchProcessing) {
  // Test batch processing of multiple molecules via GetTFDMatrices
  // clang-format off
  const std::vector<std::string> testSmiles = {
    "CCCC", "CC(C)C", "c1ccccc1", "CCO", "CCCCC",
    "CC(=O)O", "c1ccc(cc1)O", "CCCCCC", "CC(C)(C)C", "c1ccc2ccccc2c1",
  };
  // clang-format on

  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  std::vector<const RDKit::ROMol*>           molPtrs;

  for (const auto& smiles : testSmiles) {
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(smiles));
    if (mol) {
      generateConformers(*mol, 3);
      if (mol->getNumConformers() >= 2) {
        molPtrs.push_back(mol.get());
        mols.push_back(std::move(mol));
      }
    }
  }

  ASSERT_GE(mols.size(), 3u);

  nvMolKit::TFDComputeOptions options;
  auto                        results = generator_.GetTFDMatrices(molPtrs, options);

  ASSERT_EQ(results.size(), molPtrs.size());

  for (size_t i = 0; i < results.size(); ++i) {
    int numConf       = molPtrs[i]->getNumConformers();
    int expectedPairs = numConf * (numConf - 1) / 2;

    EXPECT_EQ(results[i].size(), static_cast<size_t>(expectedPairs)) << "Mismatch for molecule " << i;

    for (double tfd : results[i]) {
      EXPECT_TRUE(std::isfinite(tfd));
      EXPECT_GE(tfd, 0.0);
    }
  }
}
