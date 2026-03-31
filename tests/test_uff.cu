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

#include <ForceField/ForceField.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/ForceFieldHelpers/UFF/Builder.h>
#include <GraphMol/ForceFieldHelpers/UFF/UFF.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "device_vector.h"
#include "ff_utils.h"
#include "minimizer/bfgs_minimize.h"
#include "test_utils.h"
#include "uff.h"
#include "uff_batched_forcefield.h"
#include "uff_flattened_builder.h"

using namespace nvMolKit::UFF;
using nvMolKit::AsyncDeviceVector;

namespace {

constexpr double kEnergyTol         = 1.0e-6;
constexpr double kGradTol           = 2.0e-4;
constexpr double kMinimizeEnergyTol = 5.0e-3;

std::vector<double> positionsFromMol(const RDKit::ROMol& mol, const int confId = -1) {
  std::vector<double> positions;
  nvMolKit::confPosToVect(mol, positions, confId);
  return positions;
}

double computeEnergyViaForcefield(BatchedMolecularSystemHost& systemHost, AsyncDeviceVector<double>& positionsDevice) {
  nvMolKit::UFFBatchedForcefield forcefield(systemHost);
  AsyncDeviceVector<double>      energyOuts;
  energyOuts.resize(1);
  energyOuts.zero();
  CHECK_CUDA_RETURN(forcefield.computeEnergy(energyOuts.data(), positionsDevice.data()));
  double energy = 0.0;
  CHECK_CUDA_RETURN(cudaMemcpy(&energy, energyOuts.data(), sizeof(double), cudaMemcpyDeviceToHost));
  return energy;
}

std::vector<double> computeGradientViaForcefield(BatchedMolecularSystemHost& systemHost, AsyncDeviceVector<double>& positionsDevice) {
  nvMolKit::UFFBatchedForcefield forcefield(systemHost);
  AsyncDeviceVector<double>      gradDevice;
  gradDevice.resize(positionsDevice.size());
  gradDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()));
  std::vector<double> grad(positionsDevice.size(), 0.0);
  gradDevice.copyToHost(grad);
  cudaDeviceSynchronize();
  return grad;
}

std::unique_ptr<ForceFields::ForceField> buildReferenceForceField(RDKit::ROMol& mol, const int confId = -1) {
  auto ff = std::unique_ptr<ForceFields::ForceField>(RDKit::UFF::constructForceField(mol, 100.0, confId, true));
  ff->initialize();
  return ff;
}

std::unique_ptr<RDKit::ROMol> buildEmbeddedEdgeCaseMol(const std::string& smiles) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
  EXPECT_NE(mol, nullptr);
  auto withHs = std::unique_ptr<RDKit::ROMol>(RDKit::MolOps::addHs(*mol));
  RDKit::DGeomHelpers::EmbedMolecule(*withHs);
  return withHs;
}

}  // namespace

class UFFGpuTestFixture : public ::testing::Test {
 public:
  UFFGpuTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    mol_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(mol_, nullptr);
    RDKit::MolOps::sanitizeMol(*mol_);

    positions_ = positionsFromMol(*mol_);
    contribs_  = constructForcefieldContribs(*mol_);
    addMoleculeToBatch(contribs_, positions_, systemHost_);
    sendContribsAndIndicesToDevice(systemHost_, systemDevice_);
    systemDevice_.positions.setFromVector(systemHost_.positions);
    allocateIntermediateBuffers(systemHost_, systemDevice_);
    systemDevice_.grad.resize(systemHost_.positions.size());
    systemDevice_.grad.zero();
  }

 protected:
  std::string                         testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol>       mol_;
  std::vector<double>                 positions_;
  EnergyForceContribsHost             contribs_;
  BatchedMolecularSystemHost          systemHost_;
  BatchedMolecularDeviceBuffers       systemDevice_;
};

TEST_F(UFFGpuTestFixture, FlattenedBuilderPopulatesAllTerms) {
  EXPECT_GT(contribs_.bondTerms.idx1.size(), 0);
  EXPECT_GT(contribs_.angleTerms.idx1.size(), 0);
  EXPECT_GT(contribs_.torsionTerms.idx1.size(), 0);
  EXPECT_GT(contribs_.inversionTerms.idx1.size(), 0);
  EXPECT_GT(contribs_.vdwTerms.idx1.size(), 0);

  EXPECT_EQ(contribs_.bondTerms.idx1.size(), contribs_.bondTerms.forceConstant.size());
  EXPECT_EQ(contribs_.angleTerms.idx1.size(), contribs_.angleTerms.C2.size());
  EXPECT_EQ(contribs_.torsionTerms.idx1.size(), contribs_.torsionTerms.cosTerm.size());
  EXPECT_EQ(contribs_.inversionTerms.idx1.size(), contribs_.inversionTerms.C2.size());
  EXPECT_EQ(contribs_.vdwTerms.idx1.size(), contribs_.vdwTerms.threshold.size());
}

TEST_F(UFFGpuTestFixture, CombinedEnergyMatchesRDKit) {
  auto referenceFF = buildReferenceForceField(*mol_);
  const double wantEnergy = referenceFF->calcEnergy(positions_.data());
  const double gotEnergy  = computeEnergyViaForcefield(systemHost_, systemDevice_.positions);
  EXPECT_NEAR(gotEnergy, wantEnergy, kEnergyTol);
}

TEST_F(UFFGpuTestFixture, CombinedGradientMatchesRDKit) {
  auto referenceFF = buildReferenceForceField(*mol_);
  std::vector<double> wantGrad(referenceFF->dimension() * referenceFF->positions().size(), 0.0);
  referenceFF->calcGrad(positions_.data(), wantGrad.data());
  const std::vector<double> gotGrad = computeGradientViaForcefield(systemHost_, systemDevice_.positions);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(kGradTol), wantGrad));
}

TEST_F(UFFGpuTestFixture, BlockPerMolMatchesRDKit) {
  auto referenceFF = buildReferenceForceField(*mol_);
  systemDevice_.energyOuts.zero();
  CHECK_CUDA_RETURN(computeEnergyBlockPerMol(systemDevice_));
  double gotEnergy = 0.0;
  CHECK_CUDA_RETURN(cudaMemcpy(&gotEnergy, systemDevice_.energyOuts.data(), sizeof(double), cudaMemcpyDeviceToHost));

  std::vector<double> wantGrad(referenceFF->dimension() * referenceFF->positions().size(), 0.0);
  referenceFF->calcGrad(positions_.data(), wantGrad.data());
  systemDevice_.grad.zero();
  CHECK_CUDA_RETURN(computeGradBlockPerMol(systemDevice_));
  std::vector<double> gotGrad(systemDevice_.positions.size(), 0.0);
  systemDevice_.grad.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  EXPECT_NEAR(gotEnergy, referenceFF->calcEnergy(positions_.data()), kEnergyTol);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(kGradTol), wantGrad));
}

class UFFGpuEdgeCases : public ::testing::TestWithParam<std::string> {};

TEST_P(UFFGpuEdgeCases, CombinedEnergyAndGradient) {
  auto mol = buildEmbeddedEdgeCaseMol(GetParam());
  ASSERT_NE(mol, nullptr);

  auto positions = positionsFromMol(*mol);
  const auto contribs  = constructForcefieldContribs(*mol);
  BatchedMolecularSystemHost host;
  addMoleculeToBatch(contribs, positions, host);

  AsyncDeviceVector<double> positionsDevice;
  positionsDevice.setFromVector(host.positions);
  const double gotEnergy = computeEnergyViaForcefield(host, positionsDevice);
  const auto   gotGrad   = computeGradientViaForcefield(host, positionsDevice);

  auto referenceFF = buildReferenceForceField(*mol);
  std::vector<double> wantGrad(referenceFF->dimension() * referenceFF->positions().size(), 0.0);
  referenceFF->calcGrad(positions.data(), wantGrad.data());

  EXPECT_NEAR(gotEnergy, referenceFF->calcEnergy(positions.data()), 1.0e-5);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(2.0e-4), wantGrad));
}

INSTANTIATE_TEST_SUITE_P(UFFOneTwoAtoms,
                         UFFGpuEdgeCases,
                         ::testing::Values("C", "O", "CC", "CO", "CCC"));

TEST(UFFValidationSuite, BatchMatchesRDKitValidationSet) {
  const std::string sdfPath = getTestDataFolderPath() + "/MMFF94_dative.sdf";
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(sdfPath, mols, 25);
  ASSERT_FALSE(mols.empty());

  BatchedMolecularSystemHost host;
  std::vector<double>        wantEnergies;
  std::vector<std::vector<double>> wantGrads;
  for (size_t i = 0; i < mols.size(); ++i) {
    auto& mol = *mols[i];
    auto  positions = positionsFromMol(mol);
    auto  contribs  = constructForcefieldContribs(mol);
    addMoleculeToBatch(contribs, positions, host);

    auto ff = buildReferenceForceField(mol);
    wantEnergies.push_back(ff->calcEnergy(positions.data()));
    std::vector<double> wantGrad(ff->dimension() * ff->positions().size(), 0.0);
    ff->calcGrad(positions.data(), wantGrad.data());
    wantGrads.push_back(std::move(wantGrad));
  }

  nvMolKit::UFFBatchedForcefield forcefield(host);
  AsyncDeviceVector<double>      positionsDevice;
  AsyncDeviceVector<double>      energiesDevice;
  AsyncDeviceVector<double>      gradDevice;
  positionsDevice.setFromVector(host.positions);
  energiesDevice.resize(mols.size());
  energiesDevice.zero();
  gradDevice.resize(host.positions.size());
  gradDevice.zero();

  CHECK_CUDA_RETURN(forcefield.computeEnergy(energiesDevice.data(), positionsDevice.data()));
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()));

  std::vector<double> gotEnergies(mols.size(), 0.0);
  std::vector<double> gotGrad(host.positions.size(), 0.0);
  energiesDevice.copyToHost(gotEnergies);
  gradDevice.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < wantEnergies.size(); ++i) {
    EXPECT_NEAR(gotEnergies[i], wantEnergies[i], 1.0e-5) << "molecule " << i;
    const int atomStart = host.indices.atomStarts[i];
    const int atomEnd   = host.indices.atomStarts[i + 1];
    std::vector<double> gotGradMol(gotGrad.begin() + atomStart * 3, gotGrad.begin() + atomEnd * 3);
    EXPECT_THAT(gotGradMol, ::testing::Pointwise(::testing::DoubleNear(2.0e-4), wantGrads[i])) << "molecule " << i;
  }
}

TEST(UFFMinimizer, BatchMinimizerMatchesRDKitFinalEnergies) {
  const std::string sdfPath = getTestDataFolderPath() + "/MMFF94_dative.sdf";
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(sdfPath, mols, 8);
  ASSERT_FALSE(mols.empty());

  BatchedMolecularSystemHost host;
  std::vector<double>        referenceFinalEnergies;
  for (auto& molPtr : mols) {
    auto& mol = *molPtr;
    auto positions = positionsFromMol(mol);
    auto contribs  = constructForcefieldContribs(mol);
    addMoleculeToBatch(contribs, positions, host);

    const auto optimizeResult = RDKit::UFF::UFFOptimizeMolecule(mol, 1000, 100.0, -1, true);
    referenceFinalEnergies.push_back(optimizeResult.second);
  }

  AsyncDeviceVector<double> positionsDevice;
  AsyncDeviceVector<double> gradDevice;
  AsyncDeviceVector<double> energiesDevice;
  positionsDevice.setFromVector(host.positions);
  gradDevice.resize(host.positions.size());
  gradDevice.zero();
  energiesDevice.resize(mols.size());
  energiesDevice.zero();

  nvMolKit::UFFBatchedForcefield forcefield(host);
  nvMolKit::BfgsBatchMinimizer   minimizer;
  const bool needsMore = minimizer.minimize(1000, 1.0e-6, forcefield, positionsDevice, gradDevice, energiesDevice);
  EXPECT_FALSE(needsMore);

  std::vector<double> gotFinalEnergies(mols.size(), 0.0);
  energiesDevice.copyToHost(gotFinalEnergies);
  cudaDeviceSynchronize();
  for (size_t i = 0; i < gotFinalEnergies.size(); ++i) {
    EXPECT_NEAR(gotFinalEnergies[i], referenceFinalEnergies[i], kMinimizeEnergyTol) << "molecule " << i;
  }
}
