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

#include <memory>
#include <vector>

#include "tfd_common.h"
#include "tfd_cpu.h"
#include "tfd_kernels.h"

namespace {

constexpr double kTolerance = 1e-3;  // GPU has lower precision due to float

//! Generate conformers for a molecule using RDKit
void generateConformers(RDKit::ROMol& mol, int numConformers, int seed = 42) {
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.randomSeed                           = seed;
  params.numThreads                           = 1;
  RDKit::DGeomHelpers::EmbedMultipleConfs(mol, numConformers, params);
}

}  // namespace

// Global check for CUDA availability
static bool gCudaAvailable = false;
static bool gCudaChecked   = false;

static bool checkCudaAvailable() {
  if (!gCudaChecked) {
    gCudaChecked = true;
    try {
      int         deviceCount = 0;
      cudaError_t err         = cudaGetDeviceCount(&deviceCount);
      if (err == cudaSuccess && deviceCount > 0) {
        // Try to actually use the device
        err = cudaSetDevice(0);
        if (err == cudaSuccess) {
          gCudaAvailable = true;
        } else {
          cudaGetLastError();  // Reset error state
        }
      } else {
        cudaGetLastError();  // Reset error state
      }
    } catch (...) {
      gCudaAvailable = false;
    }
  }
  return gCudaAvailable;
}

class TFDKernelsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if CUDA is available
    if (!checkCudaAvailable()) {
      GTEST_SKIP() << "No CUDA devices available, skipping GPU tests";
    }
  }

  nvMolKit::TFDCpuGenerator cpuGenerator_;
};

// =============================================================================
// Dihedral kernel
// =============================================================================

TEST_F(TFDKernelsTest, DihedralKernelBasic) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 3);
  ASSERT_EQ(mol->getNumConformers(), 3);

  nvMolKit::TFDComputeOptions options;
  auto                        system = nvMolKit::buildTFDSystem(*mol, options);

  ASSERT_EQ(system.numMolecules(), 1);
  ASSERT_GT(system.totalConformers(), 0);
  ASSERT_GT(system.totalTorsions(), 0);
  ASSERT_GT(system.totalDihedralWorkItems(), 0);

  // Transfer to device
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<float> gpuAngles;

  {
    nvMolKit::TFDSystemDevice device;
    nvMolKit::transferToDevice(system, device, stream);

    nvMolKit::launchDihedralKernel(system.totalDihedralWorkItems(),
                                   device.positions.data(),
                                   device.confPositionStarts.data(),
                                   device.torsionAtoms.data(),
                                   device.dihedralConfIdx.data(),
                                   device.dihedralTorsIdx.data(),
                                   device.dihedralOutIdx.data(),
                                   device.dihedralAngles.data(),
                                   stream);

    gpuAngles.resize(device.dihedralAngles.size());
    device.dihedralAngles.copyToHost(gpuAngles.data(), gpuAngles.size());
    cudaStreamSynchronize(stream);
  }

  // Compute CPU reference
  auto cpuAngles = cpuGenerator_.computeDihedralAngles(system, 0);

  // Compare
  ASSERT_EQ(gpuAngles.size(), cpuAngles.size());
  for (size_t i = 0; i < cpuAngles.size(); ++i) {
    EXPECT_NEAR(gpuAngles[i], cpuAngles[i], kTolerance) << "Mismatch at angle " << i;
  }

  cudaStreamDestroy(stream);
}

// =============================================================================
// TFD pipeline (GPU vs CPU)
// =============================================================================

TEST_F(TFDKernelsTest, TFDMatrixKernelMatchesCPU) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 5);
  ASSERT_EQ(mol->getNumConformers(), 5);

  nvMolKit::TFDComputeOptions options;

  // Compute CPU reference
  auto cpuTFD = cpuGenerator_.GetTFDMatrix(*mol, options);
  ASSERT_FALSE(cpuTFD.empty());

  // Build system
  auto system = nvMolKit::buildTFDSystem(*mol, options);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<float> gpuTFD;

  {
    nvMolKit::TFDSystemDevice device;
    nvMolKit::transferToDevice(system, device, stream);

    // Compute dihedral angles first
    nvMolKit::launchDihedralKernel(system.totalDihedralWorkItems(),
                                   device.positions.data(),
                                   device.confPositionStarts.data(),
                                   device.torsionAtoms.data(),
                                   device.dihedralConfIdx.data(),
                                   device.dihedralTorsIdx.data(),
                                   device.dihedralOutIdx.data(),
                                   device.dihedralAngles.data(),
                                   stream);

    // Compute TFD matrix
    nvMolKit::launchTFDMatrixKernel(system.totalTFDWorkItems(),
                                    device.dihedralAngles.data(),
                                    device.torsionWeights.data(),
                                    device.torsionMaxDevs.data(),
                                    device.tfdAnglesI.data(),
                                    device.tfdAnglesJ.data(),
                                    device.tfdTorsStart.data(),
                                    device.tfdNumTorsions.data(),
                                    device.tfdOutIdx.data(),
                                    device.tfdOutput.data(),
                                    stream);

    gpuTFD.resize(device.tfdOutput.size());
    device.tfdOutput.copyToHost(gpuTFD.data(), gpuTFD.size());
    cudaStreamSynchronize(stream);
  }

  // Compare
  ASSERT_EQ(gpuTFD.size(), cpuTFD.size());
  for (size_t i = 0; i < cpuTFD.size(); ++i) {
    EXPECT_NEAR(gpuTFD[i], cpuTFD[i], kTolerance) << "Mismatch at TFD index " << i;
  }

  cudaStreamDestroy(stream);
}

TEST_F(TFDKernelsTest, BatchMultipleMolecules) {
  const std::vector<std::string> testSmiles = {
    "CCCC",         // n-butane
    "CCCCC",        // n-pentane
    "CCCCCC",       // n-hexane
    "c1ccccc1",     // benzene
    "CCO",          // ethanol
    "c1ccc(cc1)O",  // phenol
  };

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

  // Compute CPU reference
  auto cpuResults = cpuGenerator_.GetTFDMatrices(molPtrs, options);

  // Build batch system
  auto system = nvMolKit::buildTFDSystem(molPtrs, options);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<float> gpuTFDFlat;

  {
    nvMolKit::TFDSystemDevice device;
    nvMolKit::transferToDevice(system, device, stream);

    // Launch dihedral kernel
    nvMolKit::launchDihedralKernel(system.totalDihedralWorkItems(),
                                   device.positions.data(),
                                   device.confPositionStarts.data(),
                                   device.torsionAtoms.data(),
                                   device.dihedralConfIdx.data(),
                                   device.dihedralTorsIdx.data(),
                                   device.dihedralOutIdx.data(),
                                   device.dihedralAngles.data(),
                                   stream);

    // Launch TFD matrix kernel
    nvMolKit::launchTFDMatrixKernel(system.totalTFDWorkItems(),
                                    device.dihedralAngles.data(),
                                    device.torsionWeights.data(),
                                    device.torsionMaxDevs.data(),
                                    device.tfdAnglesI.data(),
                                    device.tfdAnglesJ.data(),
                                    device.tfdTorsStart.data(),
                                    device.tfdNumTorsions.data(),
                                    device.tfdOutIdx.data(),
                                    device.tfdOutput.data(),
                                    stream);

    gpuTFDFlat.resize(device.tfdOutput.size());
    if (!gpuTFDFlat.empty()) {
      device.tfdOutput.copyToHost(gpuTFDFlat.data(), gpuTFDFlat.size());
    }
    cudaStreamSynchronize(stream);
  }

  // Compare per molecule
  for (size_t m = 0; m < molPtrs.size(); ++m) {
    int outStart = system.tfdOutputStarts[m];
    int outEnd   = system.tfdOutputStarts[m + 1];

    ASSERT_EQ(static_cast<size_t>(outEnd - outStart), cpuResults[m].size()) << "Size mismatch for molecule " << m;

    for (int i = outStart; i < outEnd; ++i) {
      EXPECT_NEAR(gpuTFDFlat[i], cpuResults[m][i - outStart], kTolerance)
        << "Mismatch at molecule " << m << " TFD index " << (i - outStart);
    }
  }

  cudaStreamDestroy(stream);
}

// =============================================================================
// Direct RDKit reference
// =============================================================================

TEST_F(TFDKernelsTest, CompareWithRDKitReference) {
  // Compare GPU TFD output directly against pre-computed RDKit reference values.
  // Same reference values as test_tfd_cpu.cpp CompareWithRDKitReference.
  // No AddHs to avoid multi-quartet symmetric torsions (added in a later commit).
  //
  // Reference generated with RDKit Python:
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
  };
  // clang-format on

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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<float> gpuTFD;

    {
      nvMolKit::TFDSystemDevice device;
      nvMolKit::transferToDevice(system, device, stream);

      nvMolKit::launchDihedralKernel(system.totalDihedralWorkItems(),
                                     device.positions.data(),
                                     device.confPositionStarts.data(),
                                     device.torsionAtoms.data(),
                                     device.dihedralConfIdx.data(),
                                     device.dihedralTorsIdx.data(),
                                     device.dihedralOutIdx.data(),
                                     device.dihedralAngles.data(),
                                     stream);

      nvMolKit::launchTFDMatrixKernel(system.totalTFDWorkItems(),
                                      device.dihedralAngles.data(),
                                      device.torsionWeights.data(),
                                      device.torsionMaxDevs.data(),
                                      device.tfdAnglesI.data(),
                                      device.tfdAnglesJ.data(),
                                      device.tfdTorsStart.data(),
                                      device.tfdNumTorsions.data(),
                                      device.tfdOutIdx.data(),
                                      device.tfdOutput.data(),
                                      stream);

      gpuTFD.resize(device.tfdOutput.size());
      device.tfdOutput.copyToHost(gpuTFD.data(), gpuTFD.size());
      cudaStreamSynchronize(stream);
    }

    ASSERT_EQ(gpuTFD.size(), tc.reference.size());
    for (size_t i = 0; i < gpuTFD.size(); ++i) {
      EXPECT_NEAR(gpuTFD[i], tc.reference[i], kTolerance) << "TFD[" << i << "] mismatch with RDKit reference";
    }

    cudaStreamDestroy(stream);
  }
}

// =============================================================================
// Edge cases
// =============================================================================

TEST_F(TFDKernelsTest, TwoConformers) {
  // Minimum pair case: 2 conformers produce exactly 1 TFD pair
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 2);
  ASSERT_EQ(mol->getNumConformers(), 2);

  nvMolKit::TFDComputeOptions options;

  auto cpuTFD = cpuGenerator_.GetTFDMatrix(*mol, options);
  ASSERT_EQ(cpuTFD.size(), 1u);

  auto system = nvMolKit::buildTFDSystem(*mol, options);
  ASSERT_EQ(system.totalTFDWorkItems(), 1);
  ASSERT_EQ(system.totalTFDOutputs(), 1);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<float> gpuTFD;

  {
    nvMolKit::TFDSystemDevice device;
    nvMolKit::transferToDevice(system, device, stream);

    nvMolKit::launchDihedralKernel(system.totalDihedralWorkItems(),
                                   device.positions.data(),
                                   device.confPositionStarts.data(),
                                   device.torsionAtoms.data(),
                                   device.dihedralConfIdx.data(),
                                   device.dihedralTorsIdx.data(),
                                   device.dihedralOutIdx.data(),
                                   device.dihedralAngles.data(),
                                   stream);

    nvMolKit::launchTFDMatrixKernel(system.totalTFDWorkItems(),
                                    device.dihedralAngles.data(),
                                    device.torsionWeights.data(),
                                    device.torsionMaxDevs.data(),
                                    device.tfdAnglesI.data(),
                                    device.tfdAnglesJ.data(),
                                    device.tfdTorsStart.data(),
                                    device.tfdNumTorsions.data(),
                                    device.tfdOutIdx.data(),
                                    device.tfdOutput.data(),
                                    stream);

    gpuTFD.resize(device.tfdOutput.size());
    device.tfdOutput.copyToHost(gpuTFD.data(), gpuTFD.size());
    cudaStreamSynchronize(stream);
  }

  ASSERT_EQ(gpuTFD.size(), 1u);
  EXPECT_NEAR(gpuTFD[0], cpuTFD[0], kTolerance);

  cudaStreamDestroy(stream);
}

TEST_F(TFDKernelsTest, BatchWithZeroTorsionMolecule) {
  // A molecule with no torsions (ethane) should produce TFD=0
  // and not corrupt results of neighboring molecules in the batch
  auto ethane = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CC"));
  ASSERT_NE(ethane, nullptr);
  generateConformers(*ethane, 3);
  ASSERT_GE(ethane->getNumConformers(), 2);

  auto butane = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(butane, nullptr);
  generateConformers(*butane, 3);
  ASSERT_GE(butane->getNumConformers(), 2);

  // CPU reference for butane alone
  nvMolKit::TFDComputeOptions options;
  auto                        cpuButane = cpuGenerator_.GetTFDMatrix(*butane, options);

  // GPU batch: ethane first, then butane
  std::vector<const RDKit::ROMol*> molPtrs = {ethane.get(), butane.get()};
  auto                             system  = nvMolKit::buildTFDSystem(molPtrs, options);
  ASSERT_EQ(system.numMolecules(), 2);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<float> gpuTFD;

  {
    nvMolKit::TFDSystemDevice device;
    nvMolKit::transferToDevice(system, device, stream);

    nvMolKit::launchDihedralKernel(system.totalDihedralWorkItems(),
                                   device.positions.data(),
                                   device.confPositionStarts.data(),
                                   device.torsionAtoms.data(),
                                   device.dihedralConfIdx.data(),
                                   device.dihedralTorsIdx.data(),
                                   device.dihedralOutIdx.data(),
                                   device.dihedralAngles.data(),
                                   stream);

    nvMolKit::launchTFDMatrixKernel(system.totalTFDWorkItems(),
                                    device.dihedralAngles.data(),
                                    device.torsionWeights.data(),
                                    device.torsionMaxDevs.data(),
                                    device.tfdAnglesI.data(),
                                    device.tfdAnglesJ.data(),
                                    device.tfdTorsStart.data(),
                                    device.tfdNumTorsions.data(),
                                    device.tfdOutIdx.data(),
                                    device.tfdOutput.data(),
                                    stream);

    gpuTFD.resize(device.tfdOutput.size());
    if (!gpuTFD.empty()) {
      device.tfdOutput.copyToHost(gpuTFD.data(), gpuTFD.size());
    }
    cudaStreamSynchronize(stream);
  }

  // Check ethane: all TFD values should be 0 (no torsions)
  int ethaneStart = system.tfdOutputStarts[0];
  int ethaneEnd   = system.tfdOutputStarts[1];
  for (int i = ethaneStart; i < ethaneEnd; ++i) {
    EXPECT_NEAR(gpuTFD[i], 0.0f, kTolerance) << "Ethane (no torsions) should have TFD=0 at index " << i;
  }

  // Check butane: should match CPU reference (not corrupted by ethane)
  int butaneStart = system.tfdOutputStarts[1];
  int butaneEnd   = system.tfdOutputStarts[2];
  ASSERT_EQ(static_cast<size_t>(butaneEnd - butaneStart), cpuButane.size());
  for (int i = butaneStart; i < butaneEnd; ++i) {
    EXPECT_NEAR(gpuTFD[i], cpuButane[i - butaneStart], kTolerance)
      << "Butane TFD mismatch at index " << (i - butaneStart);
  }

  cudaStreamDestroy(stream);
}
