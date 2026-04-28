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

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "bfgs_mmff.h"
#include "bfgs_uff.h"
#include "cuda_error_check.h"
#include "device.h"
#include "device_coord_result.h"

using namespace nvMolKit;

namespace {

nvMolKit::BatchHardwareOptions singleThreadOptions() {
  nvMolKit::BatchHardwareOptions options;
  options.preprocessingThreads = 1;
  options.batchSize            = 64;
  options.batchesPerGpu        = 1;
  options.gpuIds               = {0};
  return options;
}

template <typename T> std::vector<T> downloadDeviceVector(const AsyncDeviceVector<T>& vec) {
  std::vector<T> host(vec.size());
  if (!host.empty()) {
    vec.copyToHost(host);
    cudaCheckError(cudaStreamSynchronize(vec.stream()));
  }
  return host;
}

std::unique_ptr<RDKit::RWMol> embeddedMol(const std::string& smiles, int seed) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(smiles));
  if (!mol) {
    return nullptr;
  }
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords                      = true;
  params.randomSeed                           = seed;
  RDKit::DGeomHelpers::EmbedMolecule(*mol, params);
  return mol;
}

}  // namespace

TEST(MMFFDeviceOutput, EthanolMatchesShape) {
  auto ethanol = embeddedMol("CCO", 1234);
  ASSERT_NE(ethanol, nullptr);
  ASSERT_GT(ethanol->getNumConformers(), 0u);
  const unsigned int nAtoms = ethanol->getNumAtoms();

  std::vector<RDKit::ROMol*>            mols = {ethanol.get()};
  std::vector<nvMolKit::MMFFProperties> props(mols.size());

  auto result = nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(mols, /*maxIters=*/50, /*gradTol=*/1e-4, props, {},
                                                          singleThreadOptions(), nvMolKit::BfgsBackend::PER_MOLECULE,
                                                          nvMolKit::CoordinateOutput::DEVICE, /*targetGpu=*/0);
  ASSERT_TRUE(result.device.has_value());
  EXPECT_TRUE(result.energies.empty());
  EXPECT_TRUE(result.converged.empty());
  const auto& dev = *result.device;
  EXPECT_EQ(dev.gpuId, 0);

  const auto positions  = downloadDeviceVector(dev.positions);
  const auto atomStarts = downloadDeviceVector(dev.atomStarts);
  const auto molIndices = downloadDeviceVector(dev.molIndices);
  const auto confIdx    = downloadDeviceVector(dev.confIndices);
  const auto energies   = downloadDeviceVector(dev.energies);
  const auto converged  = downloadDeviceVector(dev.converged);

  ASSERT_EQ(molIndices.size(), 1u);
  ASSERT_EQ(confIdx.size(), 1u);
  ASSERT_EQ(atomStarts.size(), 2u);
  ASSERT_EQ(energies.size(), 1u);
  ASSERT_EQ(converged.size(), 1u);
  EXPECT_EQ(molIndices[0], 0);
  EXPECT_EQ(confIdx[0], 0);
  EXPECT_EQ(atomStarts[0], 0);
  EXPECT_EQ(static_cast<size_t>(atomStarts[1]), nAtoms);
  ASSERT_EQ(positions.size(), static_cast<size_t>(nAtoms) * 3);

  bool anyNonZero = false;
  for (const double pos : positions) {
    EXPECT_TRUE(std::isfinite(pos));
    if (std::abs(pos) > 1e-9) {
      anyNonZero = true;
    }
  }
  EXPECT_TRUE(anyNonZero);
  EXPECT_TRUE(std::isfinite(energies[0]));
  EXPECT_TRUE(converged[0] == 0 || converged[0] == 1);

  // Host RDKit conformer must remain untouched in DEVICE mode.
  const auto& origConf = ethanol->getConformer(0);
  // Compare original conformer (pre-minimize) - it will not equal device output, but it should
  // still be the same as before we called the device-mode optimize. We sanity-check only that
  // the host conformer count is unchanged.
  EXPECT_EQ(ethanol->getNumConformers(), 1u);
  (void)origConf;
}

TEST(MMFFDeviceOutput, MultipleMoleculesCorrectIndices) {
  auto methanol = embeddedMol("CO", 7);
  auto propanol = embeddedMol("CCCO", 7);
  ASSERT_NE(methanol, nullptr);
  ASSERT_NE(propanol, nullptr);

  std::vector<RDKit::ROMol*>            mols  = {methanol.get(), propanol.get()};
  std::vector<nvMolKit::MMFFProperties> props(mols.size());

  auto result = nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(mols, 30, 1e-4, props, {}, singleThreadOptions(),
                                                          nvMolKit::BfgsBackend::PER_MOLECULE,
                                                          nvMolKit::CoordinateOutput::DEVICE, /*targetGpu=*/0);
  ASSERT_TRUE(result.device.has_value());
  const auto& dev        = *result.device;
  const auto  molIndices = downloadDeviceVector(dev.molIndices);
  const auto  atomStarts = downloadDeviceVector(dev.atomStarts);
  ASSERT_EQ(molIndices.size(), 2u);
  // Both molecules have one conformer each; output order matches input order in single-thread.
  for (size_t conformerIdx = 0; conformerIdx < molIndices.size(); ++conformerIdx) {
    const int molId          = molIndices[conformerIdx];
    const int natomsThisConf = atomStarts[conformerIdx + 1] - atomStarts[conformerIdx];
    if (molId == 0) {
      EXPECT_EQ(static_cast<unsigned int>(natomsThisConf), methanol->getNumAtoms());
    } else if (molId == 1) {
      EXPECT_EQ(static_cast<unsigned int>(natomsThisConf), propanol->getNumAtoms());
    } else {
      FAIL() << "Unexpected molId " << molId;
    }
  }
}

TEST(MMFFDeviceOutput, EmptyInputProducesEmptyResult) {
  std::vector<RDKit::ROMol*>            mols;
  std::vector<nvMolKit::MMFFProperties> props;
  auto result = nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(mols, 10, 1e-4, props, {}, singleThreadOptions(),
                                                          nvMolKit::BfgsBackend::PER_MOLECULE,
                                                          nvMolKit::CoordinateOutput::DEVICE, /*targetGpu=*/0);
  ASSERT_TRUE(result.device.has_value());
  EXPECT_EQ(result.device->positions.size(), 0u);
  // atomStarts has length n_conformers + 1; for n=0 it is a single trailing zero.
  EXPECT_EQ(result.device->atomStarts.size(), 1u);
  EXPECT_EQ(result.device->molIndices.size(), 0u);
}

TEST(UFFDeviceOutput, EthanolMatchesShape) {
  auto ethanol = embeddedMol("CCO", 99);
  ASSERT_NE(ethanol, nullptr);
  const unsigned int nAtoms = ethanol->getNumAtoms();

  std::vector<RDKit::ROMol*> mols = {ethanol.get()};
  const std::vector<double>  vdw(mols.size(), 100.0);
  const std::vector<bool>    ignore(mols.size(), false);

  auto result =
    nvMolKit::UFF::UFFMinimizeMoleculesConfs(mols, /*maxIters=*/50, /*gradTol=*/1e-4, vdw, ignore, {},
                                             singleThreadOptions(), nvMolKit::CoordinateOutput::DEVICE,
                                             /*targetGpu=*/0);
  ASSERT_TRUE(result.device.has_value());
  EXPECT_TRUE(result.energies.empty());
  const auto& dev        = *result.device;
  EXPECT_EQ(dev.gpuId, 0);
  const auto positions   = downloadDeviceVector(dev.positions);
  const auto atomStarts  = downloadDeviceVector(dev.atomStarts);
  const auto molIndices  = downloadDeviceVector(dev.molIndices);
  const auto energies    = downloadDeviceVector(dev.energies);
  const auto converged   = downloadDeviceVector(dev.converged);

  ASSERT_EQ(molIndices.size(), 1u);
  ASSERT_EQ(atomStarts.size(), 2u);
  EXPECT_EQ(static_cast<size_t>(atomStarts[1]), nAtoms);
  ASSERT_EQ(positions.size(), static_cast<size_t>(nAtoms) * 3);
  ASSERT_EQ(energies.size(), 1u);
  ASSERT_EQ(converged.size(), 1u);
  for (const double pos : positions) {
    EXPECT_TRUE(std::isfinite(pos));
  }
  EXPECT_TRUE(std::isfinite(energies[0]));
  EXPECT_TRUE(converged[0] == 0 || converged[0] == 1);
}
