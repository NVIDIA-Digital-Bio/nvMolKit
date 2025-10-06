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

#include "bfgs_mmff.h"

#include <GraphMol/ROMol.h>
#include <omp.h>

#include "bfgs_minimize.h"
#include "device.h"
#include "ff_utils.h"
#include "mmff_flattened_builder.h"
#include "openmp_helpers.h"

namespace nvMolKit::MMFF {

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                                const int                   maxIters,
                                                                const double                nonBondedThreshold,
                                                                const BatchHardwareOptions& perfOptions) {
  // Extract values from performance options
  const size_t batchSize = perfOptions.batchSize == -1 ? 500 : perfOptions.batchSize;

  for (size_t i = 0; i < mols.size(); ++i) {
    const auto* mol = mols[i];
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule pointer at index " + std::to_string(i));
    }
  }

  std::vector<int> gpuIds = perfOptions.gpuIds;
  if (gpuIds.empty()) {
    const int numDevices = countCudaDevices();
    if (numDevices == 0) {
      throw std::runtime_error("No CUDA devices found for MMFF relaxation");
    }
    gpuIds.resize(numDevices);
    std::iota(gpuIds.begin(), gpuIds.end(), 0);  // Fill with device IDs 0, 1, ..., numDevices-1
  }
  const int batchesPerGpu = perfOptions.batchesPerGpu == -1 ? 4 : perfOptions.batchesPerGpu;
  const int numThreads =
    perfOptions.batchesPerGpu > 0 ? batchesPerGpu * static_cast<int>(gpuIds.size()) : omp_get_max_threads();

  // Initialize result structure
  std::vector<std::vector<double>> moleculeEnergies(mols.size());

  // Flatten all conformers from all molecules for better load balancing
  struct ConformerInfo {
    RDKit::ROMol*     mol;
    size_t            molIdx;
    RDKit::Conformer* conformer;
    size_t            confIdx;
  };

  std::vector<ConformerInfo> allConformers;
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto* mol = mols[molIdx];
    moleculeEnergies[molIdx].resize(mol->getNumConformers());

    size_t confIdx = 0;
    for (auto confIter = mol->beginConformers(); confIter != mol->endConformers(); ++confIter, ++confIdx) {
      allConformers.push_back({mol, molIdx, &(**confIter), confIdx});
    }
  }

  // Calculate batch parameters for conformers
  const size_t totalConformers    = allConformers.size();
  const size_t effectiveBatchSize = (batchSize == 0) ? totalConformers : batchSize;

  if (totalConformers == 0) {
    return moleculeEnergies;  // Early return for empty input
  }

  // Create stream pool for better performance and profiling
  std::vector<nvMolKit::ScopedStream> streamPool;
  streamPool.reserve(numThreads);
  std::vector<int> devicesPerThread(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    const int        gpuId = gpuIds[i % gpuIds.size()];
    const WithDevice dev(gpuId);
    streamPool.emplace_back();
    devicesPerThread[i] = gpuId;  // Round-robin assignment of devices
  }
  detail::OpenMPExceptionRegistry exceptionHandler;
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) default(none) shared(allConformers,        \
                                                                                          moleculeEnergies,   \
                                                                                          totalConformers,    \
                                                                                          effectiveBatchSize, \
                                                                                          maxIters,           \
                                                                                          nonBondedThreshold, \
                                                                                          streamPool,         \
                                                                                          devicesPerThread,   \
                                                                                          exceptionHandler)
  for (size_t batchStart = 0; batchStart < totalConformers; batchStart += effectiveBatchSize) {
    try {
      const int        threadId = omp_get_thread_num();
      const WithDevice dev(devicesPerThread[threadId]);
      const size_t     batchEnd = std::min(batchStart + effectiveBatchSize, totalConformers);

      // Create batch subset of conformers
      std::vector<ConformerInfo> batchConformers(allConformers.begin() + batchStart, allConformers.begin() + batchEnd);

      // Get thread-local stream from pool
      cudaStream_t streamPtr = streamPool[threadId].stream();

      // Process this batch
      BatchedMolecularSystemHost    systemHost;
      BatchedMolecularDeviceBuffers systemDevice;
      std::vector<double>           pos;

      // Track conformer atom start positions for molecules with different sizes
      std::vector<uint32_t> conformerAtomStarts;
      uint32_t              currentAtomOffset = 0;

      // Prepare batch - each conformer becomes a separate "molecule" in the batch
      for (const auto& confInfo : batchConformers) {
        auto*          mol      = confInfo.mol;
        const uint32_t numAtoms = mol->getNumAtoms();

        auto             ffParams = constructForcefieldContribs(*mol, nonBondedThreshold);
        std::vector<int> atomNumbers;
        atomNumbers.reserve(numAtoms);
        for (uint32_t i = 0; i < numAtoms; ++i) {
          atomNumbers.push_back(mol->getAtomWithIdx(i)->getAtomicNum());
        }

        // Add this conformer to the batch
        conformerAtomStarts.push_back(currentAtomOffset);
        currentAtomOffset += numAtoms;

        nvMolKit::confPosToVect(*confInfo.conformer, pos);
        nvMolKit::MMFF::addMoleculeToBatch(ffParams, pos, systemHost, &atomNumbers);
      }

      // Send to device and set up streams
      nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost, systemDevice);
      nvMolKit::MMFF::setStreams(systemDevice, streamPtr);
      nvMolKit::MMFF::allocateIntermediateBuffers(systemHost, systemDevice);
      systemDevice.positions.setFromVector(systemHost.positions);
      systemDevice.grad.resize(systemHost.positions.size());
      systemDevice.grad.zero();

      auto eFunc = [&](const double* positions) { nvMolKit::MMFF::computeEnergy(systemDevice, positions, streamPtr); };
      auto gFunc = [&]() { nvMolKit::MMFF::computeGradients(systemDevice, streamPtr); };

      nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3, nvMolKit::DebugLevel::NONE, true, streamPtr);
      constexpr double             gradTol = 1e-4;  // hard-coded in RDKit.
      bfgsMinimizer.minimize(maxIters,
                             gradTol,
                             systemHost.indices.atomStarts,
                             systemDevice.indices.atomStarts,
                             systemDevice.positions,
                             systemDevice.grad,
                             systemDevice.energyOuts,
                             systemDevice.energyBuffer,
                             eFunc,
                             gFunc);

      std::vector<double> finalPos(systemHost.positions.size());
      systemDevice.positions.copyToHost(finalPos);

      // Compute final energies
      std::vector<double> gotEnergies(systemDevice.energyOuts.size(), 0.0);
      systemDevice.energyBuffer.zero();
      systemDevice.energyOuts.zero();
      nvMolKit::MMFF::computeEnergy(systemDevice, nullptr, streamPtr);
      systemDevice.energyOuts.copyToHost(gotEnergies);
      cudaStreamSynchronize(streamPtr);

      // Update conformer positions and store energies
      for (size_t i = 0; i < batchConformers.size(); ++i) {
        const auto&    confInfo     = batchConformers[i];
        const uint32_t numAtoms     = confInfo.mol->getNumAtoms();
        const uint32_t atomStartIdx = conformerAtomStarts[i];

        // Update conformer positions
        for (uint32_t j = 0; j < numAtoms; ++j) {
          confInfo.conformer->setAtomPos(j,
                                         RDGeom::Point3D(finalPos[3 * (atomStartIdx + j) + 0],
                                                         finalPos[3 * (atomStartIdx + j) + 1],
                                                         finalPos[3 * (atomStartIdx + j) + 2]));
        }

        // Store energy result - thread-safe since each thread writes to different indices
        moleculeEnergies[confInfo.molIdx][confInfo.confIdx] = gotEnergies[i];
      }
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }
  exceptionHandler.rethrow();
  return moleculeEnergies;
}

}  // namespace nvMolKit::MMFF
