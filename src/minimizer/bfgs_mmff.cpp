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

#include <unordered_map>

#include "bfgs_common.h"
#include "bfgs_minimize.h"
#include "ff_utils.h"
#include "mmff_batched_forcefield.h"
#include "mmff_flattened_builder.h"
#include "nvtx.h"
#include "openmp_helpers.h"

namespace nvMolKit::MMFF {

//! Cached molecule-specific preprocessing
struct CachedMoleculeData {
  EnergyForceContribsHost ffParams;
};

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                                const int                   maxIters,
                                                                const MMFFProperties&       properties,
                                                                const BatchHardwareOptions& perfOptions,
                                                                const BfgsBackend           backend) {
  return MMFFOptimizeMoleculesConfsBfgs(mols,
                                        maxIters,
                                        std::vector<MMFFProperties>(mols.size(), properties),
                                        perfOptions,
                                        backend);
}

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>&        mols,
                                                                const int                          maxIters,
                                                                const std::vector<MMFFProperties>& properties,
                                                                const BatchHardwareOptions&        perfOptions,
                                                                const BfgsBackend                  backend) {
  ScopedNvtxRange fullMinimizeRange("BFGS MMFF Optimize Molecules Confs");
  ScopedNvtxRange setupRange("BFGS MMFF Optimize Molecules Confs");

  if (properties.size() != mols.size()) {
    throw std::invalid_argument("Expected one MMFFProperties entry per molecule");
  }

  auto                             ctx = setupBatchExecution(perfOptions);
  std::vector<std::vector<double>> moleculeEnergies;
  const auto                       allConformers = flattenConformers(mols, moleculeEnergies);

  const size_t totalConformers    = allConformers.size();
  const size_t effectiveBatchSize = (ctx.batchSize == 0) ? totalConformers : ctx.batchSize;

  if (totalConformers == 0) {
    return moleculeEnergies;
  }

  std::vector<ThreadLocalBuffers> threadBuffers(ctx.numThreads);
  detail::OpenMPExceptionRegistry exceptionHandler;
  setupRange.pop();
#pragma omp parallel for num_threads(ctx.numThreads) schedule(dynamic) default(none) shared(allConformers,        \
                                                                                              moleculeEnergies,   \
                                                                                              totalConformers,    \
                                                                                              effectiveBatchSize, \
                                                                                              maxIters,           \
                                                                                              properties,         \
                                                                                              ctx,                \
                                                                                              threadBuffers,      \
                                                                                              backend,            \
                                                                                              exceptionHandler)
  for (size_t batchStart = 0; batchStart < totalConformers; batchStart += effectiveBatchSize) {
    try {
      std::unordered_map<RDKit::ROMol*, CachedMoleculeData> moleculeCache;
      ScopedNvtxRange                                       singleBatchRange("OpenMP loop thread");
      ScopedNvtxRange                                       setupBatchRange("OpenMP loop preprocessing");
      const int                                             threadId = omp_get_thread_num();
      const WithDevice                                      dev(ctx.devicesPerThread[threadId]);
      const size_t batchEnd = std::min(batchStart + effectiveBatchSize, totalConformers);

      std::vector<nvMolKit::ConformerInfo> batchConformers(allConformers.begin() + batchStart,
                                                           allConformers.begin() + batchEnd);

      cudaStream_t streamPtr = ctx.streamPool[threadId].stream();

      // Process this batch
      BatchedMolecularSystemHost    systemHost;
      BatchedMolecularDeviceBuffers systemDevice;
      BatchedForcefieldMetadata     metadata;
      std::vector<double>           pos;

      // Track conformer atom start positions for molecules with different sizes
      std::vector<uint32_t> conformerAtomStarts;
      uint32_t              currentAtomOffset = 0;

      // Prepare batch - each conformer becomes a separate "molecule" in the batch
      for (const auto& confInfo : batchConformers) {
        auto*          mol      = confInfo.mol;
        const uint32_t numAtoms = mol->getNumAtoms();

        // Look up or compute cached forcefield parameters and atom numbers
        auto it = moleculeCache.find(mol);
        if (it == moleculeCache.end()) {
          ScopedNvtxRange    computeCacheRange("Preprocess single molecule");
          CachedMoleculeData cached;
          cached.ffParams = constructForcefieldContribs(*mol, properties[confInfo.molIdx]);
          it              = moleculeCache.insert({mol, std::move(cached)}).first;
        }
        auto&           ffParams = it->second.ffParams;
        ScopedNvtxRange addToBatchRange("Add conformer to batch data");
        // Add this conformer to the batch
        conformerAtomStarts.push_back(currentAtomOffset);
        currentAtomOffset += numAtoms;

        nvMolKit::confPosToVect(*confInfo.conformer, pos);
        nvMolKit::MMFF::addMoleculeToBatch(ffParams, pos, systemHost, &metadata, confInfo.molIdx, confInfo.confIdx);
      }

      // Get thread-local buffers and ensure they have enough capacity
      auto& buffers = threadBuffers[threadId];
      buffers.ensureCapacity(systemHost.positions.size(), batchConformers.size());

      // Copy to pinned memory for async transfer
      std::copy(systemHost.positions.begin(), systemHost.positions.end(), buffers.initialPositions.begin());

      nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3, nvMolKit::DebugLevel::NONE, true, streamPtr, backend);
      constexpr double             gradTol          = 1e-4;  // hard-coded in RDKit.
      const auto                   effectiveBackend = bfgsMinimizer.resolveBackend(systemHost.indices.atomStarts);
      setupBatchRange.pop();

      if (effectiveBackend == BfgsBackend::BATCHED) {
        MMFFBatchedForcefield     forcefield(systemHost, metadata, streamPtr);
        AsyncDeviceVector<double> positionsDevice;
        AsyncDeviceVector<double> gradDevice;
        AsyncDeviceVector<double> energyOutsDevice;
        positionsDevice.setStream(streamPtr);
        gradDevice.setStream(streamPtr);
        energyOutsDevice.setStream(streamPtr);
        positionsDevice.resize(systemHost.positions.size());
        positionsDevice.copyFromHost(buffers.initialPositions.data(), systemHost.positions.size());
        gradDevice.resize(systemHost.positions.size());
        gradDevice.zero();
        energyOutsDevice.resize(batchConformers.size());
        energyOutsDevice.zero();

        bfgsMinimizer.minimize(maxIters, gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

        ScopedNvtxRange finalizeBatchRange("OpenMP loop finalizing batch");
        positionsDevice.copyToHost(buffers.positions.data(), positionsDevice.size());
        energyOutsDevice.copyToHost(buffers.energies.data(), energyOutsDevice.size());
        cudaStreamSynchronize(streamPtr);
      } else {
        nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost, systemDevice);
        nvMolKit::MMFF::setStreams(systemDevice, streamPtr);
        nvMolKit::MMFF::allocateIntermediateBuffers(systemHost, systemDevice);
        systemDevice.positions.resize(systemHost.positions.size());
        systemDevice.positions.copyFromHost(buffers.initialPositions.data(), systemHost.positions.size());
        systemDevice.grad.resize(systemHost.positions.size());
        systemDevice.grad.zero();

        bfgsMinimizer.minimizeWithMMFF(maxIters, gradTol, systemHost.indices.atomStarts, systemDevice);

        ScopedNvtxRange finalizeBatchRange("OpenMP loop finalizing batch");
        systemDevice.positions.copyToHost(buffers.positions.data(), systemDevice.positions.size());
        systemDevice.energyOuts.copyToHost(buffers.energies.data(), systemDevice.energyOuts.size());
        cudaStreamSynchronize(streamPtr);
      }

      writeBackResults(batchConformers, conformerAtomStarts, buffers, moleculeEnergies);
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }
  exceptionHandler.rethrow();
  return moleculeEnergies;
}

}  // namespace nvMolKit::MMFF
