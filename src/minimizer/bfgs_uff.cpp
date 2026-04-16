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

#include "bfgs_uff.h"

#include <GraphMol/ROMol.h>
#include <omp.h>

#include <vector>

#include "bfgs_common.h"
#include "bfgs_minimize.h"
#include "ff_utils.h"
#include "nvtx.h"
#include "openmp_helpers.h"
#include "uff_batched_forcefield.h"
#include "uff_flattened_builder.h"

namespace nvMolKit::UFF {

std::vector<std::vector<double>> UFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                               const int                   maxIters,
                                                               const std::vector<double>&  vdwThresholds,
                                                               const std::vector<bool>&    ignoreInterfragInteractions,
                                                               const BatchHardwareOptions& perfOptions) {
  ScopedNvtxRange fullMinimizeRange("BFGS UFF Optimize Molecules Confs");
  ScopedNvtxRange setupRange("BFGS UFF Optimize Molecules Confs");

  if (vdwThresholds.size() != mols.size()) {
    throw std::invalid_argument("Expected one vdw threshold per molecule");
  }
  if (ignoreInterfragInteractions.size() != mols.size()) {
    throw std::invalid_argument("Expected one interfragment interaction flag per molecule");
  }

  auto                             ctx = setupBatchExecution(perfOptions);
  std::vector<std::vector<double>> moleculeEnergies;
  const auto                       allConformers = flattenConformers(mols, moleculeEnergies);

  const size_t totalConformers    = allConformers.size();
  const size_t effectiveBatchSize = ctx.batchSize == 0 ? totalConformers : ctx.batchSize;
  if (totalConformers == 0) {
    return moleculeEnergies;
  }

  std::vector<ThreadLocalBuffers> threadBuffers(ctx.numThreads);
  detail::OpenMPExceptionRegistry exceptionHandler;
  setupRange.pop();
#pragma omp parallel for num_threads(ctx.numThreads) schedule(dynamic) default(none) \
  shared(allConformers,                                                              \
           moleculeEnergies,                                                         \
           totalConformers,                                                          \
           effectiveBatchSize,                                                       \
           maxIters,                                                                 \
           vdwThresholds,                                                            \
           ignoreInterfragInteractions,                                              \
           ctx,                                                                      \
           threadBuffers,                                                            \
           exceptionHandler)
  for (size_t batchStart = 0; batchStart < totalConformers; batchStart += effectiveBatchSize) {
    try {
      ScopedNvtxRange singleBatchRange("OpenMP loop thread");
      ScopedNvtxRange setupBatchRange("OpenMP loop preprocessing");

      const int                            threadId = omp_get_thread_num();
      const WithDevice                     dev(ctx.devicesPerThread[threadId]);
      const size_t                         batchEnd = std::min(batchStart + effectiveBatchSize, totalConformers);
      std::vector<nvMolKit::ConformerInfo> batchConformers(allConformers.begin() + batchStart,
                                                           allConformers.begin() + batchEnd);

      cudaStream_t streamPtr = ctx.streamPool[threadId].stream();

      BatchedMolecularSystemHost systemHost;
      BatchedForcefieldMetadata  metadata;
      std::vector<uint32_t>      conformerAtomStarts;
      uint32_t                   currentAtomOffset = 0;
      std::vector<double>        pos;

      for (const auto& confInfo : batchConformers) {
        const uint32_t numAtoms = confInfo.mol->getNumAtoms();
        conformerAtomStarts.push_back(currentAtomOffset);
        currentAtomOffset += numAtoms;

        nvMolKit::confPosToVect(*confInfo.conformer, pos);
        auto ffParams = constructForcefieldContribs(*confInfo.mol,
                                                    vdwThresholds[confInfo.molIdx],
                                                    confInfo.conformerId,
                                                    ignoreInterfragInteractions[confInfo.molIdx]);
        addMoleculeToBatch(ffParams, pos, systemHost, metadata, confInfo.molIdx, static_cast<int>(confInfo.confIdx));
      }

      auto& buffers = threadBuffers[threadId];
      buffers.ensureCapacity(systemHost.positions.size(), batchConformers.size());
      std::copy(systemHost.positions.begin(), systemHost.positions.end(), buffers.initialPositions.begin());

      UFFBatchedForcefield      forcefield(systemHost, metadata, streamPtr);
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

      nvMolKit::BfgsBatchMinimizer bfgsMinimizer(
        /*dataDim=*/3,
        nvMolKit::DebugLevel::NONE,
        true,
        streamPtr,
        nvMolKit::BfgsBackend::BATCHED);
      constexpr double gradTol = 1e-4;
      setupBatchRange.pop();
      bfgsMinimizer.minimize(maxIters, gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

      ScopedNvtxRange finalizeBatchRange("OpenMP loop finalizing batch");
      positionsDevice.copyToHost(buffers.positions.data(), positionsDevice.size());
      energyOutsDevice.copyToHost(buffers.energies.data(), energyOutsDevice.size());
      cudaStreamSynchronize(streamPtr);

      writeBackResults(batchConformers, conformerAtomStarts, buffers, moleculeEnergies);
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }
  exceptionHandler.rethrow();
  return moleculeEnergies;
}

}  // namespace nvMolKit::UFF
