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

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "bfgs_common.h"
#include "bfgs_minimize.h"
#include "ff_device_collect.h"
#include "ff_utils.h"
#include "nvtx.h"
#include "openmp_helpers.h"
#include "uff_batched_forcefield.h"
#include "uff_flattened_builder.h"

namespace nvMolKit::UFF {

UFFMinimizeResult UFFMinimizeMoleculesConfs(std::vector<RDKit::ROMol*>& mols,
                                            const int                   maxIters,
                                            const double                gradTol,
                                            const std::vector<double>&  vdwThresholds,
                                            const std::vector<bool>&    ignoreInterfragInteractions,
                                            const std::vector<ForceFieldConstraints::PerMolConstraints>& constraints,
                                            const BatchHardwareOptions&                                  perfOptions,
                                            const CoordinateOutput                                       output,
                                            int                                                          targetGpu,
                                            const DeviceCoordResult* deviceInput) {
  ScopedNvtxRange fullRange("BFGS UFF Minimize Molecules Confs");

  if (vdwThresholds.size() != mols.size()) {
    throw std::invalid_argument("Expected one vdw threshold per molecule");
  }
  if (ignoreInterfragInteractions.size() != mols.size()) {
    throw std::invalid_argument("Expected one interfragment interaction flag per molecule");
  }
  if (!constraints.empty() && constraints.size() != mols.size()) {
    throw std::invalid_argument("Expected one PerMolConstraints entry per molecule");
  }
  if (deviceInput != nullptr && !constraints.empty()) {
    throw std::invalid_argument("Combining device_input with constraints is not supported in this release.");
  }

  const bool deviceOutput = output == CoordinateOutput::DEVICE;

  auto ctx = setupBatchExecution(perfOptions);

  if (deviceOutput) {
    if (targetGpu < 0) {
      targetGpu = ctx.devicesPerThread.empty() ? 0 : ctx.devicesPerThread.front();
    }
    if (std::find(ctx.devicesPerThread.begin(), ctx.devicesPerThread.end(), targetGpu) ==
        ctx.devicesPerThread.end()) {
      throw std::invalid_argument(
        "targetGpu " + std::to_string(targetGpu) +
        " is not in the configured set of execution GPUs; pass it via perfOptions.gpuIds first.");
    }
  }

  std::vector<std::vector<double>> moleculeEnergies;
  const auto                       allConformers = flattenConformers(mols, moleculeEnergies);

  std::vector<std::vector<int8_t>> moleculeConverged(mols.size());
  for (size_t i = 0; i < mols.size(); ++i) {
    moleculeConverged[i].resize(moleculeEnergies[i].size(), 0);
  }

  DeviceInputIndex deviceInputIndex;
  if (deviceInput != nullptr) {
    deviceInputIndex = buildDeviceInputIndex(*deviceInput, allConformers);
  }
  const bool useDeviceInput = deviceInput != nullptr;

  const size_t totalConformers    = allConformers.size();
  const size_t effectiveBatchSize = ctx.batchSize == 0 ? totalConformers : ctx.batchSize;
  if (totalConformers == 0) {
    if (deviceOutput) {
      std::vector<FFDeviceCoordCollector> emptyCollectors;
      return {{}, {}, finalizeOnTarget(emptyCollectors, targetGpu)};
    }
    return {moleculeEnergies, moleculeConverged, std::nullopt};
  }

  std::vector<ThreadLocalBuffers>     threadBuffers(ctx.numThreads);
  std::vector<FFDeviceCoordCollector> deviceCollectors(deviceOutput ? ctx.numThreads : 0);
  if (deviceOutput) {
    for (int threadId = 0; threadId < ctx.numThreads; ++threadId) {
      auto& collector  = deviceCollectors[threadId];
      collector.gpuId  = ctx.devicesPerThread[threadId];
      collector.stream = ctx.streamPool[threadId].stream();
      collector.positions.setStream(collector.stream);
      collector.energies.setStream(collector.stream);
      collector.converged.setStream(collector.stream);
    }
  }
  detail::OpenMPExceptionRegistry exceptionHandler;
#pragma omp parallel for num_threads(ctx.numThreads) schedule(dynamic) default(none) \
  shared(allConformers,                                                              \
           moleculeEnergies,                                                         \
           moleculeConverged,                                                        \
           totalConformers,                                                          \
           effectiveBatchSize,                                                       \
           maxIters,                                                                 \
           gradTol,                                                                  \
           vdwThresholds,                                                            \
           ignoreInterfragInteractions,                                              \
           constraints,                                                              \
           ctx,                                                                      \
           threadBuffers,                                                            \
           deviceCollectors,                                                         \
           deviceOutput,                                                             \
           useDeviceInput,                                                           \
           deviceInput,                                                              \
           deviceInputIndex,                                                         \
           exceptionHandler)
  for (size_t batchStart = 0; batchStart < totalConformers; batchStart += effectiveBatchSize) {
    try {
      ScopedNvtxRange singleBatchRange("OpenMP loop thread");
      ScopedNvtxRange setupBatchRange("OpenMP loop preprocessing");

      const int                            threadId     = omp_get_thread_num();
      const int                            executingGpu = ctx.devicesPerThread[threadId];
      const WithDevice                     dev(executingGpu);
      const size_t                         batchEnd = std::min(batchStart + effectiveBatchSize, totalConformers);
      std::vector<nvMolKit::ConformerInfo> batchConformers(allConformers.begin() + batchStart,
                                                           allConformers.begin() + batchEnd);

      std::vector<int> batchSrcIndices;
      std::vector<int> batchAtomCounts;
      if (useDeviceInput) {
        batchSrcIndices.reserve(batchConformers.size());
        batchAtomCounts.reserve(batchConformers.size());
        for (size_t k = 0; k < batchConformers.size(); ++k) {
          batchSrcIndices.push_back(deviceInputIndex.conformerIndexBy[batchStart + k]);
          batchAtomCounts.push_back(static_cast<int>(batchConformers[k].mol->getNumAtoms()));
        }
      }

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
        if (!constraints.empty()) {
          constraints[confInfo.molIdx].applyTo(ffParams, pos);
        }
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
      if (useDeviceInput) {
        broadcastDeviceInputBatch(*deviceInput, deviceInputIndex, batchSrcIndices, batchAtomCounts, executingGpu,
                                  streamPtr, positionsDevice);
      }
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
      setupBatchRange.pop();
      bfgsMinimizer.minimize(maxIters, gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

      if (deviceOutput) {
        appendBatch(batchConformers,
                    positionsDevice,
                    energyOutsDevice,
                    bfgsMinimizer.statuses_,
                    deviceCollectors[threadId]);
      } else {
        ScopedNvtxRange finalizeBatchRange("OpenMP loop finalizing batch");
        positionsDevice.copyToHost(buffers.positions.data(), positionsDevice.size());
        energyOutsDevice.copyToHost(buffers.energies.data(), energyOutsDevice.size());

        std::vector<int16_t> statusesHost(batchConformers.size());
        bfgsMinimizer.statuses_.copyToHost(statusesHost.data(), batchConformers.size());
        cudaStreamSynchronize(streamPtr);

        writeBackResults(batchConformers, conformerAtomStarts, buffers, moleculeEnergies);

        for (size_t i = 0; i < batchConformers.size(); ++i) {
          const auto& confInfo                                 = batchConformers[i];
          moleculeConverged[confInfo.molIdx][confInfo.confIdx] = static_cast<int8_t>(statusesHost[i] == 0);
        }
      }
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }
  exceptionHandler.rethrow();
  if (deviceOutput) {
    return {{}, {}, finalizeOnTarget(deviceCollectors, targetGpu)};
  }
  return {moleculeEnergies, moleculeConverged, std::nullopt};
}

std::vector<std::vector<double>> UFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                               const int                   maxIters,
                                                               const std::vector<double>&  vdwThresholds,
                                                               const std::vector<bool>&    ignoreInterfragInteractions,
                                                               const BatchHardwareOptions& perfOptions) {
  return UFFMinimizeMoleculesConfs(mols, maxIters, 1e-4, vdwThresholds, ignoreInterfragInteractions, {}, perfOptions)
    .energies;
}

}  // namespace nvMolKit::UFF
