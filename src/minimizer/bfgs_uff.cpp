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

#include "bfgs_uff.h"

#include <GraphMol/ROMol.h>
#include <omp.h>

#include <numeric>
#include <vector>

#include "bfgs_minimize.h"
#include "device.h"
#include "ff_utils.h"
#include "host_vector.h"
#include "nvtx.h"
#include "openmp_helpers.h"
#include "uff_batched_forcefield.h"
#include "uff_flattened_builder.h"

namespace nvMolKit::UFF {

namespace {

struct ThreadLocalBuffers {
  PinnedHostVector<double> positions;
  PinnedHostVector<double> energies;
  PinnedHostVector<double> initialPositions;

  void ensureCapacity(const size_t positionsSize, const size_t energiesSize) {
    constexpr double extraCapacityFactor = 1.3;
    const auto       newSize = static_cast<size_t>(static_cast<double>(positionsSize) * extraCapacityFactor);
    if (positions.size() < positionsSize) {
      positions.resize(newSize);
    }
    if (energies.size() < energiesSize) {
      energies.resize(static_cast<size_t>(static_cast<double>(energiesSize) * extraCapacityFactor));
    }
    if (initialPositions.size() < positionsSize) {
      initialPositions.resize(newSize);
    }
  }
};

}  // namespace

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
  for (size_t i = 0; i < mols.size(); ++i) {
    if (mols[i] == nullptr) {
      throw std::invalid_argument("Invalid molecule pointer at index " + std::to_string(i));
    }
  }

  const size_t batchSize = perfOptions.batchSize == -1 ? 500 : perfOptions.batchSize;

  std::vector<int> gpuIds = perfOptions.gpuIds;
  if (gpuIds.empty()) {
    const int numDevices = countCudaDevices();
    if (numDevices == 0) {
      throw std::runtime_error("No CUDA devices found for UFF relaxation");
    }
    gpuIds.resize(numDevices);
    std::iota(gpuIds.begin(), gpuIds.end(), 0);
  }
  const int batchesPerGpu = perfOptions.batchesPerGpu == -1 ? 4 : perfOptions.batchesPerGpu;
  const int numThreads =
    perfOptions.batchesPerGpu > 0 ? batchesPerGpu * static_cast<int>(gpuIds.size()) : omp_get_max_threads();

  std::vector<std::vector<double>> moleculeEnergies(mols.size());
  struct ConformerInfo {
    RDKit::ROMol*     mol;
    size_t            molIdx;
    RDKit::Conformer* conformer;
    int               conformerId;
    size_t            confIdx;
  };

  std::vector<ConformerInfo> allConformers;
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto* mol = mols[molIdx];
    moleculeEnergies[molIdx].resize(mol->getNumConformers());
    size_t confIdx = 0;
    for (auto confIter = mol->beginConformers(); confIter != mol->endConformers(); ++confIter, ++confIdx) {
      allConformers.push_back({mol, molIdx, &(**confIter), static_cast<int>((*confIter)->getId()), confIdx});
    }
  }

  const size_t totalConformers    = allConformers.size();
  const size_t effectiveBatchSize = batchSize == 0 ? totalConformers : batchSize;
  if (totalConformers == 0) {
    return moleculeEnergies;
  }

  std::vector<nvMolKit::ScopedStream> streamPool;
  streamPool.reserve(numThreads);
  std::vector<int> devicesPerThread(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    const int        gpuId = gpuIds[i % gpuIds.size()];
    const WithDevice dev(gpuId);
    streamPool.emplace_back();
    devicesPerThread[i] = gpuId;
  }

  std::vector<ThreadLocalBuffers>       threadBuffers(numThreads);
  detail::OpenMPExceptionRegistry exceptionHandler;
  setupRange.pop();
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) default(none) shared(allConformers,        \
                                                                                          moleculeEnergies,   \
                                                                                          totalConformers,    \
                                                                                          effectiveBatchSize, \
                                                                                          maxIters,           \
                                                                                          vdwThresholds,      \
                                                                                          ignoreInterfragInteractions, \
                                                                                          streamPool,         \
                                                                                          devicesPerThread,   \
                                                                                          threadBuffers,      \
                                                                                          exceptionHandler)
  for (size_t batchStart = 0; batchStart < totalConformers; batchStart += effectiveBatchSize) {
    try {
      ScopedNvtxRange singleBatchRange("OpenMP loop thread");
      ScopedNvtxRange setupBatchRange("OpenMP loop preprocessing");

      const int        threadId = omp_get_thread_num();
      const WithDevice dev(devicesPerThread[threadId]);
      const size_t     batchEnd = std::min(batchStart + effectiveBatchSize, totalConformers);
      std::vector<ConformerInfo> batchConformers(allConformers.begin() + batchStart, allConformers.begin() + batchEnd);

      cudaStream_t streamPtr = streamPool[threadId].stream();

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

      UFFBatchedForcefield     forcefield(systemHost, metadata, streamPtr);
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
        /*dataDim=*/3, nvMolKit::DebugLevel::NONE, true, streamPtr, nvMolKit::BfgsBackend::BATCHED);
      constexpr double gradTol = 1e-4;
      setupBatchRange.pop();
      bfgsMinimizer.minimize(maxIters, gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

      ScopedNvtxRange finalizeBatchRange("OpenMP loop finalizing batch");
      positionsDevice.copyToHost(buffers.positions.data(), positionsDevice.size());
      energyOutsDevice.copyToHost(buffers.energies.data(), energyOutsDevice.size());
      cudaStreamSynchronize(streamPtr);

      for (size_t i = 0; i < batchConformers.size(); ++i) {
        const auto&    confInfo     = batchConformers[i];
        const uint32_t numAtoms     = confInfo.mol->getNumAtoms();
        const uint32_t atomStartIdx = conformerAtomStarts[i];
        for (uint32_t j = 0; j < numAtoms; ++j) {
          confInfo.conformer->setAtomPos(j,
                                         RDGeom::Point3D(buffers.positions[3 * (atomStartIdx + j) + 0],
                                                         buffers.positions[3 * (atomStartIdx + j) + 1],
                                                         buffers.positions[3 * (atomStartIdx + j) + 2]));
        }
        moleculeEnergies[confInfo.molIdx][confInfo.confIdx] = buffers.energies[i];
      }
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }
  exceptionHandler.rethrow();
  return moleculeEnergies;
}

}  // namespace nvMolKit::UFF
