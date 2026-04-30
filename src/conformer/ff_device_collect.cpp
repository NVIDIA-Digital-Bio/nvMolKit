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

#include "ff_device_collect.h"

#include <GraphMol/ROMol.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "coord_collect.h"
#include "cuda_error_check.h"
#include "device.h"

namespace nvMolKit {

void appendBatch(const std::vector<ConformerInfo>& batchConformers,
                 const AsyncDeviceVector<double>&  positionsDevice,
                 const AsyncDeviceVector<double>&  energiesDevice,
                 const AsyncDeviceVector<int16_t>& statusesDevice,
                 FFDeviceCoordCollector&           collector) {
  const int numConformers = static_cast<int>(batchConformers.size());
  if (numConformers == 0) {
    return;
  }
  if (energiesDevice.size() != static_cast<size_t>(numConformers)) {
    throw std::invalid_argument("energiesDevice size does not match batch size");
  }
  if (statusesDevice.size() != static_cast<size_t>(numConformers)) {
    throw std::invalid_argument("statusesDevice size does not match batch size");
  }

  const WithDevice withDevice(collector.gpuId);

  std::vector<int16_t> statusesHost(numConformers);
  statusesDevice.copyToHost(statusesHost.data(), static_cast<size_t>(numConformers));
  cudaCheckError(cudaStreamSynchronize(statusesDevice.stream()));

  int totalNewAtoms = 0;
  for (const auto& confInfo : batchConformers) {
    totalNewAtoms += static_cast<int>(confInfo.mol->getNumAtoms());
  }
  if (positionsDevice.size() != static_cast<size_t>(totalNewAtoms) * 3) {
    throw std::invalid_argument("positionsDevice size does not match batch atom count");
  }

  const size_t prevPositionsSize = collector.positions.size();
  const size_t prevConfCount     = collector.atomCounts.size();
  collector.positions.resize(prevPositionsSize + static_cast<size_t>(totalNewAtoms) * 3);
  collector.energies.resize(prevConfCount + static_cast<size_t>(numConformers));
  collector.converged.resize(prevConfCount + static_cast<size_t>(numConformers));

  cudaCheckError(cudaMemcpyAsync(collector.positions.data() + prevPositionsSize,
                                 positionsDevice.data(),
                                 static_cast<size_t>(totalNewAtoms) * 3 * sizeof(double),
                                 cudaMemcpyDeviceToDevice,
                                 collector.stream));
  cudaCheckError(cudaMemcpyAsync(collector.energies.data() + prevConfCount,
                                 energiesDevice.data(),
                                 static_cast<size_t>(numConformers) * sizeof(double),
                                 cudaMemcpyDeviceToDevice,
                                 collector.stream));

  std::vector<int8_t> convergedHost(numConformers);
  for (int i = 0; i < numConformers; ++i) {
    convergedHost[i] = static_cast<int8_t>(statusesHost[i] == 0);
  }
  cudaCheckError(cudaMemcpyAsync(collector.converged.data() + prevConfCount,
                                 convergedHost.data(),
                                 static_cast<size_t>(numConformers) * sizeof(int8_t),
                                 cudaMemcpyHostToDevice,
                                 collector.stream));

  collector.atomCounts.reserve(prevConfCount + static_cast<size_t>(numConformers));
  collector.molIds.reserve(prevConfCount + static_cast<size_t>(numConformers));
  collector.confIds.reserve(prevConfCount + static_cast<size_t>(numConformers));
  for (const auto& confInfo : batchConformers) {
    collector.atomCounts.push_back(static_cast<int>(confInfo.mol->getNumAtoms()));
    collector.molIds.push_back(static_cast<int>(confInfo.molIdx));
    collector.confIds.push_back(static_cast<int>(confInfo.confIdx));
  }
  // Sync so the pageable host-side `convergedHost` we just used in cudaMemcpyAsync is no longer
  // needed once we return.
  cudaCheckError(cudaStreamSynchronize(collector.stream));
}

DeviceCoordResult finalizeOnTarget(std::vector<FFDeviceCoordCollector>& collectors, const int targetGpu) {
  for (const auto& collector : collectors) {
    if (collector.gpuId != targetGpu && !collector.atomCounts.empty()) {
      enablePeerAccess(targetGpu, collector.gpuId);
    }
  }

  int totalConformers = 0;
  int totalAtoms      = 0;
  for (const auto& collector : collectors) {
    totalConformers += static_cast<int>(collector.atomCounts.size());
    for (const int natoms : collector.atomCounts) {
      totalAtoms += natoms;
    }
  }

  const WithDevice  withTarget(targetGpu);
  ScopedStream      targetStream("FF DeviceCoord Finalize");
  DeviceCoordResult result;
  result.gpuId       = targetGpu;
  result.positions   = AsyncDeviceVector<double>(static_cast<size_t>(totalAtoms) * 3, targetStream.stream());
  result.atomStarts  = AsyncDeviceVector<int32_t>(static_cast<size_t>(totalConformers + 1), targetStream.stream());
  result.molIndices  = AsyncDeviceVector<int32_t>(static_cast<size_t>(totalConformers), targetStream.stream());
  result.confIndices = AsyncDeviceVector<int32_t>(static_cast<size_t>(totalConformers), targetStream.stream());
  result.energies    = AsyncDeviceVector<double>(static_cast<size_t>(totalConformers), targetStream.stream());
  result.converged   = AsyncDeviceVector<int8_t>(static_cast<size_t>(totalConformers), targetStream.stream());

  std::vector<int32_t> atomStartsHost(static_cast<size_t>(totalConformers + 1), 0);
  std::vector<int32_t> molIndicesHost(static_cast<size_t>(totalConformers), 0);
  std::vector<int32_t> confIndicesHost(static_cast<size_t>(totalConformers), 0);

  int confCursor = 0;
  int atomCursor = 0;
  for (auto& collector : collectors) {
    const int numConfs = static_cast<int>(collector.atomCounts.size());
    if (numConfs == 0) {
      continue;
    }

    copyDeviceToDeviceAsync(result.positions.data() + static_cast<size_t>(atomCursor) * 3,
                            collector.positions.data(),
                            collector.positions.size() * sizeof(double),
                            collector.gpuId,
                            collector.stream,
                            targetGpu,
                            targetStream.stream());
    copyDeviceToDeviceAsync(result.energies.data() + confCursor,
                            collector.energies.data(),
                            collector.energies.size() * sizeof(double),
                            collector.gpuId,
                            collector.stream,
                            targetGpu,
                            targetStream.stream());
    copyDeviceToDeviceAsync(result.converged.data() + confCursor,
                            collector.converged.data(),
                            collector.converged.size() * sizeof(int8_t),
                            collector.gpuId,
                            collector.stream,
                            targetGpu,
                            targetStream.stream());

    for (int conformerIdx = 0; conformerIdx < numConfs; ++conformerIdx) {
      atomStartsHost[static_cast<size_t>(confCursor)]  = atomCursor;
      molIndicesHost[static_cast<size_t>(confCursor)]  = collector.molIds[conformerIdx];
      confIndicesHost[static_cast<size_t>(confCursor)] = collector.confIds[conformerIdx];
      atomCursor += collector.atomCounts[conformerIdx];
      ++confCursor;
    }
  }
  atomStartsHost[static_cast<size_t>(totalConformers)] = atomCursor;

  if (totalConformers > 0) {
    result.atomStarts.copyFromHost(atomStartsHost);
    result.molIndices.copyFromHost(molIndicesHost);
    result.confIndices.copyFromHost(confIndicesHost);
  }
  cudaCheckError(cudaStreamSynchronize(targetStream.stream()));

  // The local ScopedStream will be destroyed - rebind every result buffer to the default stream
  // so subsequent operations on the result don't dereference a freed cudaStream_t.
  result.positions.setStream(nullptr);
  result.atomStarts.setStream(nullptr);
  result.molIndices.setStream(nullptr);
  result.confIndices.setStream(nullptr);
  result.energies.setStream(nullptr);
  result.converged.setStream(nullptr);
  return result;
}

}  // namespace nvMolKit
