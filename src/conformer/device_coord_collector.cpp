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

#include "device_coord_collector.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <unordered_map>

#include "cuda_error_check.h"
#include "device.h"
#include "p2p.h"

namespace nvMolKit {
namespace detail {

DeviceCoordResult finalizeOnTarget(std::vector<DeviceCoordCollector>& collectors,
                                   const int                          targetGpu,
                                   const int                          nMols) {
  // Pre-enable peer access from target to every contributing GPU once.
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
  ScopedStream      targetStream("ETKDG DeviceCoord Finalize");
  DeviceCoordResult result;
  result.gpuId       = targetGpu;
  result.nMols       = nMols;
  result.positions   = AsyncDeviceVector<double>(static_cast<size_t>(totalAtoms) * 3, targetStream.stream());
  result.atomStarts  = AsyncDeviceVector<int32_t>(static_cast<size_t>(totalConformers + 1), targetStream.stream());
  result.molIndices  = AsyncDeviceVector<int32_t>(static_cast<size_t>(totalConformers), targetStream.stream());
  result.confIndices = AsyncDeviceVector<int32_t>(static_cast<size_t>(totalConformers), targetStream.stream());

  std::vector<int32_t> atomStartsHost(static_cast<size_t>(totalConformers + 1), 0);
  std::vector<int32_t> molIndicesHost(static_cast<size_t>(totalConformers), 0);
  std::vector<int32_t> confIndicesHost(static_cast<size_t>(totalConformers), 0);

  std::unordered_map<int, int> perMolCounter;
  int                          confCursor = 0;
  int                          atomCursor = 0;
  for (auto& collector : collectors) {
    const int numConfs = static_cast<int>(collector.atomCounts.size());
    if (numConfs == 0) {
      continue;
    }

    const size_t bytes = collector.positions.size() * sizeof(double);
    copyDeviceToDeviceAsync(result.positions.data() + static_cast<size_t>(atomCursor) * 3,
                            collector.positions.data(),
                            bytes,
                            collector.gpuId,
                            collector.stream,
                            targetGpu,
                            targetStream.stream());

    for (int conformerIdx = 0; conformerIdx < numConfs; ++conformerIdx) {
      atomStartsHost[static_cast<size_t>(confCursor)]  = atomCursor;
      const int molId                                  = collector.molIds[conformerIdx];
      molIndicesHost[static_cast<size_t>(confCursor)]  = molId;
      confIndicesHost[static_cast<size_t>(confCursor)] = perMolCounter[molId]++;
      atomCursor += collector.atomCounts[conformerIdx];
      ++confCursor;
    }
  }
  atomStartsHost[static_cast<size_t>(totalConformers)] = atomCursor;

  result.atomStarts.copyFromHost(atomStartsHost);
  if (totalConformers > 0) {
    result.molIndices.copyFromHost(molIndicesHost);
    result.confIndices.copyFromHost(confIndicesHost);
  }
  cudaCheckError(cudaStreamSynchronize(targetStream.stream()));

  // The local ScopedStream is about to be destroyed; rebind every result buffer to the default
  // stream so subsequent operations on the result do not dereference a freed cudaStream_t.
  result.positions.setStream(nullptr);
  result.atomStarts.setStream(nullptr);
  result.molIndices.setStream(nullptr);
  result.confIndices.setStream(nullptr);
  return result;
}

}  // namespace detail
}  // namespace nvMolKit
