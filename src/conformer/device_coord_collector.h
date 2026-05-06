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

#ifndef NVMOLKIT_DEVICE_COORD_COLLECTOR_H
#define NVMOLKIT_DEVICE_COORD_COLLECTOR_H

#include <cuda_runtime.h>

#include <mutex>
#include <unordered_map>
#include <vector>

#include "device_coord_result.h"
#include "device_vector.h"

namespace nvMolKit {
namespace detail {

/**
 * @brief Per-thread / per-GPU accumulator of conformer batch outputs in device-output mode.
 *
 * Each OpenMP worker writes into its own DeviceCoordCollector. Producers (ETKDG, FF, ...) append
 * conformers to @ref positions in packed 3D layout (x,y,z per atom, conformer-major) and record
 * per-conformer metadata on the host (cheap) for later stitching in @ref finalizeOnTarget.
 *
 * The accumulator and its buffers live on the GPU identified by @ref gpuId. All device operations
 * execute on @ref stream.
 */
struct DeviceCoordCollector {
  int                       gpuId  = -1;
  cudaStream_t              stream = nullptr;
  AsyncDeviceVector<double> positions;   //!< Packed 3D, length = sum(atomCounts)*3
  std::vector<int>          atomCounts;  //!< One per accumulated conformer
  std::vector<int>          molIds;      //!< Global molecule index per accumulated conformer
};

/**
 * @brief Shared cap-tracking state across DeviceCoordCollectors.
 *
 * Producers can dispatch many parallel attempts for a single molecule and they may all
 * succeed in the same iteration; only @c maxConformersPerMol of them should appear in the final
 * output. This struct provides shared bookkeeping so that worker threads collectively keep at
 * most that many conformers per molecule. The mutex guards reads/writes to @ref keptPerMol; an
 * @c -1 value of @ref maxConformersPerMol disables the cap.
 */
struct DeviceCoordCollectorCap {
  std::mutex                   mutex;
  std::unordered_map<int, int> keptPerMol;
  int                          maxConformersPerMol = -1;
};

/**
 * @brief Stitch all per-thread DeviceCoordCollectors into a single DeviceCoordResult on @p targetGpu.
 *
 * Concatenates per-thread positions onto @p targetGpu using @ref copyDeviceToDeviceAsync,
 * computes CSR `atomStarts`, and assigns per-molecule `confIndices` deterministically by walking
 * partials in the supplied order and counting per molecule. All resulting buffers are allocated
 * and live on @p targetGpu.
 *
 * @note This call is synchronous on the target stream by the time it returns: every contributing
 *       partial stream has been waited on via cross-stream events, and a final
 *       `cudaStreamSynchronize` ensures the result is visible. The returned result's `energies`
 *       and `converged` members are left empty; producers that have those values (e.g. FF)
 *       populate them on the result after this call.
 */
DeviceCoordResult finalizeOnTarget(std::vector<DeviceCoordCollector>& collectors, int targetGpu, int nMols);

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_COORD_COLLECTOR_H
