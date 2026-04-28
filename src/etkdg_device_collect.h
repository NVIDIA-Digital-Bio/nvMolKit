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

#ifndef NVMOLKIT_ETKDG_DEVICE_COLLECT_H
#define NVMOLKIT_ETKDG_DEVICE_COLLECT_H

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "device_coord_result.h"
#include "device_vector.h"
#include "etkdg_impl.h"

namespace nvMolKit {
namespace detail {

/**
 * @brief Thread-local accumulator of ETKDG batch outputs in device-output mode.
 *
 * Each OpenMP worker writes into its own DeviceCoordCollector. Successful conformers from a
 * batch are appended to @ref positions in packed 3D layout (x,y,z per atom, conformer-major)
 * by @ref appendActive. Per-conformer metadata is held on the host (cheap) for later
 * stitching in @ref finalizeOnTarget.
 *
 * The accumulator and its buffers live on the GPU identified by @ref gpuId. All device
 * operations execute on @ref stream.
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
 * ETKDG's scheduler can dispatch many parallel attempts for a single molecule and they may all
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
 * @brief Append the active subset of an ETKDG batch's positions to @p collector.
 *
 * Reads the per-conformer @ref ETKDGContext::activeThisStage flags from device, compacts the
 * surviving conformers' positions in 4D->3D layout up to the per-molecule cap encoded in
 * @p cap, and appends them to @p collector.positions via a packing kernel. Updates
 * @p collector.atomCounts and @p collector.molIds for each accepted conformer.
 *
 * @param ctx                ETKDG batch context. Used for systemDevice.positions, systemHost.atomStarts,
 *                           and activeThisStage.
 * @param dim                Source dimensionality (4 for ETKDG).
 * @param batchGlobalMolIds  Length must equal @c ctx.systemHost.atomStarts.size() - 1; entry i is
 *                           the global molecule index of batch slot i.
 * @param cap                Shared cap state across all collectors; updated atomically.
 * @param collector          Thread-local accumulator to append into.
 *
 * Postcondition: @p collector buffers are extended and ready for downstream collection. No host
 * synchronization beyond the small `active` D2H copy; the actual position pack is async on
 * @p collector.stream.
 */
void appendActive(const ETKDGContext&      ctx,
                  int                      dim,
                  const std::vector<int>&  batchGlobalMolIds,
                  DeviceCoordCollectorCap& cap,
                  DeviceCoordCollector&    collector);

/**
 * @brief ETKDG pipeline stage that appends the surviving conformers of each batch to a
 *        thread-local @ref DeviceCoordCollector via @ref appendActive.
 *
 * Use this as the final stage of the per-batch pipeline in CoordinateOutput::DEVICE mode in
 * place of @c ETKDGUpdateConformersStage. The stage holds non-owning references to the
 * batch-mol-id mapping and the collector.
 */
class ETKDGCollectDeviceCoordsStage final : public ETKDGStage {
 public:
  ETKDGCollectDeviceCoordsStage(std::vector<int>         batchGlobalMolIds,
                                int                      dim,
                                DeviceCoordCollectorCap& cap,
                                DeviceCoordCollector&    collector)
      : batchGlobalMolIds_(std::move(batchGlobalMolIds)),
        dim_(dim),
        cap_(cap),
        collector_(collector) {}

  void        execute(ETKDGContext& ctx) override { appendActive(ctx, dim_, batchGlobalMolIds_, cap_, collector_); }
  std::string name() const override { return "Collect Device Coords"; }

 private:
  std::vector<int>         batchGlobalMolIds_;
  int                      dim_;
  DeviceCoordCollectorCap& cap_;
  DeviceCoordCollector&    collector_;
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
 *       `cudaStreamSynchronize` ensures the result is visible. ETKDG has no energies/converged
 *       fields, so those members of the returned result are left empty.
 */
DeviceCoordResult finalizeOnTarget(std::vector<DeviceCoordCollector>& collectors, int targetGpu);

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_ETKDG_DEVICE_COLLECT_H
