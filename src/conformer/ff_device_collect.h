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

#ifndef NVMOLKIT_FF_DEVICE_COLLECT_H
#define NVMOLKIT_FF_DEVICE_COLLECT_H

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "conformer_info.h"
#include "device_coord_result.h"
#include "device_vector.h"

namespace nvMolKit {

/**
 * @brief Thread-local accumulator for MMFF/UFF batch outputs in device-output mode.
 *
 * Each OMP worker writes into its own FFDeviceCoordCollector. After a batch's BFGS minimization
 * completes, @ref appendBatch concatenates the batch's positions, energies, and convergence flags
 * onto the collector. Per-conformer atom counts and (molIdx, confIdx) labels are kept on the
 * host for later stitching by @ref finalizeOnTarget.
 *
 * All device buffers live on the GPU identified by @ref gpuId. All device operations execute on
 * @ref stream.
 */
struct FFDeviceCoordCollector {
  int                       gpuId  = -1;
  cudaStream_t              stream = nullptr;
  AsyncDeviceVector<double> positions;   //!< 3D, length = sum(atomCounts) * 3
  AsyncDeviceVector<double> energies;    //!< length = atomCounts.size()
  AsyncDeviceVector<int8_t> converged;   //!< length = atomCounts.size()
  std::vector<int>          atomCounts;  //!< One entry per accumulated conformer
  std::vector<int>          molIds;      //!< Original-input molecule index per conformer
  std::vector<int>          confIds;     //!< Per-molecule conformer index per conformer
};

/**
 * @brief Append the results of one MMFF/UFF batch to a thread-local collector.
 *
 * @param batchConformers   Per-conformer metadata (molIdx, confIdx) for this batch in batch order.
 * @param positionsDevice   Device buffer holding the optimized 3D positions in CSR layout
 *                          (atom-major within each conformer; conformers laid out in batch order).
 *                          Length must equal `sum(numAtoms_per_conf) * 3`.
 * @param energiesDevice    Device buffer of length @c batchConformers.size() with final energies.
 * @param statusesDevice    Device buffer of length @c batchConformers.size() with BFGS statuses
 *                          (0 = converged); will be downloaded and converted to int8_t flags.
 * @param collector         Thread-local accumulator to append into.
 *
 * The function syncs @p collector.stream once at the end so the small host-side metadata (read
 * via the statuses copy) reflects the just-completed minimization.
 */
void appendBatch(const std::vector<ConformerInfo>& batchConformers,
                 const AsyncDeviceVector<double>&  positionsDevice,
                 const AsyncDeviceVector<double>&  energiesDevice,
                 const AsyncDeviceVector<int16_t>& statusesDevice,
                 FFDeviceCoordCollector&           collector);

/**
 * @brief Stitch all per-thread FFDeviceCoordCollectors into a single DeviceCoordResult on @p targetGpu.
 *
 * Concatenates per-thread positions, energies, and convergence flags onto @p targetGpu using
 * @ref copyDeviceToDeviceAsync, computes CSR `atomStarts`, and copies the per-conformer
 * (molIndices, confIndices) host labels up to the target. All resulting buffers live on
 * @p targetGpu and are bound to the default stream of that GPU before returning, after a
 * `cudaStreamSynchronize` on the local target stream.
 */
DeviceCoordResult finalizeOnTarget(std::vector<FFDeviceCoordCollector>& collectors, int targetGpu, int nMols);

/**
 * @brief Host-side index over a DeviceCoordResult used as starting coordinates.
 *
 * Built once per minimization call; downloads atom_starts, mol_indices, and conf_indices from
 * the input device result and validates that the per-conformer (molIdx, confIdx) labels match
 * the host-side flattened conformer list one-to-one. Provides the source GPU offsets needed for
 * cross-GPU broadcasting in @ref broadcastDeviceInputBatch.
 */
struct DeviceInputIndex {
  int                  sourceGpu = -1;
  std::vector<int32_t> atomStartsHost;    //!< length n + 1
  std::vector<int32_t> conformerIndexBy;  //!< maps allConformers index -> source conformer index
};

/**
 * @brief Build and validate a DeviceInputIndex against the host-flattened conformer list.
 *
 * @param deviceInput Source device coordinates (must be non-null and on a known GPU).
 * @param allConformers Host-flattened ConformerInfo list (one entry per RDKit conformer,
 *                      input-mol order then conformer order).
 * @throws std::invalid_argument if sizes or (molIdx, confIdx) labels do not align.
 */
DeviceInputIndex buildDeviceInputIndex(const DeviceCoordResult&          deviceInput,
                                       const std::vector<ConformerInfo>& allConformers);

/**
 * @brief Copy the positions for a single batch from the input device result into @p positionsDevice.
 *
 * For each batch slot i, copies @p batchAtomCounts[i] * 3 doubles from
 * `deviceInput.positions[index.atomStartsHost[batchSrcIndices[i]] * 3 ..]` into
 * `positionsDevice[dstOffset(i) * 3 ..]`, where dstOffset(i) is the running prefix sum of
 * @p batchAtomCounts.
 *
 * Caller responsibilities:
 *   - @p positionsDevice must already be sized to hold all of the batch's atom positions
 *     (`sum(batchAtomCounts) * 3` doubles) in CSR order matching the batch.
 *   - The current device should be @p executingGpu (caller manages a WithDevice scope).
 *   - @p batchSrcIndices and @p batchAtomCounts must have the same length.
 */
void broadcastDeviceInputBatch(const DeviceCoordResult&   deviceInput,
                               const DeviceInputIndex&    index,
                               const std::vector<int>&    batchSrcIndices,
                               const std::vector<int>&    batchAtomCounts,
                               int                        executingGpu,
                               cudaStream_t               executingStream,
                               AsyncDeviceVector<double>& positionsDevice);

}  // namespace nvMolKit

#endif  // NVMOLKIT_FF_DEVICE_COLLECT_H
