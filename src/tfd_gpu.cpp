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

#include "tfd_gpu.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include "tfd_kernels.h"

namespace nvMolKit {

// ========== TFDGpuResult ==========

std::vector<double> TFDGpuResult::extractMolecule(int molIdx) const {
  if (molIdx < 0 || molIdx >= static_cast<int>(conformerCounts.size())) {
    throw std::out_of_range("Invalid molecule index: " + std::to_string(molIdx));
  }

  int outStart  = tfdOutputStarts[molIdx];
  int outEnd    = tfdOutputStarts[molIdx + 1];
  int numValues = outEnd - outStart;

  if (numValues == 0) {
    return {};
  }

  // Copy from device to host
  std::vector<float> hostFloats(numValues);
  const_cast<AsyncDeviceVector<float>&>(tfdValues).copyToHost(hostFloats.data(), numValues, 0, outStart);
  cudaStreamSynchronize(tfdValues.stream());

  // Convert to double
  std::vector<double> result(numValues);
  for (int i = 0; i < numValues; ++i) {
    result[i] = static_cast<double>(hostFloats[i]);
  }

  return result;
}

std::vector<std::vector<double>> TFDGpuResult::extractAll() const {
  int                              numMolecules = static_cast<int>(conformerCounts.size());
  std::vector<std::vector<double>> results(numMolecules);

  if (tfdValues.size() == 0) {
    return results;
  }

  // Copy all data at once
  std::vector<float> allHostFloats(tfdValues.size());
  const_cast<AsyncDeviceVector<float>&>(tfdValues).copyToHost(allHostFloats.data(), tfdValues.size());
  cudaStreamSynchronize(tfdValues.stream());

  // Extract per-molecule results
  for (int m = 0; m < numMolecules; ++m) {
    int outStart  = tfdOutputStarts[m];
    int outEnd    = tfdOutputStarts[m + 1];
    int numValues = outEnd - outStart;

    results[m].resize(numValues);
    for (int i = 0; i < numValues; ++i) {
      results[m][i] = static_cast<double>(allHostFloats[outStart + i]);
    }
  }

  return results;
}

// ========== TFDGpuGenerator ==========

TFDGpuGenerator::TFDGpuGenerator() : stream_() {
  device_.setStream(stream_.stream());
}

TFDGpuResult TFDGpuGenerator::GetTFDMatricesGpuBuffer(const std::vector<const RDKit::ROMol*>& mols,
                                                      const TFDComputeOptions&                options) {
  TFDGpuResult result;

  if (mols.empty()) {
    return result;
  }

  // Build host system data (CPU preprocessing)
  TFDSystemHost system = buildTFDSystem(mols, options);

  // Store metadata for result extraction
  result.tfdOutputStarts = system.tfdOutputStarts;
  result.conformerCounts.reserve(mols.size());
  for (size_t i = 0; i < mols.size(); ++i) {
    int numConf = system.molConformerStarts[i + 1] - system.molConformerStarts[i];
    result.conformerCounts.push_back(numConf);
  }

  // Handle edge case: no TFD outputs
  if (system.totalTFDOutputs() == 0) {
    return result;
  }

  cudaStream_t stream = stream_.stream();

  // Transfer to device (handles resize + copy + output buffer allocation)
  transferToDevice(system, device_, stream);

  // Launch dihedral kernel
  launchDihedralKernel(system.totalDihedralWorkItems(),
                       device_.positions.data(),
                       device_.confPositionStarts.data(),
                       device_.torsionAtoms.data(),
                       device_.dihedralConfIdx.data(),
                       device_.dihedralTorsIdx.data(),
                       device_.dihedralOutIdx.data(),
                       device_.dihedralAngles.data(),
                       stream);

  // Launch TFD matrix kernel
  launchTFDMatrixKernel(system.totalTFDWorkItems(),
                        device_.dihedralAngles.data(),
                        device_.torsionWeights.data(),
                        device_.torsionMaxDevs.data(),
                        device_.quartetStarts.data(),
                        device_.torsionTypes.data(),
                        device_.tfdAnglesI.data(),
                        device_.tfdAnglesJ.data(),
                        device_.tfdTorsStart.data(),
                        device_.tfdNumTorsions.data(),
                        device_.tfdOutIdx.data(),
                        device_.tfdOutput.data(),
                        stream);

  // Move output to result (transfer ownership of GPU memory)
  result.tfdValues = std::move(device_.tfdOutput);

  // Reallocate the output buffer for next use
  device_.tfdOutput = AsyncDeviceVector<float>();
  device_.tfdOutput.setStream(stream);

  return result;
}

std::vector<std::vector<double>> TFDGpuGenerator::GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                                 const TFDComputeOptions&                options) {
  TFDGpuResult gpuResult = GetTFDMatricesGpuBuffer(mols, options);
  return gpuResult.extractAll();
}

std::vector<double> TFDGpuGenerator::GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  std::vector<const RDKit::ROMol*> mols    = {&mol};
  auto                             results = GetTFDMatrices(mols, options);

  if (results.empty()) {
    return {};
  }
  return results[0];
}

}  // namespace nvMolKit
