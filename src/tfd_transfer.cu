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

#include "tfd_common.h"

namespace nvMolKit {

void TFDSystemDevice::setStream(cudaStream_t stream) {
  // Dihedral kernel inputs
  positions.setStream(stream);
  confPositionStarts.setStream(stream);
  torsionAtoms.setStream(stream);
  dihedralConfIdx.setStream(stream);
  dihedralTorsIdx.setStream(stream);
  dihedralOutIdx.setStream(stream);

  // TFD matrix kernel inputs
  torsionWeights.setStream(stream);
  torsionMaxDevs.setStream(stream);
  quartetStarts.setStream(stream);
  torsionTypes.setStream(stream);
  tfdAnglesI.setStream(stream);
  tfdAnglesJ.setStream(stream);
  tfdTorsStart.setStream(stream);
  tfdNumTorsions.setStream(stream);
  tfdOutIdx.setStream(stream);

  // Output buffers
  dihedralAngles.setStream(stream);
  tfdOutput.setStream(stream);
}

void transferToDevice(const TFDSystemHost& host, TFDSystemDevice& device, cudaStream_t stream) {
  device.setStream(stream);

  // Positions and conformer offsets
  device.positions.setFromVector(host.positions);
  device.confPositionStarts.setFromVector(host.confPositionStarts);

  // Flatten torsion atoms from array<int,4> to int*4
  std::vector<int> flatTorsionAtoms;
  flatTorsionAtoms.reserve(host.torsionAtoms.size() * 4);
  for (const auto& quartet : host.torsionAtoms) {
    flatTorsionAtoms.push_back(quartet[0]);
    flatTorsionAtoms.push_back(quartet[1]);
    flatTorsionAtoms.push_back(quartet[2]);
    flatTorsionAtoms.push_back(quartet[3]);
  }
  device.torsionAtoms.setFromVector(flatTorsionAtoms);

  // Flattened dihedral work items
  device.dihedralConfIdx.setFromVector(host.dihedralConfIdx);
  device.dihedralTorsIdx.setFromVector(host.dihedralTorsIdx);
  device.dihedralOutIdx.setFromVector(host.dihedralOutIdx);

  // TFD matrix kernel inputs
  device.torsionWeights.setFromVector(host.torsionWeights);
  device.torsionMaxDevs.setFromVector(host.torsionMaxDevs);

  // Multi-quartet support
  device.quartetStarts.setFromVector(host.quartetStarts);
  std::vector<uint8_t> types(host.torsionTypes.size());
  for (size_t i = 0; i < host.torsionTypes.size(); ++i) {
    types[i] = static_cast<uint8_t>(host.torsionTypes[i]);
  }
  device.torsionTypes.setFromVector(types);

  device.tfdAnglesI.setFromVector(host.tfdAnglesI);
  device.tfdAnglesJ.setFromVector(host.tfdAnglesJ);
  device.tfdTorsStart.setFromVector(host.tfdTorsStart);
  device.tfdNumTorsions.setFromVector(host.tfdNumTorsions);
  device.tfdOutIdx.setFromVector(host.tfdOutIdx);

  // Allocate output buffers
  int totalDihedrals = host.totalDihedrals();
  if (static_cast<int>(device.dihedralAngles.size()) < totalDihedrals) {
    device.dihedralAngles.resize(totalDihedrals);
  }

  int totalTFDOutputs = host.totalTFDOutputs();
  if (static_cast<int>(device.tfdOutput.size()) < totalTFDOutputs) {
    device.tfdOutput.resize(totalTFDOutputs);
  }
}

}  // namespace nvMolKit
