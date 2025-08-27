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

#ifndef MORGAN_FINGERPRINT_COMMON_H
#define MORGAN_FINGERPRINT_COMMON_H

#include <GraphMol/ROMol.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace nvMolKit {

constexpr int kMaxBondsPerAtom = 8;

//! Device to run fingerprint on
enum class FingerprintComputeBackend {
  CPU,
  GPU,
};

//! Output control options for Morgan fingerprint generation
struct MorganFingerprintOptions {
  std::uint32_t radius = 0;
  std::uint32_t fpSize = 2048;
};

//! Performance toggles for fingerprint computation
struct FingerprintComputeOptions {
  FingerprintComputeBackend backend       = FingerprintComputeBackend::CPU;
  //! Number of CPU threads to use for preprocessing, and for compute if backend is CPU
  std::optional<int>        numCpuThreads = std::nullopt;
  //! Number of molecules to preprocess, send to GPU, and dispatch kernels for
  std::optional<int>        gpuBatchSize  = std::nullopt;
};

struct InvariantsInfo {
  std::vector<std::uint32_t> atomInvariants;
  std::vector<std::uint32_t> bondInvariants;
  std::vector<std::int16_t>  bondAtomIndices;
  std::vector<std::int16_t>  bondOtherAtomIndices;
};

class MorganInvariantsGenerator {
 public:
  MorganInvariantsGenerator() = default;

  const InvariantsInfo& GetInvariants() const { return invariantsInfo_; }
  void                  ComputeInvariants(const std::vector<const RDKit::ROMol*>& mols, size_t maxAtoms);
  // Compute invariants directly into caller-provided buffers to avoid intermediate copies.
  // The output buffers must have sizes at least: mols.size() * maxAtoms (atom/bond invariants)
  // and mols.size() * maxAtoms * kMaxBondsPerAtom (bond indices). Unused entries are zeroed
  // for invariants and set to -1 for bond index arrays.
  static void           ComputeInvariantsInto(const std::vector<const RDKit::ROMol*>& mols,
                                              size_t                                  maxAtoms,
                                              std::uint32_t*                          atomInvariantsOut,
                                              std::uint32_t*                          bondInvariantsOut,
                                              std::int16_t*                           bondAtomIndicesOut,
                                              std::int16_t*                           bondOtherAtomIndicesOut);

 private:
  InvariantsInfo invariantsInfo_;
};

}  // namespace nvMolKit

#endif  // MORGAN_FINGERPRINT_COMMON_H
