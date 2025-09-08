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

#ifndef MORGAN_FINGERPRINT_CPU_H
#define MORGAN_FINGERPRINT_CPU_H

#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/ROMol.h>

#include <memory>

#include "morgan_fingerprint_common.h"

namespace nvMolKit {

class MorganFingerprintCpuGenerator {
 public:
  MorganFingerprintCpuGenerator(std::uint32_t radius, std::uint32_t fpSize);

  std::unique_ptr<ExplicitBitVect> GetFingerprint(
    const RDKit::ROMol&                      mol,
    std::optional<FingerprintComputeOptions> computeOptions = std::nullopt) const;
  std::vector<std::unique_ptr<ExplicitBitVect>> GetFingerprints(
    const std::vector<const RDKit::ROMol*>&  mols,
    std::optional<FingerprintComputeOptions> computeOptions = std::nullopt) const;

 private:
  std::uint32_t radius_;
  std::uint32_t fpSize_;
};

namespace internal {
std::unique_ptr<ExplicitBitVect> getFingerprintImpl(const RDKit::ROMol& mol,
                                                    const std::uint32_t radius,
                                                    const std::uint32_t fpSize);
}  // namespace internal

}  // namespace nvMolKit

#endif  // MORGAN_FINGERPRINT_CPU_H
