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

#ifndef NVMOLKIT_SUBSTRUCT_TEMPLATE_CONFIG_H
#define NVMOLKIT_SUBSTRUCT_TEMPLATE_CONFIG_H

#include <cstddef>
#include <cstdint>

namespace nvMolKit {

/**
 * @brief Enumeration of valid template configurations.
 *
 * Each configuration specifies (MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom).
 * Valid combinations require Target >= Query. Total: 24 configurations.
 *
 * Naming: Config_T{targets}_Q{queries}_B{bonds}
 */
enum class SubstructTemplateConfig : uint8_t {
  // Target 32, Query 16
  Config_T32_Q16_B4 = 0,
  Config_T32_Q16_B6,
  Config_T32_Q16_B8,
  // Target 32, Query 32
  Config_T32_Q32_B4,
  Config_T32_Q32_B6,
  Config_T32_Q32_B8,
  // Target 64, Query 16
  Config_T64_Q16_B4,
  Config_T64_Q16_B6,
  Config_T64_Q16_B8,
  // Target 64, Query 32
  Config_T64_Q32_B4,
  Config_T64_Q32_B6,
  Config_T64_Q32_B8,
  // Target 64, Query 64
  Config_T64_Q64_B4,
  Config_T64_Q64_B6,
  Config_T64_Q64_B8,
  // Target 128, Query 16
  Config_T128_Q16_B4,
  Config_T128_Q16_B6,
  Config_T128_Q16_B8,
  // Target 128, Query 32
  Config_T128_Q32_B4,
  Config_T128_Q32_B6,
  Config_T128_Q32_B8,
  // Target 128, Query 64
  Config_T128_Q64_B4,
  Config_T128_Q64_B6,
  Config_T128_Q64_B8,

  NumConfigs  ///< Total number of configurations (24)
};

/**
 * @brief Compile-time properties for a template configuration.
 */
struct TemplateConfigProperties {
  int         maxTargetAtoms;
  int         maxQueryAtoms;
  int         maxBondsPerAtom;
  std::size_t labelMatrixBits;
  std::size_t labelMatrixWords;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_TEMPLATE_CONFIG_H
