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

#ifndef NVMOLKIT_BFGS_TYPES_H
#define NVMOLKIT_BFGS_TYPES_H

namespace nvMolKit {

/// Force field type enum for BFGS minimization
enum class ForceFieldType {
  MMFF = 0,
  ETK  = 1,
  DG   = 2
};

/// Debug level for BFGS minimization
enum class DebugLevel {
  NONE     = 0,
  STEPWISE = 1,
};

/// Backend implementation type for BFGS minimization
enum class BfgsBackend {
  BATCHED      = 0,  //!< Original batched implementation
  PER_MOLECULE = 1   //!< Per-molecule kernel implementation
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_BFGS_TYPES_H
