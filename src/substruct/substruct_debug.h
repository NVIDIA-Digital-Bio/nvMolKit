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

#ifndef NVMOLKIT_SUBSTRUCT_DEBUG_H
#define NVMOLKIT_SUBSTRUCT_DEBUG_H

namespace nvMolKit {
constexpr bool kDebugAll = false;

/// @name Substructure Search Debug Flags
/// @{
constexpr bool kDebugGSI             = kDebugAll || false;  ///< Debug output in gsiBFSSearchGPU
constexpr bool kDebugDumpLabelMatrix = kDebugAll || false;  ///< Dump full label matrices after recursive preprocessing
constexpr bool kDebugPaintRecursive  = kDebugAll || false;  ///< Debug recursive bit painting kernel
constexpr bool kDebugLabelMatrix     = kDebugAll || false;  ///< Debug label matrix population
constexpr bool kDebugBoolTreeBuild   = kDebugAll || false;  ///< Debug boolean tree construction
constexpr bool kDebugEdgeConsistency = kDebugAll || false;  ///< Debug edge consistency checking
/// @}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_DEBUG_H
