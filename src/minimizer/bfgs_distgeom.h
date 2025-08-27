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

#ifndef NVMOLKIT_BFGS_DISTGEOM_H
#define NVMOLKIT_BFGS_DISTGEOM_H

#include <vector>

#include "dist_geom.h"
#include "etkdg_impl.h"

namespace nvMolKit::DistGeom {

//! Minimize the distance geometry system using BFGS
//! @param molSystemHost The host-side molecular system
//! @param molSystemDevice The device-side molecular system
//! @param context The ETKDG context containing positions and other data
//! @param maxIters Maximum number of iterations for minimization (default: 200)
//! @param gradTol Gradient tolerance for convergence (default: 1e-4)
//! @param gradTol Gradient tolerance for convergence (default: 1e-4)
//! @param repeatUntilConverged If true, will repeat minimization until all systems converge (default: false)
//! @param stream CUDA stream to use for operations (default: nullptr, uses default stream)

void DistGeomMinimizeBFGS(BatchedMolecularSystemHost&    molSystemHost,
                          BatchedMolecularDeviceBuffers& molSystemDevice,
                          detail::ETKDGContext&          context,
                          int                            maxIters             = 200,
                          double                         gradTol              = 1e-4,
                          bool                           repeatUntilConverged = false,
                          cudaStream_t                   stream               = nullptr);

}  // namespace nvMolKit::DistGeom

#endif  // NVMOLKIT_BFGS_DISTGEOM_H
