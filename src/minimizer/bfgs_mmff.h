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

#ifndef NVMOLKIT_BFGS_MMFF_H
#define NVMOLKIT_BFGS_MMFF_H

#include <optional>
#include <vector>

#include "../hardware_options.h"
#include "bfgs_minimize.h"
#include "device_coord_result.h"
#include "forcefield_constraints.h"
#include "mmff_properties.h"

namespace RDKit {
class ROMol;
}

namespace nvMolKit::MMFF {

//! \brief Optimize conformers for multiple molecules using MMFF force field
//! \param mols The molecules to optimize
//! \param maxIters The maximum number of iterations to perform
//! \param nonBondedThreshold The radius threshold for non-bonded interactions
//! \param perfOptions Performance tuning options (threading, batching)
//! \param backend The BFGS backend to use (BATCHED, PER_MOLECULE, or HYBRID which auto-selects)
//! \return A vector of vectors of energies, where each inner vector contains energies for conformers of one molecule
std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                                int                         maxIters    = 200,
                                                                const MMFFProperties&       properties  = {},
                                                                const BatchHardwareOptions& perfOptions = {},
                                                                BfgsBackend backend = BfgsBackend::HYBRID);

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>&        mols,
                                                                int                                maxIters,
                                                                const std::vector<MMFFProperties>& properties,
                                                                const BatchHardwareOptions&        perfOptions = {},
                                                                BfgsBackend backend = BfgsBackend::HYBRID);

//! \brief Result from constraint-aware MMFF minimization.
//!
//! In CoordinateOutput::RDKIT_CONFORMERS mode, @ref energies and @ref converged are populated and
//! @ref device is empty; coordinates are written back into each input molecule's RDKit conformer
//! list. In CoordinateOutput::DEVICE mode, @ref device holds the on-GPU coordinates / energies /
//! convergence flags collected onto the chosen target GPU and the host-side vectors are empty.
struct MMFFMinimizeResult {
  std::vector<std::vector<double>>  energies;   //!< Per-molecule, per-conformer final energies (RDKIT mode only).
  std::vector<std::vector<int8_t>>  converged;  //!< Per-molecule, per-conformer convergence flags (RDKIT mode only).
  std::optional<DeviceCoordResult>  device;     //!< Populated when output==DEVICE.
};

//! \brief Optimize with per-molecule constraints and return convergence status.
//! \param mols The molecules to optimize (positions written back in-place in RDKIT_CONFORMERS mode).
//! \param maxIters Maximum BFGS iterations.
//! \param gradTol Gradient convergence tolerance.
//! \param properties Per-molecule MMFF settings.
//! \param constraints Per-molecule constraint specifications.
//! \param perfOptions Hardware and batching configuration.
//! \param backend BFGS backend selection.
//! \param output Whether to write coordinates back into RDKit conformers (default) or return them
//!               on-device as a DeviceCoordResult.
//! \param targetGpu In DEVICE mode, the GPU to consolidate the result onto. -1 selects the first
//!                  configured execution GPU (or device 0).
//! \return Either host-side energies/convergence (RDKIT mode) or a populated `device` field (DEVICE mode).
MMFFMinimizeResult MMFFMinimizeMoleculesConfs(
  std::vector<RDKit::ROMol*>&                                  mols,
  int                                                          maxIters    = 200,
  double                                                       gradTol     = 1e-4,
  const std::vector<MMFFProperties>&                           properties  = {},
  const std::vector<ForceFieldConstraints::PerMolConstraints>& constraints = {},
  const BatchHardwareOptions&                                  perfOptions = {},
  BfgsBackend                                                  backend     = BfgsBackend::HYBRID,
  CoordinateOutput                                             output      = CoordinateOutput::RDKIT_CONFORMERS,
  int                                                          targetGpu   = -1);

}  // namespace nvMolKit::MMFF
#endif  // NVMOLKIT_BFGS_MMFF_H
